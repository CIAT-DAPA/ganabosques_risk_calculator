import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from utils import (
    area_ha,
    compute_intersection_area_ha_via_sindex,
    normalize_farm_id,
    parse_quarter_from_period,
    parse_year_periods,
    setup_logging,
)


# ===================== GEOMETRÍA =====================


def test_area_ha_converts_square_meters_to_hectares():
    # 100m x 100m = 10.000 m2 = 1 ha
    assert area_ha(box(0, 0, 100, 100)) == pytest.approx(1.0)
    assert area_ha(box(0, 0, 0, 0)) == 0.0


def test_compute_intersection_area_returns_zero_for_missing_or_empty_mask():
    farm = box(0, 0, 100, 100)
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3116")

    assert compute_intersection_area_ha_via_sindex(farm, None) == 0.0
    assert compute_intersection_area_ha_via_sindex(farm, empty) == 0.0


def test_compute_intersection_area_returns_zero_when_no_candidates_in_bbox():
    farm = box(0, 0, 100, 100)
    far_away = gpd.GeoDataFrame(
        {"geometry": [box(10_000, 10_000, 10_100, 10_100)]}, crs="EPSG:3116"
    )

    assert compute_intersection_area_ha_via_sindex(farm, far_away) == 0.0


def test_compute_intersection_area_returns_zero_when_candidates_do_not_intersect():
    # Un anillo cuyo hueco contiene al predio: el bounding box se solapa (por eso
    # el índice espacial lo propone como candidato) pero la geometría no interseca,
    # así que el filtro `intersects` lo descarta.
    farm = box(0, 0, 100, 100)
    ring = Polygon(
        [(-10, -10), (110, -10), (110, 110), (-10, 110)],
        [[(-5, -5), (105, -5), (105, 105), (-5, 105)]],
    )
    mask = gpd.GeoDataFrame({"geometry": [ring]}, crs="EPSG:3116")

    assert compute_intersection_area_ha_via_sindex(farm, mask) == 0.0


def test_compute_intersection_area_unions_overlapping_masks():
    farm = box(0, 0, 100, 100)
    # Dos máscaras que se solapan entre sí: la unión evita el doble conteo.
    mask = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 50, 100), box(25, 0, 75, 100)]}, crs="EPSG:3116"
    )

    area = compute_intersection_area_ha_via_sindex(farm, mask)

    # 75m x 100m = 7.500 m2 = 0,75 ha (no 1,25 ha, que sería el doble conteo)
    assert area == pytest.approx(0.75)


class _FakeMaskFrame:
    """GeoDataFrame mínimo que permite forzar las ramas de error de la función."""

    def __init__(self, geoms, sindex_raises=False):
        self._geoms = list(geoms)
        self._sindex_raises = sindex_raises
        self.empty = not self._geoms

    def __len__(self):
        return len(self._geoms)

    @property
    def sindex(self):
        if self._sindex_raises:
            raise RuntimeError("sin índice espacial")
        raise AssertionError("no debería consultarse")

    @property
    def iloc(self):
        outer = self

        class _ILoc:
            def __getitem__(self, idx):
                return _FakeMaskFrame([outer._geoms[i] for i in idx])

        return _ILoc()

    def __getitem__(self, mask):
        return self

    def intersects(self, geom):
        return [True] * len(self._geoms)

    @property
    def geometry(self):
        return self._geoms


class _ExplodingGeometry:
    """Geometría que falla al intersecar pero se puede reparar con buffer(0)."""

    def __init__(self, repaired=None):
        self._repaired = repaired

    def buffer(self, distance):
        if self._repaired is None:
            raise RuntimeError("no se puede reparar")
        return self._repaired


def test_compute_intersection_area_falls_back_when_sindex_is_unavailable():
    farm = box(0, 0, 100, 100)
    frame = _FakeMaskFrame([box(0, 0, 50, 100)], sindex_raises=True)

    area = compute_intersection_area_ha_via_sindex(farm, frame)

    assert area == pytest.approx(0.5)


def test_compute_intersection_area_repairs_invalid_geometries_with_buffer():
    farm = box(0, 0, 100, 100)
    frame = _FakeMaskFrame(
        [_ExplodingGeometry(repaired=box(0, 0, 50, 100))], sindex_raises=True
    )

    area = compute_intersection_area_ha_via_sindex(farm, frame)

    assert area == pytest.approx(0.5)


def test_compute_intersection_area_skips_geometries_that_cannot_be_repaired():
    farm = box(0, 0, 100, 100)
    frame = _FakeMaskFrame([_ExplodingGeometry(repaired=None)], sindex_raises=True)

    assert compute_intersection_area_ha_via_sindex(farm, frame) == 0.0


def test_compute_intersection_area_returns_zero_when_every_intersection_is_empty():
    farm = box(0, 0, 100, 100)
    # `intersects` está forzado a True, pero la intersección real es vacía.
    frame = _FakeMaskFrame([box(500, 500, 600, 600)], sindex_raises=True)

    assert compute_intersection_area_ha_via_sindex(farm, frame) == 0.0


# ===================== PERÍODOS =====================


def test_parse_year_periods_keeps_only_valid_ranges(caplog):
    assert parse_year_periods("") == []
    assert parse_year_periods(None) == []
    assert parse_year_periods("2017-2018") == ["2017-2018"]
    assert parse_year_periods("2017 - 2018") == ["2017-2018"]
    assert parse_year_periods("2017-2018, 2018-2019") == ["2017-2018", "2018-2019"]

    with caplog.at_level(logging.WARNING):
        # Un año suelto no es un período válido: se descarta con advertencia.
        assert parse_year_periods("2017, 2018-2019") == ["2018-2019"]

    assert "2017" in caplog.text


def test_parse_quarter_from_period_extracts_only_valid_quarters():
    assert parse_quarter_from_period("202401") == 1
    assert parse_quarter_from_period("202404") == 4
    # 05 no es un trimestre válido
    assert parse_quarter_from_period("202405") is None
    assert parse_quarter_from_period("202400") is None
    # Formatos que no son YYYYQQ
    assert parse_quarter_from_period("2024-2025") is None
    assert parse_quarter_from_period("2024") is None
    assert parse_quarter_from_period("abcdef") is None
    assert parse_quarter_from_period("202401", period_type="nad") == 1


# ===================== LOGGING =====================


def test_setup_logging_configures_handlers_and_level(tmp_path):
    log_file = tmp_path / "pipeline.log"
    root = logging.getLogger()
    previous = list(root.handlers)
    previous_level = root.level

    try:
        setup_logging(log_level="DEBUG", log_file=str(log_file))

        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2
        logging.getLogger("test").debug("mensaje de prueba")
        for handler in root.handlers:
            handler.flush()
        assert "mensaje de prueba" in log_file.read_text(encoding="utf-8")

        # Una segunda llamada reemplaza los handlers en vez de duplicarlos.
        setup_logging(log_level="INFO", log_file=str(log_file))
        assert len(root.handlers) == 2
        assert root.level == logging.INFO

        # Nivel desconocido: cae en WARNING.
        setup_logging(log_level="NIVEL_INEXISTENTE", log_file=str(log_file))
        assert root.level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in previous:
            root.addHandler(handler)
        root.setLevel(previous_level)


def test_setup_logging_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    previous = list(root.handlers)
    previous_level = root.level

    try:
        setup_logging()
        assert root.level == logging.WARNING
        assert (tmp_path / "app.log").exists()
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in previous:
            root.addHandler(handler)
        root.setLevel(previous_level)


# ===================== NORMALIZACIÓN DE IDS =====================


def test_normalize_farm_id_removes_prefix_and_padding():
    assert normalize_farm_id("FARM_ID_abc123") == "abc123"
    assert normalize_farm_id("farm_id_abc123") == "abc123"
    assert normalize_farm_id("  FARM_ID_00123  ") == "123"
    assert normalize_farm_id("123.0") == "123"
    assert normalize_farm_id("00456") == "456"
    assert normalize_farm_id(456) == "456"
    assert normalize_farm_id(456.0) == "456"


def test_normalize_farm_id_handles_null_and_nan_values():
    assert normalize_farm_id(None) == ""
    assert normalize_farm_id(float("nan")) == ""
    assert normalize_farm_id(pd.NA) == ""
    assert normalize_farm_id(np.nan) == ""
    assert normalize_farm_id("nan") == ""
    assert normalize_farm_id("NaN") == ""


def test_normalize_farm_id_respects_disabled_flags():
    assert normalize_farm_id("00123", strip_leading_zeros=False) == "00123"
    assert normalize_farm_id("123.0", strip_dot_zero=False) == "123.0"
    # Sin quitar el ".0" tampoco se quitan ceros: ya no es un dígito puro.
    assert normalize_farm_id("00123.0", strip_dot_zero=False) == "00123.0"


def test_normalize_farm_id_leaves_non_numeric_values_untouched():
    assert normalize_farm_id("abc.0") == "abc.0"
    assert normalize_farm_id("A-001") == "A-001"
    assert normalize_farm_id("") == ""


def test_normalize_farm_id_survives_unicode_digits():
    # "²".isdigit() es True pero int() falla: la conversión debe ignorarse
    # en silencio y devolver el valor original.
    assert normalize_farm_id("²") == "²"
