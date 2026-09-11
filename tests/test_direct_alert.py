import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import direct_alert


CRS = "EPSG:3116"


# ===================== utilidades de fixtures =====================


def _write_raster(path, values, crs=CRS, origin=(0, 100), resolution=10):
    """Escribe un raster pequeño; con CRS proyectado el píxel mide resolution x resolution m."""
    arr = np.array(values, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(origin[0], origin[1], resolution, resolution),
    ) as dst:
        dst.write(arr, 1)
    return path


def _write_geojson(path, geom, crs=CRS, extra=None):
    data = {"geometry": [geom]}
    if extra:
        data.update(extra)
    gpd.GeoDataFrame(data, crs=crs).to_file(path, driver="GeoJSON")
    return path


# ===================== norm / format_placeholders =====================


def test_norm_normalizes_separators_and_passthrough():
    assert direct_alert.norm("C:/tmp\\a\\b") == os.path.normpath("C:/tmp/a/b")
    assert direct_alert.norm(None) is None
    assert direct_alert.norm("") == ""
    assert direct_alert.norm(Path("a/b")) == os.path.normpath("a/b")


def test_format_placeholders_supports_both_cases():
    ctx = {"PERIODO": "annual", "YEARS": "2024"}

    assert direct_alert.format_placeholders("{PERIODO}/{YEARS}", ctx) == "annual/2024"
    assert direct_alert.format_placeholders("{periodo}/{years}", ctx) == "annual/2024"
    assert direct_alert.format_placeholders("", ctx) == ""
    assert direct_alert.format_placeholders(None, ctx) is None
    # Un placeholder desconocido se devuelve sin sustituir.
    assert direct_alert.format_placeholders("{OTRO}", ctx) == "{OTRO}"


# ===================== apply_file_range =====================


@pytest.fixture
def files():
    return [f"f{i}.geojson" for i in range(1, 6)]


def test_apply_file_range_returns_everything_for_blank_expression(files):
    assert direct_alert.apply_file_range(files, "") == files
    assert direct_alert.apply_file_range(files, None) == files
    assert direct_alert.apply_file_range(files, "   ") == files


def test_apply_file_range_selects_inclusive_ranges(files):
    assert direct_alert.apply_file_range(files, "2:4") == files[1:4]
    # Los extremos invertidos se reordenan solos.
    assert direct_alert.apply_file_range(files, "4:2") == files[1:4]


def test_apply_file_range_clamps_out_of_bounds(files):
    assert direct_alert.apply_file_range(files, "0:99") == files


def test_apply_file_range_single_number_means_first_n(files):
    assert direct_alert.apply_file_range(files, "3") == files[:3]


def test_apply_file_range_multiple_tokens_are_positions(files):
    assert direct_alert.apply_file_range(files, "1,3") == [files[0], files[2]]
    # Las posiciones fuera de rango se ignoran.
    assert direct_alert.apply_file_range(files, "1,99") == [files[0]]


def test_apply_file_range_deduplicates_preserving_order(files):
    assert direct_alert.apply_file_range(files, "1:2,2:3") == files[:3]


def test_apply_file_range_ignores_invalid_tokens(files, caplog):
    with caplog.at_level(logging.WARNING):
        assert direct_alert.apply_file_range(files, "abc,2") == [files[1]]

    assert "abc" in caplog.text


def test_apply_file_range_falls_back_to_all_when_nothing_selected(files, caplog):
    with caplog.at_level(logging.WARNING):
        assert direct_alert.apply_file_range(files, "abc") == files

    assert "se procesará todo" in caplog.text


# ===================== CRS helpers =====================


def test_crs_eq_compares_by_epsg_and_falls_back_to_strings():
    assert direct_alert._crs_eq("EPSG:3116", "EPSG:3116") is True
    assert direct_alert._crs_eq("EPSG:3116", "EPSG:4326") is False
    # Cadenas ilegibles: comparación literal.
    assert direct_alert._crs_eq("no-es-un-crs", "no-es-un-crs") is True
    assert direct_alert._crs_eq("no-es-un-crs", "otra-cosa") is False


def test_ensure_raster_crs_accepts_matching_raster(tmp_path, capsys):
    path = _write_raster(tmp_path / "r.tif", [[1, 2], [2, 1]])

    direct_alert.ensure_raster_crs(str(path), CRS)

    assert "detectado" in capsys.readouterr().out


def test_ensure_raster_crs_warns_for_different_crs(tmp_path, caplog):
    path = _write_raster(tmp_path / "r.tif", [[1, 2]], crs="EPSG:4326", origin=(-74, 4), resolution=0.001)

    with caplog.at_level(logging.WARNING):
        direct_alert.ensure_raster_crs(str(path), CRS)

    assert "WarpedVRT" in caplog.text


def test_ensure_raster_crs_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raster no encontrado"):
        direct_alert.ensure_raster_crs(str(tmp_path / "no.tif"), CRS)


def test_ensure_raster_crs_raises_without_crs(tmp_path):
    path = tmp_path / "sin_crs.tif"
    arr = np.array([[1, 2]], dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=2, count=1, dtype="uint8",
        transform=from_origin(0, 10, 1, 1),
    ) as dst:
        dst.write(arr, 1)

    with pytest.raises(ValueError, match="no tiene CRS"):
        direct_alert.ensure_raster_crs(str(path), CRS)


# ===================== listado y clasificación de archivos =====================


def test_list_files_walks_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.geojson").write_text("{}", encoding="utf-8")
    (tmp_path / "sub" / "b.tif").write_bytes(b"x")

    assert len(direct_alert._list_files(str(tmp_path))) == 2


def test_is_vector_and_is_raster():
    assert direct_alert._is_vector("x.shp") is True
    assert direct_alert._is_vector("x.GEOJSON") is True
    assert direct_alert._is_vector("x.gpkg") is True
    assert direct_alert._is_vector("x.tif") is False
    assert direct_alert._is_raster("x.tif") is True
    assert direct_alert._is_raster("x.TIFF") is True
    assert direct_alert._is_raster("x.shp") is False


# ===================== load_vector_from_paths / load_bundle =====================


def test_load_vector_from_paths_merges_and_reprojects(tmp_path):
    a = _write_geojson(tmp_path / "a.geojson", box(0, 0, 10, 10))
    b = _write_geojson(tmp_path / "b.geojson", box(-74, 4, -73.99, 4.01), crs="EPSG:4326")

    gdf = direct_alert.load_vector_from_paths([str(a), str(b)], CRS)

    assert len(gdf) == 2
    assert gdf.crs.to_string() == CRS


def test_load_vector_from_paths_skips_empty_and_crsless_layers(tmp_path, monkeypatch):
    a = _write_geojson(tmp_path / "a.geojson", box(0, 0, 10, 10))
    b = _write_geojson(tmp_path / "b.geojson", box(20, 0, 30, 10))

    respuestas = iter(
        [
            gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS),
            gpd.GeoDataFrame({"geometry": [box(20, 0, 30, 10)]}),  # sin CRS
        ]
    )
    monkeypatch.setattr(direct_alert.gpd, "read_file", lambda *a, **k: next(respuestas))

    assert direct_alert.load_vector_from_paths([str(a), str(b)], CRS) is None


def test_load_vector_from_paths_skips_unreadable_files(tmp_path):
    a = _write_geojson(tmp_path / "a.geojson", box(0, 0, 10, 10))
    roto = tmp_path / "b.geojson"
    roto.write_text("{ roto", encoding="utf-8")

    gdf = direct_alert.load_vector_from_paths([str(a), str(roto)], CRS)

    assert len(gdf) == 1


def test_load_vector_from_paths_returns_none_without_paths():
    assert direct_alert.load_vector_from_paths([], CRS) is None


def test_load_vector_from_paths_returns_none_when_all_geometries_are_empty(tmp_path, monkeypatch):
    a = _write_geojson(tmp_path / "a.geojson", box(0, 0, 10, 10))
    vacia = gpd.GeoDataFrame({"geometry": [box(0, 0, 0, 0).buffer(0)]}, crs=CRS)
    monkeypatch.setattr(direct_alert.gpd, "read_file", lambda *a, **k: vacia)

    assert direct_alert.load_vector_from_paths([str(a)], CRS) is None


def test_load_bundle_lists_vectors_and_rasters(tmp_path):
    _write_geojson(tmp_path / "a.geojson", Point(0, 0))
    _write_raster(tmp_path / "r.tif", [[1, 2], [3, 4]])

    gdf, rasters, summary = direct_alert.load_bundle(
        str(tmp_path), {"PERIODO": "annual", "YEARS": "2024"}, CRS
    )

    assert summary["vec_files"] == 1
    assert summary["ras_files"] == 1
    assert gdf is not None
    assert len(rasters) == 1


def test_load_bundle_returns_empty_for_missing_template():
    gdf, rasters, summary = direct_alert.load_bundle(None, {}, CRS)

    assert (gdf, rasters) == (None, [])
    assert summary["folder"] is None


def test_load_bundle_returns_empty_for_missing_folder(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        gdf, rasters, _ = direct_alert.load_bundle(str(tmp_path / "no"), {}, CRS)

    assert (gdf, rasters) == (None, [])
    assert "No se encontró carpeta" in caplog.text


def test_load_bundle_without_vectors_returns_none_gdf(tmp_path):
    _write_raster(tmp_path / "r.tif", [[1]])

    gdf, rasters, summary = direct_alert.load_bundle(str(tmp_path), {}, CRS)

    assert gdf is None
    assert summary["ras_files"] == 1


# ===================== filtros y apertura de rasters =====================


def test_filter_rasters_same_crs_accepts_all_readable(tmp_path, caplog):
    igual = _write_raster(tmp_path / "a.tif", [[1]])
    distinto = _write_raster(
        tmp_path / "b.tif", [[1]], crs="EPSG:4326", origin=(-74, 4), resolution=0.001
    )
    roto = tmp_path / "c.tif"
    roto.write_bytes(b"no es un tif")

    with caplog.at_level(logging.WARNING):
        ok = direct_alert._filter_rasters_same_crs([str(igual), str(distinto), str(roto)], CRS)

    # Los dos legibles se aceptan; el corrupto se descarta.
    assert len(ok) == 2
    assert "virtualmente" in caplog.text


def test_filter_rasters_same_crs_accepts_raster_without_crs(tmp_path):
    path = tmp_path / "sin_crs.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=1, count=1, dtype="uint8",
        transform=from_origin(0, 1, 1, 1),
    ) as dst:
        dst.write(np.array([[1]], dtype="uint8"), 1)

    # Sin CRS entra por la rama del warning, pero igualmente se acepta.
    assert direct_alert._filter_rasters_same_crs([str(path)], CRS) == [str(path)]


def test_open_raster_returns_dataset_for_matching_crs(tmp_path):
    path = _write_raster(tmp_path / "r.tif", [[1, 2]])

    with direct_alert.open_raster(str(path), target_crs=CRS) as src:
        assert src.crs.to_string() == CRS


def test_open_raster_wraps_in_vrt_for_other_crs(tmp_path):
    path = _write_raster(
        tmp_path / "r.tif", [[1, 2]], crs="EPSG:4326", origin=(-74, 4), resolution=0.001
    )

    vrt = direct_alert.open_raster(str(path), target_crs=CRS)
    try:
        assert vrt.crs.to_string() == CRS
    finally:
        vrt.close()


def test_open_raster_raises_without_crs(tmp_path):
    path = tmp_path / "sin_crs.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=1, count=1, dtype="uint8",
        transform=from_origin(0, 1, 1, 1),
    ) as dst:
        dst.write(np.array([[1]], dtype="uint8"), 1)

    with pytest.raises(ValueError, match="no tiene CRS"):
        direct_alert.open_raster(str(path), target_crs=CRS)


def test_open_rasters_skips_failures(tmp_path, caplog):
    bueno = _write_raster(tmp_path / "a.tif", [[1]])
    roto = tmp_path / "b.tif"
    roto.write_bytes(b"no es un tif")

    with caplog.at_level(logging.WARNING):
        srcs = direct_alert.open_rasters([str(bueno), str(roto)], target_crs=CRS)

    try:
        assert len(srcs) == 1
    finally:
        for s in srcs:
            s.close()


def test_open_rasters_handles_none():
    assert direct_alert.open_rasters(None) == []


# ===================== pixel_area_m2_approx_for_vrt =====================


def test_pixel_area_approximation_for_geographic_raster(tmp_path):
    path = _write_raster(
        tmp_path / "r.tif", [[1, 2]], crs="EPSG:4326", origin=(-74, 4), resolution=0.001
    )

    with rasterio.open(path) as src:
        area = direct_alert.pixel_area_m2_approx_for_vrt(src)

    # 0.001° x 0.001° cerca del ecuador ≈ 111 m x 110 m ≈ 12.300 m2
    assert area == pytest.approx(12_300, rel=0.1)


def test_pixel_area_approximation_falls_back_on_error():
    class SrcRoto:
        transform = type("T", (), {"a": 0.001, "e": -0.001})()

        @property
        def bounds(self):
            raise RuntimeError("sin bounds")

    # El fallback usa 111.000 m/grado.
    assert direct_alert.pixel_area_m2_approx_for_vrt(SrcRoto()) == pytest.approx(
        0.001 * 0.001 * 111000.0**2
    )


# ===================== area_from_vectors =====================


def test_area_from_vectors_returns_intersection_area():
    geom = box(0, 0, 100, 100)
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 50, 100)]}, crs=CRS)

    assert direct_alert.area_from_vectors(geom, gdf) == pytest.approx(0.5)


def test_area_from_vectors_returns_zero_for_missing_layer():
    geom = box(0, 0, 100, 100)
    vacio = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS)

    assert direct_alert.area_from_vectors(geom, None) == 0.0
    assert direct_alert.area_from_vectors(geom, vacio) == 0.0


def test_area_from_vectors_returns_zero_without_candidates():
    geom = box(0, 0, 100, 100)
    lejos = gpd.GeoDataFrame({"geometry": [box(10_000, 10_000, 10_100, 10_100)]}, crs=CRS)

    assert direct_alert.area_from_vectors(geom, lejos) == 0.0


def test_area_from_vectors_returns_zero_when_intersections_are_empty():
    from shapely.geometry import Polygon

    geom = box(0, 0, 100, 100)
    anillo = Polygon(
        [(-10, -10), (110, -10), (110, 110), (-10, 110)],
        [[(-5, -5), (105, -5), (105, 105), (-5, 105)]],
    )
    gdf = gpd.GeoDataFrame({"geometry": [anillo]}, crs=CRS)

    assert direct_alert.area_from_vectors(geom, gdf) == 0.0


def test_area_from_vectors_returns_zero_on_error(caplog):
    class GdfRoto:
        empty = False

        @property
        def sindex(self):
            raise RuntimeError("sin índice")

    with caplog.at_level(logging.WARNING):
        assert direct_alert.area_from_vectors(box(0, 0, 1, 1), GdfRoto()) == 0.0

    assert "Error área vectorial" in caplog.text


# ===================== calculate_deforestation_metrics_from_src =====================


@pytest.fixture
def raster_10x10(tmp_path):
    """Raster 10x10 píxeles de 10m: la mitad izquierda es deforestación (valor 2)."""
    valores = [[2] * 5 + [1] * 5 for _ in range(10)]
    return _write_raster(tmp_path / "defo.tif", valores, origin=(0, 100), resolution=10)


def test_deforestation_metrics_counts_pixels(raster_10x10):
    geom = box(0, 0, 100, 100)

    with rasterio.open(raster_10x10) as src:
        intersecta, ha, prop = direct_alert.calculate_deforestation_metrics_from_src(src, geom, 2)

    assert intersecta is True
    # 50 píxeles de 100 m2 = 5.000 m2 = 0,5 ha
    assert ha == pytest.approx(0.5)
    assert prop == pytest.approx(0.5)


def test_deforestation_metrics_returns_false_without_matching_pixels(raster_10x10):
    geom = box(0, 0, 100, 100)

    with rasterio.open(raster_10x10) as src:
        assert direct_alert.calculate_deforestation_metrics_from_src(src, geom, 9) == (
            False,
            0.0,
            0.0,
        )


def test_deforestation_metrics_precise_mode_uses_real_intersection(raster_10x10):
    geom = box(0, 0, 50, 100)

    with rasterio.open(raster_10x10) as src:
        intersecta, ha, prop = direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 2, use_precise_area=True
        )

    assert intersecta is True
    assert ha == pytest.approx(0.5)
    assert prop == pytest.approx(1.0)


def test_deforestation_metrics_precise_mode_without_matches(raster_10x10):
    geom = box(0, 0, 100, 100)

    with rasterio.open(raster_10x10) as src:
        assert direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 9, use_precise_area=True
        ) == (False, 0.0, 0.0)


def test_deforestation_metrics_precise_mode_without_overlap(tmp_path):
    # La deforestación está lejos del predio: la intersección real es vacía.
    valores = [[2, 2], [2, 2]]
    raster = _write_raster(tmp_path / "defo.tif", valores, origin=(1000, 1100), resolution=10)
    geom = box(0, 0, 50, 50)

    with rasterio.open(raster) as src:
        resultado = direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 2, use_precise_area=True
        )

    assert resultado == (False, 0.0, 0.0)


def test_deforestation_metrics_reprojects_geometry_for_geographic_raster(tmp_path):
    valores = [[2, 2], [2, 2]]
    raster = _write_raster(
        tmp_path / "defo.tif", valores, crs="EPSG:4326", origin=(-74, 4.002), resolution=0.001
    )
    # Un predio en 3116 que cae sobre ese raster geográfico.
    geom_4326 = box(-73.9995, 4.0005, -73.9985, 4.0015)
    geom_3116 = (
        gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(CRS).iloc[0]
    )

    with rasterio.open(raster) as src:
        intersecta, ha, prop = direct_alert.calculate_deforestation_metrics_from_src(
            src, geom_3116, 2
        )

    assert intersecta is True
    assert ha > 0


def test_deforestation_metrics_clamps_proportion_to_one(tmp_path):
    # Predio diminuto sobre un raster totalmente deforestado: la proporción
    # calculada excede 1 y debe recortarse.
    valores = [[2] * 10 for _ in range(10)]
    raster = _write_raster(tmp_path / "defo.tif", valores, origin=(0, 100), resolution=10)
    geom = box(44, 44, 46, 46)

    with rasterio.open(raster) as src:
        _, _, prop = direct_alert.calculate_deforestation_metrics_from_src(src, geom, 2)

    assert prop == 1.0


def test_deforestation_metrics_returns_zero_for_degenerate_geometry(raster_10x10):
    # Geometría de área nula: la proporción se fuerza a 0 en vez de dividir por cero.
    geom = box(0, 0, 100, 100)

    class SrcConGeomVacia:
        pass

    with rasterio.open(raster_10x10) as src:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(direct_alert, "area_ha", lambda g: 0.0)
            intersecta, ha, prop = direct_alert.calculate_deforestation_metrics_from_src(
                src, geom, 2
            )

    assert intersecta is True
    assert prop == 0.0


def test_deforestation_metrics_returns_zero_on_error(caplog):
    class SrcRoto:
        crs = None
        transform = None

    with caplog.at_level(logging.WARNING):
        resultado = direct_alert.calculate_deforestation_metrics_from_src(
            SrcRoto(), box(0, 0, 1, 1), 2
        )

    assert resultado == (False, 0.0, 0.0)
    assert "Error calculando deforestación" in caplog.text


# ===================== area_from_rasters_open =====================


def test_area_from_rasters_open_sums_valid_pixels(raster_10x10):
    geom = box(0, 0, 100, 100)

    with rasterio.open(raster_10x10) as src:
        total = direct_alert.area_from_rasters_open(geom, [src])

    # Los 100 píxeles de 100 m2 = 10.000 m2 = 1 ha
    assert total == pytest.approx(1.0)


def test_area_from_rasters_open_handles_empty_inputs():
    assert direct_alert.area_from_rasters_open(box(0, 0, 1, 1), None) == 0.0
    assert direct_alert.area_from_rasters_open(box(0, 0, 1, 1), []) == 0.0


def test_area_from_rasters_open_skips_geometries_outside_raster(raster_10x10, caplog):
    geom = box(10_000, 10_000, 10_100, 10_100)

    with rasterio.open(raster_10x10) as src:
        with caplog.at_level(logging.WARNING):
            total = direct_alert.area_from_rasters_open(geom, [src])

    assert total == 0.0


def test_area_from_rasters_open_reprojects_for_geographic_raster(tmp_path):
    raster = _write_raster(
        tmp_path / "r.tif", [[1, 1], [1, 1]], crs="EPSG:4326", origin=(-74, 4.002), resolution=0.001
    )
    geom_3116 = (
        gpd.GeoSeries([box(-73.9995, 4.0005, -73.9985, 4.0015)], crs="EPSG:4326")
        .to_crs(CRS)
        .iloc[0]
    )

    with rasterio.open(raster) as src:
        total = direct_alert.area_from_rasters_open(geom_3116, [src])

    assert total > 0


def test_area_from_rasters_open_skips_rasters_outside_the_geometry(tmp_path):
    # El predio queda fuera del raster: mask() falla y la fuente se descarta.
    raster = _write_raster(tmp_path / "r.tif", [[1]], origin=(0, 10), resolution=10)
    geom = box(1000, 1000, 1010, 1010)

    with rasterio.open(raster) as src:
        assert direct_alert.area_from_rasters_open(geom, [src]) == 0.0


def test_area_from_rasters_open_skips_fully_masked_windows(tmp_path):
    # Raster con nodata: dentro del predio no queda ningún píxel válido.
    path = tmp_path / "nodata.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8",
        crs=CRS, transform=from_origin(0, 20, 10, 10), nodata=0,
    ) as dst:
        dst.write(np.zeros((2, 2), dtype="uint8"), 1)

    with rasterio.open(path) as src:
        assert direct_alert.area_from_rasters_open(box(0, 0, 20, 20), [src]) == 0.0


# ===================== process_row_option1 =====================


def test_process_row_option1_rounds_and_flags(raster_10x10):
    fila = {"farm_id": "123", "geometry": box(0, 0, 100, 100)}

    with rasterio.open(raster_10x10) as src:
        resultado = direct_alert.process_row_option1(fila, src, 2)

    assert resultado == {
        "id": "123",
        "intersect_deforestation": True,
        "deforested_ha": 0.5,
        "deforested_prop": 0.5,
        "direct_alert": True,
    }


def test_print_availability_option(capsys):
    direct_alert.print_availability_option("Raster", True, "→ x.tif")
    direct_alert.print_availability_option("Raster", False)

    salida = capsys.readouterr().out
    assert "Raster: Sí → x.tif" in salida
    assert "Raster: No" in salida


# ===================== make_out_dir_and_paths =====================


def test_make_out_dir_and_paths_builds_new_structure(tmp_path):
    plantilla = str(tmp_path / "results" / "direct_alerts" / "salida.csv")
    ctx = {"PERIODO": "annual", "YEARS": "2013-2014"}

    out_dir, out_csv = direct_alert.make_out_dir_and_paths(
        plantilla, ctx, "smbyc", deforestation_type="annual"
    )

    assert Path(out_dir) == tmp_path / "results" / "smbyc" / "annual" / "direct_alerts"
    assert Path(out_csv).name == "smbyc_direct_alert_annual_2013-2014.csv"
    assert Path(out_dir).is_dir()


def test_make_out_dir_and_paths_is_idempotent(tmp_path):
    ctx = {"PERIODO": "annual", "YEARS": "2024"}
    plantilla = str(tmp_path / "results" / "direct_alerts" / "salida.csv")

    primero, _ = direct_alert.make_out_dir_and_paths(plantilla, ctx, "smbyc", "annual")
    # Al pasar de nuevo un directorio ya en la estructura nueva no se duplica.
    segundo, _ = direct_alert.make_out_dir_and_paths(
        os.path.join(primero, "x.csv"), ctx, "smbyc", "annual"
    )

    assert Path(primero) == Path(segundo)


def test_make_out_dir_and_paths_uses_legacy_layout_without_type(tmp_path):
    ctx = {"PERIODO": "annual", "YEARS": "2024"}
    plantilla = str(tmp_path / "salida.csv")

    out_dir, out_csv = direct_alert.make_out_dir_and_paths(plantilla, ctx, "smbyc")

    assert Path(out_dir) == tmp_path / "SMBYC" / "2024"
    assert Path(out_csv).name == "smbyc_direct_alert_annual_2024.csv"


def test_make_out_dir_and_paths_uppercases_unknown_sources(tmp_path):
    ctx = {"PERIODO": "annual", "YEARS": "2024"}

    out_dir, _ = direct_alert.make_out_dir_and_paths(str(tmp_path / "s.csv"), ctx, "otra")

    assert "OTRA" in out_dir


def test_write_reason_log_appends(tmp_path, capsys):
    direct_alert.write_reason_log(str(tmp_path), "motivo.log", "  hola  ")
    direct_alert.write_reason_log(str(tmp_path), "motivo.log", "adios")

    assert (tmp_path / "motivo.log").read_text(encoding="utf-8").splitlines() == ["hola", "adios"]
    assert "Se escribió log" in capsys.readouterr().out


# ===================== calculate_direct_alerts =====================


@pytest.fixture
def alert_env(tmp_path, monkeypatch):
    """Carpeta de predios + raster + plantilla de salida, con el paquete simulado."""
    farms = tmp_path / "farms"
    farms.mkdir()
    _write_geojson(farms / "A1.geojson", box(0, 0, 100, 100))
    _write_geojson(farms / "A2.geojson", box(200, 0, 300, 100))

    raster = _write_raster(
        tmp_path / "defo.tif", [[2] * 5 + [1] * 5 for _ in range(10)], origin=(0, 100), resolution=10
    )

    salida = str(tmp_path / "results" / "direct_alerts" / "out.csv")

    def fake_pkg(**kwargs):
        plots = kwargs["plots"]
        return pd.DataFrame(
            {
                "id": list(plots["id"]),
                "direct_alert": [True] * len(plots),
                "deforested_ha": [0.5] * len(plots),
                "deforested_prop": [0.5] * len(plots),
            }
        )

    monkeypatch.setattr(direct_alert, "pkg_alert_direct", fake_pkg)
    monkeypatch.setattr(direct_alert, "setup_logging", lambda *a, **k: None, raising=False)

    return {
        "tmp": tmp_path,
        "farms": farms,
        "raster": str(raster),
        "output": salida,
        "log": str(tmp_path / "pipeline.log"),
    }


def _log_text(alert_env):
    """calculate_direct_alerts reconfigura el logger raíz (y elimina el handler de
    caplog), así que las advertencias se comprueban leyendo su archivo de log."""
    for handler in list(logging.getLogger().handlers):
        handler.flush()
    path = Path(alert_env["log"])
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _call(alert_env, **kwargs):
    defaults = dict(
        source="smbyc",
        period_type="annual",
        years=["2024"],
        farm_folder=str(alert_env["farms"]),
        raster_template="",
        output_csv=alert_env["output"],
        batch_size=10,
        log_file=alert_env["log"],
        raster_paths_dict={"2024": alert_env["raster"]},
    )
    defaults.update(kwargs)
    return direct_alert.calculate_direct_alerts(**defaults)


def test_calculate_direct_alerts_writes_csv_from_folder(alert_env):
    result = _call(alert_env)

    assert result["success"] is True
    assert result["periods_processed"] == 1
    assert result["farms_processed"] == 2

    salida = (
        Path(alert_env["tmp"])
        / "results" / "smbyc" / "annual" / "direct_alerts"
        / "smbyc_direct_alert_annual_2024.csv"
    )
    df = pd.read_csv(salida)
    assert sorted(df["id"].astype(str)) == ["A1", "A2"]
    assert list(df.columns) == [
        "id",
        "intersect_deforestation",
        "deforested_ha",
        "deforested_prop",
        "direct_alert",
    ]


def test_calculate_direct_alerts_uses_raster_template_when_no_dict(alert_env):
    plantilla = str(Path(alert_env["raster"]).parent / "{YEARS}.tif")
    os.rename(alert_env["raster"], str(Path(alert_env["raster"]).parent / "2024.tif"))

    result = _call(alert_env, raster_template=plantilla, raster_paths_dict=None)

    assert result["success"] is True


def test_calculate_direct_alerts_raises_for_missing_farm_folder(alert_env):
    with pytest.raises(RuntimeError, match="No existe la carpeta de geojsons"):
        _call(alert_env, farm_folder=str(alert_env["tmp"] / "no_existe"))


def test_calculate_direct_alerts_raises_for_empty_folder(alert_env, tmp_path):
    vacia = tmp_path / "vacia"
    vacia.mkdir()

    with pytest.raises(RuntimeError, match="ERROR CRÍTICO"):
        _call(alert_env, farm_folder=str(vacia))


def test_calculate_direct_alerts_returns_failure_without_valid_rasters(alert_env):
    result = _call(alert_env, raster_paths_dict={"2024": str(alert_env["tmp"] / "no.tif")})

    assert result["success"] is False
    assert result["periods_processed"] == 0


def test_calculate_direct_alerts_reports_skipped_periods(alert_env, capsys):
    result = _call(
        alert_env,
        years=["2024", "2025"],
        raster_paths_dict={"2024": alert_env["raster"], "2025": str(alert_env["tmp"] / "no.tif")},
    )

    assert result["periods_processed"] == 1
    assert "Períodos omitidos por raster faltante: 2025" in capsys.readouterr().out


def test_calculate_direct_alerts_removes_previous_output(alert_env):
    _call(alert_env)
    salida = (
        Path(alert_env["tmp"])
        / "results" / "smbyc" / "annual" / "direct_alerts"
        / "smbyc_direct_alert_annual_2024.csv"
    )
    primera = len(pd.read_csv(salida))

    _call(alert_env)

    # La segunda corrida reemplaza el archivo en vez de acumular filas.
    assert len(pd.read_csv(salida)) == primera


def test_calculate_direct_alerts_applies_farm_limit(alert_env):
    result = _call(alert_env, _farm_limit=1)

    assert result["farms_processed"] == 1


def test_calculate_direct_alerts_ignores_limit_above_available(alert_env, capsys):
    result = _call(alert_env, _farm_limit=99)

    assert result["farms_processed"] == 2
    assert "no alcanzado" in capsys.readouterr().out


def test_calculate_direct_alerts_applies_farm_range(alert_env):
    result = _call(alert_env, farm_range="1:1")

    assert result["farms_processed"] == 1


def test_calculate_direct_alerts_applies_worker_filter(alert_env):
    result = _call(alert_env, _farm_range_filter={"A2.geojson"})

    assert result["farms_processed"] == 1


def test_calculate_direct_alerts_accepts_explicit_file_subset(alert_env, capsys):
    result = _call(alert_env, _farm_files_subset=["A1.geojson", "no_geojson.txt"])

    assert result["farms_processed"] == 1
    assert "GeoJSONs del chunk recibidos: 1" in capsys.readouterr().out


def test_calculate_direct_alerts_splits_into_batches(alert_env, capsys):
    _call(alert_env, batch_size=1)

    assert "Lote 2/2" in capsys.readouterr().out


def test_calculate_direct_alerts_uses_unknown_source_tag(alert_env):
    result = _call(alert_env, source="otra")

    assert result["success"] is True
    salida = (
        Path(alert_env["tmp"])
        / "results" / "otra" / "annual" / "direct_alerts"
        / "otra_direct_alert_annual_2024.csv"
    )
    assert salida.exists()


def test_calculate_direct_alerts_announces_precise_mode(alert_env, capsys):
    _call(alert_env, use_precise_area=True, pixel_divisions=3)

    assert "Modo PRECISO habilitado" in capsys.readouterr().out


def test_calculate_direct_alerts_skips_geojsons_without_crs(alert_env, monkeypatch):
    respuestas = iter(
        [
            gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}),  # sin CRS
            gpd.GeoDataFrame({"geometry": [box(20, 0, 30, 10)]}, crs=CRS),
        ]
    )
    monkeypatch.setattr(direct_alert.gpd, "read_file", lambda *a, **k: next(respuestas))

    result = _call(alert_env)

    assert result["success"] is True
    assert "no tiene CRS" in _log_text(alert_env)


def test_calculate_direct_alerts_reprojects_geojsons(alert_env, monkeypatch):
    farms = alert_env["tmp"] / "farms4326"
    farms.mkdir()
    _write_geojson(farms / "A1.geojson", box(-74, 4, -73.99, 4.01), crs="EPSG:4326")

    result = _call(alert_env, farm_folder=str(farms))

    assert result["success"] is True


def test_calculate_direct_alerts_extracts_id_from_prefixed_filenames(alert_env):
    farms = alert_env["tmp"] / "farms_prefijo"
    farms.mkdir()
    _write_geojson(farms / "sitcode_123.geojson", box(0, 0, 100, 100))
    _write_geojson(farms / "FARM_456.geojson", box(0, 0, 100, 100))

    _call(alert_env, farm_folder=str(farms))

    salida = (
        Path(alert_env["tmp"])
        / "results" / "smbyc" / "annual" / "direct_alerts"
        / "smbyc_direct_alert_annual_2024.csv"
    )
    assert sorted(pd.read_csv(salida)["id"].astype(str)) == ["123", "456"]


def test_calculate_direct_alerts_skips_unreadable_geojsons(alert_env):
    (alert_env["farms"] / "roto.geojson").write_text("{ roto", encoding="utf-8")

    result = _call(alert_env)

    assert result["success"] is True
    assert "Error leyendo" in _log_text(alert_env)


def test_calculate_direct_alerts_reports_batch_without_valid_geometries(alert_env, capsys):
    farms = alert_env["tmp"] / "solo_rotos"
    farms.mkdir()
    (farms / "roto.geojson").write_text("{ roto", encoding="utf-8")

    result = _call(alert_env, farm_folder=str(farms))

    assert result["success"] is True
    assert "sin geometrías válidas" in capsys.readouterr().out


def test_calculate_direct_alerts_skips_empty_package_results(alert_env, monkeypatch):
    monkeypatch.setattr(direct_alert, "pkg_alert_direct", lambda **k: pd.DataFrame())

    result = _call(alert_env)

    salida = (
        Path(alert_env["tmp"])
        / "results" / "smbyc" / "annual" / "direct_alerts"
        / "smbyc_direct_alert_annual_2024.csv"
    )
    assert result["success"] is True
    assert not salida.exists()


# ---- rama con farms_metadata y DataManager ----


class FakeDataManagerConCache:
    def __init__(self, geojsons_dir, geometries):
        self.geojsons_dir = Path(geojsons_dir)
        self._geometries_cache = geometries

    def ensure_geojson_available(self, farm_meta):
        sitcode = farm_meta.get("sitcode")
        path = self.geojsons_dir / f"{sitcode}.geojson"
        return str(path) if path.exists() else None

    def get_geometry(self, farm_id):
        return self._geometries_cache.get(farm_id)


def test_calculate_direct_alerts_uses_metadata_and_geometry_cache(alert_env):
    dm = FakeDataManagerConCache(
        alert_env["farms"], {"A1": box(0, 0, 100, 100), "A2": box(200, 0, 300, 100)}
    )

    result = _call(
        alert_env,
        _farms_metadata=[{"sitcode": "A1", "mongo_id": "m1"}, {"sitcode": "A2", "mongo_id": "m2"}],
        _data_manager=dm,
    )

    assert result["farms_processed"] == 2


def test_calculate_direct_alerts_reports_farms_without_geojson(alert_env, capsys):
    dm = FakeDataManagerConCache(alert_env["farms"], {"A1": box(0, 0, 100, 100)})

    result = _call(
        alert_env,
        _farms_metadata=[
            {"sitcode": "A1", "mongo_id": "m1"},
            {"sitcode": None, "mongo_id": "sin_sitcode"},
            {"sitcode": "NO_EXISTE", "mongo_id": "m3"},
        ],
        _data_manager=dm,
    )

    salida = capsys.readouterr().out
    assert "Farms sin sitcode o sin geojson: 2/3" in salida
    assert result["farms_processed"] == 1


def test_calculate_direct_alerts_metadata_without_data_manager(alert_env, capsys):
    result = _call(
        alert_env,
        _farms_metadata=[
            {"sitcode": "A1", "mongo_id": "m1"},
            {"sitcode": "NO_EXISTE", "mongo_id": "m2"},
        ],
    )

    assert result["farms_processed"] == 1
    assert "(no existe)" in capsys.readouterr().out


def test_calculate_direct_alerts_raises_when_metadata_has_no_geojsons(alert_env):
    with pytest.raises(RuntimeError, match="No se encontraron GeoJSONs para los"):
        _call(alert_env, _farms_metadata=[{"sitcode": "NO_EXISTE", "mongo_id": "m1"}])


def test_calculate_direct_alerts_counts_freshly_downloaded_geojsons(alert_env, capsys):
    # ensure_geojson_available devuelve un archivo recién creado: se reporta como descarga.
    nuevo = alert_env["farms"] / "A3.geojson"
    _write_geojson(nuevo, box(0, 0, 100, 100))
    dm = FakeDataManagerConCache(alert_env["farms"], {"A3": box(0, 0, 100, 100)})

    _call(alert_env, _farms_metadata=[{"sitcode": "A3", "mongo_id": "m3"}], _data_manager=dm)

    assert "Geojsons descargados desde MongoDB: 1" in capsys.readouterr().out


def test_calculate_direct_alerts_skips_farms_missing_from_geometry_cache(alert_env):
    dm = FakeDataManagerConCache(alert_env["farms"], {"A1": box(0, 0, 100, 100)})

    result = _call(
        alert_env,
        _farms_metadata=[{"sitcode": "A1"}, {"sitcode": "A2"}],
        _data_manager=dm,
    )

    assert result["success"] is True
    assert "no tiene geometr" in _log_text(alert_env)


def test_calculate_direct_alerts_handles_geometry_cache_errors(alert_env):
    class DmQueFalla(FakeDataManagerConCache):
        def get_geometry(self, farm_id):
            raise RuntimeError("caché corrupta")

    dm = DmQueFalla(alert_env["farms"], {"A1": box(0, 0, 100, 100)})

    result = _call(alert_env, _farms_metadata=[{"sitcode": "A1"}], _data_manager=dm)

    assert result["success"] is True
    assert "Error obteniendo geometr" in _log_text(alert_env)


# ---- ramas defensivas del modo preciso ----


def test_precise_mode_returns_false_when_second_mask_has_no_deforestation(raster_10x10, monkeypatch):
    """La segunda lectura (sobre la geometría sin reproyectar) puede no contener
    píxeles de la clase buscada; en ese caso se descarta el predio."""
    geom = box(0, 0, 100, 100)
    real_mask = direct_alert.mask
    llamadas = {"n": 0}

    def mask_espia(src, shapes_, **kwargs):
        llamadas["n"] += 1
        salida, transform = real_mask(src, shapes_, **kwargs)
        if llamadas["n"] >= 2:
            salida = np.ma.MaskedArray(np.zeros_like(salida.data), mask=False)
        return salida, transform

    monkeypatch.setattr(direct_alert, "mask", mask_espia)

    with rasterio.open(raster_10x10) as src:
        assert direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 2, use_precise_area=True
        ) == (False, 0.0, 0.0)


def test_precise_mode_returns_false_without_vectorized_pixels(raster_10x10, monkeypatch):
    geom = box(0, 0, 100, 100)
    # shapes() devuelve solo geometrías de otra clase: no hay polígonos que unir.
    monkeypatch.setattr(
        direct_alert, "shapes", lambda arr, mask=None, transform=None: iter([({"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]}, 99)])
    )

    with rasterio.open(raster_10x10) as src:
        assert direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 2, use_precise_area=True
        ) == (False, 0.0, 0.0)


def test_precise_mode_returns_false_when_intersection_is_empty(raster_10x10, monkeypatch):
    geom = box(0, 0, 100, 100)
    # La unión de píxeles cae fuera del predio: intersección vacía.
    monkeypatch.setattr(direct_alert, "unary_union", lambda geoms: box(5000, 5000, 5100, 5100))

    with rasterio.open(raster_10x10) as src:
        assert direct_alert.calculate_deforestation_metrics_from_src(
            src, geom, 2, use_precise_area=True
        ) == (False, 0.0, 0.0)


def test_calculate_direct_alerts_warns_about_unusable_farm_limit(alert_env, capsys):
    # Un límite negativo no filtra nada, pero el pipeline lo reporta.
    result = _call(alert_env, _farm_limit=-1)

    assert result["farms_processed"] == 2
    assert "farm_limit recibido pero no aplicado" in capsys.readouterr().out


def test_calculate_direct_alerts_tolerates_raster_close_errors(alert_env, monkeypatch):
    real_open = direct_alert.open_raster

    class SrcQueFallaAlCerrar:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            raise RuntimeError("no se pudo cerrar")

    monkeypatch.setattr(
        direct_alert, "open_raster", lambda path, target_crs=None: SrcQueFallaAlCerrar(real_open(path, target_crs=target_crs))
    )

    result = _call(alert_env)

    assert result["success"] is True
