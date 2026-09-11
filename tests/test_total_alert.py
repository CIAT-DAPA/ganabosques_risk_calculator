import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import total_alert
from orm_doubles import Ref, make_document


CRS = "EPSG:3116"
MOV_COLS = [
    "n_total_mov",
    "n_in",
    "n_out",
    "n_indirect_in",
    "n_indirect_out",
    "indirect_alert_in",
    "indirect_alert_out",
]


# ===================== helpers de placeholders y rutas =====================


def test_derive_mov_year_picks_the_right_end_of_the_range():
    assert total_alert.derive_mov_year("2017-2020", "anual") == "2017"
    assert total_alert.derive_mov_year("2017-2020", "cumulative") == "2020"
    assert total_alert.derive_mov_year("2017-2020", "cum") == "2020"
    assert total_alert.derive_mov_year("2017-2020", "acumulado") == "2020"
    # Cualquier otro periodo usa el primer año.
    assert total_alert.derive_mov_year("2017-2020", "nad") == "2017"
    # Sin rango se devuelve el valor tal cual.
    assert total_alert.derive_mov_year("2020", "anual") == "2020"
    assert total_alert.derive_mov_year("", "anual") == ""
    assert total_alert.derive_mov_year(None, "anual") == ""


def test_format_placeholders_supports_both_cases_and_missing_keys(caplog):
    ctx = {"EMPRESA": "ACME", "PERIODO": "annual", "YEARS": "2024", "MOV_YEAR": "2024"}

    assert total_alert.format_placeholders("{EMPRESA}/{YEARS}", ctx) == "ACME/2024"
    assert total_alert.format_placeholders("{empresa}/{years}", ctx) == "ACME/2024"
    assert total_alert.format_placeholders(None, ctx) is None
    # Un placeholder desconocido no rompe: se devuelve la plantilla sin tocar.
    assert total_alert.format_placeholders("{DESCONOCIDO}", ctx) == "{DESCONOCIDO}"


def test_sanitize_empresa_folder_normalizes_names():
    assert total_alert.sanitize_empresa_folder("Mi Empresa") == "mi_empresa"
    assert total_alert.sanitize_empresa_folder("Frigorífico S.A.") == "frigor_fico_s.a."
    assert total_alert.sanitize_empresa_folder("A/B\\C") == "a_b_c"


def test_ensure_dir_is_idempotent(tmp_path):
    destino = tmp_path / "a" / "b"
    total_alert.ensure_dir(str(destino))
    total_alert.ensure_dir(str(destino))

    assert destino.is_dir()


def test_write_reason_log_appends_lines(tmp_path):
    carpeta = tmp_path / "logs"

    total_alert.write_reason_log(str(carpeta), "motivo.log", "  primera  ")
    total_alert.write_reason_log(str(carpeta), "motivo.log", "segunda")

    contenido = (carpeta / "motivo.log").read_text(encoding="utf-8").splitlines()
    assert contenido == ["primera", "segunda"]


def test_list_geojsons_walks_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.geojson").write_text("{}", encoding="utf-8")
    (tmp_path / "sub" / "b.GEOJSON").write_text("{}", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    encontrados = total_alert.list_geojsons(str(tmp_path))

    assert [Path(p).name for p in encontrados] == ["a.geojson", "b.GEOJSON"]


@pytest.mark.parametrize(
    "fn,key",
    [
        (total_alert.alerts_dir_from_output_csv, "OUTPUT_CSV"),
        (total_alert.movement_dir_from_output_csv, "MOVEMENT_RISK_OUTPUT_CSV"),
        (total_alert.total_out_dir_from_output_csv, "TOTAL_RISK_OUTPUT_CSV"),
    ],
)
def test_directory_helpers_append_source_and_company(fn, key):
    ctx = {"EMPRESA": "ACME", "PERIODO": "annual", "YEARS": "2024", "MOV_YEAR": "2024"}

    ruta = fn("/base/{PERIODO}/archivo.csv", ctx, "smbyc", "Mi Empresa")

    assert ruta.endswith(os.path.join("SMBYC", "mi_empresa"))
    assert "annual" in ruta


def test_directory_helpers_uppercase_unknown_sources():
    ctx = {"EMPRESA": "ACME", "PERIODO": "annual", "YEARS": "2024", "MOV_YEAR": ""}

    ruta = total_alert.alerts_dir_from_output_csv("archivo.csv", ctx, "otra", "ACME")

    assert "OTRA" in ruta


# ===================== geo helpers =====================


def test_assert_crs_exact_accepts_matching_crs():
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs=CRS)

    total_alert.assert_crs_exact(gdf, CRS, "Capa")


def test_assert_crs_exact_rejects_missing_crs():
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})

    with pytest.raises(RuntimeError, match="sin CRS definido"):
        total_alert.assert_crs_exact(gdf, CRS, "Capa")


def test_assert_crs_exact_rejects_different_crs():
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

    with pytest.raises(RuntimeError, match="difiere de esperado"):
        total_alert.assert_crs_exact(gdf, CRS, "Capa")


def _write_farm(path, geom, crs=CRS):
    gpd.GeoDataFrame({"geometry": [geom]}, crs=crs).to_file(path, driver="GeoJSON")


def test_load_farms_geoms_extracts_id_from_filename(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    _write_farm(carpeta / "finca_123.geojson", box(0, 0, 10, 10))
    _write_farm(carpeta / "456.geojson", box(10, 0, 20, 10))

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert sorted(farms["id"].tolist()) == ["123", "456"]


def test_load_farms_geoms_falls_back_to_id_column_then_filename(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    # Sin dígitos en el nombre: usa la columna `id` cuando hay una sola fila.
    gpd.GeoDataFrame({"id": ["desde_columna"], "geometry": [box(0, 0, 5, 5)]}, crs=CRS).to_file(
        carpeta / "sin_digitos.geojson", driver="GeoJSON"
    )
    # Sin dígitos y sin columna id: usa el nombre del archivo.
    _write_farm(carpeta / "otra_finca.geojson", box(10, 0, 15, 5))

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert sorted(farms["id"].tolist()) == ["desde_columna", "otra_finca"]


def test_load_farms_geoms_reprojects_when_needed(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    _write_farm(carpeta / "1.geojson", box(-74, 4, -73.9, 4.1), crs="EPSG:4326")

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert farms.crs.to_string() == CRS


def test_load_farms_geoms_dissolves_duplicate_ids(tmp_path, capsys):
    carpeta = tmp_path / "fincas"
    (carpeta / "sub").mkdir(parents=True)
    _write_farm(carpeta / "finca_1.geojson", box(0, 0, 10, 10))
    _write_farm(carpeta / "sub" / "finca_1.geojson", box(20, 0, 30, 10))

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert len(farms) == 1
    assert "IDs duplicados: 1" in capsys.readouterr().out


def test_load_farms_geoms_skips_unreadable_files(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    _write_farm(carpeta / "1.geojson", box(0, 0, 10, 10))
    (carpeta / "2.geojson").write_text("{ roto", encoding="utf-8")

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert farms["id"].tolist() == ["1"]


def test_load_farms_geoms_skips_layers_without_crs(tmp_path, monkeypatch):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    _write_farm(carpeta / "1.geojson", box(0, 0, 10, 10))
    _write_farm(carpeta / "2.geojson", box(20, 0, 30, 10))

    # El primer archivo se lee sin CRS: no se puede reproyectar y se descarta.
    respuestas = iter(
        [
            gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}),
            gpd.GeoDataFrame({"geometry": [box(20, 0, 30, 10)]}, crs=CRS),
        ]
    )
    monkeypatch.setattr(total_alert.gpd, "read_file", lambda *a, **k: next(respuestas))

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert farms["id"].tolist() == ["2"]


def test_load_farms_geoms_skips_empty_layers(tmp_path, monkeypatch):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    _write_farm(carpeta / "1.geojson", box(0, 0, 10, 10))
    _write_farm(carpeta / "2.geojson", box(20, 0, 30, 10))

    respuestas = iter(
        [
            gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS),
            gpd.GeoDataFrame({"geometry": [box(20, 0, 30, 10)]}, crs=CRS),
        ]
    )
    monkeypatch.setattr(total_alert.gpd, "read_file", lambda *a, **k: next(respuestas))

    farms = total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)

    assert farms["id"].tolist() == ["2"]


def test_load_farms_geoms_raises_for_missing_folder(tmp_path):
    with pytest.raises(RuntimeError, match="No existe carpeta de fincas"):
        total_alert.load_farms_geoms_reproject(str(tmp_path / "no"), "ACME", CRS)


def test_load_farms_geoms_raises_for_empty_folder(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()

    with pytest.raises(RuntimeError, match="No se encontraron GeoJSONs"):
        total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)


def test_load_farms_geoms_raises_when_every_file_fails(tmp_path):
    carpeta = tmp_path / "fincas"
    carpeta.mkdir()
    (carpeta / "1.geojson").write_text("{ roto", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No se pudo cargar ninguna finca"):
        total_alert.load_farms_geoms_reproject(str(carpeta), "ACME", CRS)


def test_load_mask_gdf_strict_reads_layer(tmp_path):
    path = tmp_path / "mascara.gpkg"
    gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs=CRS).to_file(path, driver="GPKG")

    gdf = total_alert.load_mask_gdf_strict(str(path), CRS, "Frontera")

    assert gdf is not None and len(gdf) == 1


def test_load_mask_gdf_strict_returns_none_for_missing_file(tmp_path):
    assert total_alert.load_mask_gdf_strict(str(tmp_path / "no.gpkg"), CRS, "Frontera") is None


def test_load_mask_gdf_strict_returns_none_for_wrong_crs(tmp_path):
    path = tmp_path / "mascara.gpkg"
    gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs="EPSG:4326").to_file(
        path, driver="GPKG"
    )

    assert total_alert.load_mask_gdf_strict(str(path), CRS, "Frontera") is None


def test_load_mask_gdf_strict_returns_none_for_empty_layer(tmp_path, monkeypatch):
    path = tmp_path / "mascara.gpkg"
    gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs=CRS).to_file(path, driver="GPKG")
    monkeypatch.setattr(
        total_alert.gpd,
        "read_file",
        lambda *a, **k: gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS),
    )

    assert total_alert.load_mask_gdf_strict(str(path), CRS, "Frontera") is None


def test_load_mask_gdf_strict_survives_missing_sindex(tmp_path, monkeypatch):
    path = tmp_path / "mascara.gpkg"
    base = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs=CRS)
    base.to_file(path, driver="GPKG")

    class SinIndice(gpd.GeoDataFrame):
        @property
        def sindex(self):
            raise RuntimeError("rtree no disponible")

    monkeypatch.setattr(total_alert.gpd, "read_file", lambda *a, **k: SinIndice(base))

    assert total_alert.load_mask_gdf_strict(str(path), CRS, "Frontera") is not None


# ===================== scan_ids_needed =====================


def test_scan_ids_needed_collects_from_direct_and_movement(tmp_path, monkeypatch):
    alerts = tmp_path / "annual" / "SMBYC" / "acme"
    movs = tmp_path / "mov" / "annual" / "SMBYC" / "acme"
    alerts.mkdir(parents=True)
    movs.mkdir(parents=True)
    pd.DataFrame({"id": ["FARM_ID_00123", "000456"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    pd.DataFrame({"id": ["789"]}).to_csv(
        movs / "smbyc_movement_alerts_ACME_2024.csv", index=False
    )

    monkeypatch.setitem(total_alert.config, "OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "f.csv"))
    monkeypatch.setitem(
        total_alert.config, "MOVEMENT_RISK_OUTPUT_CSV", str(tmp_path / "mov" / "{PERIODO}" / "f.csv")
    )

    ids = total_alert.scan_ids_needed(["2024"], ["smbyc"], "ACME", "annual")

    assert ids == {"123", "456", "789"}


def test_scan_ids_needed_ignores_unreadable_csvs(tmp_path, monkeypatch):
    alerts = tmp_path / "annual" / "SMBYC" / "acme"
    alerts.mkdir(parents=True)
    # Sin columna `id`: usecols falla y el archivo se ignora.
    pd.DataFrame({"otra": ["x"]}).to_csv(alerts / "smbyc_direct_alert_ACME_2024.csv", index=False)

    movs = tmp_path / "mov" / "annual" / "SMBYC" / "acme"
    movs.mkdir(parents=True)
    pd.DataFrame({"otra": ["y"]}).to_csv(
        movs / "smbyc_movement_alerts_ACME_2024.csv", index=False
    )

    monkeypatch.setitem(total_alert.config, "OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "f.csv"))
    monkeypatch.setitem(
        total_alert.config, "MOVEMENT_RISK_OUTPUT_CSV", str(tmp_path / "mov" / "{PERIODO}" / "f.csv")
    )

    assert total_alert.scan_ids_needed(["2024"], ["smbyc"], "ACME", "annual") == set()


def test_scan_ids_needed_returns_empty_without_files(tmp_path, monkeypatch):
    monkeypatch.setitem(total_alert.config, "OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "f.csv"))
    monkeypatch.setitem(
        total_alert.config, "MOVEMENT_RISK_OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "m.csv")
    )

    assert total_alert.scan_ids_needed(["2024"], ["smbyc"], "ACME", "annual") == set()


# ===================== métricas espaciales y caché =====================


@pytest.fixture
def farms_gdf():
    return gpd.GeoDataFrame(
        {"id": ["1", "2"], "geometry": [box(0, 0, 100, 100), box(200, 0, 300, 100)]}, crs=CRS
    )


def test_compute_metrics_cache_computes_proportions(farms_gdf):
    frontier = gpd.GeoDataFrame({"geometry": [box(0, 0, 50, 100)]}, crs=CRS)
    protected = gpd.GeoDataFrame({"geometry": [box(0, 0, 20, 100)]}, crs=CRS)

    df = total_alert.compute_metrics_cache(farms_gdf, {"1"}, frontier, protected)

    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["id"] == "1"
    # 50m x 100m de 100m x 100m = 0,5 ha de 1 ha
    assert fila["farming_in_ha"] == pytest.approx(0.5)
    assert fila["farming_in_prop"] == pytest.approx(0.5)
    assert fila["farming_out_ha"] == pytest.approx(0.5)
    assert fila["protected_ha"] == pytest.approx(0.2)
    assert fila["protected_prop"] == pytest.approx(0.2)


def test_compute_metrics_cache_without_reference_layers(farms_gdf):
    df = total_alert.compute_metrics_cache(farms_gdf, {"1", "2"}, None, None)

    assert df["farming_in_ha"].tolist() == [0.0, 0.0]
    assert df["protected_ha"].tolist() == [0.0, 0.0]
    # Sin frontera, toda el área queda "fuera".
    assert df["farming_out_prop"].tolist() == [1.0, 1.0]


def test_compute_metrics_cache_handles_zero_area_geometries():
    farms = gpd.GeoDataFrame({"id": ["1"], "geometry": [box(0, 0, 0, 0)]}, crs=CRS)

    df = total_alert.compute_metrics_cache(farms, {"1"}, None, None)

    assert df.iloc[0]["farming_in_prop"] == 0.0
    assert df.iloc[0]["farming_out_prop"] == 0.0
    assert df.iloc[0]["protected_prop"] == 0.0


def test_cache_dir_for_total_creates_directory(tmp_path):
    destino = total_alert.cache_dir_for_total(
        str(tmp_path / "{PERIODO}" / "out.csv"), "Mi Empresa", "annual"
    )

    assert Path(destino).is_dir()
    assert destino.endswith(os.path.join("__cache__", "mi_empresa"))


def test_load_or_build_cache_writes_then_reuses(tmp_path, monkeypatch, farms_gdf):
    monkeypatch.setitem(
        total_alert.config, "TOTAL_RISK_OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "out.csv")
    )

    primero = total_alert.load_or_build_cache("ACME", "annual", farms_gdf, {"1", "2"}, None, None)
    assert len(primero) == 2

    # La segunda llamada lee de disco: no debe recalcular.
    monkeypatch.setattr(
        total_alert,
        "compute_metrics_cache",
        lambda *a, **k: pytest.fail("no debería recalcular con caché completa"),
    )
    segundo = total_alert.load_or_build_cache("ACME", "annual", farms_gdf, {"1", "2"}, None, None)

    assert len(segundo) == 2


def test_load_or_build_cache_completes_partial_cache(tmp_path, monkeypatch, farms_gdf):
    monkeypatch.setitem(
        total_alert.config, "TOTAL_RISK_OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "out.csv")
    )
    total_alert.load_or_build_cache("ACME", "annual", farms_gdf, {"1"}, None, None)

    completado = total_alert.load_or_build_cache(
        "ACME", "annual", farms_gdf, {"1", "2"}, None, None
    )

    assert sorted(completado["id"].astype(str)) == ["1", "2"]


def test_load_or_build_cache_rebuilds_on_corrupt_file(tmp_path, monkeypatch, farms_gdf, capsys):
    monkeypatch.setitem(
        total_alert.config, "TOTAL_RISK_OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "out.csv")
    )
    cdir = total_alert.cache_dir_for_total(
        str(tmp_path / "{PERIODO}" / "out.csv"), "ACME", "annual"
    )
    # Caché sin columna `id`: se detecta como inválida.
    pd.DataFrame({"otra": ["x"]}).to_csv(Path(cdir) / "spatial_metrics_ACME.csv", index=False)

    df = total_alert.load_or_build_cache("ACME", "annual", farms_gdf, {"1"}, None, None)

    assert "Caché inválida" in capsys.readouterr().out
    assert df["id"].tolist() == ["1"]


# ===================== build_mongo_maps =====================


@pytest.fixture
def orm_farms(monkeypatch):
    Farm = make_document("Farm")
    FarmPolygons = make_document("FarmPolygons")

    class ExtIdMatcher:
        """Filtra por ext_id__source y ext_id__ext_code__in como hace mongoengine."""

        def __init__(self, store):
            self.store = store

        def __call__(self, **filters):
            source = filters.get("ext_id__source")
            codes = set(filters.get("ext_id__ext_code__in", []))
            farm_ids = filters.get("farm_id__in")

            if farm_ids is not None:
                claves = {str(getattr(f, "id", f)) for f in farm_ids}
                items = [
                    d
                    for d in self.store
                    if str(getattr(d.farm_id, "id", d.farm_id)) in claves
                ]
            else:
                items = [
                    d
                    for d in self.store
                    if any(
                        e.source == source and e.ext_code in codes for e in (d.ext_id or [])
                    )
                ]

            class QS:
                def __init__(self, its):
                    self._its = its

                def only(self, *fields):
                    return self

                def __iter__(self):
                    return iter(self._its)

            return QS(items)

    monkeypatch.setattr(total_alert, "HAS_ORM", True)
    monkeypatch.setattr(total_alert, "Farm", Farm, raising=False)
    monkeypatch.setattr(total_alert, "FarmPolygons", FarmPolygons, raising=False)

    return {"Farm": Farm, "FarmPolygons": FarmPolygons, "Matcher": ExtIdMatcher}


class Ext:
    def __init__(self, source, ext_code):
        self.source = source
        self.ext_code = ext_code


class EnumLike:
    def __init__(self, value):
        self.value = value


def test_build_mongo_maps_returns_empty_frame_without_ids():
    df = total_alert.build_mongo_maps(set())

    assert df.empty
    assert list(df.columns) == ["id", "farm_id", "farm_poligons_id"]


def test_build_mongo_maps_returns_blank_columns_without_orm(monkeypatch):
    monkeypatch.setattr(total_alert, "HAS_ORM", False)

    df = total_alert.build_mongo_maps({"123"})

    assert df["id"].tolist() == ["123"]
    assert df["farm_id"].tolist() == [""]


def test_build_mongo_maps_resolves_sitcode_and_polygons(orm_farms, monkeypatch):
    Farm, FarmPolygons = orm_farms["Farm"], orm_farms["FarmPolygons"]
    Farm.reset([Farm(id="farm1", ext_id=[Ext("SIT_CODE", "123")])])
    FarmPolygons.reset([FarmPolygons(id="poly1", farm_id=Ref("farm1"))])
    monkeypatch.setattr(Farm, "objects", orm_farms["Matcher"](Farm._store))
    monkeypatch.setattr(FarmPolygons, "objects", orm_farms["Matcher"](FarmPolygons._store))

    df = total_alert.build_mongo_maps({"123"})

    assert df.iloc[0].tolist() == ["123", "farm1", "poly1"]


def test_build_mongo_maps_falls_back_to_geofarmer_ids(orm_farms, monkeypatch):
    Farm, FarmPolygons = orm_farms["Farm"], orm_farms["FarmPolygons"]
    # El ext_code viene con el prefijo FARM_ID_ que normalize_farm_id retira.
    Farm.reset([Farm(id="farm9", ext_id=[Ext("GEOFARMER_ID", "FARM_ID_777")])])
    FarmPolygons.reset([])
    monkeypatch.setattr(Farm, "objects", orm_farms["Matcher"](Farm._store))
    monkeypatch.setattr(FarmPolygons, "objects", orm_farms["Matcher"](FarmPolygons._store))

    df = total_alert.build_mongo_maps({"777"})

    assert df.iloc[0]["farm_id"] == "farm9"
    assert df.iloc[0]["farm_poligons_id"] == ""


def test_build_mongo_maps_accepts_enum_like_sources(orm_farms, monkeypatch):
    Farm, FarmPolygons = orm_farms["Farm"], orm_farms["FarmPolygons"]

    class EnumMatcher(orm_farms["Matcher"]):
        def __call__(self, **filters):
            # Compara por .value cuando el source es un enum.
            source = filters.get("ext_id__source")
            codes = set(filters.get("ext_id__ext_code__in", []))
            if "farm_id__in" in filters:
                return super().__call__(**filters)
            items = [
                d
                for d in self.store
                if any(
                    getattr(e.source, "value", e.source) == source and e.ext_code in codes
                    for e in (d.ext_id or [])
                )
            ]

            class QS:
                def __init__(self, its):
                    self._its = its

                def only(self, *fields):
                    return self

                def __iter__(self):
                    return iter(self._its)

            return QS(items)

    Farm.reset([Farm(id="farm1", ext_id=[Ext(EnumLike("SIT_CODE"), "123")])])
    FarmPolygons.reset([])
    monkeypatch.setattr(Farm, "objects", EnumMatcher(Farm._store))
    monkeypatch.setattr(FarmPolygons, "objects", EnumMatcher(FarmPolygons._store))

    df = total_alert.build_mongo_maps({"123"})

    assert df.iloc[0]["farm_id"] == "farm1"


def test_build_mongo_maps_reports_unmatched_ids(orm_farms, monkeypatch, capsys):
    Farm, FarmPolygons = orm_farms["Farm"], orm_farms["FarmPolygons"]
    Farm.reset([])
    FarmPolygons.reset([])
    monkeypatch.setattr(Farm, "objects", orm_farms["Matcher"](Farm._store))
    monkeypatch.setattr(FarmPolygons, "objects", orm_farms["Matcher"](FarmPolygons._store))

    df = total_alert.build_mongo_maps({"999"})

    assert df.iloc[0]["farm_id"] == ""
    assert "sin match" in capsys.readouterr().out


def test_build_mongo_maps_skips_farms_with_broken_ext_ids(orm_farms, monkeypatch):
    Farm, FarmPolygons = orm_farms["Farm"], orm_farms["FarmPolygons"]

    class ExtRoto:
        @property
        def source(self):
            raise RuntimeError("ext_id corrupto")

    class MatcherTodo(orm_farms["Matcher"]):
        def __call__(self, **filters):
            if "farm_id__in" in filters:
                return super().__call__(**filters)

            class QS:
                def __init__(self, its):
                    self._its = its

                def only(self, *fields):
                    return self

                def __iter__(self):
                    return iter(self._its)

            return QS(list(self.store))

    Farm.reset([Farm(id="farm1", ext_id=[ExtRoto()])])
    FarmPolygons.reset([])
    monkeypatch.setattr(Farm, "objects", MatcherTodo(Farm._store))
    monkeypatch.setattr(FarmPolygons, "objects", MatcherTodo(FarmPolygons._store))

    df = total_alert.build_mongo_maps({"123"})

    assert df.iloc[0]["farm_id"] == ""


def test_build_mongo_maps_returns_blank_columns_on_query_error(orm_farms, monkeypatch):
    class Explota:
        @staticmethod
        def objects(**kwargs):
            raise RuntimeError("mongo caído")

    monkeypatch.setattr(total_alert, "Farm", Explota)

    df = total_alert.build_mongo_maps({"123"})

    assert df["farm_id"].tolist() == [""]


# ===================== process_year_source =====================


@pytest.fixture
def legacy_dirs(tmp_path, monkeypatch):
    monkeypatch.setitem(total_alert.config, "OUTPUT_CSV", str(tmp_path / "{PERIODO}" / "d.csv"))
    monkeypatch.setitem(
        total_alert.config, "MOVEMENT_RISK_OUTPUT_CSV", str(tmp_path / "mov" / "{PERIODO}" / "m.csv")
    )
    monkeypatch.setitem(
        total_alert.config, "TOTAL_RISK_OUTPUT_CSV", str(tmp_path / "tot" / "{PERIODO}" / "t.csv")
    )
    alerts = tmp_path / "annual" / "SMBYC" / "acme"
    movs = tmp_path / "mov" / "annual" / "SMBYC" / "acme"
    total = tmp_path / "tot" / "annual" / "SMBYC" / "acme"
    alerts.mkdir(parents=True)
    movs.mkdir(parents=True)
    return tmp_path, alerts, movs, total


def _metrics_cache(ids=("123",)):
    return pd.DataFrame(
        {
            "id": list(ids),
            "farming_in_ha": [1.0] * len(ids),
            "farming_in_prop": [0.1] * len(ids),
            "farming_out_ha": [9.0] * len(ids),
            "farming_out_prop": [0.9] * len(ids),
            "protected_ha": [0.0] * len(ids),
            "protected_prop": [0.0] * len(ids),
        }
    )


def test_process_year_source_merges_direct_movement_and_metrics(legacy_dirs):
    _, alerts, movs, total = legacy_dirs
    pd.DataFrame({"id": ["00123"], "direct_alert": ["True"], "deforested_ha": ["2.5"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    pd.DataFrame({"id": ["123"], "n_in": ["2"], "indirect_alert_in": ["True"]}).to_csv(
        movs / "smbyc_movement_alerts_ACME_2024.csv", index=False
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(),
        pd.DataFrame({"id": ["123"], "farm_id": ["f1"], "farm_poligons_id": ["p1"]}),
    )

    df = pd.read_csv(salida, dtype=str)
    assert df.iloc[0]["id"] == "123"
    assert df.iloc[0]["farm_id"] == "f1"
    assert df.iloc[0]["n_in"] == "2"
    assert df.iloc[0]["farming_in_ha"] == "1.0"


def test_process_year_source_fills_no_info_without_movement_file(legacy_dirs):
    _, alerts, movs, total = legacy_dirs
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
    )

    df = pd.read_csv(salida, dtype=str)
    assert df.iloc[0]["n_total_mov"] == "no_info"
    assert df.iloc[0]["farm_id"] != df.iloc[0]["farm_id"] or df.iloc[0].get("farm_id", "") in ("", None) or pd.isna(df.iloc[0]["farm_id"])


def test_process_year_source_derives_direct_alert_from_alternative_columns(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"id": ["123"], "intersect_early_warnings": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
    )

    assert pd.read_csv(salida, dtype=str).iloc[0]["direct_alert"] == "True"


def test_process_year_source_defaults_direct_alert_to_false(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"id": ["123"], "otra": ["x"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
    )

    df = pd.read_csv(salida, dtype=str)
    assert df.iloc[0]["direct_alert"] == "False"
    assert df.iloc[0]["deforested_ha"] == "no data"


def test_process_year_source_uses_geofarmer_fallback(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"id": ["777"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    # La finca 777 aparece en el mapeo pero sin farm_id resuelto por sit_code;
    # otra fila la relaciona por GEOFARMER_ID y esa es la que rellena el hueco.
    mapa = pd.DataFrame(
        {
            "id": ["777", "otro"],
            "farm_id": ["", "f9"],
            "farm_poligons_id": ["", "p9"],
            "GEOFARMER_ID": ["", "777"],
        }
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(["777"]), mapa
    )

    df = pd.read_csv(salida, dtype=str)
    assert df.iloc[0]["farm_id"] == "f9"
    assert df.iloc[0]["farm_poligons_id"] == "p9"


def test_process_year_source_geofarmer_fallback_ignores_unmatched_ids(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"id": ["777"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    # OJO: si el id no está en absoluto en el mapeo, el merge deja NaN (no ""),
    # y la máscara `== ""` de esta función no lo detecta, así que el respaldo por
    # GEOFARMER_ID no se aplica. La API nueva (calculate_total_risk) sí contempla
    # el NaN; se fija aquí la diferencia de comportamiento.
    mapa = pd.DataFrame(
        {"id": ["otro"], "farm_id": ["f9"], "farm_poligons_id": ["p9"], "GEOFARMER_ID": ["777"]}
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(["777"]), mapa
    )

    df = pd.read_csv(salida)
    assert pd.isna(df.iloc[0]["farm_id"])


def test_process_year_source_recreates_columns_lost_to_merge_suffixes(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    # Las métricas ya traen farm_id/farm_poligons_id, así que el merge con el
    # mapeo genera sufijos _x/_y y las columnas base desaparecen: la función
    # debe volver a crearlas vacías.
    metrics = _metrics_cache()
    metrics["farm_id"] = "previo"
    metrics["farm_poligons_id"] = "previo"

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", metrics,
        pd.DataFrame({"id": ["123"], "farm_id": ["f1"], "farm_poligons_id": ["p1"]}),
    )

    df = pd.read_csv(salida, dtype=str, keep_default_na=False)
    assert df.iloc[0]["farm_id"] == ""
    assert "farm_id_x" in df.columns


def test_process_year_source_returns_none_without_direct_file(legacy_dirs):
    tmp_path, _, _, total = legacy_dirs

    assert (
        total_alert.process_year_source(
            "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
        )
        is None
    )
    assert (total / "smbyc_total_log_ACME_2024.txt").exists()


def test_process_year_source_returns_none_when_direct_lacks_id(legacy_dirs):
    _, alerts, _, _ = legacy_dirs
    pd.DataFrame({"otra": ["x"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )

    assert (
        total_alert.process_year_source(
            "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
        )
        is None
    )


def test_process_year_source_ignores_movement_without_id(legacy_dirs):
    _, alerts, movs, _ = legacy_dirs
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    pd.DataFrame({"otra": ["x"]}).to_csv(
        movs / "smbyc_movement_alerts_ACME_2024.csv", index=False
    )

    salida = total_alert.process_year_source(
        "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
    )

    assert pd.read_csv(salida, dtype=str).iloc[0]["n_in"] == "no_info"


def test_process_year_source_returns_none_when_saving_fails(legacy_dirs, monkeypatch):
    _, alerts, _, total = legacy_dirs
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        alerts / "smbyc_direct_alert_ACME_2024.csv", index=False
    )
    monkeypatch.setattr(
        pd.DataFrame, "to_csv", lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno"))
    )

    assert (
        total_alert.process_year_source(
            "ACME", "annual", "2024", "2024", "smbyc", _metrics_cache(), pd.DataFrame()
        )
        is None
    )


# ===================== main legacy =====================


@pytest.fixture
def legacy_main(monkeypatch, tmp_path):
    for key, value in {
        "EMPRESA": "ACME",
        "PERIODO": "annual",
        "YEARS": "2017-2018",
        "CRS_METROS": CRS,
        "FOLDER_GEOJSONS": str(tmp_path / "fincas"),
        "FARMING_FRONTIER_SHP": "",
        "SHP_PROTECTED": "",
        "TOTAL_RISK_OUTPUT_CSV": str(tmp_path / "tot" / "{PERIODO}" / "t.csv"),
    }.items():
        monkeypatch.setitem(total_alert.config, key, value)
    monkeypatch.setattr(total_alert, "setup_logging", lambda *a, **k: None)
    return tmp_path


def test_legacy_main_stops_without_ids(legacy_main, monkeypatch, capsys):
    monkeypatch.setattr(total_alert, "scan_ids_needed", lambda *a, **k: set())

    total_alert.main()

    assert "No hay IDs en insumos" in capsys.readouterr().out


def test_legacy_main_generates_outputs(legacy_main, monkeypatch, capsys):
    monkeypatch.setattr(total_alert, "scan_ids_needed", lambda *a, **k: {"123"})
    monkeypatch.setattr(total_alert, "build_mongo_maps", lambda ids: pd.DataFrame({"id": ["123"]}))
    monkeypatch.setattr(
        total_alert,
        "load_farms_geoms_reproject",
        lambda *a, **k: gpd.GeoDataFrame({"id": ["123"], "geometry": [box(0, 0, 10, 10)]}, crs=CRS),
    )
    monkeypatch.setattr(total_alert, "load_mask_gdf_strict", lambda *a, **k: None)
    monkeypatch.setattr(total_alert, "load_or_build_cache", lambda *a, **k: _metrics_cache())
    monkeypatch.setattr(
        total_alert, "process_year_source", lambda *a, **k: "/salida/archivo.csv"
    )

    total_alert.main()

    assert "Archivos generados" in capsys.readouterr().out


def test_legacy_main_warns_when_nothing_is_generated(legacy_main, monkeypatch, capsys):
    monkeypatch.setattr(total_alert, "scan_ids_needed", lambda *a, **k: {"123"})
    monkeypatch.setattr(total_alert, "build_mongo_maps", lambda ids: pd.DataFrame({"id": ["123"]}))
    monkeypatch.setattr(
        total_alert,
        "load_farms_geoms_reproject",
        lambda *a, **k: gpd.GeoDataFrame({"id": ["123"], "geometry": [box(0, 0, 10, 10)]}, crs=CRS),
    )
    monkeypatch.setattr(total_alert, "load_mask_gdf_strict", lambda *a, **k: None)
    monkeypatch.setattr(total_alert, "load_or_build_cache", lambda *a, **k: _metrics_cache())
    monkeypatch.setattr(total_alert, "process_year_source", lambda *a, **k: None)

    total_alert.main()

    assert "No se generaron archivos" in capsys.readouterr().out


def test_legacy_main_uses_raw_years_when_not_a_range(legacy_main, monkeypatch):
    monkeypatch.setitem(total_alert.config, "YEARS", "2024")
    vistos = []
    monkeypatch.setattr(
        total_alert, "scan_ids_needed", lambda years, *a, **k: (vistos.append(years), set())[1]
    )

    total_alert.main()

    assert vistos[0] == ["2024"]


# ===================== calculate_total_risk (API actual) =====================


@pytest.fixture
def modern_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "alertas"
    direct = ws / "results" / "smbyc" / "annual" / "direct_alerts"
    indirect = ws / "results" / "smbyc" / "annual" / "indirect_alerts"
    metrics = ws / "metrics"
    for d in (direct, indirect, metrics):
        d.mkdir(parents=True)
    monkeypatch.setattr(total_alert, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        total_alert,
        "pkg_total_risk",
        lambda **kwargs: kwargs["direct_df"][["id", "direct_alert"]].copy(),
    )
    return ws, direct, indirect, metrics


def test_calculate_total_risk_writes_one_file_per_period(modern_workspace):
    ws, direct, indirect, metrics = modern_workspace
    pd.DataFrame({"id": ["00123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )
    pd.DataFrame({"id": ["123"], "indirect_alert_in": ["True"]}).to_csv(
        indirect / "smbyc_indirect_alert_annual_2024.csv", index=False
    )
    pd.DataFrame({"id": ["123"], "farming_in_ha": ["1.0"]}).to_csv(
        metrics / "spatial_metrics.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc",
        period_type="annual",
        periods=["2024"],
        workspace_dir=str(ws),
        mongo_map_df=pd.DataFrame(
            {
                "id": ["123"],
                "farm_id": ["f1"],
                "farm_poligons_id": ["p1"],
                "GEOFARMER_ID": ["123"],
            }
        ),
    )

    assert result["success"] is True
    assert result["periods_processed"] == 1
    generado = pd.read_csv(Path(result["files_generated"][0]), dtype=str)
    assert generado.iloc[0]["farm_id"] == "f1"


def test_calculate_total_risk_without_precalculated_metrics(modern_workspace, capsys):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc",
        period_type="annual",
        periods=["2024"],
        workspace_dir=str(ws),
        use_precalculated_metrics=False,
        mongo_map_df=pd.DataFrame(),
    )

    assert result["success"] is True
    assert "No hay métricas precalculadas" in capsys.readouterr().out


def test_calculate_total_risk_handles_unreadable_metrics(modern_workspace, monkeypatch, capsys):
    ws, direct, _, metrics = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )
    (metrics / "spatial_metrics.csv").write_text("id\n123\n", encoding="utf-8")

    original = pd.read_csv

    def selective(path, *args, **kwargs):
        if "spatial_metrics" in str(path):
            raise ValueError("métricas corruptas")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(total_alert.pd, "read_csv", selective)

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    assert result["success"] is True
    assert "Error cargando métricas" in capsys.readouterr().out


def test_calculate_total_risk_skips_period_without_direct_alerts(modern_workspace):
    ws, _, _, _ = modern_workspace

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    assert result["success"] is False
    assert result["files_generated"] == []


def test_calculate_total_risk_skips_direct_csv_without_id(modern_workspace):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"otra": ["x"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    assert result["periods_processed"] == 0


def test_calculate_total_risk_derives_direct_alert_column(modern_workspace):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "intersect_deforestation": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    generado = pd.read_csv(Path(result["files_generated"][0]), dtype=str)
    assert generado.iloc[0]["direct_alert"] == "True"


def test_calculate_total_risk_defaults_direct_alert_to_false(modern_workspace):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "otra": ["x"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    generado = pd.read_csv(Path(result["files_generated"][0]), dtype=str)
    assert generado.iloc[0]["direct_alert"] == "False"


def test_calculate_total_risk_handles_indirect_csv_without_id(modern_workspace):
    ws, direct, indirect, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )
    pd.DataFrame({"otra": ["x"]}).to_csv(
        indirect / "smbyc_indirect_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    assert result["periods_processed"] == 1


def test_calculate_total_risk_ignores_unreadable_alert_csvs(modern_workspace, monkeypatch):
    ws, direct, indirect, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )
    pd.DataFrame({"id": ["123"]}).to_csv(
        indirect / "smbyc_indirect_alert_annual_2024.csv", index=False
    )

    original = pd.read_csv
    fallos = {"n": 0}

    def selective(path, *args, **kwargs):
        # Solo falla durante el escaneo inicial de IDs (usecols=['id']).
        if kwargs.get("usecols") == ["id"]:
            fallos["n"] += 1
            raise ValueError("csv ilegible")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(total_alert.pd, "read_csv", selective)

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    assert fallos["n"] == 2
    assert result["success"] is True


def test_calculate_total_risk_builds_mongo_map_when_not_provided(modern_workspace, monkeypatch):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )
    llamadas = []
    monkeypatch.setattr(
        total_alert,
        "build_mongo_maps",
        lambda ids: (llamadas.append(ids), pd.DataFrame({"id": ["123"], "farm_id": ["f1"], "farm_poligons_id": ["p1"]}))[1],
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"], workspace_dir=str(ws)
    )

    assert llamadas == [{"123"}]
    generado = pd.read_csv(Path(result["files_generated"][0]), dtype=str)
    assert generado.iloc[0]["farm_id"] == "f1"


def test_calculate_total_risk_leaves_columns_blank_without_mapping(modern_workspace):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws), mongo_map_df=pd.DataFrame(),
    )

    generado = pd.read_csv(Path(result["files_generated"][0]))
    assert generado["farm_id"].isna().all()


def test_calculate_total_risk_recreates_columns_lost_to_merge_suffixes(modern_workspace, monkeypatch):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    # El paquete devuelve ya una columna farm_id: el merge posterior la renombra
    # con sufijos y la función debe reponer las columnas esperadas.
    def pkg_con_farm_id(**kwargs):
        df = kwargs["direct_df"][["id", "direct_alert"]].copy()
        df["farm_id"] = "previo"
        df["farm_poligons_id"] = "previo"
        return df

    monkeypatch.setattr(total_alert, "pkg_total_risk", pkg_con_farm_id)

    result = total_alert.calculate_total_risk(
        source="smbyc", period_type="annual", periods=["2024"],
        workspace_dir=str(ws),
        mongo_map_df=pd.DataFrame(
            {
                "id": ["123"],
                "farm_id": ["f1"],
                "farm_poligons_id": ["p1"],
                "GEOFARMER_ID": ["123"],
            }
        ),
    )

    generado = pd.read_csv(Path(result["files_generated"][0]), dtype=str, keep_default_na=False)
    # Las columnas originales quedaron como farm_id_x / farm_id_y; la función
    # repone `farm_id` vacía y el respaldo por GEOFARMER_ID acaba rellenándola.
    assert "farm_id_x" in generado.columns
    assert "farm_id_y" in generado.columns
    assert generado.iloc[0]["farm_id"] == "f1"
    assert generado.iloc[0]["farm_poligons_id"] == "p1"


def test_calculate_total_risk_requires_geofarmer_column_in_supplied_map(modern_workspace):
    ws, direct, _, _ = modern_workspace
    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(
        direct / "smbyc_direct_alert_annual_2024.csv", index=False
    )

    # OJO: el filtrado inicial asume que un mapeo precargado siempre trae la
    # columna GEOFARMER_ID, cosa que build_mongo_maps no produce. Si el llamador
    # entrega un mapeo sin esa columna, la función revienta en vez de degradarse.
    with pytest.raises(KeyError, match="GEOFARMER_ID"):
        total_alert.calculate_total_risk(
            source="smbyc", period_type="annual", periods=["2024"],
            workspace_dir=str(ws),
            mongo_map_df=pd.DataFrame({"id": ["123"], "farm_id": ["f1"]}),
        )
