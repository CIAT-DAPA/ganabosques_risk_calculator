import io
import sys
import types
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import box

import spatial_metrics


CRS = "EPSG:3116"


# ===================== dobles =====================


class FakeDataManager:
    """DataManager mínimo con la superficie que usa spatial_metrics."""

    def __init__(self, workspace_dir, geometries=None, cache_loaded=False):
        self.workspace_dir = Path(workspace_dir)
        self.geojsons_dir = self.workspace_dir / "farms" / "geojsons"
        self.geojsons_dir.mkdir(parents=True, exist_ok=True)
        self._geometries = geometries or {}
        self._cache_loaded = cache_loaded

    def is_geometry_cache_loaded(self):
        return self._cache_loaded

    def get_all_cached_geometries(self):
        return self._geometries

    def get_farms_geodataframe(self, expected_crs=CRS):
        return gpd.GeoDataFrame(
            {"id": list(self._geometries), "geometry": list(self._geometries.values())},
            crs=expected_crs,
        )


def _shapezip_bytes(gdf, name="capa"):
    """Construye en memoria un ZIP con un shapefile, como devuelve el WFS."""
    import tempfile

    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        shp = Path(tmp) / f"{name}.shp"
        gdf.to_file(shp)
        with zipfile.ZipFile(buffer, "w") as zf:
            for part in Path(tmp).iterdir():
                zf.write(part, part.name)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload=b"", status_error=None):
        self._payload = payload
        self._status_error = status_error
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


@pytest.fixture
def frontier_gdf():
    return gpd.GeoDataFrame({"name": ["frontera"], "geometry": [box(0, 0, 50, 100)]}, crs=CRS)


# ===================== download_and_extract_wfs =====================


def test_download_and_extract_wfs_returns_cached_file_without_network(tmp_path, monkeypatch):
    out_dir = tmp_path / "reference_layers" / "capa"
    out_dir.mkdir(parents=True)
    cached = out_dir / "capa.gpkg"
    cached.write_bytes(b"gpkg")

    def fail(*a, **k):
        raise AssertionError("no debería descargar si hay caché")

    monkeypatch.setattr(spatial_metrics.requests, "get", fail)

    assert spatial_metrics.download_and_extract_wfs("http://x", out_dir, "capa", CRS) == cached


def test_download_and_extract_wfs_writes_geopackage(tmp_path, monkeypatch, frontier_gdf):
    payload = _shapezip_bytes(frontier_gdf.to_crs("EPSG:4326"), "capa")
    monkeypatch.setattr(
        spatial_metrics.requests, "get", lambda url, stream=True, timeout=120: FakeResponse(payload)
    )

    out_dir = tmp_path / "reference_layers" / "capa"
    result = spatial_metrics.download_and_extract_wfs("http://x", out_dir, "capa", CRS)

    assert result == out_dir / "capa.gpkg"
    written = gpd.read_file(result)
    assert written.crs.to_string() == CRS
    assert len(written) == 1


def test_download_and_extract_wfs_assigns_crs_when_shapefile_has_none(tmp_path, monkeypatch, frontier_gdf):
    sin_crs = frontier_gdf.copy()
    sin_crs.crs = None
    payload = _shapezip_bytes(sin_crs, "capa")
    monkeypatch.setattr(
        spatial_metrics.requests, "get", lambda url, stream=True, timeout=120: FakeResponse(payload)
    )

    result = spatial_metrics.download_and_extract_wfs(
        "http://x", tmp_path / "capa", "capa", CRS
    )

    assert result is not None
    assert gpd.read_file(result).crs.to_string() == CRS


def test_download_and_extract_wfs_returns_none_on_network_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        spatial_metrics.requests,
        "get",
        lambda url, stream=True, timeout=120: FakeResponse(
            status_error=requests.exceptions.ConnectionError("sin red")
        ),
    )

    assert spatial_metrics.download_and_extract_wfs("http://x", tmp_path / "c", "c", CRS) is None


def test_download_and_extract_wfs_returns_none_for_corrupt_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        spatial_metrics.requests,
        "get",
        lambda url, stream=True, timeout=120: FakeResponse(b"esto no es un zip"),
    )

    assert spatial_metrics.download_and_extract_wfs("http://x", tmp_path / "c", "c", CRS) is None


def test_download_and_extract_wfs_returns_none_when_zip_has_no_shapefile(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("leeme.txt", "sin shapefile")
    monkeypatch.setattr(
        spatial_metrics.requests,
        "get",
        lambda url, stream=True, timeout=120: FakeResponse(buffer.getvalue()),
    )

    assert spatial_metrics.download_and_extract_wfs("http://x", tmp_path / "c", "c", CRS) is None


# ===================== load_reference_layer =====================


def test_load_reference_layer_returns_none_for_missing_file(tmp_path):
    assert spatial_metrics.load_reference_layer(str(tmp_path / "no.gpkg"), CRS, "Test") is None


def test_load_reference_layer_reads_and_indexes(tmp_path, frontier_gdf):
    path = tmp_path / "capa.gpkg"
    frontier_gdf.to_file(path, driver="GPKG")

    gdf = spatial_metrics.load_reference_layer(str(path), CRS, "Frontera")

    assert gdf is not None
    assert len(gdf) == 1
    assert gdf.crs.to_string() == CRS


def test_load_reference_layer_reprojects_when_crs_differs(tmp_path, frontier_gdf):
    path = tmp_path / "capa.gpkg"
    frontier_gdf.to_crs("EPSG:4326").to_file(path, driver="GPKG")

    gdf = spatial_metrics.load_reference_layer(str(path), CRS, "Frontera")

    assert gdf.crs.to_string() == CRS


def test_load_reference_layer_returns_none_for_empty_layer(tmp_path, monkeypatch, frontier_gdf):
    path = tmp_path / "capa.gpkg"
    frontier_gdf.to_file(path, driver="GPKG")

    vacio = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS)
    monkeypatch.setattr(spatial_metrics.gpd, "read_file", lambda *a, **k: vacio)

    assert spatial_metrics.load_reference_layer(str(path), CRS, "Frontera") is None


def test_load_reference_layer_returns_none_when_crs_is_missing(tmp_path, monkeypatch, frontier_gdf):
    path = tmp_path / "capa.gpkg"
    frontier_gdf.to_file(path, driver="GPKG")

    sin_crs = frontier_gdf.copy()
    sin_crs.crs = None
    monkeypatch.setattr(spatial_metrics.gpd, "read_file", lambda *a, **k: sin_crs)

    assert spatial_metrics.load_reference_layer(str(path), CRS, "Frontera") is None


def test_load_reference_layer_survives_missing_spatial_index(tmp_path, monkeypatch, frontier_gdf):
    path = tmp_path / "capa.gpkg"
    frontier_gdf.to_file(path, driver="GPKG")

    class SinIndice(gpd.GeoDataFrame):
        @property
        def sindex(self):
            raise RuntimeError("rtree no disponible")

    monkeypatch.setattr(
        spatial_metrics.gpd, "read_file", lambda *a, **k: SinIndice(frontier_gdf.copy())
    )

    gdf = spatial_metrics.load_reference_layer(str(path), CRS, "Frontera")

    assert gdf is not None


# ===================== load_or_download_reference_layer =====================


def test_load_or_download_uses_local_cache(tmp_path, frontier_gdf):
    reference_dir = tmp_path / "reference_layers" / "upra_boundaries"
    reference_dir.mkdir(parents=True)
    frontier_gdf.to_file(reference_dir / "upra_boundaries.gpkg", driver="GPKG")

    gdf = spatial_metrics.load_or_download_reference_layer(
        tmp_path, "upra_boundaries", "http://x", CRS, "Frontera"
    )

    assert gdf is not None


def test_load_or_download_downloads_when_cache_is_missing(tmp_path, monkeypatch, frontier_gdf):
    def fake_download(url, output_dir, layer_name, expected_crs):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{layer_name}.gpkg"
        frontier_gdf.to_file(path, driver="GPKG")
        return path

    monkeypatch.setattr(spatial_metrics, "download_and_extract_wfs", fake_download)

    gdf = spatial_metrics.load_or_download_reference_layer(
        tmp_path, "pnn_areas", "http://x", CRS, "PNN"
    )

    assert gdf is not None


def test_load_or_download_returns_none_when_download_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(spatial_metrics, "download_and_extract_wfs", lambda *a, **k: None)

    assert (
        spatial_metrics.load_or_download_reference_layer(
            tmp_path, "pnn_areas", "http://x", CRS, "PNN"
        )
        is None
    )


# ===================== load_farm_geometries =====================


def _write_geojson(path, geom, crs=CRS):
    gpd.GeoDataFrame({"geometry": [geom]}, crs=crs).to_file(path, driver="GeoJSON")


def test_load_farm_geometries_prefers_memory_cache(tmp_path):
    dm = FakeDataManager(
        tmp_path,
        geometries={"FARM_ID_00123": box(0, 0, 10, 10), "": box(0, 0, 1, 1)},
        cache_loaded=True,
    )

    gdf = spatial_metrics.load_farm_geometries(None, dm, CRS)

    # El ID vacío se descarta y el prefijo se normaliza.
    assert gdf["id"].tolist() == ["123"]


def test_load_farm_geometries_reads_from_disk_using_metadata(tmp_path):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    _write_geojson(dm.geojsons_dir / "A2.geojson", box(10, 0, 20, 10))

    metadata = [
        {"sitcode": "A1"},
        {"sitcode": "A2"},
        {"sitcode": None},          # sin sitcode
        {"sitcode": "NO_EXISTE"},   # sin geojson en disco
        {"sitcode": "nan"},         # se normaliza a cadena vacía
    ]

    gdf = spatial_metrics.load_farm_geometries(metadata, dm, CRS)

    assert sorted(gdf["id"].tolist()) == ["A1", "A2"]


def test_load_farm_geometries_offline_lists_folder_and_applies_limit(tmp_path):
    dm = FakeDataManager(tmp_path)
    for name in ("A1", "A2", "A3"):
        _write_geojson(dm.geojsons_dir / f"{name}.geojson", box(0, 0, 5, 5))

    gdf = spatial_metrics.load_farm_geometries(None, dm, CRS, farm_limit=2)

    assert len(gdf) == 2


def test_load_farm_geometries_reprojects_and_assigns_missing_crs(tmp_path):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(-74, 4, -73.9, 4.1), crs="EPSG:4326")

    gdf = spatial_metrics.load_farm_geometries([{"sitcode": "A1"}], dm, CRS)

    assert gdf.crs.to_string() == CRS


def test_load_farm_geometries_dissolves_multiple_polygons_per_farm(tmp_path):
    dm = FakeDataManager(tmp_path)
    gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)]}, crs=CRS
    ).to_file(dm.geojsons_dir / "A1.geojson", driver="GeoJSON")

    gdf = spatial_metrics.load_farm_geometries([{"sitcode": "A1"}], dm, CRS)

    assert len(gdf) == 1
    assert gdf.iloc[0]["geometry"].geom_type == "MultiPolygon"


def test_load_farm_geometries_skips_unreadable_geojsons(tmp_path):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    (dm.geojsons_dir / "A2.geojson").write_text("{ no es json", encoding="utf-8")

    gdf = spatial_metrics.load_farm_geometries(
        [{"sitcode": "A1"}, {"sitcode": "A2"}], dm, CRS
    )

    assert gdf["id"].tolist() == ["A1"]


def test_load_farm_geometries_skips_empty_geojsons(tmp_path, monkeypatch):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))

    vacio = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS)
    monkeypatch.setattr(spatial_metrics.gpd, "read_file", lambda *a, **k: vacio)

    with pytest.raises(RuntimeError, match="No se pudo cargar ninguna geometría"):
        spatial_metrics.load_farm_geometries([{"sitcode": "A1"}], dm, CRS)


def test_load_farm_geometries_raises_when_nothing_loads(tmp_path):
    dm = FakeDataManager(tmp_path)

    with pytest.raises(RuntimeError, match="No se pudo cargar ninguna geometría"):
        spatial_metrics.load_farm_geometries([{"sitcode": "NO_EXISTE"}], dm, CRS)


# ===================== calculate_spatial_metrics =====================


def _metrics_frame(ids):
    return pd.DataFrame(
        {
            "id": list(ids),
            "total_ha": [10.0] * len(ids),
            "farming_in_ha": [4.0] * len(ids),
            "farming_in_prop": [0.4] * len(ids),
            "farming_out_ha": [6.0] * len(ids),
            "farming_out_prop": [0.6] * len(ids),
            "protected_ha": [1.0] * len(ids),
            "protected_prop": [0.1] * len(ids),
        }
    )


@pytest.fixture
def stub_reference_layers(monkeypatch, frontier_gdf):
    monkeypatch.setattr(
        spatial_metrics, "load_or_download_reference_layer", lambda **kwargs: frontier_gdf
    )
    return frontier_gdf


def test_calculate_spatial_metrics_sequential_writes_csv(tmp_path, monkeypatch, stub_reference_layers):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 100, 100))

    monkeypatch.setattr(spatial_metrics, "pkg_spatial_metrics", lambda **kwargs: _metrics_frame(["A1"]))

    result = spatial_metrics.calculate_spatial_metrics([{"sitcode": "A1"}], dm)

    assert result["success"] is True
    assert result["farms_processed"] == 1
    output = Path(result["output_file"])
    assert output.name == "spatial_metrics.csv"
    assert output.parent == dm.workspace_dir / "metrics"
    assert output.exists()
    assert result["metrics"]["avg_total_ha"] == pytest.approx(10.0)
    assert result["metrics"]["farms_in_frontier"] == 1
    assert result["metrics"]["farms_in_protected"] == 1


def test_calculate_spatial_metrics_honours_custom_output(tmp_path, monkeypatch, stub_reference_layers):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    monkeypatch.setattr(spatial_metrics, "pkg_spatial_metrics", lambda **kwargs: _metrics_frame(["A1"]))

    destino = tmp_path / "salida"
    result = spatial_metrics.calculate_spatial_metrics(
        [{"sitcode": "A1"}], dm, output_dir=str(destino), output_name="metricas.csv"
    )

    assert Path(result["output_file"]) == destino / "metricas.csv"


def test_calculate_spatial_metrics_uses_parallel_path(tmp_path, monkeypatch, stub_reference_layers):
    dm = FakeDataManager(tmp_path, geometries={"A1": box(0, 0, 10, 10)}, cache_loaded=True)

    fake_module = types.ModuleType("parallel_processor")
    llamadas = []

    def fake_run(params, num_workers):
        llamadas.append((params, num_workers))
        return {"success": True, "results_df": _metrics_frame(["A1"])}

    fake_module.run_parallel_spatial_metrics = fake_run
    monkeypatch.setitem(sys.modules, "parallel_processor", fake_module)

    result = spatial_metrics.calculate_spatial_metrics(
        None, dm, use_parallel=True, num_workers=3
    )

    assert result["success"] is True
    params, workers = llamadas[0]
    assert workers == 3
    assert params["_geometries_cache"] == {"A1": dm._geometries["A1"]}
    assert params["crs"] == "EPSG:3116"


def test_calculate_spatial_metrics_raises_when_parallel_fails(tmp_path, monkeypatch, stub_reference_layers):
    dm = FakeDataManager(tmp_path, geometries={"A1": box(0, 0, 1, 1)}, cache_loaded=True)

    fake_module = types.ModuleType("parallel_processor")
    fake_module.run_parallel_spatial_metrics = lambda params, num_workers: {
        "success": False,
        "error": "sin caché",
    }
    monkeypatch.setitem(sys.modules, "parallel_processor", fake_module)

    with pytest.raises(RuntimeError, match="sin caché"):
        spatial_metrics.calculate_spatial_metrics(None, dm, use_parallel=True)


def test_calculate_spatial_metrics_falls_back_to_sequential_without_cache(tmp_path, monkeypatch, stub_reference_layers):
    # use_parallel=True pero el caché no está cargado: debe usar la ruta secuencial.
    dm = FakeDataManager(tmp_path, cache_loaded=False)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    monkeypatch.setattr(spatial_metrics, "pkg_spatial_metrics", lambda **kwargs: _metrics_frame(["A1"]))

    result = spatial_metrics.calculate_spatial_metrics([{"sitcode": "A1"}], dm, use_parallel=True)

    assert result["success"] is True


def test_calculate_spatial_metrics_requires_at_least_one_reference_layer(tmp_path, monkeypatch):
    dm = FakeDataManager(tmp_path)
    monkeypatch.setattr(spatial_metrics, "load_or_download_reference_layer", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="No se pudo cargar ninguna capa de referencia"):
        spatial_metrics.calculate_spatial_metrics([{"sitcode": "A1"}], dm)


# ===================== main =====================


@pytest.fixture
def stub_main_dependencies(monkeypatch, tmp_path):
    """Sustituye DataManager y el cálculo real para poder ejercitar main()."""
    dm_module = types.ModuleType("data_manager")
    estado = {"farms": [{"sitcode": "A1"}], "db_error": None}

    class StubDataManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.workspace_dir = tmp_path

        def load_farms_metadata(self, limit=None):
            estado["limit"] = limit
            return estado["farms"], estado["db_error"]

        def prepare_geojsons(self, farms_metadata):
            return len(farms_metadata)

    dm_module.DataManager = StubDataManager
    monkeypatch.setitem(sys.modules, "data_manager", dm_module)

    pp_module = types.ModuleType("parallel_processor")
    pp_module.run_parallel_spatial_metrics = lambda *a, **k: {"success": True}
    monkeypatch.setitem(sys.modules, "parallel_processor", pp_module)

    return estado


def test_main_runs_full_flow(monkeypatch, stub_main_dependencies, capsys):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py"])
    monkeypatch.setattr(spatial_metrics, "calculate_spatial_metrics", lambda *a, **k: {"success": True})

    spatial_metrics.main()

    assert "Proceso completado exitosamente" in capsys.readouterr().out
    assert stub_main_dependencies["limit"] is None


def test_main_reads_farm_limit_from_argv(monkeypatch, stub_main_dependencies):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py", "--farm-limit", "25"])
    monkeypatch.setattr(spatial_metrics, "calculate_spatial_metrics", lambda *a, **k: {"success": True})

    spatial_metrics.main()

    assert stub_main_dependencies["limit"] == 25


def test_main_ignores_farm_limit_without_value(monkeypatch, stub_main_dependencies):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py", "--farm-limit"])
    monkeypatch.setattr(spatial_metrics, "calculate_spatial_metrics", lambda *a, **k: {"success": True})

    spatial_metrics.main()

    assert stub_main_dependencies["limit"] is None


def test_main_exits_on_database_error(monkeypatch, stub_main_dependencies):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py"])
    stub_main_dependencies["db_error"] = "sin conexión"

    with pytest.raises(SystemExit) as exc:
        spatial_metrics.main()

    assert exc.value.code == 1


def test_main_exits_when_no_farms_found(monkeypatch, stub_main_dependencies):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py"])
    stub_main_dependencies["farms"] = []

    with pytest.raises(SystemExit) as exc:
        spatial_metrics.main()

    assert exc.value.code == 1


def test_main_exits_when_metrics_fail(monkeypatch, stub_main_dependencies):
    monkeypatch.setattr(sys, "argv", ["spatial_metrics.py"])
    monkeypatch.setattr(spatial_metrics, "calculate_spatial_metrics", lambda *a, **k: {"success": False})

    with pytest.raises(SystemExit) as exc:
        spatial_metrics.main()

    assert exc.value.code == 1


def test_load_farm_geometries_assigns_crs_to_geojson_without_projection(tmp_path, monkeypatch):
    dm = FakeDataManager(tmp_path)
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))

    # Un GeoJSON sin CRS declarado: se asume que ya viene en el CRS de trabajo.
    sin_crs = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]})
    monkeypatch.setattr(spatial_metrics.gpd, "read_file", lambda *a, **k: sin_crs)

    gdf = spatial_metrics.load_farm_geometries([{"sitcode": "A1"}], dm, CRS)

    assert gdf.crs.to_string() == CRS
    assert gdf["id"].tolist() == ["A1"]
