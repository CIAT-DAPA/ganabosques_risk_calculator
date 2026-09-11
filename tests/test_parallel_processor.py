import importlib
import sys
import types
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

# process_metrics_chunk importa esta función de forma perezosa dentro del worker,
# así que hay que sustituirla en el módulo de origen. Se resuelve con
# import_module y no con 'import ganabosques_risk_package.spatial_metrics as X'
# porque el __init__ del paquete re-exporta la función con el mismo nombre que el
# submódulo: con la forma 'import ... as' el nombre acabaría apuntando a la
# función y no al módulo, según la versión del paquete que haya instalada.
pkg_spatial_metrics_module = importlib.import_module(
    'ganabosques_risk_package.spatial_metrics'
)
import parallel_processor


# ===================== dobles compartidos =====================


class DummyPool:
    """Pool que ejecuta los chunks en el proceso actual, en orden."""

    instances = []

    def __init__(self, processes=None):
        self.processes = processes
        DummyPool.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, chunks):
        for chunk in chunks:
            yield func(chunk)


@pytest.fixture(autouse=True)
def reset_pool_instances():
    DummyPool.instances = []
    yield
    DummyPool.instances = []


@pytest.fixture
def stub_direct_alert(monkeypatch, tmp_path):
    """Sustituye el módulo direct_alert que parallel_processor importa perezosamente."""
    module = types.ModuleType("direct_alert")
    module.calls = []

    def make_out_dir_and_paths(template, ctx, source, deforestation_type=None):
        out_dir = tmp_path / "results" / source / (deforestation_type or "") / "direct_alerts"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{source}_direct_alert_{deforestation_type}_{ctx['YEARS']}.csv"
        return str(out_dir), str(out_dir / name)

    def calculate_direct_alerts(**kwargs):
        module.calls.append(kwargs)
        return {"success": True, "periods_processed": 1, "farms_processed": len(
            kwargs.get("_farm_files_subset") or []
        )}

    module.make_out_dir_and_paths = make_out_dir_and_paths
    module.calculate_direct_alerts = calculate_direct_alerts
    module.norm = lambda value: value
    module.format_placeholders = lambda value, ctx: value
    monkeypatch.setitem(sys.modules, "direct_alert", module)
    return module


# ===================== process_farm_chunk =====================


def test_process_farm_chunk_fails_without_temp_dir(stub_direct_alert):
    result = parallel_processor.process_farm_chunk((0, ["a.geojson"], {"farm_folder": "x"}))

    assert result == {"chunk_id": 0, "success": False, "error": "No temp_dir provided"}


def test_process_farm_chunk_isolates_output_and_forwards_subset(stub_direct_alert, tmp_path):
    params = {
        "_temp_dir": str(tmp_path / "temp"),
        "_workspace_dir": str(tmp_path),
        "farm_folder": str(tmp_path / "farms"),
        "farm_range": "1:100",
        "output_csv": str(tmp_path / "global.csv"),
        "source": "smbyc",
        "raster_paths_dict": {"2024": "raster.tif"},
    }

    result = parallel_processor.process_farm_chunk((3, ["a.geojson", "b.geojson"], params))

    assert result["success"] is True
    assert result["chunk_id"] == 3
    assert result["temp_dir"] == str(tmp_path / "temp" / "chunk_3")

    call = stub_direct_alert.calls[-1]
    # Cada worker escribe en su propio directorio para no pisarse.
    assert call["output_csv"].startswith(str(tmp_path / "temp" / "chunk_3"))
    # El rango global se anula: el reparto lo define el chunk.
    assert call["farm_range"] == ""
    assert call["_farm_files_subset"] == ["a.geojson", "b.geojson"]
    # Los rasters ya resueltos se propagan para no volver a descargarlos.
    assert call["raster_paths_dict"] == {"2024": "raster.tif"}
    # Los parámetros internos no se reenvían.
    assert "_temp_dir" not in call
    assert "_workspace_dir" not in call


def test_process_farm_chunk_captures_worker_errors(stub_direct_alert, tmp_path):
    def boom(**kwargs):
        raise RuntimeError("worker reventó")

    stub_direct_alert.calculate_direct_alerts = boom
    params = {"_temp_dir": str(tmp_path / "temp"), "farm_folder": "x"}

    result = parallel_processor.process_farm_chunk((1, [], params))

    assert result["success"] is False
    assert result["error"] == "worker reventó"
    assert result["chunk_id"] == 1
    assert result["temp_dir"].endswith("chunk_1")


# ===================== combine_csv_results =====================


def _chunk_with_csv(base, chunk_id, name, rows):
    path = base / f"chunk_{chunk_id}" / "worker_output"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / name, index=False)
    return str(base / f"chunk_{chunk_id}")


def test_combine_csv_results_merges_chunks(tmp_path):
    name = "smbyc_direct_alert_annual_2024.csv"
    dirs = [
        _chunk_with_csv(tmp_path, 0, name, {"id": ["1"], "deforested_ha": [1.0]}),
        _chunk_with_csv(tmp_path, 1, name, {"id": ["2"], "deforested_ha": [2.0]}),
    ]
    output = tmp_path / "final" / "merged.csv"

    count = parallel_processor.combine_csv_results(
        dirs, str(output), "2024", "smbyc", "annual"
    )

    assert count == 2
    merged = pd.read_csv(output)
    assert sorted(merged["id"].tolist()) == [1, 2]


def test_combine_csv_results_drops_duplicate_ids(tmp_path):
    name = "smbyc_direct_alert_annual_2024.csv"
    dirs = [
        _chunk_with_csv(tmp_path, 0, name, {"id": ["1"], "deforested_ha": [1.0]}),
        _chunk_with_csv(tmp_path, 1, name, {"id": ["1"], "deforested_ha": [9.9]}),
    ]
    output = tmp_path / "merged.csv"

    count = parallel_processor.combine_csv_results(
        dirs, str(output), "2024", "smbyc", "annual"
    )

    assert count == 1
    # Se conserva la primera aparición.
    assert pd.read_csv(output)["deforested_ha"].tolist() == [1.0]


def test_combine_csv_results_uses_tolerant_fallback_pattern(tmp_path):
    # El nombre no coincide con el prefijo esperado, pero sí con el patrón laxo.
    dirs = [_chunk_with_csv(tmp_path, 0, "otra_direct_alert_xx_2024.csv", {"id": ["7"]})]
    output = tmp_path / "merged.csv"

    count = parallel_processor.combine_csv_results(
        dirs, str(output), "2024", "smbyc", "annual"
    )

    assert count == 1


def test_combine_csv_results_returns_zero_without_matching_files(tmp_path):
    empty = tmp_path / "chunk_0"
    empty.mkdir()

    count = parallel_processor.combine_csv_results(
        [str(empty)], str(tmp_path / "merged.csv"), "2024", "smbyc", "annual"
    )

    assert count == 0
    assert not (tmp_path / "merged.csv").exists()


def test_combine_csv_results_skips_unreadable_files(tmp_path, monkeypatch):
    name = "smbyc_direct_alert_annual_2024.csv"
    good = _chunk_with_csv(tmp_path, 0, name, {"id": ["1"]})
    bad = _chunk_with_csv(tmp_path, 1, name, {"id": ["2"]})

    original = pd.read_csv

    def selective(path, *args, **kwargs):
        if "chunk_1" in str(path):
            raise ValueError("csv corrupto")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(parallel_processor.pd, "read_csv", selective)

    count = parallel_processor.combine_csv_results(
        [good, bad], str(tmp_path / "merged.csv"), "2024", "smbyc", "annual"
    )

    # El chunk ilegible se descarta pero el lote no se pierde.
    assert count == 1


# ===================== cleanup_temp_dirs =====================


def test_cleanup_temp_dirs_removes_directories(tmp_path):
    target = tmp_path / "chunk_0"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "f.txt").write_text("x", encoding="utf-8")

    parallel_processor.cleanup_temp_dirs([str(target)])

    assert not target.exists()


def test_cleanup_temp_dirs_tolerates_failures(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(OSError("bloqueado"))
    )

    # No debe propagar la excepción.
    parallel_processor.cleanup_temp_dirs([str(tmp_path / "no_existe")])


# ===================== run_parallel_direct_alerts =====================


def _base_params(tmp_path, farm_folder):
    return {
        "farm_folder": str(farm_folder),
        "years": ["2024"],
        "source": "smbyc",
        "period_type": "annual",
        "output_csv": str(tmp_path / "out.csv"),
        "_workspace_dir": str(tmp_path),
    }


def test_run_parallel_direct_alerts_splits_folder_listing(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    for i in range(5):
        (farm_folder / f"farm{i}.geojson").write_text("{}", encoding="utf-8")
    (farm_folder / "ignorar.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    seen_chunks = []

    def fake_chunk(chunk):
        seen_chunks.append(chunk)
        return {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path / f"t{chunk[0]}")}

    monkeypatch.setattr(parallel_processor, "process_farm_chunk", fake_chunk)
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 5)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    result = parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=2
    )

    assert result["success"] is True
    assert result["num_workers"] == 2
    assert result["chunks_processed"] == 2
    assert result["chunks_failed"] == 0
    assert result["farms_per_period"] == 5
    assert result["combined_stats"] == {"2024": 5}
    # Los .txt se ignoran y el reparto cubre los 5 geojson sin solapamiento.
    repartidos = [f for chunk in seen_chunks for f in chunk[1]]
    assert sorted(repartidos) == [f"farm{i}.geojson" for i in range(5)]
    assert DummyPool.instances[0].processes == 2


def test_run_parallel_direct_alerts_prefers_database_metadata(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "ignorada.geojson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    seen = []
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: (seen.append(chunk), {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path)})[1],
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 2)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    params = _base_params(tmp_path, farm_folder)
    params["_farms_metadata"] = [
        {"sitcode": "A1"},
        {"sitcode": "A2"},
        {"sitcode": None},  # sin sitcode: se descarta
    ]

    result = parallel_processor.run_parallel_direct_alerts(params, num_workers=1)

    assert result["farms_per_period"] == 2
    assert seen[0][1] == ["A1.geojson", "A2.geojson"]


def test_run_parallel_direct_alerts_applies_farm_limit(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    for i in range(10):
        (farm_folder / f"farm{i:02d}.geojson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path)},
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 3)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    params = _base_params(tmp_path, farm_folder)
    params["_farm_limit"] = 3

    result = parallel_processor.run_parallel_direct_alerts(params, num_workers=1)

    assert result["farms_per_period"] == 3


def test_run_parallel_direct_alerts_stops_creating_empty_chunks(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "unica.geojson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path)},
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    # 4 workers para 1 finca: solo debe crearse un chunk.
    result = parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=4
    )

    assert result["chunks_processed"] == 1


def test_run_parallel_direct_alerts_reports_failed_chunks(tmp_path, monkeypatch, capsys, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    for i in range(2):
        (farm_folder / f"farm{i}.geojson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: (
            {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path)}
            if chunk[0] == 0
            else {"success": False, "chunk_id": chunk[0], "error": "raster ausente"}
        ),
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    result = parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=2
    )

    assert result["chunks_processed"] == 1
    assert result["chunks_failed"] == 1
    assert "raster ausente" in capsys.readouterr().out


def test_run_parallel_direct_alerts_defaults_workers_to_capped_cpu_count(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "farm.geojson").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "cpu_count", lambda: 32)
    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path)},
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    result = parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=None
    )

    # El tope es 4 aunque la máquina tenga más núcleos.
    assert result["num_workers"] == 4


def test_run_parallel_direct_alerts_cleans_temp_dirs_when_requested(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "farm.geojson").write_text("{}", encoding="utf-8")

    cleaned = []
    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path / "t0")},
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: cleaned.extend(dirs))

    parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=1, cleanup=True
    )
    assert cleaned == [str(tmp_path / "t0")]

    cleaned.clear()
    parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=1, cleanup=False
    )
    assert cleaned == []


# ===================== process_metrics_chunk =====================


def test_process_metrics_chunk_builds_geodataframe_from_cache(monkeypatch):
    captured = {}

    def fake_pkg(plots, farming_areas, protected_areas, crs, id_column, show_progress):
        captured["plots"] = plots
        captured["crs"] = crs
        return pd.DataFrame({"id": list(plots["id"]), "total_ha": [1.0] * len(plots)})

    monkeypatch.setattr(
        pkg_spatial_metrics_module,
        "spatial_metrics",
        fake_pkg,
    )

    params = {
        "_geometries_cache": {"1": box(0, 0, 10, 10), "2": box(10, 0, 20, 10)},
        "_frontier_gdf": None,
        "_protected_gdf": None,
        "crs": "EPSG:3116",
    }

    result = parallel_processor.process_metrics_chunk((0, ["1", "2"], params))

    assert result["success"] is True
    assert result["farms_processed"] == 2
    assert result["results"][0]["id"] == "1"
    assert isinstance(captured["plots"], gpd.GeoDataFrame)
    assert captured["crs"] == "EPSG:3116"


def test_process_metrics_chunk_fails_when_cache_has_no_geometries():
    params = {"_geometries_cache": {"1": box(0, 0, 1, 1)}}

    result = parallel_processor.process_metrics_chunk((2, ["desconocida"], params))

    assert result == {
        "chunk_id": 2,
        "success": False,
        "error": "No geometries found in cache",
        "results": [],
    }


def test_process_metrics_chunk_captures_errors(monkeypatch):
    monkeypatch.setattr(
        pkg_spatial_metrics_module,
        "spatial_metrics",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("métricas rotas")),
    )

    params = {"_geometries_cache": {"1": box(0, 0, 1, 1)}}
    result = parallel_processor.process_metrics_chunk((0, ["1"], params))

    assert result["success"] is False
    assert result["error"] == "métricas rotas"
    assert result["results"] == []


# ===================== run_parallel_spatial_metrics =====================


def test_run_parallel_spatial_metrics_combines_chunk_results(monkeypatch):
    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_metrics_chunk",
        lambda chunk: {
            "success": True,
            "chunk_id": chunk[0],
            "results": [{"id": farm_id, "total_ha": 1.0} for farm_id in chunk[1]],
        },
    )

    params = {"_geometries_cache": {"1": None, "2": None, "3": None}}
    result = parallel_processor.run_parallel_spatial_metrics(params, num_workers=2)

    assert result["success"] is True
    assert result["chunks_processed"] == 2
    assert result["farms_processed"] == 3
    assert sorted(result["results_df"]["id"].tolist()) == ["1", "2", "3"]


def test_run_parallel_spatial_metrics_requires_geometry_cache():
    assert parallel_processor.run_parallel_spatial_metrics({}) == {
        "success": False,
        "error": "No geometry cache available",
    }


def test_run_parallel_spatial_metrics_reports_failed_chunks(monkeypatch, capsys):
    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_metrics_chunk",
        lambda chunk: (
            {"success": True, "chunk_id": chunk[0], "results": [{"id": "1"}]}
            if chunk[0] == 0
            else {"success": False, "chunk_id": chunk[0], "error": "sin geometrías"}
        ),
    )

    params = {"_geometries_cache": {"1": None, "2": None}}
    result = parallel_processor.run_parallel_spatial_metrics(params, num_workers=2)

    assert result["chunks_failed"] == 1
    assert "sin geometrías" in capsys.readouterr().out


def test_run_parallel_spatial_metrics_defaults_workers(monkeypatch):
    monkeypatch.setattr(parallel_processor, "cpu_count", lambda: 2)
    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_metrics_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "results": []},
    )

    result = parallel_processor.run_parallel_spatial_metrics(
        {"_geometries_cache": {"1": None}}, num_workers=None
    )

    assert result["num_workers"] == 2


def test_run_parallel_direct_alerts_tolerates_non_empty_temp_base(tmp_path, monkeypatch, stub_direct_alert):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "farm.geojson").write_text("{}", encoding="utf-8")

    # Un residuo de una corrida anterior impide borrar el directorio base;
    # la limpieza debe seguir adelante sin romper el pipeline.
    temp_base = tmp_path / "alertas" / "temp_parallel"
    temp_base.mkdir(parents=True)
    (temp_base / "residuo.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(
        parallel_processor,
        "process_farm_chunk",
        lambda chunk: {"success": True, "chunk_id": chunk[0], "temp_dir": str(tmp_path / "t0")},
    )
    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda dirs: None)

    result = parallel_processor.run_parallel_direct_alerts(
        _base_params(tmp_path, farm_folder), num_workers=1, cleanup=True
    )

    assert result["success"] is True
    assert temp_base.exists()
