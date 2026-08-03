import sys
import types
from pathlib import Path

import pandas as pd

import parallel_processor


def test_parallel_processor_combines_csvs(tmp_path):
    temp_dir = tmp_path / "chunk1"
    temp_dir.mkdir(parents=True)
    (temp_dir / "nested").mkdir()
    pd.DataFrame({"id": ["1"], "value": [10]}).to_csv(temp_dir / "nested" / "smbyc_direct_alert_annual_2024.csv", index=False)

    output_csv = tmp_path / "merged.csv"
    count = parallel_processor.combine_csv_results([str(temp_dir)], str(output_csv), "2024", "smbyc", "annual")

    assert count == 1
    merged = pd.read_csv(output_csv)
    assert merged.iloc[0]["value"] == 10


def test_run_parallel_direct_alerts_uses_chunk_results(tmp_path, monkeypatch):
    farm_folder = tmp_path / "farms"
    farm_folder.mkdir()
    (farm_folder / "farm1.geojson").write_text("{}", encoding="utf-8")
    (farm_folder / "farm2.geojson").write_text("{}", encoding="utf-8")

    class DummyPool:
        def __init__(self, processes=None):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, chunks):
            for chunk in chunks:
                yield func(chunk)

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(parallel_processor, "process_farm_chunk", lambda chunk: {"success": True, "temp_dir": str(tmp_path / "tmp"), "chunk_id": chunk[0]})

    dummy_direct_alert = types.ModuleType("direct_alert")
    dummy_direct_alert.make_out_dir_and_paths = lambda template, ctx, source, deforestation_type=None: (Path(tmp_path), str(tmp_path / "out.csv"))
    dummy_direct_alert.norm = lambda value: value
    dummy_direct_alert.format_placeholders = lambda value, ctx: value
    monkeypatch.setitem(sys.modules, "direct_alert", dummy_direct_alert)

    monkeypatch.setattr(parallel_processor, "combine_csv_results", lambda *args, **kwargs: 1)
    monkeypatch.setattr(parallel_processor, "cleanup_temp_dirs", lambda temp_dirs: None)

    result = parallel_processor.run_parallel_direct_alerts(
        {"farm_folder": str(farm_folder), "years": ["2024"], "source": "smbyc", "period_type": "annual", "output_csv": str(tmp_path / "output.csv")},
        num_workers=2,
        cleanup=True,
    )

    assert result["success"] is True
    assert result["chunks_processed"] == 2
    assert result["combined_stats"]["2024"] == 1


def test_run_parallel_spatial_metrics_returns_combined_df(tmp_path, monkeypatch):
    class DummyPool:
        def __init__(self, processes=None):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, chunks):
            for chunk in chunks:
                yield func(chunk)

    monkeypatch.setattr(parallel_processor, "Pool", DummyPool)
    monkeypatch.setattr(parallel_processor, "process_metrics_chunk", lambda chunk: {"success": True, "results": [{"id": "1"}], "chunk_id": chunk[0]})

    result = parallel_processor.run_parallel_spatial_metrics({"_geometries_cache": {"1": "geom"}}, num_workers=1)

    assert result["success"] is True
    assert result["farms_processed"] == 1
    assert result["results_df"].iloc[0]["id"] == "1"
