import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import total_alert


def test_total_alert_helpers_and_cache(tmp_path, monkeypatch):
    assert total_alert.derive_mov_year("2017-2020", "anual") == "2017"
    assert total_alert.derive_mov_year("2017-2020", "cumulative") == "2020"
    assert total_alert.format_placeholders("{EMPRESA}/{YEARS}", {"EMPRESA": "ACME", "YEARS": "2024"}) == "ACME/2024"
    assert total_alert.sanitize_empresa_folder("Mi Empresa") == "mi_empresa"

    log_dir = tmp_path / "logs"
    total_alert.write_reason_log(str(log_dir), "test.log", "hello")
    assert (log_dir / "test.log").exists()

    files = [tmp_path / "a.geojson", tmp_path / "b.geojson", tmp_path / "c.txt"]
    for path in files[:-1]:
        path.write_text("{}", encoding="utf-8")
    files[-1].write_text("x", encoding="utf-8")
    assert total_alert.list_geojsons(str(tmp_path)) == [str(files[0]), str(files[1])]

    monkeypatch.setitem(total_alert.config, "TOTAL_RISK_OUTPUT_CSV", str(tmp_path / "out.csv"))
    farms = gpd.GeoDataFrame({"id": ["1", "2"], "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)]}, crs="EPSG:3116")
    frontier = gpd.GeoDataFrame({"name": ["f"], "geometry": [box(0, 0, 5, 10)]}, crs="EPSG:3116")
    protected = gpd.GeoDataFrame({"name": ["p"], "geometry": [box(0, 0, 2, 10)]}, crs="EPSG:3116")

    metrics = total_alert.compute_metrics_cache(farms, {"1"}, frontier, protected)
    assert metrics.iloc[0]["id"] == "1"
    assert metrics.iloc[0]["farming_in_ha"] >= 0

    cached = total_alert.load_or_build_cache("ACME", "annual", farms, {"1", "2"}, frontier, protected)
    assert len(cached) == 2


def test_scan_ids_needed_reads_csvs(tmp_path, monkeypatch):
    base = tmp_path / "results" / "annual" / "direct_alert" / "SMBYC" / "acme"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["FARM_ID_00123", "000456"]}).to_csv(base / "smbyc_direct_alert_ACME_2024.csv", index=False)

    movement_dir = tmp_path / "results" / "annual" / "movement" / "SMBYC" / "acme"
    movement_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["789"]}).to_csv(movement_dir / "smbyc_movement_alerts_ACME_2024.csv", index=False)

    monkeypatch.setitem(total_alert.config, "OUTPUT_CSV", str(tmp_path / "results" / "{PERIODO}" / "direct_alert" / "file.csv"))
    monkeypatch.setitem(total_alert.config, "MOVEMENT_RISK_OUTPUT_CSV", str(tmp_path / "results" / "{PERIODO}" / "movement" / "file.csv"))

    ids = total_alert.scan_ids_needed(["2024"], ["smbyc"], "ACME", "annual")
    assert ids == {"123", "456", "789"}


def test_calculate_total_risk_writes_outputs(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    results_dir = workspace / "results" / "smbyc" / "annual" / "direct_alerts"
    indirect_dir = workspace / "results" / "smbyc" / "annual" / "indirect_alerts"
    output_dir = workspace / "results" / "smbyc" / "annual" / "total_risk"
    metrics_dir = workspace / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    indirect_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": ["123"], "direct_alert": ["True"]}).to_csv(results_dir / "smbyc_direct_alert_annual_2024.csv", index=False)
    pd.DataFrame({"id": ["123"], "indirect_alert_in": ["True"]}).to_csv(indirect_dir / "smbyc_indirect_alert_annual_2024.csv", index=False)
    pd.DataFrame({"id": ["123"], "farming_in_ha": [1.0]}).to_csv(metrics_dir / "spatial_metrics.csv", index=False)

    def fake_pkg_total_risk(**kwargs):
        return pd.DataFrame({"id": ["123"], "direct_alert": ["True"], "indirect_alert_in": ["True"]})

    monkeypatch.setattr(total_alert, "pkg_total_risk", fake_pkg_total_risk)

    result = total_alert.calculate_total_risk(
        source="smbyc",
        period_type="annual",
        periods=["2024"],
        workspace_dir=str(workspace),
        use_precalculated_metrics=True,
        mongo_map_df=pd.DataFrame({"id": ["123"], "farm_id": ["farm1"], "farm_poligons_id": ["poly1"], "GEOFARMER_ID": ["123"]}),
    )

    assert result["success"] is True
    assert len(result["files_generated"]) == 1
    generated = pd.read_csv(output_dir / "smbyc_total_risk_annual_2024.csv")
    assert generated.iloc[0]["farm_id"] == "farm1"
