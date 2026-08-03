import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import direct_alert
import enterprise_alert
import indirect_alert
import parallel_processor
import total_alert


def test_direct_alert_helpers_and_file_range():
    expected = os.path.normpath("C:/tmp/a/b")
    assert direct_alert.norm("C:/tmp\\a\\b") == expected
    assert direct_alert.format_placeholders("{PERIODO}/{YEARS}", {"PERIODO": "annual", "YEARS": "2024"}) == "annual/2024"

    files = ["f1.geojson", "f2.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "2:3") == ["f2.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "1,3") == ["f1.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "99") == files


def test_load_bundle_and_open_raster(tmp_path):
    vector_path = tmp_path / "sample.geojson"
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    gdf.to_file(vector_path, driver="GeoJSON")

    raster_path = tmp_path / "sample.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-1, 1, 1, 1),
    ) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype="uint8"), 1)

    bundle_gdf, raster_paths, summary = direct_alert.load_bundle(str(tmp_path), {"PERIODO": "annual", "YEARS": "2024"}, "EPSG:3116")

    assert summary["vec_files"] == 1
    assert summary["ras_files"] == 1
    assert bundle_gdf is not None
    assert str(raster_path) in raster_paths

    with direct_alert.open_raster(raster_path, target_crs="EPSG:3116") as src:
        assert src.crs is not None

    srcs = direct_alert.open_rasters([str(raster_path)], target_crs="EPSG:3116")
    assert len(srcs) == 1
    assert direct_alert.pixel_area_m2_approx_for_vrt(srcs[0]) > 0


def test_area_from_vectors_uses_intersection_and_returns_ha():
    geom = box(0, 0, 10, 10)
    gdf = gpd.GeoDataFrame({"name": ["mask"]}, geometry=[box(0, 0, 5, 5)], crs="EPSG:3116")
    area = direct_alert.area_from_vectors(geom, gdf)
    assert area > 0


def test_indirect_alert_loading_and_enterprise_extraction(tmp_path):
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["00123", "00456"], "direct_alert": ["true", "false"]}).to_csv(direct_csv, index=False)

    result = indirect_alert.load_direct_alerts(str(direct_csv), {"strip_leading_zeros": True, "strip_dot_zero": True})
    assert result["id"].tolist() == ["123", "456"]
    assert result["direct_alert"].tolist() == [True, False]

    movements = pd.DataFrame(
        {
            "origen_id": ["123", "456"],
            "destination_id": ["999", "123"],
            "tipo_origen": ["FARM", "SLAUGHTERHOUSE"],
            "tipo_destino": ["SLAUGHTERHOUSE", "FARM"],
            "producer_id_origen": ["p1", "p2"],
            "producer_id_destino": ["p3", "p4"],
        }
    )
    enterprise_df = indirect_alert.extract_enterprise_alerts(
        movements,
        farms_with_alert={"123"},
        period="202401",
        year="2024",
        quarter=1,
    )
    assert not enterprise_df.empty
    assert set(enterprise_df["typemove"]) == {"in", "out"}


def test_enterprise_alerts_period_writes_csv(tmp_path, monkeypatch):
    direct_dir = tmp_path / "direct"
    movement_dir = tmp_path / "movement"
    output_dir = tmp_path / "out"
    direct_dir.mkdir(parents=True)
    movement_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    pd.DataFrame({"id": ["00123"], "direct_alert": [True]}).to_csv(direct_dir / "smbyc_direct_alert_nad_202401.csv", index=False)
    pd.DataFrame({
        "SIT_CODE_ORIGEN": ["123"],
        "SIT_CODE_DESTINO": ["999"],
        "TIPO_ORIGEN": ["FARM"],
        "TIPO_DESTINO": ["SLAUGHTERHOUSE"],
        "PRODUCER_ID_ORIGEN": ["p1"],
        "PRODUCER_ID_DESTINO": ["p2"],
        "DATE": ["2024-01-10"],
    }).to_csv(movement_dir / "movement_data_base_2024.csv", index=False)

    def fake_pkg_alert_enterprise(total_risk_df, movements_df, id_column, normalize_ids=True, show_progress=True):
        return pd.DataFrame({
            "idpro": ["E1"],
            "id_farm": ["123"],
            "typemove": ["in"],
            "enterprise_type": ["SLAUGHTERHOUSE"],
        })

    monkeypatch.setattr(enterprise_alert, "pkg_alert_enterprise", fake_pkg_alert_enterprise)
    monkeypatch.setattr(enterprise_alert, "load_enterprise_mapping", lambda *args, **kwargs: {})

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is True
    output_file = output_dir / "smbyc_enterprise_alert_nad_202401.csv"
    assert output_file.exists()
    written = pd.read_csv(output_file)
    assert written.iloc[0]["typemove"] == "in"


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
    cached_again = total_alert.load_or_build_cache("ACME", "annual", farms, {"1", "2"}, frontier, protected)
    assert len(cached_again) == 2


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
