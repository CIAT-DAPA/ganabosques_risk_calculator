import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import direct_alert


def test_direct_alert_helpers_and_file_range():
    expected = os.path.normpath("C:/tmp/a/b")
    assert direct_alert.norm("C:/tmp\\a\\b") == expected
    assert direct_alert.format_placeholders("{PERIODO}/{YEARS}", {"PERIODO": "annual", "YEARS": "2024"}) == "annual/2024"

    files = ["f1.geojson", "f2.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "2:3") == ["f2.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "1,3") == ["f1.geojson", "f3.geojson"]
    assert direct_alert.apply_file_range(files, "99") == files


def test_area_from_vectors_uses_intersection_and_returns_ha():
    geom = box(0, 0, 10, 10)
    gdf = gpd.GeoDataFrame({"name": ["mask"]}, geometry=[box(0, 0, 5, 5)], crs="EPSG:3116")
    area = direct_alert.area_from_vectors(geom, gdf)
    assert area > 0


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
