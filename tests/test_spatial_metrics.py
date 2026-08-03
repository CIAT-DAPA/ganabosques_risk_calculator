import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import spatial_metrics


def test_load_reference_layer_returns_none_for_missing_file(tmp_path):
    missing = tmp_path / "missing.gpkg"
    assert spatial_metrics.load_reference_layer(str(missing), "EPSG:3116", "Test") is None


def test_download_and_extract_wfs_returns_none_on_failure(monkeypatch, tmp_path):
    class DummyResponse:
        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            raise RuntimeError("boom")

    def fake_get(url, stream=True, timeout=120):
        return DummyResponse()

    monkeypatch.setattr(spatial_metrics.requests, "get", fake_get)
    assert spatial_metrics.download_and_extract_wfs("http://example.com", tmp_path, "layer", "EPSG:3116") is None
