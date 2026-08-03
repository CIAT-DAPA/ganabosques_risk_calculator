import json
from types import SimpleNamespace

import data_manager
from data_manager import DataManager


def test_data_manager_initializes_expected_directories(tmp_path):
    manager = DataManager(str(tmp_path), "http://example.com", "user", "pass")

    assert manager.geojsons_dir.exists()
    assert manager.rasters_dir.exists()
    assert manager.results_dir.exists()
    assert manager.workspace_dir.name == "alertas"


def test_chunked_and_normalize_external_farm_id():
    chunks = DataManager._chunked([1, 2, 3, 4], 2)
    assert chunks == [[1, 2], [3, 4]]
    assert DataManager._normalize_external_farm_id("FARM_ID_00123") == "00123"
    assert DataManager._normalize_external_farm_id(None) == ""


def test_data_manager_cache_helpers_and_geojson_sanitization(tmp_path):
    manager = DataManager(str(tmp_path), "http://example.com", "user", "pass")

    assert manager._list_local_geojson_stems() == set()
    manager._invalidate_geojson_stems_cache()
    assert manager._list_local_geojson_stems() == set()
    assert DataManager._farm_ref_to_str(SimpleNamespace(id="abc")) == "abc"
    assert DataManager._farm_ref_to_str("xyz") == "xyz"
    assert manager.get_farmrisk_cache_for_analysis(None) == {}
    manager._farmrisk_cache_by_analysis["42"] = {"foo": "bar"}
    assert manager.get_farmrisk_cache_for_analysis("42") == {"foo": "bar"}

    geojson_path = manager.geojsons_dir / "sample.geojson"
    with geojson_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"centroid": [1, 2], "name": "x"}}]}, handle)

    manager._sanitize_geojson_file(geojson_path)
    saved = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert "centroid" not in saved["features"][0]["properties"]
    assert saved["features"][0]["properties"]["name"] == "x"


def test_prepare_geojsons_and_raster_download(tmp_path, monkeypatch):
    manager = DataManager(str(tmp_path), "http://example.com", "user", "pass")
    geojson_path = manager.geojsons_dir / "farm1.geojson"
    geojson_path.write_text("{}", encoding="utf-8")

    available = manager.prepare_geojsons([{"mongo_id": "1", "sitcode": "farm1"}])
    assert available == 1
    assert manager.count_available_geojsons([{"sitcode": "farm1"}]) == 1

    class DummyResponse:
        status_code = 200
        headers = {"content-type": "image/geotiff"}
        text = ""

        def iter_content(self, chunk_size=8192):
            yield b"tiff-data"

    def fake_get(url, params=None, auth=None, stream=True, timeout=300):
        return DummyResponse()

    monkeypatch.setattr(data_manager.requests, "get", fake_get)
    raster_path = manager.ensure_raster_available("smbyc", "annual", "layer", "2024", "2024")
    assert raster_path is not None
    assert manager.ensure_raster_available("smbyc", "annual", "layer", "2024", "2024") == raster_path
    assert manager.extract_time_filter_from_name("smbyc_deforestation_annual_2017-2018") == "2017-2018"
