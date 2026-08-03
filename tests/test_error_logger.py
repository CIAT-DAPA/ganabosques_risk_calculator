from error_logger import ErrorLogger


def test_error_logger_writes_csv(tmp_path):
    logger = ErrorLogger(str(tmp_path), prefix="errors", append_timestamp=False)
    logger.log_error("direct_alert", "RASTER_ERROR", "Raster missing", farm_id="123", period="2024")
    logger.log_geojson_not_found("456", stage="direct_alert")

    assert logger.filepath.exists()
    assert logger.get_summary()["total_errors"] == 2
    assert logger.get_summary()["errors_by_type"]["RASTER_ERROR"] == 1
