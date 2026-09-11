import csv
import json

import error_logger
from error_logger import (
    ErrorLogger,
    get_error_logger,
    init_error_logger,
    log_error,
)


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_error_logger_creates_file_with_timestamped_name(tmp_path):
    logger = ErrorLogger(str(tmp_path), prefix="errors")

    assert logger.filepath.parent == tmp_path
    assert logger.filepath.name.startswith("errors_")
    assert logger.filepath.suffix == ".csv"
    # El archivo se crea de forma perezosa, en el primer error.
    assert not logger.filepath.exists()


def test_error_logger_can_skip_the_timestamp_suffix(tmp_path):
    logger = ErrorLogger(str(tmp_path), prefix="errores", append_timestamp=False)

    assert logger.filepath.name == "errores.csv"


def test_error_logger_creates_missing_output_directory(tmp_path):
    target = tmp_path / "nested" / "logs"
    ErrorLogger(str(target), append_timestamp=False)

    assert target.exists()


def test_error_logger_writes_header_and_rows(tmp_path):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)

    logger.log_error(
        stage="direct_alert",
        error_type="RASTER_ERROR",
        error_message="raster ilegible",
        farm_id="farm-1",
        period="202401",
    )

    rows = _read_rows(logger.filepath)
    assert rows[0] == [
        "timestamp",
        "stage",
        "farm_id",
        "period",
        "error_type",
        "error_message",
        "additional_info",
    ]
    assert rows[1][1:] == [
        "direct_alert",
        "farm-1",
        "202401",
        "RASTER_ERROR",
        "raster ilegible",
        "",
    ]


def test_error_logger_serializes_additional_info_as_json(tmp_path):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)

    logger.log_error(
        stage="save_db",
        error_type="DB_SAVE",
        error_message="duplicado",
        additional_info={"analysis": "a1", "detalle": "clave duplicada"},
    )

    rows = _read_rows(logger.filepath)
    assert json.loads(rows[1][6]) == {"analysis": "a1", "detalle": "clave duplicada"}
    # farm_id y period ausentes se guardan como cadena vacía, no como "None".
    assert rows[1][2] == ""
    assert rows[1][3] == ""


def test_error_logger_only_writes_the_header_once(tmp_path):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)

    logger.log_error(stage="s", error_type="UNKNOWN", error_message="uno")
    logger.log_error(stage="s", error_type="UNKNOWN", error_message="dos")

    rows = _read_rows(logger.filepath)
    assert len(rows) == 3
    assert rows[1][5] == "uno"
    assert rows[2][5] == "dos"


def test_error_logger_shortcuts_write_the_expected_error_types(tmp_path):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)

    logger.log_geojson_not_found("farm-1")
    logger.log_raster_error(period="202401", error="sin banda")
    logger.log_db_error(error="timeout", farm_id="farm-2")
    logger.log_geometry_error(farm_id="farm-3", error="geometría inválida", period="202402")

    rows = _read_rows(logger.filepath)[1:]
    types = [row[4] for row in rows]
    stages = [row[1] for row in rows]

    assert types == [
        "GEOJSON_NOT_FOUND",
        "RASTER_ERROR",
        "DB_SAVE",
        "GEOMETRY_ERROR",
    ]
    assert stages == ["direct_alert", "direct_alert", "save_db", "direct_alert"]
    assert rows[0][2] == "farm-1"
    assert rows[1][3] == "202401"
    assert rows[3][2] == "farm-3"
    assert rows[3][3] == "202402"


def test_error_logger_summary_counts_by_type(tmp_path):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)

    assert logger.get_summary() == {
        "total_errors": 0,
        "errors_by_type": {},
        "output_file": str(logger.filepath),
        "file_exists": False,
    }

    logger.log_geojson_not_found("farm-1")
    logger.log_geojson_not_found("farm-2")
    logger.log_db_error(error="timeout")

    summary = logger.get_summary()
    assert summary["total_errors"] == 3
    assert summary["errors_by_type"] == {"GEOJSON_NOT_FOUND": 2, "DB_SAVE": 1}
    assert summary["file_exists"] is True


def test_error_logger_print_summary_lists_known_and_unknown_types(tmp_path, capsys):
    logger = ErrorLogger(str(tmp_path), append_timestamp=False)
    logger.log_geojson_not_found("farm-1")
    logger.log_error(stage="s", error_type="TIPO_NO_CATALOGADO", error_message="x")

    logger.print_summary()

    output = capsys.readouterr().out
    assert "RESUMEN DE ERRORES (2 total)" in output
    # Los tipos conocidos se imprimen con su descripción del catálogo.
    assert "GEOJSON_NOT_FOUND: 1 (GeoJSON de finca no encontrado)" in output
    # Los desconocidos caen de vuelta en el propio código.
    assert "TIPO_NO_CATALOGADO: 1 (TIPO_NO_CATALOGADO)" in output
    assert str(logger.filepath) in output


def test_global_error_logger_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(error_logger, "_global_error_logger", None)

    # Sin inicializar, log_error es un no-op y get_error_logger devuelve None.
    assert get_error_logger() is None
    log_error("direct_alert", "UNKNOWN", "se descarta en silencio")

    logger = init_error_logger(str(tmp_path), prefix="pipeline")
    assert get_error_logger() is logger
    assert logger.filepath.name.startswith("pipeline_")

    log_error(
        "direct_alert",
        "GEOJSON_NOT_FOUND",
        "sin geojson",
        farm_id="farm-9",
        period="202401",
    )

    rows = _read_rows(logger.filepath)
    assert rows[1][2] == "farm-9"
    assert logger.get_summary()["total_errors"] == 1

    # Reinicializar reemplaza la instancia global.
    other = init_error_logger(str(tmp_path), prefix="otro")
    assert get_error_logger() is other
    assert other is not logger
