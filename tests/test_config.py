import importlib

import dotenv
import pytest

import config as config_module
from config import RiskLevel, classify_risk, config, getenv_clean


@pytest.fixture
def reload_config(monkeypatch):
    """Recarga `config` leyendo solo de os.environ.

    `config.py` llama a `load_dotenv(override=True)`, así que sin neutralizarlo
    los valores de `src/.env` pisarían los del test y el resultado dependería
    de la máquina donde se ejecute la suite.
    """
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: True)

    def _reload():
        return importlib.reload(config_module)

    yield _reload

    monkeypatch.undo()
    importlib.reload(config_module)


def test_getenv_clean_returns_trimmed_value(monkeypatch):
    monkeypatch.setenv("GANABOSQUES_TEST_KEY", "  valor  ")
    assert getenv_clean("GANABOSQUES_TEST_KEY") == "valor"


def test_getenv_clean_strips_wrapping_quotes(monkeypatch):
    monkeypatch.setenv("GANABOSQUES_TEST_KEY", '"D:/ruta con espacios/"')
    assert getenv_clean("GANABOSQUES_TEST_KEY") == "D:/ruta con espacios/"

    monkeypatch.setenv("GANABOSQUES_TEST_KEY", "'otra/ruta'")
    assert getenv_clean("GANABOSQUES_TEST_KEY") == "otra/ruta"

    # Comillas desbalanceadas se dejan tal cual.
    monkeypatch.setenv("GANABOSQUES_TEST_KEY", '"sin cerrar')
    assert getenv_clean("GANABOSQUES_TEST_KEY") == '"sin cerrar'


def test_getenv_clean_raises_when_variable_is_missing(monkeypatch):
    monkeypatch.delenv("GANABOSQUES_TEST_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GANABOSQUES_TEST_KEY"):
        getenv_clean("GANABOSQUES_TEST_KEY")


def test_getenv_clean_raises_when_variable_is_blank(monkeypatch):
    monkeypatch.setenv("GANABOSQUES_TEST_KEY", "   ")

    with pytest.raises(RuntimeError, match="Falta variable requerida"):
        getenv_clean("GANABOSQUES_TEST_KEY")


def test_classify_risk_thresholds():
    assert classify_risk(3.0) is RiskLevel.HIGH
    assert classify_risk(2.5) is RiskLevel.HIGH
    assert classify_risk(2.49) is RiskLevel.MEDIUM
    assert classify_risk(1.5) is RiskLevel.MEDIUM
    assert classify_risk(1.49) is RiskLevel.LOW
    assert classify_risk(0.01) is RiskLevel.LOW
    assert classify_risk(0.0) is RiskLevel.NO_RISK
    assert classify_risk(-1.0) is RiskLevel.NO_RISK


def test_risk_level_numeric_values():
    assert RiskLevel.HIGH.value == 3
    assert RiskLevel.MEDIUM.value == 2
    assert RiskLevel.LOW.value == 1
    assert RiskLevel.NO_RISK.value == 0


def test_config_exposes_mongo_aliases_and_defaults():
    assert config["CONNECTION_URI"] == config["MONGO_URI"]
    assert config["CONNECTION_DB"] == config["MONGO_DB_NAME"]
    # Los numéricos se convierten a int, no quedan como string.
    assert isinstance(config["MAX_DIST"], int)
    assert isinstance(config["DEFOREST_VALUE"], int)
    assert isinstance(config["BATCH_SIZE"], int)
    assert isinstance(config["TASK_MAX_WORKERS"], int)
    assert isinstance(config["MERGE_OUTPUT"], bool)
    assert isinstance(config["MERGE_TOTAL_OUTPUT"], bool)


def test_config_reload_applies_environment_overrides(monkeypatch, reload_config):
    monkeypatch.setenv("CRS_METROS", "EPSG:9377")
    monkeypatch.setenv("MAX_DIST", "12345")
    monkeypatch.setenv("MERGE_OUTPUT", "TRUE")
    monkeypatch.setenv("MERGE_TOTAL_OUTPUT", "false")
    monkeypatch.setenv("TASK_MAX_WORKERS", "16")
    # Rutas entrecomilladas: el módulo debe limpiarlas al recargar.
    monkeypatch.setenv("TOTAL_ALERT_BASE_DIR", '"D:/datos/alertas"')
    monkeypatch.setenv("EMPRESA_ALERT_OUT_DIR", "'D:/datos/empresas'")

    reloaded = reload_config()

    assert reloaded.config["CRS_METROS"] == "EPSG:9377"
    assert reloaded.config["MAX_DIST"] == 12345
    assert reloaded.config["TASK_MAX_WORKERS"] == 16
    assert reloaded.config["MERGE_OUTPUT"] is True
    assert reloaded.config["MERGE_TOTAL_OUTPUT"] is False
    assert reloaded.config["TOTAL_ALERT_BASE_DIR"] == "D:/datos/alertas"
    assert reloaded.config["EMPRESA_ALERT_OUT_DIR"] == "D:/datos/empresas"


def test_config_reload_uses_documented_defaults(monkeypatch, reload_config):
    for key in (
        "CRS_METROS",
        "MAX_DIST",
        "DEFOREST_VALUE",
        "BATCH_SIZE",
        "TASK_MAX_WORKERS",
        "LOG_LEVEL",
        "LOG_FILE",
        "URL_GEO",
        "MERGE_OUTPUT",
    ):
        monkeypatch.delenv(key, raising=False)

    reloaded = reload_config()

    assert reloaded.config["CRS_METROS"] == "EPSG:3116"
    assert reloaded.config["MAX_DIST"] == 50000
    assert reloaded.config["DEFOREST_VALUE"] == 2
    assert reloaded.config["BATCH_SIZE"] == 1000
    assert reloaded.config["TASK_MAX_WORKERS"] == 8
    assert reloaded.config["LOG_LEVEL"] == "WARNING"
    assert reloaded.config["LOG_FILE"] == "risk_analysis.log"
    assert reloaded.config["GEOSERVER_URL"] == "http://localhost:8600/geoserver"
    assert reloaded.config["MERGE_OUTPUT"] is False


def test_config_reload_keeps_unset_optional_paths_as_none(monkeypatch, reload_config):
    monkeypatch.delenv("TOTAL_ALERT_BASE_DIR", raising=False)
    monkeypatch.delenv("EMPRESA_ALERT_OUT_DIR", raising=False)
    monkeypatch.delenv("SOURCES_AVAILABLE", raising=False)

    reloaded = reload_config()

    assert reloaded.config["TOTAL_ALERT_BASE_DIR"] is None
    assert reloaded.config["EMPRESA_ALERT_OUT_DIR"] is None
    assert reloaded.config["SOURCES_AVAILABLE"] is None


def test_config_reload_fails_fast_without_mongo_settings(monkeypatch, reload_config):
    monkeypatch.delenv("MONGO_URI", raising=False)

    with pytest.raises(RuntimeError, match="MONGO_URI"):
        reload_config()
