"""Configuración compartida de la suite de pruebas.

`src/config.py` valida en tiempo de importación que existan `MONGO_URI` y
`MONGO_DB_NAME`, y las lee de `src/.env`, que no está versionado. Sin estos
valores por defecto la suite ni siquiera se puede recolectar en un checkout
limpio (por ejemplo, en CI). Se definen aquí porque conftest.py se importa
antes que los módulos de prueba, y por tanto antes que `config`.

Ninguna prueba se conecta realmente a MongoDB ni a GeoServer: los valores son
marcadores de posición y todo acceso externo se sustituye con monkeypatch.
"""

import os
import sys
from pathlib import Path

_ENV_DEFAULTS = {
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB_NAME": "ganabosques_test",
}

for _key, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

# pytest.ini ya añade src/ al pythonpath; esto mantiene los imports funcionando
# cuando la suite se ejecuta con otra configuración (por ejemplo, desde un IDE).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    """Workspace con la estructura de carpetas que espera el pipeline."""
    base = tmp_path / "data" / "alertas"
    for sub in ("farms/geojsons", "rasters", "results", "movements", "metrics"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def results_dir(workspace):
    """Devuelve una función que crea `results/{source}/{period_type}/{stage}/`."""

    def _make(stage, source="smbyc", period_type="annual"):
        path = workspace / "results" / source / period_type / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    return _make
