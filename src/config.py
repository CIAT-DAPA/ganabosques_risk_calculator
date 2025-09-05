# config.py (estricto)
import os
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

# Carga .env junto al config.py (no depende del cwd)
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

def getenv_clean(key: str) -> str:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        raise RuntimeError(f"Falta variable requerida en .env: {key}")
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v

config = {}

# Mongo
config['MONGO_URI'] = getenv_clean("MONGO_URI")
config['MONGO_DB_NAME'] = getenv_clean("MONGO_DB_NAME")

# Parámetros
config['CRS_METROS'] = os.getenv("CRS_METROS", "EPSG:3116")
config['MAX_DIST'] = int(os.getenv("MAX_DIST", "50000"))
config['DEFOREST_VALUE'] = int(os.getenv("DEFOREST_VALUE", "2"))
config['EMPRESA'] = getenv_clean("EMPRESA")
config['YEARS']   = getenv_clean("YEARS")    # admite múltiples periodos con coma
config['PERIODO'] = getenv_clean("PERIODO")
config['BATCH_SIZE'] = int(os.getenv("BATCH_SIZE", "12"))

# Rutas (obligatorias, sin default)
config['FOLDER_GEOJSONS'] = getenv_clean("FOLDER_GEOJSONS")
config['RASTER_DEFOREST'] = getenv_clean("RASTER_DEFOREST")
config['SHP_PROTECTED']   = getenv_clean("SHP_PROTECTED")
config['OUTPUT_CSV']      = getenv_clean("OUTPUT_CSV")

# Logging (opcionales)
config['LOG_LEVEL'] = os.getenv("LOG_LEVEL", "WARNING")
config['LOG_FILE']  = os.getenv("LOG_FILE",  "risk_analysis_intersections.log")


# --- Movement ---
# Si DIRECT_RISK_CSV no se expandió en .env, usa OUTPUT_CSV como fallback
config['DIRECT_RISK_CSV'] = os.getenv("DIRECT_RISK_CSV", config['OUTPUT_CSV'])
config['MOVEMENT_INPUT_CSV'] = getenv_clean("MOVEMENT_INPUT_CSV")
config['MOVEMENT_RISK_OUTPUT_CSV'] = getenv_clean("MOVEMENT_RISK_OUTPUT_CSV")

config['MERGE_OUTPUT'] = os.getenv("MERGE_OUTPUT", "false").lower() == "true"
config['OUTPUT_CSV_ALL'] = os.getenv("OUTPUT_CSV_ALL", None)
config['TASK_MAX_WORKERS'] = int(os.getenv("TASK_MAX_WORKERS", "8"))


# --- Total Risk ---
from pathlib import Path  # (ya lo tienes arriba)

# Salida por periodo (obligatoria)
config['TOTAL_RISK_OUTPUT_CSV'] = getenv_clean("TOTAL_RISK_OUTPUT_CSV")

# Combinado (opcional)
config['MERGE_TOTAL_OUTPUT'] = os.getenv("MERGE_TOTAL_OUTPUT", "false").lower() == "true"
config['TOTAL_RISK_OUTPUT_ALL'] = os.getenv("TOTAL_RISK_OUTPUT_ALL", None)


class RiskLevel(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NO_RISK = 0

def classify_risk(score: float) -> "RiskLevel":
    """
    Clasifica el puntaje de riesgo compuesto.
    Umbrales fijos (puedes hacerlos configurables más adelante si quieres):
      - >= 2.5 -> HIGH
      - >= 1.5 -> MEDIUM
      -  > 0   -> LOW
      -  else  -> NO_RISK
    """
    if score >= 2.5:
        return RiskLevel.HIGH
    elif score >= 1.5:
        return RiskLevel.MEDIUM
    elif score > 0:
        return RiskLevel.LOW
    else:
        return RiskLevel.NO_RISK


config['FARM_FILE_RANGE'] = os.getenv("FARM_FILE_RANGE", "")