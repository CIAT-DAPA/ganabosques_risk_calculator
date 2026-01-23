# src/config.py  -- VERSION CORREGIDA
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
    # quitar comillas envolventes si existen
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v

# objeto de configuración que exportamos
config = {}

# --- Mongo ---
config['MONGO_URI'] = getenv_clean("MONGO_URI")
config['MONGO_DB_NAME'] = getenv_clean("MONGO_DB_NAME")

# Aliases de compatibilidad (muchos scripts esperan estos nombres)
config['CONNECTION_URI'] = config['MONGO_URI']
config['CONNECTION_DB']  = config['MONGO_DB_NAME']

# --- Parámetros (algunos con default) ---
config['CRS_METROS'] = os.getenv("CRS_METROS", "EPSG:3116")
config['MAX_DIST'] = int(os.getenv("MAX_DIST", "50000"))
config['DEFOREST_VALUE'] = int(os.getenv("DEFOREST_VALUE", "2"))
config['EMPRESA'] = getenv_clean("EMPRESA")
config['YEARS']   = getenv_clean("YEARS")
config['PERIODO'] = getenv_clean("PERIODO")
config['BATCH_SIZE'] = int(os.getenv("BATCH_SIZE", "12"))

# --- Rutas (obligatorias) ---
config['FOLDER_GEOJSONS'] = getenv_clean("FOLDER_GEOJSONS")
config['RASTER_DEFOREST'] = getenv_clean("RASTER_DEFOREST")
config['ALERTAS_DIR'] = os.getenv("ALERTAS_DIR", "").strip()
config['NUCLEOS_DIR'] = os.getenv("NUCLEOS_DIR", "").strip()
config['SHP_PROTECTED']   = getenv_clean("SHP_PROTECTED")
config['OUTPUT_CSV']      = getenv_clean("OUTPUT_CSV")
config['FARMING_FRONTIER_SHP'] = getenv_clean("FARMING_FRONTIER_SHP")

# Logging (opcionales)
config['LOG_LEVEL'] = os.getenv("LOG_LEVEL", "WARNING")
config['LOG_FILE']  = os.getenv("LOG_FILE",  "risk_analysis_intersections.log")

# --- Movement ---
config['DIRECT_RISK_CSV'] = os.getenv("DIRECT_RISK_CSV", config['OUTPUT_CSV'])
config['MOVEMENT_INPUT_CSV'] = getenv_clean("MOVEMENT_INPUT_CSV")
config['MOVEMENT_RISK_OUTPUT_CSV'] = getenv_clean("MOVEMENT_RISK_OUTPUT_CSV")

config['MERGE_OUTPUT'] = os.getenv("MERGE_OUTPUT", "false").lower() == "true"
config['OUTPUT_CSV_ALL'] = os.getenv("OUTPUT_CSV_ALL", None)
config['TASK_MAX_WORKERS'] = int(os.getenv("TASK_MAX_WORKERS", "8"))

# Opcionales para alertas
config['TOTAL_ALERT_BASE_DIR'] = os.getenv("TOTAL_ALERT_BASE_DIR", None)
config['SOURCES_AVAILABLE'] = os.getenv("SOURCES_AVAILABLE", None)
config['EMPRESA_ALERT_OUT_DIR'] = os.getenv("EMPRESA_ALERT_OUT_DIR", None)

# Total Risk
config['TOTAL_RISK_OUTPUT_CSV'] = getenv_clean("TOTAL_RISK_OUTPUT_CSV")
config['MERGE_TOTAL_OUTPUT'] = os.getenv("MERGE_TOTAL_OUTPUT", "false").lower() == "true"
config['TOTAL_RISK_OUTPUT_ALL'] = os.getenv("TOTAL_RISK_OUTPUT_ALL", None)

# Utilidades para clasificación
class RiskLevel(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NO_RISK = 0

def classify_risk(score: float) -> "RiskLevel":
    if score >= 2.5:
        return RiskLevel.HIGH
    elif score >= 1.5:
        return RiskLevel.MEDIUM
    elif score > 0:
        return RiskLevel.LOW
    else:
        return RiskLevel.NO_RISK

config['FARM_FILE_RANGE'] = os.getenv("FARM_FILE_RANGE", "")

# Normaliza posibles comillas en algunas rutas
for k in ['TOTAL_ALERT_BASE_DIR', 'EMPRESA_ALERT_OUT_DIR']:
    if config.get(k):
        v = config[k].strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        config[k] = v
