#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
⚠️  ARCHIVO DEPRECADO - NO USAR ⚠️

Este archivo es una variante con buffers que no se usa en producción.
Usar en su lugar: total_alert.py

Fecha de deprecación: Enero 2026
"""

import warnings
warnings.warn(
    "total_alert_buffers.py está DEPRECADO. Usar total_alert.py en su lugar.",
    DeprecationWarning,
    stacklevel=2
)

import os
import re
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

#"2010-2011,2010-2012,2010-2013,2010-2014,2010-2015,2010-2016,2010-2017,2010-2018,2010-2019,2010-2020,2010-2021,2010-2022,2010-2023", #
#"2010-2012,2012-2013,2013-2014,2014-2015,2015-2016,2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022,2022-2023,2023-2024"
# ===================== CONFIG =====================
config = {
    # Parámetros
    "PERIODO": "annual",                         # "annual" o "cumulative"
    "YEARS": "2010-2012,2012-2013,2013-2014,2014-2015,2015-2016,2016-2017,2017-2018,2018-2019,2019-2020,2020-2021,2021-2022,2022-2023,2023-2024", # múltiple, separado por coma
     
    # Rutas (Windows; usa r'' para backslashes)
    "DIRECT_DIR":   r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\{PERIODO}\direct_alert\SMBYC\{YEARS}",
    "DIRECT_NAME":  "smbyc_direct_alert_{PERIODO}_{YEARS}.csv",

    "MOVE_DIR":     r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\{PERIODO}\indirect_alert\SMBYC\{YEARS}",
    "MOVE_NAME":    "smbyc_movement_alerts_{YEARS}.csv",

    "METRICS_PATH": r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\metrics\metricas.csv",
    "OUTPUT_ROOT":  r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\result_alerts",

    # Mongo
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB":  "ganabosques",
    "COL_FARM":  "farm",
    "COL_POLY":  "farmpolygons",

    # Lotes para las consultas a Mongo (para no hacer una por id)
    "MONGO_BATCH": 5000,

    # Logging
    "LOG_LEVEL": "INFO",
}

# ===================== LOGGING =====================
def setup_logging():
    lvl = getattr(logging, str(config.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(lvl)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(sh)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ===================== HELPERS =====================
def parse_years_list(raw: str) -> List[str]:
    if not raw:
        return []
    out = []
    for p in str(raw).split(","):
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(r"\d{4}\s*-\s*\d{4}", p):
            out.append(p.replace(" ", ""))
        else:
            logging.warning(f"YEARS ignorado (usa AAAA-AAAA): '{p}'")
    return out

def id_preserve_zeros(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            i = int(float(s))
            s2 = str(i)
            if re.fullmatch(r"\d+\.0", s):
                return s[:-2]  # "000123.0" -> "000123"
            return s2
        except Exception:
            return s[:-2]
    return s

def load_csv_if_exists(path: str, need_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        logging.error(f"No existe: {path}")
        return None
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if need_cols:
            for c in need_cols:
                if c not in df.columns:
                    logging.warning(f"CSV '{os.path.basename(path)}' sin columna '{c}'.")
        return df
    except Exception as e:
        logging.error(f"No se pudo leer {path}: {e}")
        return None

# ===================== MONGO =====================
def build_mongo_client() -> MongoClient:
    return MongoClient(config["MONGO_URI"], serverSelectionTimeoutMS=5000)

def map_id_to_farm_and_polygons(df_ids: pd.Series) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Devuelve dos diccionarios:
      id -> farm_id (ObjectId como string)
      id -> farm_poligons_id (si hay varios, el primero)
    """
    ids = [id_preserve_zeros(x) for x in df_ids.dropna().astype(str).tolist()]
    ids = [i for i in ids if i != ""]
    if not ids:
        return {}, {}

    client = build_mongo_client()
    db = client[config["MONGO_DB"]]
    farm_col = db[config["COL_FARM"]]
    poly_col = db[config["COL_POLY"]]

    batch = int(config.get("MONGO_BATCH", 5000))

    id2farm: Dict[str, ObjectId] = {}
    # ---- 1) Buscar farms por ext_id (SIT_CODE) usando $elemMatch + $in en lotes
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        q = {"ext_id": {"$elemMatch": {"source": "SIT_CODE", "ext_code": {"$in": chunk}}}}
        cur = farm_col.find(q, {"_id": 1, "ext_id": 1})
        found = 0
        for doc in cur:
            ext = doc.get("ext_id", [])
            match_codes = [e.get("ext_code") for e in ext if isinstance(e, dict) and e.get("source") == "SIT_CODE" and e.get("ext_code") in chunk]
            if not match_codes:
                continue
            for code in match_codes:
                if code not in id2farm:
                    id2farm[code] = doc["_id"]
                    found += 1
        logging.info(f"[MONGO] Lote farms {i+1}-{i+len(chunk)} / {len(ids)} → asignados: {found}")

    # ---- 2) Buscar farmpolygons para los farm_ids encontrados
    farm_ids = list({v for v in id2farm.values()})
    farm_id2poly_first: Dict[ObjectId, ObjectId] = {}
    farm_id2poly_all: Dict[ObjectId, List[ObjectId]] = {}
    for i in range(0, len(farm_ids), batch):
        chunk = farm_ids[i:i+batch]
        cur = poly_col.find({"farm_id": {"$in": chunk}}, {"_id": 1, "farm_id": 1})
        grouped: Dict[ObjectId, List[ObjectId]] = {}
        for d in cur:
            fid = d.get("farm_id")
            pid = d.get("_id")
            if fid is None or pid is None:
                continue
            grouped.setdefault(fid, []).append(pid)
        for fid, plist in grouped.items():
            farm_id2poly_all[fid] = plist
            if plist:
                farm_id2poly_first[fid] = plist[0]
        logging.info(f"[MONGO] Lote polys {i+1}-{i+len(chunk)} / {len(farm_ids)} → farms con polígonos: {len(grouped)}")

    # ---- 3) Construir mapas finales id->...
    id2farm_str = {k: str(v) for k, v in id2farm.items()}
    id2poly_str = {}
    for code, farm_oid in id2farm.items():
        pid = farm_id2poly_first.get(farm_oid)
        if pid is not None:
            id2poly_str[code] = str(pid)

    client.close()
    return id2farm_str, id2poly_str

# ===================== PIPELINE =====================
def process_year(periodo: str, years: str, metrics_df: pd.DataFrame) -> Optional[str]:
    ctx = {"PERIODO": periodo, "YEARS": years}

    # --- Cargar DIRECT
    direct_path = os.path.join(
        config["DIRECT_DIR"].format(**ctx),
        config["DIRECT_NAME"].format(**ctx)
    )
    df_direct = load_csv_if_exists(direct_path)
    if df_direct is None or "id" not in df_direct.columns:
        logging.error(f"[DIRECT] No se pudo usar: {direct_path}")
        return None
    df_direct["id"] = df_direct["id"].map(id_preserve_zeros)

    # compat: asegurar columnas
    if "direct_alert" not in df_direct.columns:
        for c in ["intersect_deforestation","intersect_early_warnings","intersect_active_hotspots"]:
            if c in df_direct.columns:
                df_direct["direct_alert"] = df_direct[c]
                break
    if "direct_alert" not in df_direct.columns:
        df_direct["direct_alert"] = "False"
    for c in ["intersect_deforestation","deforested_ha","deforested_prop"]:
        if c not in df_direct.columns:
            df_direct[c] = "no data"

    # --- Cargar MOVEMENT
    move_path = os.path.join(
        config["MOVE_DIR"].format(**ctx),
        config["MOVE_NAME"].format(**ctx)
    )
    df_move = load_csv_if_exists(move_path)
    mov_cols = ["n_total_mov","n_in","n_out","n_indirect_in","n_indirect_out","indirect_alert_in","indirect_alert_out"]
    if df_move is None or "id" not in df_move.columns:
        df_move = pd.DataFrame({"id": df_direct["id"].astype(str), **{c: "no_info" for c in mov_cols}})
    else:
        df_move["id"] = df_move["id"].map(id_preserve_zeros)
        for c in mov_cols:
            if c not in df_move.columns:
                df_move[c] = "no_info"

    # --- Unir direct + movement
    merged = pd.merge(
        df_direct[["id","direct_alert","intersect_deforestation","deforested_ha","deforested_prop"]],
        df_move[["id"] + mov_cols],
        on="id", how="outer", copy=False
    )

    # --- Unir métricas
    if "id" not in metrics_df.columns:
        logging.error("[METRICS] La tabla de métricas no tiene columna 'id'")
        return None
    metrics_df_local = metrics_df.copy()
    metrics_df_local["id"] = metrics_df_local["id"].map(id_preserve_zeros)

    final = pd.merge(merged, metrics_df_local, on="id", how="left", copy=False)

    # --- Enriquecer con Mongo (farm_id y farm_poligons_id)
    unique_ids = final["id"].dropna().astype(str).unique().tolist()
    logging.info(f"[MONGO] Consultando farm y farmpolygons para {len(unique_ids)} ids…")
    id2farm, id2poly = map_id_to_farm_and_polygons(final["id"])

    final["farm_id"] = final["id"].map(id2farm).fillna("")
    final["farm_poligons_id"] = final["id"].map(id2poly).fillna("")

    # --- Guardar (CAMBIO DE NOMBRE DE ARCHIVO)
    out_dir = os.path.join(config["OUTPUT_ROOT"], periodo, years)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"deforestation_{periodo}_{years}.csv")
    final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Guardado: {out_path}")
    return out_path

def main():
    setup_logging()
    t0 = time.perf_counter()

    periodo = str(config["PERIODO"]).strip()
    years_list = parse_years_list(str(config["YEARS"]))
    if not years_list:
        logging.error("No hay YEARS válidos.")
        return

    # Cargar métricas UNA sola vez
    metrics_path = config["METRICS_PATH"]
    df_metrics = load_csv_if_exists(metrics_path)
    if df_metrics is None:
        logging.error("No se pudo cargar la tabla de métricas.")
        return

    produced: List[str] = []
    for y in years_list:
        print(f"\n▶️ Procesando YEARS={y} | PERIODO={periodo}")
        out = process_year(periodo, y, df_metrics)
        if out:
            produced.append(out)

    if produced:
        print("\n🎉 Archivos generados:")
        for p in produced:
            print("  -", p)

    print(f"\n⏱ Tiempo total: {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
