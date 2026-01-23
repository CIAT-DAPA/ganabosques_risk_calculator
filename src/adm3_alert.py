# -*- coding: utf-8 -*-
"""
adm3_alert_modified.py — ADM3 risk con farm_amount_total
"""

import os
from typing import List, Dict, Any, Tuple
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from pymongo import MongoClient
from bson import ObjectId
import pandas as pd

# Progreso opcional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# --- Conexión ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ganabosques"

# --- Salida ---
BASE_OUTDIR = r"D:\OneDrive - CGIAR\Desktop\ganabosques\ganabosques_results\04_etl_alerts\alert_adm3"

# --- Colecciones ---
COL_DEFORESTATION = "deforestation"
COL_ANALYSIS      = "analysis"
COL_FARM          = "farm"
COL_FARMRISK      = "farmrisk"

# Campo de área dentro de farmrisk
FARMRISK_DEF_HA_PATH = ("deforestation", "ha")  # farmrisk.deforestation.ha


# ---------------- Acceso base ----------------
def _get_analysis_ids(db, name_def: str) -> List[ObjectId]:
    def_doc = db[COL_DEFORESTATION].find_one({"name": name_def}, {"_id": 1})
    if not def_doc:
        return []
    def_id = def_doc["_id"]
    return [doc["_id"] for doc in db[COL_ANALYSIS].find({"deforestation_id": def_id}, {"_id": 1})]


def _get_all_adm3_ids(db) -> List[ObjectId]:
    pipeline = [
        {"$match": {"adm3_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$adm3_id"}},
        {"$match": {"_id": {"$ne": None}}},
    ]
    return [g["_id"] for g in db[COL_FARM].aggregate(pipeline, allowDiskUse=True)]


def _aggregate_adm3_risk(db, analysis_ids: List[ObjectId]) -> List[Dict[str, Any]]:
    if not analysis_ids:
        return []

    ha_field_path = f"{FARMRISK_DEF_HA_PATH[0]}.{FARMRISK_DEF_HA_PATH[1]}"

    pipeline = [
        {
            "$match": {
                "analysis_id": {"$in": analysis_ids},
                "$or": [
                    {"risk_direct": True},
                    {"risk_input": True},
                    {"risk_output": True},
                ],
            }
        },
        {
            "$project": {
                "farm_id": 1,
                "analysis_id": 1,
                "risk_direct": 1,
                "risk_input": 1,
                "risk_output": 1,
                ha_field_path: 1,
            }
        },
        {
            "$lookup": {
                "from": COL_FARM,
                "localField": "farm_id",
                "foreignField": "_id",
                "as": "farm"
            }
        },
        {"$unwind": "$farm"},
        {"$match": {"farm.adm3_id": {"$exists": True, "$ne": None}}},
        {
            "$set": {
                "direct_ha": {
                    "$cond": [
                        {"$eq": ["$risk_direct", True]},
                        f"${ha_field_path}",
                        0
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "adm3_id": "$farm.adm3_id",
                    "analysis_id": "$analysis_id",
                },
                "def_ha_sum": {"$sum": {"$ifNull": ["$direct_ha", 0]}},
                "farms_set": {"$addToSet": "$farm_id"},  # fincas con riesgo (distinct)
                "risk_any": {"$max": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "adm3_id": "$_id.adm3_id",
                "analysis_id": "$_id.analysis_id",
                "def_ha": {"$ifNull": ["$def_ha_sum", 0]},
                "risk_total": {"$gt": ["$risk_any", 0]},
                "farm_amount": {"$size": {"$ifNull": ["$farms_set", []]}},
            }
        }
    ]

    return list(db[COL_FARMRISK].aggregate(pipeline, allowDiskUse=True))


def _compute_rows_for_name(db, name_def: str) -> List[Dict[str, Any]]:
    analysis_ids = _get_analysis_ids(db, name_def)
    print(f"[{name_def}] analysis_ids: {len(analysis_ids)}")
    if not analysis_ids:
        return []

    all_adm3_ids = _get_all_adm3_ids(db)
    print(f"[{name_def}] adm3 totales: {len(all_adm3_ids)}")

    # Diccionario: adm3_id -> número total de fincas (con y sin riesgo)
    adm3_total_farms = {}
    pipeline_farms = [
        {"$match": {"adm3_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$adm3_id", "total_farms": {"$addToSet": "$_id"}}},
    ]
    for g in db[COL_FARM].aggregate(pipeline_farms, allowDiskUse=True):
        adm3_id = str(g["_id"])
        adm3_total_farms[adm3_id] = len(g["total_farms"])

    agg_rows = _aggregate_adm3_risk(db, analysis_ids)
    print(f"[{name_def}] combos con riesgo retornados por agregación: {len(agg_rows)}")

    # Indexar resultado por (adm3_id, analysis_id)
    key_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in agg_rows:
        adm3_id = r.get("adm3_id")
        aid = r.get("analysis_id")
        if isinstance(adm3_id, ObjectId):
            adm3_id = str(adm3_id)
        if isinstance(aid, ObjectId):
            aid = str(aid)
        key_map[(adm3_id, aid)] = {
            "adm3_id": adm3_id,
            "analysis_id": aid,
            "def_ha": float(r.get("def_ha", 0.0)),
            "risk_total": bool(r.get("risk_total", False)),
            "farm_amount": int(r.get("farm_amount", 0)),
        }

    # Construir filas para TODAS las combinaciones
    rows: List[Dict[str, Any]] = []
    all_adm3_str = [str(x) for x in all_adm3_ids]
    analysis_ids_str = [str(x) for x in analysis_ids]

    iterator = tqdm(all_adm3_str, desc=f"[{name_def}] completando malla", unit="adm3", leave=False) if HAS_TQDM else all_adm3_str
    for adm3_id_str in iterator:
        for aid in analysis_ids_str:
            k = (adm3_id_str, aid)
            if k in key_map:
                row = key_map[k]
            else:
                row = {
                    "adm3_id": adm3_id_str,
                    "analysis_id": aid,
                    "def_ha": 0.0,
                    "risk_total": False,
                    "farm_amount": 0,
                }
            # Agregar farm_amount_total
            row["farm_amount_total"] = adm3_total_farms.get(adm3_id_str, 0)
            rows.append(row)

    return rows


def _task_for_name(mongo_uri: str, db_name: str, nd: str) -> Tuple[str, List[Dict[str, Any]]]:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    rows = _compute_rows_for_name(db, nd)
    return nd, rows


def adm3_risk(
    name_def: List[str],
    workers: int = 1,
) -> List[str]:
    os.makedirs(BASE_OUTDIR, exist_ok=True)
    max_w = max(1, min(workers, cpu_count()))
    print("\n================ ADM3 RISK (AGGREGATION) ================")
    print(f"MongoDB: {MONGO_URI} | DB: {MONGO_DB_NAME}")
    print(f"Workers: {max_w}\n")

    paths: List[str] = []

    if max_w == 1:
        for nd in name_def:
            nd, rows = _task_for_name(MONGO_URI, MONGO_DB_NAME, nd)
            out_path = os.path.join(BASE_OUTDIR, f"{nd}_ADM3_RISK.csv")
            df = pd.DataFrame(rows, columns=["adm3_id", "analysis_id", "def_ha", "risk_total", "farm_amount", "farm_amount_total"])
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[OK] CSV generado: {out_path}")
            paths.append(out_path)
    else:
        with ProcessPoolExecutor(max_workers=max_w) as ex:
            futs = {ex.submit(_task_for_name, MONGO_URI, MONGO_DB_NAME, nd): nd for nd in name_def}
            iterator = tqdm(as_completed(futs), total=len(futs), desc="[Progreso name_def]", unit="nd") if HAS_TQDM else as_completed(futs)
            for fut in iterator:
                nd, rows = fut.result()
                out_path = os.path.join(BASE_OUTDIR, f"{nd}_ADM3_RISK.csv")
                df = pd.DataFrame(rows, columns=["adm3_id", "analysis_id", "def_ha", "risk_total", "farm_amount", "farm_amount_total"])
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"[OK] CSV generado: {out_path}")
                paths.append(out_path)

    print("\nArchivos generados:")
    for p in paths:
        print(" -", p)
    return paths


if __name__ == "__main__":
    name_def = [
        "smbyc_deforestation_cumulative_2010-2011",
       "smbyc_deforestation_cumulative_2010-2012",
        "smbyc_deforestation_cumulative_2010-2013",
        "smbyc_deforestation_cumulative_2010-2014",
        "smbyc_deforestation_cumulative_2010-2015",
        "smbyc_deforestation_cumulative_2010-2016",
        "smbyc_deforestation_cumulative_2010-2017",
        "smbyc_deforestation_cumulative_2010-2018",
        "smbyc_deforestation_cumulative_2010-2019",
        "smbyc_deforestation_cumulative_2010-2020",
        "smbyc_deforestation_cumulative_2010-2021",
        "smbyc_deforestation_cumulative_2010-2022",
        "smbyc_deforestation_cumulative_2010-2023",
    ]
    adm3_risk(name_def, workers=3)


        #"smbyc_deforestation_annual_2010-2012",
        #"smbyc_deforestation_annual_2012-2013",
        #"smbyc_deforestation_annual_2013-2014",
        #"smbyc_deforestation_annual_2014-2015",
        #"smbyc_deforestation_annual_2015-2016",
        #"smbyc_deforestation_annual_2016-2017",
        #"smbyc_deforestation_annual_2017-2018",
        #"smbyc_deforestation_annual_2018-2019",
        #"smbyc_deforestation_annual_2019-2020",
        #"smbyc_deforestation_annual_2020-2021",
        #"smbyc_deforestation_annual_2021-2022",
        #"smbyc_deforestation_annual_2022-2023",
        #"smbyc_deforestation_annual_2023-2024",





       #"smbyc_deforestation_cumulative_2010-2011",
       #"smbyc_deforestation_cumulative_2010-2012",
       # "smbyc_deforestation_cumulative_2010-2013",
       # "smbyc_deforestation_cumulative_2010-2014",
       # "smbyc_deforestation_cumulative_2010-2015",
       # "smbyc_deforestation_cumulative_2010-2016",
       # "smbyc_deforestation_cumulative_2010-2017",
       # "smbyc_deforestation_cumulative_2010-2018",
       # "smbyc_deforestation_cumulative_2010-2019",
       # "smbyc_deforestation_cumulative_2010-2020",
       # "smbyc_deforestation_cumulative_2010-2021",
       # "smbyc_deforestation_cumulative_2010-2022",
       # "smbyc_deforestation_cumulative_2010-2023",