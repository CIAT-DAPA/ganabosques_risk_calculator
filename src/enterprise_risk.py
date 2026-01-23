# -*- coding: utf-8 -*-
"""
enterprise_risk.py  (paralelizado + barra de progreso + heartbeat 5 min)

Flujo por cada par (name_def[i], MOV_YEAR[i]):
  1) deforestation.name == name_def[i] -> _id_deforestation
  2) analysis.deforestation_id == _id_deforestation -> analysis_id(s)
  3) enterprise: filtra por tipo_empresa e itera TODAS las empresas (en paralelo)
  4) movement: año == MOV_YEAR[i] y enterprise_id_destination == _id_empresa
     -> junta farm_id_origin únicos
  5) farmrisk: farm_id ∈ lista y analysis_id ∈ analysis_ids y algún riesgo True
     -> listas ids_directa / ids_indirect_in / ids_indirect_out por analysis_id
  6) CSV por par con columnas:
     name_enterprise, analysis_id, ids_directa, ids_indirect_in, ids_indirect_out
"""

import os
import time
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from pymongo import MongoClient
from bson import ObjectId
import pandas as pd

# Barra de progreso opcional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# --- Conexión ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ganabosques"

# --- Salida ---
BASE_OUTDIR = r"D:\OneDrive - CGIAR\Desktop\ganabosques\ganabosques_results\04_etl_alerts\alert_enterprise\SLAUGHTERHOUSE"


@dataclass
class Buckets:
    direct: Set[str] = field(default_factory=set)
    indirect_in: Set[str] = field(default_factory=set)
    indirect_out: Set[str] = field(default_factory=set)


def _year_range(year: int) -> Tuple[datetime, datetime]:
    """Rango [start, end) para un año calendario (UTC)."""
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return start, end


# -------------- Worker: procesa UNA empresa --------------
def _worker_process_enterprise(args) -> Dict[str, Any]:
    """
    Ejecuta en un proceso aparte para una empresa.
    Cada worker crea su propio MongoClient.
    Devuelve filas para el CSV y métricas para logs.
    """
    (
        idx, total_enterprises,
        ent_id_str, ent_name,
        year_start, year_end,
        analysis_ids_str,
        mongo_uri, db_name
    ) = args

    client = MongoClient(mongo_uri)
    db = client[db_name]
    col_movement = db["movement"]
    col_farmrisk = db["farmrisk"]

    ent_id = ObjectId(ent_id_str)
    analysis_ids = [ObjectId(s) for s in analysis_ids_str]

    # Movements año + destino empresa
    mv_query = {
        "enterprise_id_destination": ent_id,
        "date": {"$gte": year_start, "$lt": year_end},
    }
    mv_cursor = col_movement.find(mv_query, {"farm_id_origin": 1})
    farms_set: Set[ObjectId] = set()
    mv_count = 0
    for mv in mv_cursor:
        mv_count += 1
        fid = mv.get("farm_id_origin")
        if isinstance(fid, ObjectId):
            farms_set.add(fid)
        elif isinstance(fid, str):
            try:
                farms_set.add(ObjectId(fid))
            except Exception:
                pass

    if not farms_set:
        return {
            "index": idx,
            "log": {
                "ent_name": ent_name,
                "ent_id": ent_id_str,
                "mv_count": mv_count,
                "farms_count": 0,
                "frisk_count": 0,
                "analyses_with_risk": 0,
                "had_risk": False,
                "total": total_enterprises,
            },
            "rows": [],
        }

    # farmrisk
    fr_query = {
        "farm_id": {"$in": list(farms_set)},
        "analysis_id": {"$in": analysis_ids},
        "$or": [
            {"risk_direct": True},
            {"risk_input": True},
            {"risk_output": True},
        ],
    }
    fr_cursor = col_farmrisk.find(
        fr_query,
        {"farm_id": 1, "analysis_id": 1, "risk_direct": 1, "risk_input": 1, "risk_output": 1}
    )

    grouped: Dict[str, Buckets] = {}
    fr_count = 0
    for fr in fr_cursor:
        fr_count += 1
        farm_id = fr.get("farm_id")
        analysis_id = fr.get("analysis_id")
        if farm_id is None or analysis_id is None:
            continue
        key = str(analysis_id)
        b = grouped.setdefault(key, Buckets())
        s_farm = str(farm_id)
        if fr.get("risk_direct", False) is True:
            b.direct.add(s_farm)
        if fr.get("risk_input", False) is True:
            b.indirect_in.add(s_farm)
        if fr.get("risk_output", False) is True:
            b.indirect_out.add(s_farm)

    rows = []
    for analysis_key, b in grouped.items():
        rows.append({
            "name_enterprise": ent_name,
            "analysis_id": analysis_key,
            "ids_directa": ",".join(sorted(b.direct)),
            "ids_indirect_in": ",".join(sorted(b.indirect_in)),
            "ids_indirect_out": ",".join(sorted(b.indirect_out)),
        })

    return {
        "index": idx,
        "log": {
            "ent_name": ent_name,
            "ent_id": ent_id_str,
            "mv_count": mv_count,
            "farms_count": len(farms_set),
            "frisk_count": fr_count,
            "analyses_with_risk": len(grouped),
            "had_risk": len(grouped) > 0,
            "total": total_enterprises,
        },
        "rows": rows,
    }


# -------------- Orquestador --------------
def enterprise_risk(
    name_def: List[str],
    MOV_YEAR: List[int],
    tipo_empresa: str,
    workers: int = 1
) -> List[str]:
    """
    Ejecuta el flujo para todos los pares (name_def[i], MOV_YEAR[i]) y todas las empresas del tipo.
    Paraleliza por empresa con 'workers' procesos.
    Devuelve la lista de rutas de CSV generados.
    """
    if len(name_def) != len(MOV_YEAR):
        raise ValueError("Los vectores name_def y MOV_YEAR deben tener la MISMA longitud (pares 1 a 1).")

    max_w = max(1, min(workers, cpu_count()))
    print("\n================ ENTERPRISE RISK (PARALELIZADO) =================", flush=True)
    print(f"Conectando a MongoDB: {MONGO_URI} | DB: {MONGO_DB_NAME}", flush=True)
    print(f"Workers (procesos): {max_w}\n", flush=True)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]

    col_deforestation = db["deforestation"]
    col_analysis = db["analysis"]
    col_enterprise = db["enterprise"]

    print(f"[1/5] Buscando enterprises con type_enterprise = '{tipo_empresa}' ...", flush=True)
    enterprises = list(col_enterprise.find({"type_enterprise": tipo_empresa}, {"name": 1}))
    total_enterprises = len(enterprises)
    print(f"  -> Empresas encontradas: {total_enterprises}\n", flush=True)
    if total_enterprises == 0:
        print("  !! No hay empresas para ese tipo. No se generarán CSV.", flush=True)
        return []

    os.makedirs(BASE_OUTDIR, exist_ok=True)
    generated_paths: List[str] = []

    # Iterar por pares
    for pair_idx, (name_i, year_i) in enumerate(zip(name_def, MOV_YEAR), start=1):
        print(f"========== PAR {pair_idx}/{len(name_def)}: name_def='{name_i}' | MOV_YEAR={year_i} ==========", flush=True)

        print("[2/5] Buscando en 'deforestation' por name ...", flush=True)
        def_doc = col_deforestation.find_one({"name": name_i}, {"_id": 1})
        if not def_doc:
            print(f"  !! No se encontró deforestation.name = '{name_i}'. Se omite este par.", flush=True)
            continue
        def_id: ObjectId = def_doc["_id"]
        print(f"  -> _id_deforestation = {def_id}", flush=True)

        print("[3/5] Buscando 'analysis' por deforestation_id ...", flush=True)
        analysis_ids = [doc["_id"] for doc in col_analysis.find({"deforestation_id": def_id}, {"_id": 1})]
        print(f"  -> analysis_id(s) encontrados: {len(analysis_ids)}", flush=True)
        if not analysis_ids:
            print("  !! No hay analysis para ese deforestation. Se omite este par.", flush=True)
            continue

        year_start, year_end = _year_range(int(year_i))
        print(f"[4/5] Filtro de año en 'movement': {year_start.date()} <= date < {year_end.date()}", flush=True)

        # Enviar trabajos
        analysis_ids_str = [str(aid) for aid in analysis_ids]
        tasks = []
        start_collect = time.time()
        with ProcessPoolExecutor(max_workers=max_w) as ex:
            for e_idx, ent in enumerate(enterprises, start=1):
                ent_id_str = str(ent["_id"])
                ent_name = ent.get("name", "")
                args = (
                    e_idx, total_enterprises,
                    ent_id_str, ent_name,
                    year_start, year_end,
                    analysis_ids_str,
                    MONGO_URI, MONGO_DB_NAME
                )
                tasks.append(ex.submit(_worker_process_enterprise, args))

            # Recolección con barra de progreso y heartbeat cada 5 min
            results = []
            processed = 0
            last_heartbeat = time.time()
            if HAS_TQDM:
                pbar = tqdm(total=total_enterprises, desc="[Progreso empresas]", unit="emp", leave=True)
            else:
                pbar = None
                print("[Progreso empresas] 0/{} ...".format(total_enterprises), end="\r", flush=True)

            for fut in as_completed(tasks):
                res = fut.result()
                results.append(res)
                processed += 1

                if pbar:
                    pbar.update(1)
                else:
                    print("[Progreso empresas] {}/{} ...".format(processed, total_enterprises), end="\r", flush=True)

                # Heartbeat cada 5 minutos
                now = time.time()
                if now - last_heartbeat > 300:
                    elapsed = int(now - start_collect)
                    print(f"\n[Sigue trabajando] {processed}/{total_enterprises} empresas procesadas | {elapsed//60} min transcurridos", flush=True)
                    last_heartbeat = now

            if pbar:
                pbar.close()
            else:
                print()  # salto de línea del contador simple

        # Ordenar para mantener el estilo de impresión final
        results.sort(key=lambda d: d["index"])

        rows: List[Dict[str, Any]] = []
        total_mv = total_farms = total_frisk = 0
        enterprises_with_risk = 0

        # Imprimir logs detallados por empresa (en orden)
        for res in results:
            lg = res["log"]
            ent_name = lg["ent_name"]
            ent_id_str = lg["ent_id"]
            mv_count = lg["mv_count"]
            farms_count = lg["farms_count"]
            frisk_count = lg["frisk_count"]
            analyses_with_risk = lg["analyses_with_risk"]
            had_risk = lg["had_risk"]

            print(f"\n[2/5] ({res['index']}/{total_enterprises}) Empresa: {ent_name} | _id={ent_id_str}", flush=True)
            print("      - Buscando movements por enterprise_id_destination ...", flush=True)
            print(f"      -> Movements encontrados: {mv_count}", flush=True)
            print(f"      -> Fincas únicas (farm_id_origin): {farms_count}", flush=True)
            print("      - Consultando farmrisk para esas fincas con algún riesgo TRUE ...", flush=True)
            print(f"      -> Registros en farmrisk con riesgo TRUE: {frisk_count}", flush=True)
            if had_risk:
                print(f"      -> Análisis con riesgos para esta empresa: {analyses_with_risk}", flush=True)
            else:
                print("      -> No hubo riesgos TRUE para esta empresa.", flush=True)

            total_mv += mv_count
            total_farms += farms_count
            total_frisk += frisk_count
            if had_risk:
                enterprises_with_risk += 1

            rows.extend(res["rows"])

        # Guardar CSV del par
        out_path = os.path.join(BASE_OUTDIR, f"{name_i}_MOV_YEAR_{year_i}.csv")
        print("\n[5/5] Escribiendo CSV ...", flush=True)
        if rows:
            df = pd.DataFrame(rows, columns=[
                "name_enterprise", "analysis_id",
                "ids_directa", "ids_indirect_in", "ids_indirect_out"
            ])
        else:
            df = pd.DataFrame(columns=[
                "name_enterprise", "analysis_id",
                "ids_directa", "ids_indirect_in", "ids_indirect_out"
            ])
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Resumen del par
        print("\n================ RESUMEN =================", flush=True)
        print(f"Empresas del tipo                  : {total_enterprises}", flush=True)
        print(f"Empresas con algún riesgo TRUE     : {enterprises_with_risk}", flush=True)
        print(f"Movements revisados (total)        : {total_mv}", flush=True)
        print(f"Fincas únicas consideradas (total) : {total_farms}", flush=True)
        print(f"Registros farmrisk con riesgo TRUE : {total_frisk}", flush=True)
        print(f"CSV generado en: {out_path}", flush=True)
        print("=========================================\n", flush=True)

        generated_paths.append(out_path)

    print("Archivos generados:")
    for p in generated_paths:
        print(" -", p)
    return generated_paths


if __name__ == "__main__":
    # Ejemplo (ajústalo con tus pares reales):
    name_def = ["smbyc_deforestation_annual_2014-2015"]
    MOV_YEAR = [2014]
    tipo_empresa = "SLAUGHTERHOUSE"

    # Ejecuta con 3 workers (cores)
    enterprise_risk(name_def, MOV_YEAR, tipo_empresa, workers=3)


#,"smbyc_deforestation_cumulative_2010-2015","smbyc_deforestation_cumulative_2010-2016",
#                "smbyc_deforestation_cumulative_2010-2017","smbyc_deforestation_cumulative_2010-2018","smbyc_deforestation_cumulative_2010-2019",
 #               "smbyc_deforestation_cumulative_2010-2020","smbyc_deforestation_cumulative_2010-2021","smbyc_deforestation_cumulative_2010-2022",
#                "smbyc_deforestation_cumulative_2010-2023",
#,2015,2016,2017,2018,2019,2020,2021,2022,2023