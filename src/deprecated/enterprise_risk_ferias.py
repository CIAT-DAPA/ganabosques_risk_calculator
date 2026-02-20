# -*- coding: utf-8 -*-
"""
enterprise_risk_cattle_fair_io.py
Genera, por par (deforestation_name, MOV_YEAR) y por empresa de un tipo,
un CSV con fincas con riesgo separadas por dirección (entrada/salida) y tipo de riesgo.

Formato de salida (por analysis_id y empresa):
name_enterprise,analysis_id,
entrada_ids_directa,entrada_ids_indirect_in,entrada_ids_indirect_out,
salida_ids_directa,salida_ids_indirect_in,salida_ids_indirect_out

Optimizado:
- distinct() para obtener fincas únicas inbound/outbound
- Proyecciones mínimas
- Consultas farmrisk por lotes (CHUNK_SIZE)
- Paralelización por empresa (ProcessPoolExecutor)
- Variables de entorno para conexión y salida
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

# --- Configuración ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ganabosques")
BASE_OUTDIR = os.getenv(
    "ENTERPRISE_RISK_OUTDIR",
    r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\enterprise_risk_ferias_new"
)
# Lote para consultas farmrisk (ajústalo según RAM/índices)
CHUNK_SIZE = int(os.getenv("FARMRISK_CHUNK_SIZE", "2000"))

# --- Constantes movement ---
FIELD_ENT_DEST = "enterprise_id_destination"
FIELD_ENT_ORIG = "enterprise_id_origin"
FIELD_FARM_ORIG = "farm_id_origin"
FIELD_FARM_DEST = "farm_id_destination"
FIELD_DATE     = "date"


@dataclass
class BucketsIO:
    # ENTRADA (fincas -> empresa)
    entrada_direct: Set[str] = field(default_factory=set)
    entrada_ind_in: Set[str] = field(default_factory=set)
    entrada_ind_out: Set[str] = field(default_factory=set)
    # SALIDA (empresa -> fincas)
    salida_direct: Set[str] = field(default_factory=set)
    salida_ind_in: Set[str] = field(default_factory=set)
    salida_ind_out: Set[str] = field(default_factory=set)


def _year_range(year: int) -> Tuple[datetime, datetime]:
    """Rango [start, end) para un año calendario (UTC)."""
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return start, end


def _to_objectid_maybe(x) -> ObjectId | None:
    """Convierte x en ObjectId si es posible; devuelve None si no."""
    if isinstance(x, ObjectId):
        return x
    if isinstance(x, str):
        try:
            return ObjectId(x)
        except Exception:
            return None
    return None


def _distinct_valid_ids(col, field: str, match: Dict[str, Any]) -> Set[ObjectId]:
    """
    Devuelve un set de ObjectId válidos provenientes de un distinct.
    Filtra None y strings no casteables.
    """
    vals = col.distinct(field, match)
    out: Set[ObjectId] = set()
    for v in vals:
        oid = _to_objectid_maybe(v)
        if oid is not None:
            out.add(oid)
    return out


def _format_list(ids: Set[str]) -> str:
    """Ordena y une en CSV; si vacío, devuelve ''."""
    if not ids:
        return ""
    return ",".join(sorted(ids))


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
        tipo_empresa_upper,
        mongo_uri, db_name
    ) = args

    client = MongoClient(mongo_uri)
    db = client[db_name]
    col_movement = db["movement"]
    col_farmrisk = db["farmrisk"]

    ent_id = ObjectId(ent_id_str)
    analysis_ids = [ObjectId(s) for s in analysis_ids_str]

    # --------- Recolección de fincas (entrada/salida) ---------
    # ENTRADA: empresa es DESTINO -> fincas origen
    mv_in_match = {
        FIELD_ENT_DEST: ent_id,
        FIELD_DATE: {"$gte": year_start, "$lt": year_end},
    }
    inbound_farms: Set[ObjectId] = _distinct_valid_ids(col_movement, FIELD_FARM_ORIG, mv_in_match)

    # SALIDA: solo ferias -> empresa es ORIGEN -> fincas destino
    outbound_farms: Set[ObjectId] = set()
    if tipo_empresa_upper == "CATTLE_FAIR":
        mv_out_match = {
            FIELD_ENT_ORIG: ent_id,
            FIELD_DATE: {"$gte": year_start, "$lt": year_end},
        }
        outbound_farms = _distinct_valid_ids(col_movement, FIELD_FARM_DEST, mv_out_match)

    # Si no hay fincas, devolver log y sin filas
    if not inbound_farms and not outbound_farms:
        return {
            "index": idx,
            "log": {
                "ent_name": ent_name,
                "ent_id": ent_id_str,
                "in_farms": 0,
                "out_farms": 0,
                "frisk_count": 0,
                "analyses_with_risk": 0,
                "had_risk": False,
                "total": total_enterprises,
            },
            "rows": [],
        }

    # Prepara sets de string para pruebas rápidas de dirección
    inbound_str: Set[str] = {str(x) for x in inbound_farms}
    outbound_str: Set[str] = {str(x) for x in outbound_farms}

    # --------- farmrisk (algún riesgo True) por lotes ---------
    all_farms: List[ObjectId] = list(inbound_farms | outbound_farms)
    rows: List[Dict[str, Any]] = []
    fr_count_global = 0
    analyses_with_risk_global = 0

    if not all_farms:
        # No debería ocurrir por el if anterior, pero por seguridad
        return {
            "index": idx,
            "log": {
                "ent_name": ent_name,
                "ent_id": ent_id_str,
                "in_farms": len(inbound_farms),
                "out_farms": len(outbound_farms),
                "frisk_count": 0,
                "analyses_with_risk": 0,
                "had_risk": False,
                "total": total_enterprises,
            },
            "rows": [],
        }

    # Procesar por analysis_id para filas limpias y sets dirigidos
    for aid in analysis_ids:
        buckets = BucketsIO()

        # procesar por lotes de farm_id para no hacer un $in enorme
        for start in range(0, len(all_farms), CHUNK_SIZE):
            chunk = all_farms[start:start + CHUNK_SIZE]
            fr_query = {
                "farm_id": {"$in": chunk},
                "analysis_id": aid,
                "$or": [
                    {"risk_direct": True},
                    {"risk_input": True},
                    {"risk_output": True},
                ],
            }
            # Proyección mínima
            fr_cursor = col_farmrisk.find(
                fr_query,
                {"farm_id": 1, "risk_direct": 1, "risk_input": 1, "risk_output": 1}
            )
            for fr in fr_cursor:
                fr_count_global += 1
                s_farm = str(fr.get("farm_id"))
                # ENTRADA
                if s_farm in inbound_str:
                    if fr.get("risk_direct", False):
                        buckets.entrada_direct.add(s_farm)
                    if fr.get("risk_input", False):
                        buckets.entrada_ind_in.add(s_farm)
                    if fr.get("risk_output", False):
                        buckets.entrada_ind_out.add(s_farm)
                # SALIDA
                if s_farm in outbound_str:
                    if fr.get("risk_direct", False):
                        buckets.salida_direct.add(s_farm)
                    if fr.get("risk_input", False):
                        buckets.salida_ind_in.add(s_farm)
                    if fr.get("risk_output", False):
                        buckets.salida_ind_out.add(s_farm)

        # ¿Este analysis_id tuvo al menos un riesgo?
        any_risk = any([
            buckets.entrada_direct, buckets.entrada_ind_in, buckets.entrada_ind_out,
            buckets.salida_direct,  buckets.salida_ind_in,  buckets.salida_ind_out
        ])
        if any_risk:
            analyses_with_risk_global += 1
            rows.append({
                "name_enterprise": ent_name,
                "analysis_id": str(aid),
                "entrada_ids_directa":     _format_list(buckets.entrada_direct),
                "entrada_ids_indirect_in": _format_list(buckets.entrada_ind_in),
                "entrada_ids_indirect_out":_format_list(buckets.entrada_ind_out),
                "salida_ids_directa":      _format_list(buckets.salida_direct),
                "salida_ids_indirect_in":  _format_list(buckets.salida_ind_in),
                "salida_ids_indirect_out": _format_list(buckets.salida_ind_out),
            })
        # Si no hubo riesgos para este analysis_id, no se agrega fila (requisito tuyo)

    return {
        "index": idx,
        "log": {
            "ent_name": ent_name,
            "ent_id": ent_id_str,
            "in_farms": len(inbound_farms),
            "out_farms": len(outbound_farms),
            "frisk_count": fr_count_global,
            "analyses_with_risk": analyses_with_risk_global,
            "had_risk": analyses_with_risk_global > 0,
            "total": total_enterprises,
        },
        "rows": rows,
    }


# -------------- Orquestador --------------
def enterprise_risk(
    name_def: List[str],
    MOV_YEAR: List[int],
    tipo_empresa: str,          # "CATTLE_FAIR" o "SLAUGHTERHOUSE"
    workers: int = 3
) -> List[str]:
    """
    Ejecuta el flujo para todos los pares (name_def[i], MOV_YEAR[i]) y todas las empresas del tipo.
    Paraleliza por empresa con 'workers' procesos.
    Devuelve la lista de rutas de CSV generados.
    """
    if len(name_def) != len(MOV_YEAR):
        raise ValueError("Los vectores name_def y MOV_YEAR deben tener la MISMA longitud.")

    tupper = tipo_empresa.upper()
    max_w = max(1, min(workers, cpu_count()))
    print("\n================ ENTERPRISE RISK (PARALELIZADO) =================", flush=True)
    print(f"Conectando a MongoDB: {MONGO_URI} | DB: {MONGO_DB_NAME}", flush=True)
    print(f"Workers (procesos): {max_w}\n", flush=True)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]

    col_deforestation = db["deforestation"]
    col_analysis      = db["analysis"]
    col_enterprise    = db["enterprise"]

    print(f"[1/5] Buscando enterprises con type_enterprise = '{tupper}' ...", flush=True)
    # Solo traemos _id y name
    enterprises = list(col_enterprise.find({"type_enterprise": tupper}, {"name": 1}))
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
                    tupper,
                    MONGO_URI, MONGO_DB_NAME
                )
                tasks.append(ex.submit(_worker_process_enterprise, args))

            # Recolección con barra de progreso y heartbeat
            results = []
            processed = 0
            last_heartbeat = time.time()
            pbar = tqdm(total=total_enterprises, desc="[Progreso empresas]", unit="emp", leave=True) if HAS_TQDM else None
            if not pbar:
                print("[Progreso empresas] 0/{} ...".format(total_enterprises), end="\r", flush=True)

            for fut in as_completed(tasks):
                res = fut.result()
                results.append(res)
                processed += 1

                if pbar:
                    pbar.update(1)
                else:
                    print("[Progreso empresas] {}/{} ...".format(processed, total_enterprises), end="\r", flush=True)

                # Heartbeat cada 5 min
                now = time.time()
                if now - last_heartbeat > 300:
                    elapsed = int(now - start_collect)
                    print(f"\n[Sigue trabajando] {processed}/{total_enterprises} empresas | {elapsed//60} min", flush=True)
                    last_heartbeat = now

            if pbar:
                pbar.close()
            else:
                print()  # salto de línea

        # Ordenar resultados por índice de empresa
        results.sort(key=lambda d: d["index"])

        rows: List[Dict[str, Any]] = []
        sum_in_farms = sum_out_farms = sum_frisk = 0
        enterprises_with_risk = 0

        # Logs por empresa (en orden)
        for res in results:
            lg = res["log"]
            ent_name = lg["ent_name"]
            ent_id_str = lg["ent_id"]
            in_farms = lg["in_farms"]
            out_farms = lg["out_farms"]
            frisk_count = lg["frisk_count"]
            analyses_with_risk = lg["analyses_with_risk"]
            had_risk = lg["had_risk"]

            print(f"\n[2/5] ({res['index']}/{total_enterprises}) Empresa: {ent_name} | _id={ent_id_str}", flush=True)
            print("      - Fincas inbound (recibe de):", in_farms, flush=True)
            if tupper == "CATTLE_FAIR":
                print("      - Fincas outbound (envía a) :", out_farms, flush=True)
            print("      - Registros farmrisk con algún riesgo TRUE:", frisk_count, flush=True)
            if had_risk:
                print(f"      -> Análisis con riesgos para esta empresa: {analyses_with_risk}", flush=True)
            else:
                print("      -> No hubo riesgos TRUE para esta empresa.", flush=True)

            sum_in_farms  += in_farms
            sum_out_farms += out_farms
            sum_frisk     += frisk_count
            if had_risk:
                enterprises_with_risk += 1

            rows.extend(res["rows"])

        # Guardar CSV del par
        out_path = os.path.join(BASE_OUTDIR, f"{name_i}_MOV_YEAR_{year_i}_{tupper}.csv")
        print("\n[5/5] Escribiendo CSV ...", flush=True)
        if rows:
            df = pd.DataFrame(rows, columns=[
                "name_enterprise", "analysis_id",
                "entrada_ids_directa","entrada_ids_indirect_in","entrada_ids_indirect_out",
                "salida_ids_directa", "salida_ids_indirect_in", "salida_ids_indirect_out",
            ])
        else:
            df = pd.DataFrame(columns=[
                "name_enterprise", "analysis_id",
                "entrada_ids_directa","entrada_ids_indirect_in","entrada_ids_indirect_out",
                "salida_ids_directa", "salida_ids_indirect_in", "salida_ids_indirect_out",
            ])
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Resumen del par
        print("\n================ RESUMEN =================", flush=True)
        print(f"Empresas del tipo                  : {total_enterprises}", flush=True)
        print(f"Fincas inbound (suma por empresas) : {sum_in_farms}", flush=True)
        if tupper == "CATTLE_FAIR":
            print(f"Fincas outbound (suma por empresas): {sum_out_farms}", flush=True)
        print(f"Registros farmrisk con riesgo TRUE : {sum_frisk}", flush=True)
        print(f"Empresas con algún riesgo TRUE     : {enterprises_with_risk}", flush=True)
        print(f"CSV generado en: {out_path}", flush=True)
        print("=========================================\n", flush=True)

        generated_paths.append(out_path)

    print("Archivos generados:")
    for p in generated_paths:
        print(" -", p)
    return generated_paths


if __name__ == "__main__":
    # Ejemplo
    name_def = [
        "smbyc_deforestation_cumulative_2010-2014","smbyc_deforestation_cumulative_2010-2015",
        "smbyc_deforestation_cumulative_2010-2016","smbyc_deforestation_cumulative_2010-2017",
        "smbyc_deforestation_cumulative_2010-2018","smbyc_deforestation_cumulative_2010-2019",
        "smbyc_deforestation_cumulative_2010-2020","smbyc_deforestation_cumulative_2010-2021",
        "smbyc_deforestation_cumulative_2010-2022","smbyc_deforestation_cumulative_2010-2023",
        
    ]
    MOV_YEAR = [2014,2015,2016,2017,2018,2019, 2020,2021,2022,2023]
    tipo_empresa = "CATTLE_FAIR"  # o "SLAUGHTERHOUSE"

    enterprise_risk(name_def, MOV_YEAR, tipo_empresa, workers=3)
