import os
import re
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# -----------------------
# CONFIGURACIÓN GLOBAL
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ganabosques")

BASE_OUT_DEFAULT = r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\enterprise_risk"
OUT_ENCODING = "utf-8-sig"

COL_DEFORESTATION = "deforestation"
COL_ANALYSIS = "analysis"
COL_ENTERPRISE = "enterprise"
COL_FARM = "farm"
COL_FARMRISK = "farmrisk"
COL_MOVEMENT = "movement"

CHUNK_SIZE = 1000
MAX_WORKERS = 3

# -----------------------
# UTILIDADES
# -----------------------
DIGITS_RE = re.compile(r"(\d+)")

@dataclass
class Buckets:
    direct: Set[str] = field(default_factory=set)
    indirect_in: Set[str] = field(default_factory=set)
    indirect_out: Set[str] = field(default_factory=set)


def _year_range(year: int) -> Tuple[datetime, datetime]:
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)

# -----------------------
# OPCIÓN 1: ENTERPRISE (sin cambios)
# -----------------------

def _extract_codes_from_folder(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"  !! Carpeta no encontrada: {folder}")
        return []
    codes: List[str] = []
    for name in os.listdir(folder):
        if not name.lower().endswith('.geojson'):
            continue
        m = DIGITS_RE.search(os.path.splitext(name)[0])
        if m:
            codes.append(m.group(1))
    return sorted(set(codes))


def _find_analysis_ids(db, name_def: str) -> List[ObjectId]:
    def_doc = db[COL_DEFORESTATION].find_one({"name": name_def}, {"_id": 1})
    if not def_doc:
        print(f"  !! No se encontró en 'deforestation' el name='{name_def}'")
        return []
    def_id = def_doc["_id"]
    analysis_ids = [doc["_id"] for doc in db[COL_ANALYSIS].find({"deforestation_id": def_id}, {"_id": 1})]
    return analysis_ids


def _rows_for_enterprise(db, enterprise_name: str, analysis_ids: List[ObjectId], codes_from_folder: List[str]) -> Dict[str, Any]:
    if not codes_from_folder:
        return {"rows": [], "farms_count": 0, "farms_ids": []}
    farm_cursor = db[COL_FARM].find({"ext_id.ext_code": {"$in": codes_from_folder}}, {"_id": 1})
    farm_ids: List[ObjectId] = [doc["_id"] for doc in farm_cursor]
    if not farm_ids:
        return {"rows": [], "farms_count": 0, "farms_ids": []}

    fr_query = {
        "farm_id": {"$in": farm_ids},
        "analysis_id": {"$in": analysis_ids},
        "$or": [{"risk_direct": True}, {"risk_input": True}, {"risk_output": True}],
    }
    fr_fields = {"farm_id": 1, "analysis_id": 1, "risk_direct": 1, "risk_input": 1, "risk_output": 1}
    fr_cursor = db[COL_FARMRISK].find(fr_query, fr_fields)

    grouped: Dict[str, Buckets] = {}
    for fr in fr_cursor:
        farm_id = fr.get("farm_id")
        analysis_id = fr.get("analysis_id")
        if farm_id is None or analysis_id is None:
            continue
        key = str(analysis_id)
        b = grouped.setdefault(key, Buckets())
        s_farm = str(farm_id)
        if fr.get("risk_direct", False): b.direct.add(s_farm)
        if fr.get("risk_input", False): b.indirect_in.add(s_farm)
        if fr.get("risk_output", False): b.indirect_out.add(s_farm)

    rows: List[Dict[str, Any]] = []
    for analysis_key, b in grouped.items():
        rows.append({
            "name_enterprise": enterprise_name,
            "analysis_id": analysis_key,
            "ids_directa": ",".join(sorted(b.direct)),
            "ids_indirect_in": ",".join(sorted(b.indirect_in)),
            "ids_indirect_out": ",".join(sorted(b.indirect_out)),
        })
    return {"rows": rows, "farms_count": len(farm_ids), "farms_ids": farm_ids}


def enterprise_risk_for_enterprises(name_def: str, tipo_empresa: str, enterprise_dirs: Dict[str, str], out_path_base: str):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    analysis_ids = _find_analysis_ids(db, name_def)
    if not analysis_ids:
        print("  !! No se encontraron analysis_id.")
        return {}

    enterprises = list(db[COL_ENTERPRISE].find({"type_enterprise": tipo_empresa}, {"name": 1}))
    names_in_db = {e.get("name", ""): e["_id"] for e in enterprises}

    os.makedirs(out_path_base, exist_ok=True)
    csv_paths_by_enterprise: Dict[str, List[str]] = {}

    for ent_name, folder in enterprise_dirs.items():
        if ent_name not in names_in_db:
            continue
        codes = _extract_codes_from_folder(folder)
        res = _rows_for_enterprise(db, ent_name, analysis_ids, codes)
        rows = res["rows"]
        if not rows:
            csv_paths_by_enterprise[ent_name] = []
            continue
        by_analysis: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_analysis.setdefault(r["analysis_id"], []).append(r)
        generated: List[str] = []
        for analysis_id, group_rows in by_analysis.items():
            df = pd.DataFrame(group_rows, columns=["name_enterprise", "analysis_id", "ids_directa", "ids_indirect_in", "ids_indirect_out"])
            empresa_lower = ent_name.lower().replace(" ", "_")
            fname = f"alert_enterprise_{empresa_lower}_{analysis_id}.csv"
            out_path = os.path.join(out_path_base, fname)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            generated.append(out_path)
        csv_paths_by_enterprise[ent_name] = generated
    return csv_paths_by_enterprise


def run_enterprise_interactive():
    print("=== Opción 1: Enterprise (COLACTEOS/CARNATURAL) ===")
    print("1) COLACTEOS\n2) CARNATURAL")
    choice = input("Selecciona numero (1/2): ").strip()
    if choice == '1':
        empresa = 'COLACTEOS'
    elif choice == '2':
        empresa = 'CARNATURAL'
    else:
        return
    carpeta_geo = input("Ruta carpeta donde están los GEOJSON: ").strip()
    if not carpeta_geo or not os.path.isdir(carpeta_geo):
        return
    name_defs_input = input("Ingrese los name_def separados por coma:\n")
    name_defs = [s.strip() for s in name_defs_input.split(',') if s.strip()]
    out_path = input(f"Ruta carpeta salida (enter para usar {BASE_OUT_DEFAULT}): ").strip() or BASE_OUT_DEFAULT
    os.makedirs(out_path, exist_ok=True)
    enterprise_dirs = {empresa: carpeta_geo}
    for nd in name_defs:
        enterprise_risk_for_enterprises(nd, 'ENTERPRISE', enterprise_dirs, out_path)

# -----------------------
# OPCIÓN 2 Y 3: MOVEMENT paralelizado por empresa
# -----------------------

def _process_enterprise_full(args):
    ent_id, ent_name, analysis_ids, year_start, year_end, chunk_size, tipo_empresa_upper, mongo_uri, db_name = args
    client = MongoClient(mongo_uri)
    db = client[db_name]
    col_movement = db[COL_MOVEMENT]
    col_farmrisk = db[COL_FARMRISK]

    mv_in_match = {"enterprise_id_destination": ent_id, "date": {"$gte": year_start, "$lt": year_end}}
    inbound_vals = col_movement.distinct("farm_id_origin", mv_in_match)
    inbound_farms = set(ObjectId(v) if isinstance(v, str) and ObjectId.is_valid(v) else v for v in inbound_vals)

    outbound_farms = set()
    if tipo_empresa_upper == "CATTLE_FAIR":
        mv_out_match = {"enterprise_id_origin": ent_id, "date": {"$gte": year_start, "$lt": year_end}}
        outbound_vals = col_movement.distinct("farm_id_destination", mv_out_match)
        outbound_farms = set(ObjectId(v) if isinstance(v, str) and ObjectId.is_valid(v) else v for v in outbound_vals)

    if not inbound_farms and not outbound_farms:
        return {"ent_name": ent_name, "rows": []}

    all_farms = list(inbound_farms | outbound_farms)
    inbound_str = {str(x) for x in inbound_farms}
    outbound_str = {str(x) for x in outbound_farms}

    chunks = [all_farms[i:i+chunk_size] for i in range(0, len(all_farms), chunk_size)]
    combined_map: Dict[str, Dict[str, Set[str]]] = {}

    for chunk in chunks:
        q = {
            "farm_id": {"$in": chunk},
            "analysis_id": {"$in": analysis_ids},
            "$or": [{"risk_direct": True}, {"risk_input": True}, {"risk_output": True}]
        }
        proj = {"farm_id": 1, "analysis_id": 1, "risk_direct": 1, "risk_input": 1, "risk_output": 1}
        for fr in col_farmrisk.find(q, proj):
            aid = fr.get("analysis_id")
            fid = fr.get("farm_id")
            if not aid or not fid:
                continue
            akey = str(aid)
            s_farm = str(fid)
            entry = combined_map.setdefault(akey, {
                "entrada_direct": set(), "entrada_ind_in": set(), "entrada_ind_out": set(),
                "salida_direct": set(), "salida_ind_in": set(), "salida_ind_out": set()
            })
            if s_farm in inbound_str:
                if fr.get("risk_direct", False): entry["entrada_direct"].add(s_farm)
                if fr.get("risk_input", False): entry["entrada_ind_in"].add(s_farm)
                if fr.get("risk_output", False): entry["entrada_ind_out"].add(s_farm)
            if s_farm in outbound_str:
                if fr.get("risk_direct", False): entry["salida_direct"].add(s_farm)
                if fr.get("risk_input", False): entry["salida_ind_in"].add(s_farm)
                if fr.get("risk_output", False): entry["salida_ind_out"].add(s_farm)

    rows = []
    for aid, b in combined_map.items():
        if not any(b.values()):
            continue
        rows.append({
            "name_enterprise": ent_name,
            "analysis_id": aid,
            "entrada_ids_directa": ",".join(sorted(b["entrada_direct"])),
            "entrada_ids_indirect_in": ",".join(sorted(b["entrada_ind_in"])),
            "entrada_ids_indirect_out": ",".join(sorted(b["entrada_ind_out"])),
            "salida_ids_directa": ",".join(sorted(b["salida_direct"])),
            "salida_ids_indirect_in": ",".join(sorted(b["salida_ind_in"])),
            "salida_ids_indirect_out": ",".join(sorted(b["salida_ind_out"])),
        })
    return {"ent_name": ent_name, "rows": rows}


def run_movement_parallel(name_def: List[str], mov_year: List[int], tipo_empresa: str, out_base: str,
                          chunk_size: int = CHUNK_SIZE, n_workers: int = MAX_WORKERS):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    col_def = db[COL_DEFORESTATION]
    col_analysis = db[COL_ANALYSIS]
    col_enterprise = db[COL_ENTERPRISE]

    enterprises = list(col_enterprise.find({"type_enterprise": tipo_empresa.upper()}, {"name": 1}))
    if not enterprises:
        print(f"No se encontraron empresas tipo {tipo_empresa.upper()}")
        return

    for nd, year in zip(name_def, mov_year):
        def_doc = col_def.find_one({"name": nd}, {"_id": 1})
        if not def_doc:
            print(f"No se encontró deforestation.name={nd}")
            continue
        def_id = def_doc["_id"]
        analysis_ids = [doc["_id"] for doc in col_analysis.find({"deforestation_id": def_id}, {"_id": 1})]
        if not analysis_ids:
            print(f"No hay analysis para deforestation {nd}")
            continue

        year_start, year_end = _year_range(year)
        print(f"\nProcesando {nd} - {year} | empresas totales: {len(enterprises)}")

        tasks = [
            (ent["_id"], ent.get("name", ""), analysis_ids, year_start, year_end,
             chunk_size, tipo_empresa.upper(), MONGO_URI, MONGO_DB_NAME)
            for ent in enterprises
        ]

        results_rows = []
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
            futures = [ex.submit(_process_enterprise_full, t) for t in tasks]
            if HAS_TQDM:
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[{nd}-{year}] Procesando empresas", ncols=100):
                    res = fut.result()
                    results_rows.extend(res.get("rows", []))
            else:
                for fut in as_completed(futures):
                    res = fut.result()
                    results_rows.extend(res.get("rows", []))

        if results_rows:
            df = pd.DataFrame(results_rows)
            os.makedirs(out_base, exist_ok=True)
            out_file = os.path.join(out_base, f"{nd}_{year}_{tipo_empresa.upper()}_by_enterprise_parallel.csv")
            df.to_csv(out_file, index=False, encoding=OUT_ENCODING)
            print(f"CSV guardado en {out_file}")
        else:
            print(f"No se generaron filas para {nd} - {year} - {tipo_empresa.upper()}")

# -----------------------
# INTERFAZ PRINCIPAL
# -----------------------

def main():
    print("=== MENU DE TIPOS DE EMPRESA ===")
    print("1 - Empresa (ENTERPRISE)")
    print("2 - Matadero (SLAUGHTERHOUSE)")
    print("3 - Feria Ganadera (CATTLE_FAIR)")
    choice = input("Selecciona el tipo (1,2,3): ").strip()
    if choice == "1":
        run_enterprise_interactive()
    elif choice in ("2", "3"):
        name_def = input("Ingresa name_def (vector separados por coma): ").strip()
        name_defs = [s.strip() for s in name_def.split(",") if s.strip()]
        mov_year_input = input("Ingresa mov_year (vector de años separados por coma): ").strip()
        mov_year = [int(s.strip()) for s in mov_year_input.split(',') if s.strip()]
        out_path = input("Ruta donde se guardarán los CSV: ").strip()
        tipo = "SLAUGHTERHOUSE" if choice == "2" else "CATTLE_FAIR"
        run_movement_parallel(name_defs, mov_year, tipo, out_path)
    else:
        print("Opción inválida.")


if __name__ == "__main__":
    main()
