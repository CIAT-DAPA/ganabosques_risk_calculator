# -*- coding: utf-8 -*-
"""
etl_farmrisk_with_adm3.py

ETL combinado:
1) Crear Analysis en bulk
2) Procesar carpeta recursivamente y guardar FarmRisk
3) Procesar carpeta de ADM3-Risk y guardar/upsert en MongoDB
"""

import os
import re
from pathlib import Path
from datetime import datetime
from bson import ObjectId
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne
from mongoengine import connect

import config
from ganabosques_orm.collections.analysis import Analysis
from ganabosques_orm.collections.deforestation import Deforestation
from ganabosques_orm.collections.farmrisk import FarmRisk
from ganabosques_orm.collections.protectedareas import ProtectedAreas
from ganabosques_orm.collections.farmingareas import FarmingAreas

# ---------------- CONFIG ----------------
CSV_CHUNK_ROWS = int(os.getenv("CSV_CHUNK_ROWS", "50000"))
MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "20000"))

# Columnas esperadas en CSV de farmrisk
COL_FARM_ID         = "farm_id"
COL_FARM_POLY_ID    = "farm_polygons_id"
COL_FARM_POLY_ID_ALT= "farm_poligons_id"
COL_RISK_DIRECT     = "direct_alert"
COL_RISK_INPUT      = "indirect_alert_in"
COL_RISK_OUTPUT     = "indirect_alert_out"
COL_DEF_HA          = "deforested_ha"
COL_DEF_PROP        = "deforested_prop"
COL_FIN_HA          = "farming_in_ha"
COL_FIN_PROP        = "farming_in_prop"
COL_FOUT_HA         = "farming_out_ha"
COL_FOUT_PROP       = "farming_out_prop"
COL_PROT_HA         = "protected_ha"
COL_PROT_PROP       = "protected_prop"

# ---------------- UTILIDADES ----------------
def normalize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\.csv$", "", s, flags=re.IGNORECASE)
    s = re.sub(r",$", "", s).strip()
    return s

def to_bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "si", "sí", "y", "yes"}: return True
    if s in {"false", "f", "0", "no", "n", ""}: return False
    try: return float(s) != 0.0
    except Exception: return False

def to_float(v):
    if v is None or v == "": return 0.0
    try: return float(v)
    except Exception: return 0.0

def to_objectid_or_none(v):
    if v is None or v == "": return None
    try: return ObjectId(str(v))
    except Exception: return None

# ---------------- CONEXIÓN ----------------
def init_connection():
    uri = config.config.get('CONNECTION_URI') or config.config.get('MONGO_URI')
    dbname = config.config.get('CONNECTION_DB') or config.config.get('MONGO_DB_NAME')
    if not uri or not dbname:
        raise RuntimeError("Falta CONNECTION_URI/CONNECTION_DB en config")
    connect(host=uri, db=dbname)

# ---------------- ANALYSIS BULK ----------------
def create_analysis_fast():
    init_connection()
    prot = ProtectedAreas.objects.first()
    farm = FarmingAreas.objects.first()
    if not prot or not farm:
        raise RuntimeError("Faltan documentos en protectedareas o farmingareas.")

    def_ids = [d.id for d in Deforestation.objects().only('id')]
    print(f"Creando Analysis para {len(def_ids)} deforestations...")

    user_id = ObjectId()
    now = datetime.utcnow()
    ops = [{"protected_areas_id": prot.id,
            "farming_areas_id": farm.id,
            "deforestation_id": did,
            "user_id": user_id,
            "date": now} for did in def_ids]

    if not ops:
        print("Nada que crear.")
        return

    coll = Analysis._get_collection()
    inserted = 0
    for i in tqdm(range(0, len(ops), MONGO_BATCH_SIZE), desc="Insertando Analysis", unit="lote"):
        coll.insert_many(ops[i:i+MONGO_BATCH_SIZE], ordered=False)
        inserted += len(ops[i:i+MONGO_BATCH_SIZE])
    print(f"Listo. Insertados {inserted} Analysis.")

# ---------------- FARMRISK ----------------
def save_farm_risk_fast(input_file: str, analysis_id: str) -> int:
    try: analysis_oid = ObjectId(analysis_id)
    except Exception as e:
        raise ValueError(f"analysis_id inválido: {analysis_id}") from e

    reader = pd.read_csv(input_file, dtype=str, chunksize=CSV_CHUNK_ROWS, low_memory=False)
    print(f"Ingiere {Path(input_file).name} → analysis_id={analysis_oid}")

    coll = FarmRisk._get_collection()
    total_inserted = 0

    for chunk_df in reader:
        if COL_FARM_POLY_ID_ALT in chunk_df.columns and COL_FARM_POLY_ID not in chunk_df.columns:
            chunk_df = chunk_df.rename(columns={COL_FARM_POLY_ID_ALT: COL_FARM_POLY_ID})

        for col in [COL_FARM_ID, COL_FARM_POLY_ID, COL_RISK_DIRECT, COL_RISK_INPUT, COL_RISK_OUTPUT,
                    COL_DEF_HA, COL_DEF_PROP, COL_FIN_HA, COL_FIN_PROP, COL_FOUT_HA, COL_FOUT_PROP,
                    COL_PROT_HA, COL_PROT_PROP]:
            if col not in chunk_df.columns: chunk_df[col] = ""

        farm_ids = chunk_df[COL_FARM_ID].astype(str).map(to_objectid_or_none)
        poly_ids = chunk_df[COL_FARM_POLY_ID].astype(str).map(to_objectid_or_none)
        risk_direct = chunk_df[COL_RISK_DIRECT].map(to_bool)
        risk_input  = chunk_df[COL_RISK_INPUT].map(to_bool)
        risk_output = chunk_df[COL_RISK_OUTPUT].map(to_bool)
        def_ha   = chunk_df[COL_DEF_HA].map(to_float)
        def_prop = chunk_df[COL_DEF_PROP].map(to_float)
        fin_ha   = chunk_df[COL_FIN_HA].map(to_float)
        fin_prop = chunk_df[COL_FIN_PROP].map(to_float)
        fout_ha  = chunk_df[COL_FOUT_HA].map(to_float)
        fout_prop= chunk_df[COL_FOUT_PROP].map(to_float)
        prot_ha  = chunk_df[COL_PROT_HA].map(to_float)
        prot_prop= chunk_df[COL_PROT_PROP].map(to_float)

        docs = []
        for i in range(len(chunk_df)):
            if farm_ids.iat[i] is None: continue
            docs.append({
                "analysis_id":      analysis_oid,
                "farm_id":          farm_ids.iat[i],
                "farm_polygons_id": poly_ids.iat[i],
                "risk_direct":      bool(risk_direct.iat[i]),
                "risk_input":       bool(risk_input.iat[i]),
                "risk_output":      bool(risk_output.iat[i]),
                "deforestation":    {"ha": def_ha.iat[i],  "prop": def_prop.iat[i]},
                "farming_in":       {"ha": fin_ha.iat[i],  "prop": fin_prop.iat[i]},
                "farming_out":      {"ha": fout_ha.iat[i], "prop": fout_prop.iat[i]},
                "protected":        {"ha": prot_ha.iat[i], "prop": prot_prop.iat[i]},
            })

        for j in tqdm(range(0, len(docs), MONGO_BATCH_SIZE), desc=f"Insertando {Path(input_file).name}", unit="lote", leave=False):
            coll.insert_many(docs[j:j+MONGO_BATCH_SIZE], ordered=False)
            total_inserted += len(docs[j:j+MONGO_BATCH_SIZE])

    return total_inserted

def resolve_analysis_id_by_filename(csv_path: Path):
    base_name = normalize_name(csv_path.stem)
    defo = Deforestation.objects(name=base_name).only('id').first()
    if not defo: return None
    ana = Analysis.objects(deforestation_id=defo.id).order_by('-id').only('id').first()
    if not ana: return None
    return str(ana.id)

def process_farmrisk_folder_recursive(root_folder: str):
    root = Path(root_folder)
    if not root.exists(): return print(f"No existe la carpeta: {root}")

    init_connection()
    processed_files = set()
    counts_by_analysis = {}

    for dirpath, _, filenames in os.walk(root):
        dirp = Path(dirpath)
        csvs = [dirp / f for f in filenames if f.lower().endswith(".csv")]
        if not csvs: continue

        print(f"\n[CARPETA] Procesando: {dirp} → {len(csvs)} CSV(s)")
        for csv_path in csvs:
            abs_path = str(csv_path.resolve())
            if abs_path in processed_files: continue

            ana_id = resolve_analysis_id_by_filename(csv_path)
            if not ana_id:
                print(f"  [WARN] No se resolvió analysis para '{csv_path.name}'. Se omite.")
                continue

            inserted = save_farm_risk_fast(str(csv_path), ana_id)
            processed_files.add(abs_path)
            counts_by_analysis.setdefault(ana_id, 0)
            counts_by_analysis[ana_id] += inserted
            print(f"  {csv_path.name}: Guardados {inserted} docs para analysis_id={ana_id}")

    print("\n=== RESUMEN FINAL ===")
    total_all = sum(counts_by_analysis.values())
    for ana_id, cnt in counts_by_analysis.items():
        print(f"analysis_id={ana_id} → {cnt} documentos")
    print(f"TOTAL DOCUMENTOS GUARDADOS: {total_all}")
    print("🏁 Proceso finalizado.")

# ---------------- ADM3-RISK ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "ganabosques")
COL_ADM3RISK = "adm3risk"
CSV_CHUNK_ROWS_ADM3 = 10000
MONGO_BATCH_SIZE_ADM3 = 5000

def to_oid(x):
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    try: return ObjectId(s)
    except Exception: return None

def to_int(x):
    try: return int(float(str(x).strip()))
    except: return 0

def get_db():
    cli = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    cli.admin.command("ping")
    return cli, cli[MONGO_DB]

def ensure_indexes(db):
    db[COL_ADM3RISK].create_index([("analysis_id", 1), ("adm3_id", 1)], unique=True)
    db[COL_ADM3RISK].create_index([("adm3_id", 1)])
    db[COL_ADM3RISK].create_index([("risk_total", 1)])
    db[COL_ADM3RISK].create_index([("farm_amount", 1)])

def iter_csv_chunks(path: Path):
    return pd.read_csv(
        path, dtype=str, chunksize=CSV_CHUNK_ROWS_ADM3,
        engine="python", sep=None, encoding="utf-8-sig", on_bad_lines="skip"
    )

def upsert_adm3risk_from_csv(csv_path: Path) -> int:
    cli, db = get_db()
    ensure_indexes(db)
    total_changed = 0

    try:
        print(f"Procesando: {csv_path}")
        for chunk in tqdm(iter_csv_chunks(csv_path), desc=f"[Chunks] {csv_path.name}", unit="chunk"):
            chunk.rename(columns=lambda c: str(c).replace("\ufeff", "").strip(), inplace=True)
            for col in ["adm3_id", "analysis_id", "def_ha", "risk_total", "farm_amount"]:
                if col not in chunk.columns: chunk[col] = ""

            adm3_oid = chunk["adm3_id"].map(to_oid)
            ana_oid  = chunk["analysis_id"].map(to_oid)
            def_ha   = chunk["def_ha"].map(to_float)
            risk_bool= chunk["risk_total"].map(to_bool)
            farm_amount = chunk["farm_amount"].map(to_int)

            ops = []
            for i in range(len(chunk)):
                a = ana_oid.iat[i]
                m = adm3_oid.iat[i]
                if a is None or m is None: continue
                doc = {
                    "analysis_id": a,
                    "adm3_id": m,
                    "def_ha": float(def_ha.iat[i]),
                    "risk_total": bool(risk_bool.iat[i]),
                    "farm_amount": int(farm_amount.iat[i]),
                }
                ops.append(UpdateOne({"analysis_id": a, "adm3_id": m}, {"$set": doc}, upsert=True))

            for j in range(0, len(ops), MONGO_BATCH_SIZE_ADM3):
                res = db[COL_ADM3RISK].bulk_write(ops[j:j+MONGO_BATCH_SIZE_ADM3], ordered=False)
                total_changed += (res.upserted_count + res.modified_count)
    finally:
        cli.close()

    print(f"✅ {csv_path.name}: upserts/modificados = {total_changed}")
    return total_changed

def process_adm3_folder(folder_path: str):
    path = Path(folder_path)
    if not path.exists(): return print(f"No existe la carpeta: {path}")

    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not files: return print(f"No hay CSV en {path}")

    total = 0
    print(f"Encontrados {len(files)} CSV en {path}")
    for f in tqdm(files, desc="[Archivos]", unit="csv"):
        total += upsert_adm3risk_from_csv(f)
    print(f"🏁 Terminado. Total filas afectadas: {total}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("=== ETL Combinado ===")
    print("1) Crear Analysis en bulk")
    print("2) Procesar carpeta FarmRisk (recursivo)")
    print("3) Procesar carpeta ADM3-Risk")
    opt = input("Elige opción (1/2/3): ").strip()
    if opt == "1":
        create_analysis_fast()
    elif opt == "2":
        folder = input("Ruta raíz con subfolders de FarmRisk: ").strip()
        process_farmrisk_folder_recursive(folder)
    elif opt == "3":
        folder = input("Ruta con CSV de ADM3-Risk: ").strip()
        process_adm3_folder(folder)
    else:
        print("Opción inválida.")
