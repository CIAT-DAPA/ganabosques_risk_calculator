# -*- coding: utf-8 -*-
"""
etl_two_phase_fast.py
Pipeline en 2 fases optimizada:
 - FASE A: combinar direct + indirect + metrics por subfolder -> combined_dir
 - FASE B: (rápida) bulk query a Mongo para todos los codes, construir code2farm y farm2poly,
          luego mapear cada combined en chunks (vectorizado) y guardar final.
Muestra barras de progreso por cada tabla y permite cache en disco y procesamiento paralelo.
"""

import sys
import time
import math
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import csv
import argparse

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm

# ---------- CONFIG DEFAULTS (ajusta según tu entorno) ----------
PERIODO = "cumulative"
EMPRESA = "SMBYC"
ROOT_BASE = Path(r"D:\OneDrive - CGIAR\Desktop\ganabosques\ganabosques_results\04_etl_alerts")
METRICS_FILE_DEFAULT = ROOT_BASE / "metrics" / "metricas.csv"
COMBINED_DIR_DEFAULT = ROOT_BASE / "combined_temp"
OUTPUT_DIR_DEFAULT = ROOT_BASE / "cleaned_output_final"

MONGO_URI_DEFAULT = "mongodb://localhost:27017"
MONGO_DB_DEFAULT = "ganabosques"

# columnas
COL_SIT_CODE = "id"
OUT_FARM_ID = "farm_id"
OUT_FARM_POLY = "farm_poligons_id"

# tuning
CHUNK_QUERY = 5000         # tamaño $in para consultas a Mongo en bulk (ajustar: 5k-10k suele ir bien)
CSV_CHUNK_SIZE = 20000     # filas por chunk al mapear archivos grandes
COMBINED_FILE_SUFFIX = ".combined.csv"

# paralelismo para fase B (mapeo de archivos). 1 = secuencial (recomendado para barras claras).
PARALLEL_WORKERS = 1

# cache en disco (pickle) para code2farm y farm2poly (acelera runs repetidos)
CACHE_ENABLED = True
CACHE_DIR = Path("etl_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CODE2FARM_CACHE_PATH = CACHE_DIR / "code2farm.pkl"
FARM2POLY_CACHE_PATH = CACHE_DIR / "farm2poly.pkl"

# lectura CSV kwargs
READ_KWARGS = dict(dtype=str, engine="python", sep=None, encoding="utf-8-sig", on_bad_lines="skip")

# ---------- UTIL ----------
def find_first_csv_in_folder(folder: Path):
    if folder is None:
        return None
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return None
    for entry in sorted(p.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            return entry
    return None

def read_csv_safe(path: Path):
    if path is None:
        return None
    try:
        return pd.read_csv(path, **READ_KWARGS)
    except Exception as e:
        print(f"[ERROR] No se pudo leer {path}: {e}")
        return None

# ---------- MONGO HELPERS ----------
def get_pymongo_client(uri):
    cli = MongoClient(uri, serverSelectionTimeoutMS=5000)
    cli.admin.command("ping")
    return cli

def choose_farm(docs):
    if not docs:
        return None
    geos = [d for d in docs if d.get("farm_source") == "GEOFARMER"]
    pool = geos if geos else docs
    return sorted(pool, key=lambda x: x["_id"], reverse=True)[0]

def query_farms_bulk(db, codes: List[str], chunk_size: int = CHUNK_QUERY, progress_by: str = "codes") -> Dict[str, str]:
    """
    Bulk query ext_id.ext_code -> farm_id (str).
    progress_by: "codes" (barra hasta len(codes)) o "chunks" (barra por chunks).
    """
    out = {}
    if not codes:
        return out

    n = len(codes)
    n_chunks = math.ceil(n / chunk_size)
    if progress_by == "codes":
        pbar = tqdm(total=n, desc="  farms (codes)", unit="code")
    else:
        pbar = tqdm(total=n_chunks, desc="  farms (chunks)", unit="chunk")

    start_total = time.time()
    try:
        for i in range(0, n, chunk_size):
            chunk = codes[i:i + chunk_size]
            cur = db["farm"].find(
                {"ext_id.ext_code": {"$in": chunk}},
                {"_id": 1, "farm_source": 1, "ext_id": 1}
            )
            bucket = {}
            for d in cur:
                for e in (d.get("ext_id") or []):
                    code = e.get("ext_code")
                    if code in chunk:
                        bucket.setdefault(code, []).append(d)
            for code, docs in bucket.items():
                chosen = choose_farm(docs)
                if chosen:
                    out[code] = str(chosen["_id"])
            if progress_by == "codes":
                pbar.update(len(chunk))
            else:
                pbar.update(1)
    finally:
        pbar.close()
    elapsed = time.time() - start_total
    tqdm.write(f"    -> farms encontrados: {len(out)} en {elapsed:.2f}s")
    return out

def query_polygons_bulk(db, farm_ids: List[str], chunk_size: int = CHUNK_QUERY, progress_by: str = "chunks") -> Dict[str, str]:
    """
    Bulk query farm_id -> last polygon_id
    progress_by: "ids" or "chunks"
    """
    out = {}
    if not farm_ids:
        return out

    # convertir a ObjectId
    farm_ids_obj = []
    for f in farm_ids:
        try:
            farm_ids_obj.append(ObjectId(str(f)))
        except Exception:
            continue
    if not farm_ids_obj:
        return out

    n = len(farm_ids_obj)
    n_chunks = math.ceil(n / chunk_size)
    if progress_by == "ids":
        pbar = tqdm(total=n, desc="  polygons (ids)", unit="id")
    else:
        pbar = tqdm(total=n_chunks, desc="  polygons (chunks)", unit="chunk")

    start_total = time.time()
    try:
        for i in range(0, n, chunk_size):
            chunk = farm_ids_obj[i:i + chunk_size]
            pipeline = [
                {"$match": {"farm_id": {"$in": chunk}}},
                {"$sort": {"_id": -1}},
                {"$group": {"_id": "$farm_id", "poly": {"$first": "$_id"}}}
            ]
            for g in db["farmpolygons"].aggregate(pipeline, allowDiskUse=True):
                out[str(g["_id"])] = str(g["poly"])
            if progress_by == "ids":
                pbar.update(len(chunk))
            else:
                pbar.update(1)
    finally:
        pbar.close()
    elapsed = time.time() - start_total
    tqdm.write(f"    -> polygons encontrados: {len(out)} en {elapsed:.2f}s")
    return out

# ---------- FASE A: combinar direct+indirect+metrics por subfolder ----------
def find_common_key(df1, df2):
    if df1 is None or df2 is None:
        return None
    common = list(set(df1.columns).intersection(df2.columns))
    if not common:
        return None
    preferred = ["id", "farm_id", "adm3_id", "deforestation_id", "name"]
    for p in preferred:
        if p in common:
            return p
    return common[0]

def _prefix_conflicting_cols(df_metrics, existing_cols_set, prefix="metric_"):
    if df_metrics is None:
        return df_metrics
    df = df_metrics.copy()
    new_names = {}
    for col in df.columns:
        if col in existing_cols_set:
            new_names[col] = f"{prefix}{col}"
    if new_names:
        df = df.rename(columns=new_names)
    return df

def cbind_with_metrics(direct_df, indirect_df, metrics_df):
    # Copiado y conservado de tu versión original (mismo comportamiento)
    if direct_df is None:
        return None, "direct_df vacío"
    if indirect_df is None:
        indirect_df = pd.DataFrame()
    n = len(direct_df)

    if metrics_df is None or metrics_df.empty:
        if len(indirect_df) not in (0, n):
            if len(indirect_df) == 1:
                ind_used = pd.DataFrame([indirect_df.iloc[0].to_dict()] * n)
            else:
                ind_used = indirect_df.reindex(range(n)).reset_index(drop=True)
        else:
            ind_used = indirect_df.reset_index(drop=True)
        res = pd.concat([direct_df.reset_index(drop=True), ind_used.reset_index(drop=True)], axis=1)
        return res, "No metrics: concatenado direct+indirect"

    m = len(metrics_df)
    if m == 1:
        metrics_row = pd.DataFrame([metrics_df.iloc[0].to_dict()] * n)
        metrics_row = _prefix_conflicting_cols(metrics_row, set(direct_df.columns).union(indirect_df.columns))
        if len(indirect_df) == 1:
            ind_used = pd.DataFrame([indirect_df.iloc[0].to_dict()] * n)
        elif len(indirect_df) == 0:
            ind_used = pd.DataFrame([{}] * n)
        else:
            ind_used = indirect_df.reindex(range(n)).reset_index(drop=True)
        res = pd.concat([direct_df.reset_index(drop=True), ind_used.reset_index(drop=True), metrics_row], axis=1)
        return res, "metrics: 1 fila -> broadcast"

    if m == n:
        metrics_pref = _prefix_conflicting_cols(metrics_df.reset_index(drop=True), set(direct_df.columns).union(indirect_df.columns))
        if len(indirect_df) == 1:
            ind_used = pd.DataFrame([indirect_df.iloc[0].to_dict()] * n)
        elif len(indirect_df) == 0:
            ind_used = pd.DataFrame([{}] * n)
        else:
            ind_used = indirect_df.reset_index(drop=True)
        res = pd.concat([direct_df.reset_index(drop=True), ind_used.reset_index(drop=True), metrics_pref], axis=1)
        return res, "metrics: mismo número filas -> concat col-wise"

    key = find_common_key(direct_df, metrics_df)
    if key:
        merged = pd.merge(direct_df, metrics_df, how="left", on=key, suffixes=("","_m"))
        if len(indirect_df) == len(merged):
            ind_used = indirect_df.reset_index(drop=True)
        elif len(indirect_df) == 1:
            ind_used = pd.DataFrame([indirect_df.iloc[0].to_dict()] * len(merged))
        else:
            ind_used = indirect_df.reindex(range(len(merged))).reset_index(drop=True)
        merged = pd.concat([merged.reset_index(drop=True), ind_used.reset_index(drop=True)], axis=1)
        return merged, f"metrics: merge left por '{key}'"

    metrics_row = pd.DataFrame([metrics_df.iloc[0].to_dict()] * n)
    metrics_row = _prefix_conflicting_cols(metrics_row, set(direct_df.columns).union(indirect_df.columns))
    if len(indirect_df) == 1:
        ind_used = pd.DataFrame([indirect_df.iloc[0].to_dict()] * n)
    elif len(indirect_df) == 0:
        ind_used = pd.DataFrame([{}] * n)
    else:
        ind_used = indirect_df.reindex(range(n)).reset_index(drop=True)
    res = pd.concat([direct_df.reset_index(drop=True), ind_used.reset_index(drop=True), metrics_row], axis=1)
    return res, "metrics: fallback -> broadcast primera fila"

def create_combined_by_subfolder(periodo: str, empresa: str, root_base: Path, metrics_file: Path, combined_dir: Path) -> List[Path]:
    """
    FASE A: por cada subfolder en direct_alert/{empresa}, combina direct+indirect+metrics
    y escribe a combined_dir. Devuelve lista de archivos combined creados.
    """
    root_base = Path(root_base)
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    direct_base = root_base / periodo / "direct_alert" / empresa
    indirect_base = root_base / periodo / "indirect_alert" / empresa

    if not direct_base.exists() or not indirect_base.exists():
        print(f"[ERROR] direct/indirect base no existen:\n  {direct_base}\n  {indirect_base}")
        return []

    metrics_df = None
    if Path(metrics_file).exists():
        try:
            metrics_df = pd.read_csv(metrics_file, **READ_KWARGS)
        except Exception as e:
            print(f"[WARN] No se pudo leer metrics: {e}")
            metrics_df = None
    else:
        print(f"[WARN] metrics file no existe: {metrics_file} -> se procederá sin metrics")

    subfolders = sorted([p for p in direct_base.iterdir() if p.is_dir()])
    if not subfolders:
        print(f"[WARN] No hay subfolders en {direct_base}")
        return []

    saved_files = []
    for sub in tqdm(subfolders, desc="FASE A - Subfolders", unit="folder"):
        subname = sub.name
        direct_csv_path = find_first_csv_in_folder(direct_base / subname)
        indirect_csv_path = find_first_csv_in_folder(indirect_base / subname)

        if direct_csv_path is None and indirect_csv_path is None:
            continue

        direct_df = read_csv_safe(direct_csv_path) if direct_csv_path else pd.DataFrame()
        indirect_df = read_csv_safe(indirect_csv_path) if indirect_csv_path else pd.DataFrame()

        if (direct_df is None or direct_df.empty) and (indirect_df is None or indirect_df.empty):
            continue

        combined_df, info = cbind_with_metrics(direct_df if direct_df is not None else pd.DataFrame(),
                                              indirect_df if indirect_df is not None else pd.DataFrame(),
                                              metrics_df)
        if combined_df is None or combined_df.empty:
            continue

        out_name = f"{empresa.lower()}_deforestation_{periodo}_{subname}{COMBINED_FILE_SUFFIX}"
        out_path = Path(combined_dir) / out_name
        try:
            combined_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            saved_files.append(out_path)
        except Exception as e:
            print(f"[ERROR] Guardando combined {out_path}: {e}")
    return saved_files

# ---------- util: extraer códigos de un CSV por streaming (rápido) ----------
def extract_codes_from_csv(path: Path, sit_col: str = COL_SIT_CODE) -> List[str]:
    """
    Extrae códigos únicos leyendo línea a línea sin cargar todo en memoria.
    Maneja columnas duplicadas (toma la primera columna con nombre sit_col).
    """
    codes = set()
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                first_idx = header.index(sit_col)
            except ValueError:
                return []
            for row in reader:
                if first_idx < len(row):
                    v = (row[first_idx] or "").strip()
                    if v:
                        codes.add(v)
    except Exception as e:
        print(f"[WARN] Falló extracción de codes de {path}: {e}")
    return sorted(codes)

# ---------- FASE B: construir caches globales (bulk queries) ----------
def build_global_mappings(db, combined_files: List[Path], chunk_query: int = CHUNK_QUERY,
                          use_cache: bool = True, progress_by_farms: str = "codes", progress_by_polygons: str = "chunks"
                         ) -> Tuple[Dict[str,str], Dict[str,str]]:
    """
    1) Extrae todos los SIT codes únicos (streaming) de combined_files.
    2) Usa query_farms_bulk y query_polygons_bulk para construir maps:
        code2farm: sit_code -> farm_id (str)
        farm2poly: farm_id (str) -> polygon_id (str)
    3) Opción de cargar/guardar cache en disco.
    """
    # intentar cargar cache si existe y se desea
    if use_cache and CODE2FARM_CACHE_PATH.exists() and FARM2POLY_CACHE_PATH.exists():
        try:
            code2farm = pickle.loads(CODE2FARM_CACHE_PATH.read_bytes())
            farm2poly = pickle.loads(FARM2POLY_CACHE_PATH.read_bytes())
            print("[CACHE] Cargadas caches desde disco.")
            return code2farm, farm2poly
        except Exception:
            print("[CACHE] Error al cargar cache, se reconstruirá.")

    # 1) extraer todos los codes (streaming)
    all_codes = set()
    tqdm.write("FASE B: extrayendo códigos únicos de todos los combined (streaming)...")
    for p in tqdm(combined_files, desc="Extrayendo codes por archivo", unit="archivo"):
        codes = extract_codes_from_csv(p, sit_col=COL_SIT_CODE)
        all_codes.update(codes)
    all_codes = sorted(all_codes)
    tqdm.write(f"  Total códigos únicos para mapear: {len(all_codes)}")

    # 2) bulk query farms
    code2farm = {}
    if all_codes:
        tqdm.write("  Consultando farms en Mongo en bloques...")
        code2farm = query_farms_bulk(db, all_codes, chunk_size=chunk_query, progress_by=progress_by_farms)

    # 3) construir farm_ids (únicos) y bulk query polygons
    farm_ids = sorted({v for v in code2farm.values() if v})
    farm2poly = {}
    if farm_ids:
        tqdm.write(f"  Farm ids únicos: {len(farm_ids)} -> consultando polígonos...")
        farm2poly = query_polygons_bulk(db, farm_ids, chunk_size=chunk_query, progress_by=progress_by_polygons)

    # 4) guardar cache en disco si se desea
    if use_cache:
        try:
            CODE2FARM_CACHE_PATH.write_bytes(pickle.dumps(code2farm))
            FARM2POLY_CACHE_PATH.write_bytes(pickle.dumps(farm2poly))
            tqdm.write("[CACHE] Guardados code2farm y farm2poly en disco.")
        except Exception as e:
            tqdm.write(f"[CACHE] No se pudo guardar cache: {e}")

    return code2farm, farm2poly

# ---------- mapeo de un archivo single (chunked, vectorizado) ----------
def map_single_combined_file(args):
    """
    Función diseñada para usar en procesos workers (ProcessPoolExecutor).
    args: tuple(file_path_str, out_dir_str, code2farm, farm2poly, csv_chunk_size)
    Retorna: (file_name, rows_written, elapsed_seconds)
    """
    file_path_str, out_dir_str, code2farm, farm2poly, csv_chunk_size = args
    p = Path(file_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = p.name.replace(COMBINED_FILE_SUFFIX, ".csv") if p.name.endswith(COMBINED_FILE_SUFFIX) else p.name
    out_path = out_dir / out_name

    t0_total = time.time()
    first_chunk = True
    rows_written = 0

    # Intentar obtener número de filas una sola vez para barra
    total_rows = None
    try:
        total_rows = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
    except Exception:
        total_rows = None

    if total_rows:
        rows_bar = tqdm(total=total_rows, desc=f"  mapeando {p.name}", unit="fila")
    else:
        rows_bar = tqdm(desc=f"  mapeando {p.name}", unit="fila")

    try:
        reader = pd.read_csv(p, chunksize=csv_chunk_size, dtype=str, engine="python", sep=None, encoding="utf-8-sig", on_bad_lines="skip")
        for chunk in reader:
            cols = list(chunk.columns)
            try:
                first_idx = cols.index(COL_SIT_CODE)
                sit_series = chunk.iloc[:, first_idx].fillna("").astype(str).str.strip()
            except ValueError:
                sit_series = pd.Series([""] * len(chunk), index=chunk.index)

            # vectorizado: map desde dicts (muy rápido)
            chunk[OUT_FARM_ID] = sit_series.map(code2farm).fillna("").astype(str).values
            chunk[OUT_FARM_POLY] = chunk[OUT_FARM_ID].map(farm2poly).fillna("").astype(str).values

            chunk.to_csv(out_path, index=False, header=first_chunk, mode="w" if first_chunk else "a", encoding="utf-8-sig")
            first_chunk = False
            rows_written += len(chunk)
            rows_bar.update(len(chunk))
    finally:
        rows_bar.close()

    elapsed = time.time() - t0_total
    return (p.name, rows_written, elapsed, str(out_path))

# ---------- FASE B: mapear todos los combined usando los mappings globales ----------
def map_all_combined_files(db, combined_files: List[Path], output_dir: Path,
                           code2farm: Dict[str,str], farm2poly: Dict[str,str],
                           csv_chunk_size: int = CSV_CHUNK_SIZE, parallel_workers: int = PARALLEL_WORKERS):
    """
    Mapea cada combined file usando code2farm/farm2poly construidos y escribe archivos finales.
    Si parallel_workers > 1 usa ProcessPoolExecutor (con duplicación de memoria para dicts).
    Por defecto parallel_workers=1 (secuencial) -> barras limpias por tabla.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for p in combined_files:
        tasks.append((str(p), str(out_dir), code2farm, farm2poly, csv_chunk_size))

    results = []
    if parallel_workers and parallel_workers > 1:
        # ADVERTENCIA: pasar grandes dicts a procesos duplicará memoria; usar con cuidado.
        with ProcessPoolExecutor(max_workers=parallel_workers) as exe:
            futures = {exe.submit(map_single_combined_file, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="FASE B - archivos (paral.)", unit="archivo"):
                try:
                    res = fut.result()
                    results.append(res)
                    tqdm.write(f"Archivo {res[0]} -> filas={res[1]} tiempo={res[2]:.2f}s salida={res[3]}")
                except Exception as e:
                    tqdm.write(f"[ERROR] worker fallo: {e}")
    else:
        # secuencial, barras por tabla limpias
        for t in tasks:
            res = map_single_combined_file(t)
            results.append(res)
            tqdm.write(f"Archivo {res[0]} -> filas={res[1]} tiempo={res[2]:.2f}s salida={res[3]}")

    return results

# ---------- MAIN ENTRY ----------
def main():
    parser = argparse.ArgumentParser(description="ETL 2-fase rápido: combina y mapea farm_id/polygon.")
    parser.add_argument("--periodo", default=PERIODO)
    parser.add_argument("--empresa", default=EMPRESA)
    parser.add_argument("--root_base", default=str(ROOT_BASE))
    parser.add_argument("--metrics_file", default=str(METRICS_FILE_DEFAULT))
    parser.add_argument("--combined_dir", default=str(COMBINED_DIR_DEFAULT))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--mongo_uri", default=MONGO_URI_DEFAULT)
    parser.add_argument("--mongo_db", default=MONGO_DB_DEFAULT)
    parser.add_argument("--chunk_query", type=int, default=CHUNK_QUERY)
    parser.add_argument("--csv_chunk_size", type=int, default=CSV_CHUNK_SIZE)
    parser.add_argument("--parallel_workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--use_cache", action="store_true", help="Usar/guardar cache en disco (code2farm/farm2poly)")
    args = parser.parse_args()

    periodo = args.periodo
    empresa = args.empresa
    root_base = Path(args.root_base)
    metrics_file = Path(args.metrics_file)
    combined_dir = Path(args.combined_dir)
    output_dir = Path(args.output_dir)
    mongo_uri = args.mongo_uri
    mongo_db = args.mongo_db

    print(f"Parametros usados:\n  periodo={periodo}\n  empresa={empresa}\n  root_base={root_base}\n  metrics_file={metrics_file}\n  combined_dir={combined_dir}\n  output_dir={output_dir}\n  mongo_uri={mongo_uri}\n  mongo_db={mongo_db}\n  chunk_query={args.chunk_query}\n  csv_chunk_size={args.csv_chunk_size}\n  parallel_workers={args.parallel_workers}\n  use_cache={args.use_cache}\n")

    # conectar Mongo
    client = get_pymongo_client(mongo_uri)
    db = client[mongo_db]

    # FASE A: combinar y guardar combined
    t0 = time.time()
    combined_files = create_combined_by_subfolder(periodo, empresa, root_base, metrics_file, combined_dir)
    t1 = time.time()
    print(f"FASE A completada: {len(combined_files)} archivos combined creados en {t1 - t0:.2f}s")

    if not combined_files:
        print("No hay archivos combined para procesar. Saliendo.")
        client.close()
        return

    # FASE B: construir mappings globales (bulk queries)
    t0 = time.time()
    code2farm, farm2poly = build_global_mappings(db, combined_files, chunk_query=args.chunk_query,
                                                 use_cache=args.use_cache, progress_by_farms="codes",
                                                 progress_by_polygons="chunks")
    t1 = time.time()
    print(f"FASE B (bulk queries) completada en {t1 - t0:.2f}s -> farms={len(code2farm)} polygons={len(farm2poly)}")

    # FASE B (segunda parte): mapear archivos usando los mappings ya calculados
    t0 = time.time()
    results = map_all_combined_files(db, combined_files, output_dir, code2farm, farm2poly,
                                     csv_chunk_size=args.csv_chunk_size, parallel_workers=args.parallel_workers)
    t1 = time.time()
    print(f"FASE B (mapeo archivos) completada en {t1 - t0:.2f}s")

    # resumen
    total_rows = sum(r[1] for r in results) if results else 0
    print("\n=== RESUMEN FINAL ===")
    print(f"Combined files procesados: {len(results)}")
    print(f"Filas totales escritas: {total_rows}")
    print(f"Salida en: {output_dir.resolve()}")

    client.close()

if __name__ == "__main__":
    main()
