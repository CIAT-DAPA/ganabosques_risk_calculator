#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
alert_summary.py  — versión para movement + enterprise
------------------------------------------------------
Modos:
  - empresa: igual que antes, lee <risk>/<periodo>/total_risk/<SOURCE>/<empresa>/*.csv
  - institucion:
      * Lista instituciones desde la colección 'enterprise'
      * Para una institución: busca su _id en 'enterprise' y luego en 'movement' los docs donde
        enterprise_id_origin o enterprise_id_destination coincidan.
      * Si el otro lado del movimiento es FARM, recoge farm_id_origin/destination.
      * Cruza esos farm_ids con total_risk (annual+cumulative, todas las empresas).
  - --all-institutions: procesa todas las instituciones de 'enterprise'

Salida (única): <out_root>/empresa_alert_summary_all.csv
  actor_type, nombre, periodo, source, empresa_alert,
  n_predios_alerta, n_predios_directa, n_predios_indirect_in, n_predios_indirect_out,
  ids_alerta, ids_directa, ids_indirect_in, ids_indirect_out, YEARS

Ejemplos:
  # Empresa
  py src\alert_summary.py --mode empresa --nombre carnatural --sources smbyc --out-root "D:\...\empresa_alert\instituciones"

  # Todas las instituciones
  py src\alert_summary.py --mode institucion --all-institutions --inst-type SLAUGHTERHOUSE --sources smbyc --out-root "D:\...\empresa_alert\instituciones"

Requisitos .env (cargado por config.py):
  MONGO_URI, MONGO_DB_NAME, TOTAL_ALERT_BASE_DIR, (opcional) SOURCES_AVAILABLE, EMPRESA_ALERT_OUT_DIR
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from pymongo import MongoClient, errors as mongo_errors

from config import config

# ================== Constantes / columnas ==================
SOURCE_DIRNAME: Dict[str, str] = {"smbyc": "SMBYC", "atd": "ATD", "nad": "NAD"}
PERIODOS = ["annual", "cumulative"]
SUMMARY_COLS = [
    "actor_type", "nombre", "periodo", "source", "empresa_alert",
    "n_predios_alerta", "n_predios_directa", "n_predios_indirect_in", "n_predios_indirect_out",
    "ids_alerta", "ids_directa", "ids_indirect_in", "ids_indirect_out", "YEARS",
]

# ================== Helpers ==================
def sanitize_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).lower()

def normalize_token(s: Any) -> str:
    import re as _re
    s = "" if s is None else str(s)
    return _re.sub(r"[^a-z0-9]+", "", s.lower())

def s2bool(x) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "t", "yes", "y", "si", "sí"}:  return True
    if s in {"false", "0", "f", "no", "n"}:               return False
    if s in {"", "na", "nan", "none", "null", "no_info"}: return None
    return None

def extract_years_from_name(fname: str) -> str:
    m = re.search(r"(\d{4}(?:-\d{4})?)\.(?:csv|CSV)$", fname)
    return m.group(1) if m else ""

def resolve_risk_root() -> Path:
    base = config.get("TOTAL_ALERT_BASE_DIR")
    print(f"[DEBUG] TOTAL_ALERT_BASE_DIR: {base}")
    if not base:
        raise RuntimeError("Falta TOTAL_ALERT_BASE_DIR en .env/config")
    p = Path(base).resolve()
    print(f"[DEBUG] Resolviendo risk root desde: {p}")
    if not p.exists():
        raise RuntimeError(f"Ruta no existe: {p}")
    parts = [x.lower() for x in p.parts]
    if p.name.lower() == "risk":
        print(f"[DEBUG] Usando risk root directo: {p}")
        return p
    if "total_risk" in parts:
        idx = parts.index("total_risk")
        risk_period = Path(*p.parts[:idx])
        if risk_period.name.lower() not in {"annual", "cumulative"}:
            raise RuntimeError(f"No pude identificar el periodo en {risk_period}")
        root = risk_period.parent
        print(f"[DEBUG] risk root final: {root}")
        return root
    raise RuntimeError(f"TOTAL_ALERT_BASE_DIR debe apuntar a .../risk o .../risk/<periodo>/total_risk. Valor: {p}")

def chosen_sources(cli_sources: Optional[str]) -> List[str]:
    if cli_sources:
        out = [s.strip().lower() for s in cli_sources.split(",") if s.strip()]
        print(f"[DEBUG] Fuentes por CLI: {out}")
        return out
    env_src = config.get("SOURCES_AVAILABLE")
    if env_src:
        out = [s.strip().lower() for s in str(env_src).split(",") if s.strip()]
        print(f"[DEBUG] Fuentes por .env: {out}")
        return out
    print("[DEBUG] Fuentes por defecto: ['smbyc']")
    return ["smbyc"]

def get_out_root(cli_out: Optional[str]) -> Path:
    if cli_out:
        out = Path(cli_out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] out_root por CLI: {out}")
        return out
    env_out = config.get("EMPRESA_ALERT_OUT_DIR")
    if env_out:
        out = Path(env_out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] out_root por .env: {out}")
        return out
    out = resolve_risk_root() / "empresa_alert"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] out_root por defecto: {out}")
    return out

def safe_read_csv(fp: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(fp, dtype=str, low_memory=False, sep=None, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e1:
        print(f"   [WARN] Sniffer falló en {fp.name}: {e1} → intento simple")
        try:
            df = pd.read_csv(fp, dtype=str, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e2:
            print(f"   [ERROR] Lectura CSV {fp.name}: {e2}")
            return None

def ensure_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].apply(s2bool)
    alt = {normalize_token(c): c for c in df.columns}
    key = normalize_token(col)
    if key in alt:
        return df[alt[key]].apply(s2bool)
    return pd.Series([None] * len(df), index=df.index)

def _quote_join(items: Iterable[str]) -> str:
    return ";".join(f"'{str(x)}'" for x in items)

# ================== Alertas (total_risk) ==================
def iter_total_risk_csvs_for_all_empresas(risk_root: Path, periodo: str, source_key: str) -> Iterable[Path]:
    src_dir = risk_root / periodo / "total_risk" / SOURCE_DIRNAME.get(source_key, source_key.upper())
    print(f"[DEBUG] Buscando CSVs en: {src_dir}")
    if not src_dir.is_dir():
        print(f"[DEBUG] Directorio inexistente: {src_dir}")
        return []
    for emp_dir in sorted([d for d in src_dir.iterdir() if d.is_dir()]):
        for fp in sorted(emp_dir.glob("*.csv")):
            yield fp

def collect_alerts_for_farm_ids(
    farm_ids: Set[str], risk_root: Path, periodo: str, source_key: str
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    print(f"[DEBUG] collect_alerts_for_farm_ids: periodo={periodo}, source={source_key}, n_farm_ids={len(farm_ids)}")
    ids_all, ids_dir, ids_in, ids_out = set(), set(), set(), set()
    years_set = set()
    if not farm_ids:
        print("[DEBUG] No hay farm_ids → retorno vacío.")
        return ids_all, ids_dir, ids_in, ids_out, years_set

    n_files = 0
    for fp in iter_total_risk_csvs_for_all_empresas(risk_root, periodo, source_key):
        n_files += 1
        df = safe_read_csv(fp)
        if df is None:
            continue
        colmap = {normalize_token(c): c for c in df.columns}
        id_col = None
        for k in ["id", "idpredio", "predioid", "predio_id"]:
            if k in colmap:
                id_col = colmap[k]; break
        if not id_col:
            print(f"   [DEBUG] {fp.name} sin columna ID")
            continue
        df["id"] = df[id_col].astype(str).str.strip()
        df = df[df["id"].isin(farm_ids)]
        if df.empty:
            continue
        df["direct_alert"]       = ensure_bool_series(df, "direct_alert")
        df["indirect_alert_in"]  = ensure_bool_series(df, "indirect_alert_in")
        df["indirect_alert_out"] = ensure_bool_series(df, "indirect_alert_out")
        years = extract_years_from_name(fp.name)
        mask_any = (df["direct_alert"] == True) | (df["indirect_alert_in"] == True) | (df["indirect_alert_out"] == True)
        n_alert = int(mask_any.sum())
        print(f"   [DEBUG] {fp.name}: match_ids={len(df)}, en_alerta={n_alert}, YEARS={years}")
        if not n_alert:
            continue
        sub = df.loc[mask_any, ["id", "direct_alert", "indirect_alert_in", "indirect_alert_out"]].copy()
        ids_all.update(sub["id"].astype(str).tolist())
        years_set.add(years)
        ids_dir.update(sub.loc[sub["direct_alert"] == True, "id"].astype(str).tolist())
        ids_in.update(sub.loc[sub["indirect_alert_in"] == True, "id"].astype(str).tolist())
        ids_out.update(sub.loc[sub["indirect_alert_out"] == True, "id"].astype(str).tolist())
    print(f"[DEBUG] Archivos revisados: {n_files} | ids_all={len(ids_all)} | years={sorted(years_set)}")
    return ids_all, ids_dir, ids_in, ids_out, years_set

# ================== Mongo ==================
def get_mongo() -> MongoClient:
    uri = config["MONGO_URI"]
    print(f"[DEBUG] Conectando a Mongo: {uri}")
    return MongoClient(uri)

def list_institutions_from_enterprise(
    client: MongoClient,
    enterprise_coll: str,
    inst_type: Optional[str] = None,
    name_regex: Optional[str] = None,
) -> List[str]:
    """
    Lista instituciones leyendo la colección 'enterprise':
      - Si inst_type: filtra por type_enterprise (regex i)
      - Si no: devuelve todos los names no vacíos
    """
    db = client[config["MONGO_DB_NAME"]]
    coll = db[enterprise_coll]

    cond: Dict[str, Any] = {}
    if inst_type:
        cond["type_enterprise"] = {"$regex": inst_type, "$options": "i"}

    try:
        cursor = coll.find(cond, projection={"name": 1, "type_enterprise": 1})
    except mongo_errors.PyMongoError as e:
        print(f"[ERROR] list_institutions_from_enterprise falló: {e}")
        return []

    names: Set[str] = set()
    for doc in cursor:
        name = str(doc.get("name", "")).strip()
        if not name:
            continue
        if name_regex:
            if not re.search(name_regex, name, flags=re.IGNORECASE):
                continue
        names.add(name)

    print(f"[DEBUG] Instituciones detectadas en '{enterprise_coll}': {len(names)}")
    return sorted(names)

def enterprise_ids_by_name(
    client: MongoClient, enterprise_coll: str, name: str, inst_type: Optional[str] = None
) -> List[Any]:
    """Devuelve la lista de _id en enterprise con ese nombre (y tipo si se pasa)."""
    db = client[config["MONGO_DB_NAME"]]
    coll = db[enterprise_coll]
    cond: Dict[str, Any] = {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
    if inst_type:
        cond["type_enterprise"] = {"$regex": inst_type, "$options": "i"}
    ids = []
    try:
        for doc in coll.find(cond, projection={"_id": 1}):
            ids.append(doc["_id"])
    except mongo_errors.PyMongoError as e:
        print(f"[ERROR] enterprise_ids_by_name falló: {e}")
    print(f"[DEBUG] enterprise_ids_by_name('{name}'): {len(ids)} id(s)")
    return ids

def fetch_farm_ids_for_institution(
    client: MongoClient,
    institucion: str,
    inst_type: Optional[str],
    movement_coll: str,
    enterprise_coll: str,
) -> Set[str]:
    """
    Busca _id(s) de la institución en 'enterprise', luego consulta 'movement' donde
    enterprise_id_origin o enterprise_id_destination ∈ esos ids. Si el otro lado es FARM,
    agrega el farm_id correspondiente.
    """
    db = client[config["MONGO_DB_NAME"]]
    mov = db[movement_coll]

    ent_ids = enterprise_ids_by_name(client, enterprise_coll, institucion, inst_type)
    if not ent_ids:
        print(f"[DEBUG] No hay enterprise IDs para '{institucion}'.")
        return set()

    q = {"$or": [
        {"enterprise_id_origin": {"$in": ent_ids}},
        {"enterprise_id_destination": {"$in": ent_ids}},
    ]}
    proj = {
        "type_origin": 1, "type_destination": 1,
        "farm_id_origin": 1, "farm_id_destination": 1,
        "enterprise_id_origin": 1, "enterprise_id_destination": 1,
    }

    farm_ids: Set[str] = set()
    n_docs = 0
    try:
        cursor = mov.find(q, projection=proj)
        for doc in cursor:
            n_docs += 1
            to = str(doc.get("type_origin", "")).strip().upper()
            td = str(doc.get("type_destination", "")).strip().upper()
            if to == "FARM" and doc.get("farm_id_origin") is not None:
                farm_ids.add(str(doc["farm_id_origin"]))
            if td == "FARM" and doc.get("farm_id_destination") is not None:
                farm_ids.add(str(doc["farm_id_destination"]))
    except mongo_errors.PyMongoError as e:
        print(f"[ERROR] fetch_farm_ids_for_institution falló: {e}")
        return set()

    print(f"[DEBUG] Docs movement leídos: {n_docs} | farm_ids únicos: {len(farm_ids)}")
    return {fid for fid in farm_ids if fid}

# ================== Builders ==================
def build_rows_for_empresa(empresa: str, sources: List[str]) -> List[Dict[str, Any]]:
    risk_root = resolve_risk_root()
    rows: List[Dict[str, Any]] = []
    print(f"[DEBUG] === EMPRESA: {empresa} ===")
    for periodo in PERIODOS:
        for src in sources:
            folder = risk_root / periodo / "total_risk" / SOURCE_DIRNAME.get(src, src.upper()) / sanitize_name(empresa)
            print(f"[DEBUG] Explorando: {folder}")
            ids_all, ids_dir, ids_in, ids_out, years = set(), set(), set(), set(), set()
            if folder.is_dir():
                files = sorted(folder.glob("*.csv"))
                print(f"[DEBUG] CSV encontrados: {len(files)}")
                for fp in files:
                    print(f"   [DEBUG] Leyendo {fp.name}")
                    df = safe_read_csv(fp)
                    if df is None: continue
                    print(f"   [DEBUG] Filas totales: {len(df)}")
                    colmap = {normalize_token(c): c for c in df.columns}
                    id_col = None
                    for k in ["id", "idpredio", "predioid", "predio_id"]:
                        if k in colmap: id_col = colmap[k]; break
                    if not id_col:
                        print("   [DEBUG] No hay columna ID → salto.")
                        continue
                    df["id"] = df[id_col].astype(str).str.strip()
                    df["direct_alert"]       = ensure_bool_series(df, "direct_alert")
                    df["indirect_alert_in"]  = ensure_bool_series(df, "indirect_alert_in")
                    df["indirect_alert_out"] = ensure_bool_series(df, "indirect_alert_out")
                    mask = (df["direct_alert"] == True) | (df["indirect_alert_in"] == True) | (df["indirect_alert_out"] == True)
                    n_alert = int(mask.sum())
                    print(f"   [DEBUG] Filas en alerta: {n_alert}")
                    if not n_alert: continue
                    years.add(extract_years_from_name(fp.name))
                    sub = df.loc[mask, ["id","direct_alert","indirect_alert_in","indirect_alert_out"]]
                    ids_all.update(sub["id"].astype(str).tolist())
                    ids_dir.update(sub.loc[sub["direct_alert"]==True,"id"].astype(str).tolist())
                    ids_in.update(sub.loc[sub["indirect_alert_in"]==True,"id"].astype(str).tolist())
                    ids_out.update(sub.loc[sub["indirect_alert_out"]==True,"id"].astype(str).tolist())
            else:
                print(f"[DEBUG] Carpeta no encontrada: {folder}")

            empresa_alert = len(ids_all) > 0
            years_list = sorted([y for y in years if y])
            print(f"[DEBUG] Resumen parcial: periodo={periodo}, source={src}, empresa_alert={empresa_alert}, ids={len(ids_all)}, years={years_list}")

            if not years_list:
                rows.append({
                    "actor_type":"empresa","nombre":empresa,"periodo":periodo,"source":SOURCE_DIRNAME.get(src,src.upper()),
                    "empresa_alert":empresa_alert,
                    "n_predios_alerta":len(ids_all),"n_predios_directa":len(ids_dir),
                    "n_predios_indirect_in":len(ids_in),"n_predios_indirect_out":len(ids_out),
                    "ids_alerta":_quote_join(ids_all),"ids_directa":_quote_join(ids_dir),
                    "ids_indirect_in":_quote_join(ids_in),"ids_indirect_out":_quote_join(ids_out),
                    "YEARS":""
                })
            else:
                for y in years_list:
                    rows.append({
                        "actor_type":"empresa","nombre":empresa,"periodo":periodo,"source":SOURCE_DIRNAME.get(src,src.upper()),
                        "empresa_alert":empresa_alert,
                        "n_predios_alerta":len(ids_all),"n_predios_directa":len(ids_dir),
                        "n_predios_indirect_in":len(ids_in),"n_predios_indirect_out":len(ids_out),
                        "ids_alerta":_quote_join(ids_all),"ids_directa":_quote_join(ids_dir),
                        "ids_indirect_in":_quote_join(ids_in),"ids_indirect_out":_quote_join(ids_out),
                        "YEARS":y
                    })
    return rows

def build_rows_for_institucion(
    institucion: str, inst_type: Optional[str], sources: List[str],
    client: MongoClient, movement_coll: str, enterprise_coll: str
) -> List[Dict[str, Any]]:
    print(f"[DEBUG] === INSTITUCION: {institucion} (tipo={inst_type or 'ANY'}) ===")
    risk_root = resolve_risk_root()

    farm_ids = fetch_farm_ids_for_institution(client, institucion, inst_type, movement_coll, enterprise_coll)
    print(f"[DEBUG] FARM IDs para '{institucion}': {len(farm_ids)}")

    rows: List[Dict[str, Any]] = []
    for periodo in PERIODOS:
        for src in sources:
            ids_all, ids_dir, ids_in, ids_out, years = collect_alerts_for_farm_ids(farm_ids, risk_root, periodo, src)
            institucion_alert = len(ids_all) > 0
            years_list = sorted([y for y in years if y])
            print(f"[DEBUG] Resumen parcial: periodo={periodo}, source={src}, institucion_alert={institucion_alert}, ids={len(ids_all)}, years={years_list}")

            if not years_list:
                rows.append({
                    "actor_type":"institucion","nombre":institucion,"periodo":periodo,"source":SOURCE_DIRNAME.get(src,src.upper()),
                    "empresa_alert":institucion_alert,
                    "n_predios_alerta":len(ids_all),"n_predios_directa":len(ids_dir),
                    "n_predios_indirect_in":len(ids_in),"n_predios_indirect_out":len(ids_out),
                    "ids_alerta":_quote_join(ids_all),"ids_directa":_quote_join(ids_dir),
                    "ids_indirect_in":_quote_join(ids_in),"ids_indirect_out":_quote_join(ids_out),
                    "YEARS":""
                })
            else:
                for y in years_list:
                    rows.append({
                        "actor_type":"institucion","nombre":institucion,"periodo":periodo,"source":SOURCE_DIRNAME.get(src,src.upper()),
                        "empresa_alert":institucion_alert,
                        "n_predios_alerta":len(ids_all),"n_predios_directa":len(ids_dir),
                        "n_predios_indirect_in":len(ids_in),"n_predios_indirect_out":len(ids_out),
                        "ids_alerta":_quote_join(ids_all),"ids_directa":_quote_join(ids_dir),
                        "ids_indirect_in":_quote_join(ids_in),"ids_indirect_out":_quote_join(ids_out),
                        "YEARS":y
                    })
    return rows

# ================== Escritura ==================
def write_summary_all(out_root: Path, new_rows: List[Dict[str, Any]]) -> Path:
    out_csv = out_root / "empresa_alert_summary_all.csv"
    df_new = pd.DataFrame(new_rows, columns=SUMMARY_COLS)
    print(f"[DEBUG] Filas nuevas a escribir: {len(df_new)}")
    if out_csv.exists():
        print(f"[DEBUG] Archivo existente: {out_csv} → merge/upsert")
        df_old = pd.read_csv(out_csv, dtype=str)
        print(f"[DEBUG] Filas existentes: {len(df_old)}")
        key_cols = ["actor_type","nombre","periodo","source","YEARS"]
        df = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(subset=key_cols, keep="last")
        )
        print(f"[DEBUG] Filas tras upsert: {len(df)}")
    else:
        print(f"[DEBUG] Creando nuevo archivo: {out_csv}")
        df = df_new

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DEBUG] Escrito: {out_csv}")
    return out_csv

# ================== CLI ==================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resumen de alertas para empresa o institución (con movement + enterprise).")
    ap.add_argument("--mode", choices=["empresa","institucion"], required=True)
    ap.add_argument("--nombre", help="Nombre de empresa o institución (requerido salvo --all-institutions)")
    ap.add_argument("--inst-type", help="Tipo de institución (ej. SLAUGHTERHOUSE, FAIR, PLANT) para filtrar en enterprise/movement")
    ap.add_argument("--sources", help="Fuentes (ej. smbyc,atd,nad). Por defecto: SOURCES_AVAILABLE o 'smbyc'.")
    ap.add_argument("--out-root", help="Carpeta salida (default: <risk>/empresa_alert)")
    # nuevas: nombres reales de colecciones
    ap.add_argument("--movement-coll", default="movement", help="Colección de movimientos (default: movement)")
    ap.add_argument("--enterprise-coll", default="enterprise", help="Colección de instituciones (default: enterprise)")
    ap.add_argument("--all-institutions", action="store_true", help="Procesa todas las instituciones desde enterprise")
    ap.add_argument("--name-regex", help="Regex para filtrar nombres con --all-institutions (case-insensitive)")
    return ap.parse_args()

def main():
    args = parse_args()
    sources = chosen_sources(args.sources)
    out_root = get_out_root(args.out_root)

    print(f"[DEBUG] === INICIO === mode={args.mode} | sources={sources} | out_root={out_root}")

    if args.mode == "empresa":
        if not args.nombre:
            raise SystemExit("--nombre es requerido en modo empresa")
        rows = build_rows_for_empresa(args.nombre, sources)
        out_csv = write_summary_all(out_root, rows)
        print(f"[OK] Resumen (empresa: {args.nombre}) actualizado: {out_csv}")
        return

    # modo institucion
    try:
        client = get_mongo()
        client.admin.command('ping')
        print("[DEBUG] Conexión a Mongo OK.")
    except mongo_errors.PyMongoError as e:
        print(f"[ERROR] No se pudo conectar a Mongo: {e}")
        raise SystemExit(1)

    try:
        if args.all_institutions:
            inst_names = list_institutions_from_enterprise(
                client, enterprise_coll=args.enterprise_coll,
                inst_type=args.inst_type, name_regex=args.name_regex
            )
            print(f"[INFO] Total instituciones a procesar: {len(inst_names)}")
            total_rows: List[Dict[str, Any]] = []
            for i, inst in enumerate(inst_names, 1):
                print(f"[INFO] ({i}/{len(inst_names)}) Procesando: {inst}")
                total_rows.extend(
                    build_rows_for_institucion(inst, args.inst_type, sources, client,
                                               movement_coll=args.movement_coll,
                                               enterprise_coll=args.enterprise_coll)
                )
            out_csv = write_summary_all(out_root, total_rows)
            print(f"[OK] Resumen (todas las instituciones) actualizado: {out_csv}")
        else:
            if not args.nombre:
                raise SystemExit("--nombre es requerido si no usas --all-institutions en modo institucion")
            rows = build_rows_for_institucion(
                args.nombre, args.inst_type, sources, client,
                movement_coll=args.movement_coll, enterprise_coll=args.enterprise_coll
            )
            out_csv = write_summary_all(out_root, rows)
            print(f"[OK] Resumen (institucion: {args.nombre}) actualizado: {out_csv}")
    finally:
        client.close()
        print("[DEBUG] Conexión a Mongo cerrada.")

if __name__ == "__main__":
    main()
