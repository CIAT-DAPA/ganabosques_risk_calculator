#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from config import config

# --- MONGO
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # permite correr sin pymongo (deja columnas vacías)

# ===================== logging =====================
def setup_logging():
    lvl = getattr(logging, str(config.get("LOG_LEVEL", "WARNING")).upper(), logging.WARNING)
    logging.basicConfig(
        filename=config.get("LOG_FILE", "risk_postprocess.log"),
        level=lvl,
        format="%(asctime)s %(levelname)s:%(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

# ===================== normalización de IDs =====================
def _norm_id(s) -> str:
    """Normaliza IDs: str + trim, elimina '.0', quita ceros a la izquierda si es numérico, upper()."""
    if s is None:
        return ""
    s = str(s).strip()
    if s.endswith(".0"):
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    if s.isdigit():
        try:
            s = str(int(s))
        except Exception:
            s = s.lstrip("0") or "0"
    return s.upper()

# ===================== placeholders =====================
def parse_year_periods(years_raw: str) -> List[str]:
    if not years_raw:
        return []
    parts = [p.strip() for p in str(years_raw).split(",") if p.strip()]
    valids = []
    for p in parts:
        if re.match(r"^\d{4}\s*-\s*\d{4}$", p):
            valids.append(p.replace(" ", ""))
        else:
            logging.warning(f"YEARS ignorado por formato no válido: '{p}' (usa AAAA-AAAA)")
    return valids

def derive_mov_year(years_range: str, periodo: str) -> str:
    if not years_range or "-" not in years_range:
        return str(years_range or "")
    first_year, last_year = years_range.split("-", 1)
    p = (periodo or "").strip().lower()
    if p == "anual":
        return first_year
    if p in {"cumulative", "cum", "acumulado"}:
        return last_year
    return first_year

def format_placeholders(template: Optional[str], ctx: Dict[str, Any]) -> Optional[str]:
    if template is None:
        return None
    mixed = {
        "EMPRESA": ctx.get("EMPRESA"),
        "PERIODO": ctx.get("PERIODO"),
        "YEARS": ctx.get("YEARS"),
        "MOV_YEAR": ctx.get("MOV_YEAR"),
        "empresa": ctx.get("EMPRESA"),
        "periodo": ctx.get("PERIODO"),
        "years": ctx.get("YEARS"),
        "mov_year": ctx.get("MOV_YEAR"),
    }
    try:
        return template.format(**mixed)
    except KeyError as e:
        logging.warning(f"Placeholder faltante {e} en: {template}")
        return template

def sanitize_empresa_folder(empresa: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(empresa)).lower()

# ===================== IO helpers =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_reason_log(folder: str, fname: str, msg: str):
    ensure_dir(folder)
    path = os.path.join(folder, fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(f"📝 Log: {path}")

def list_geojsons(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".geojson"):
                out.append(os.path.join(root, f))
    return sorted(out)

# ===================== geo helpers =====================
def area_ha(geom: BaseGeometry) -> float:
    return float(geom.area / 10_000.0)

def assert_crs_exact(gdf: gpd.GeoDataFrame, expected_crs: str, label: str):
    if gdf.crs is None:
        raise RuntimeError(f"{label}: capa sin CRS definido.")
    crs_str = gdf.crs.to_string()
    if crs_str != expected_crs:
        raise RuntimeError(f"{label}: CRS={crs_str} difiere de esperado {expected_crs}. Corrige/reproyecta offline.")

def compute_intersection_area_ha_via_sindex(farm_geom: BaseGeometry, mask_gdf: gpd.GeoDataFrame) -> float:
    if mask_gdf is None or mask_gdf.empty:
        return 0.0
    try:
        sidx = mask_gdf.sindex
    except Exception:
        sidx = None
    cand_idx = list(sidx.intersection(farm_geom.bounds)) if sidx is not None else list(range(len(mask_gdf)))
    if not cand_idx:
        return 0.0
    mask_sub = mask_gdf.iloc[cand_idx]
    mask_sub = mask_sub[mask_sub.intersects(farm_geom)]
    if mask_sub.empty:
        return 0.0
    covered = None
    for mg in mask_sub.geometry:
        try:
            inter = farm_geom.intersection(mg)
        except Exception:
            try:
                inter = farm_geom.buffer(0).intersection(mg.buffer(0))
            except Exception:
                continue
        if inter.is_empty:
            continue
        covered = inter if covered is None else covered.union(inter)
    if covered is None or covered.is_empty:
        return 0.0
    return area_ha(covered)

def load_farms_geoms_reproject(folder_tpl: str, empresa: str, expected_crs: str) -> gpd.GeoDataFrame:
    base_ctx = {"EMPRESA": empresa, "PERIODO": "", "YEARS": "", "MOV_YEAR": ""}
    folder = format_placeholders(folder_tpl, base_ctx)
    if not os.path.isdir(folder):
        raise RuntimeError(f"No existe carpeta de fincas: {folder}")

    print(f"▶️  Listando GeoJSONs de fincas en: {folder}")
    t_list = time.perf_counter()
    files = list_geojsons(folder)
    print(f"⏱ Listado fincas: {time.perf_counter() - t_list:.2f}s – {len(files)} archivos")

    if not files:
        raise RuntimeError(f"No se encontraron GeoJSONs en: {folder}")

    frames = []
    t_read = time.perf_counter()
    for fp in tqdm(files, desc="Leyendo fincas", unit="archivo"):
        try:
            g = gpd.read_file(fp)
            if g.crs is None:
                raise RuntimeError(f"Finca {fp}: capa sin CRS; define su CRS para poder reproyectar.")
            if g.crs.to_string() != expected_crs:
                g = g.to_crs(expected_crs)
            if "geometry" not in g.columns or g.empty:
                continue
            fname = os.path.basename(fp)
            m = re.search(r"[_-](\d+)\.geojson$", fname, flags=re.IGNORECASE) or re.search(r"(\d+)\.geojson$", fname, flags=re.IGNORECASE)
            farm_id = str(m.group(1)) if m else (str(g.get("id").iloc[0]) if "id" in g.columns and len(g) == 1 else os.path.splitext(fname)[0])
            gg = g[["geometry"]].copy()
            gg["id"] = str(farm_id).strip()
            frames.append(gg[["id","geometry"]])
        except Exception as e:
            logging.warning(f"No se pudo leer finca {fp}: {e}")
    print(f"⏱ Lectura+reproyección fincas: {time.perf_counter() - t_read:.2f}s")

    if not frames:
        raise RuntimeError("No se pudo cargar ninguna finca.")
    farms = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=expected_crs)
    farms = farms[farms.geometry.notna() & ~farms.geometry.is_empty].copy()

    dup = farms["id"].duplicated().sum()
    print(f"ℹ️  Fincas: {len(farms)} features | IDs duplicados: {dup}")
    if dup > 0:
        t_diss = time.perf_counter()
        farms = farms.dissolve(by="id", as_index=False)
        print(f"⏱ Disolver por id: {time.perf_counter() - t_diss:.2f}s")
    print(f"✔️  Fincas listas: {len(farms)} ids únicos")
    return farms

def load_mask_gdf_strict(path: str, expected_crs: str, label: str) -> Optional[gpd.GeoDataFrame]:
    if not os.path.isfile(path):
        logging.warning(f"{label}: no existe vector: {path}")
        return None
    print(f"▶️  Cargando {label}: {path}")
    t0 = time.perf_counter()
    try:
        gdf = gpd.read_file(path)
        assert_crs_exact(gdf, expected_crs, f"{label} ({path})")
        if gdf.empty or "geometry" not in gdf.columns:
            logging.warning(f"{label}: vector vacío o sin geometría: {path}")
            return None
        print(f"⏱ {label} leído: {time.perf_counter() - t0:.2f}s | {len(gdf)} features")
        t_sidx = time.perf_counter()
        try:
            _ = gdf.sindex
            print(f"⏱ {label} sindex listo: {time.perf_counter() - t_sidx:.2f}s")
        except Exception:
            print(f"⚠ {label} sin sindex disponible (rtree/pygeos no instalado); fallback lineal")
        return gdf
    except Exception as e:
        logging.warning(f"{label}: no se pudo leer {path}: {e}")
        return None

# ===================== paths por fuente =====================
SOURCE_MAP = {"smbyc": "SMBYC", "atd": "ATD", "nad": "NAD"}

def alerts_dir_from_output_csv(output_csv_tpl: str, ctx: Dict[str, Any], source_tag: str, empresa: str) -> str:
    formatted = format_placeholders(output_csv_tpl, ctx)
    base_dir = os.path.dirname(formatted) or "."
    return os.path.join(base_dir, SOURCE_MAP.get(source_tag, source_tag.upper()), sanitize_empresa_folder(empresa))

def movement_dir_from_output_csv(movement_out_tpl: str, ctx: Dict[str, Any], source_tag: str, empresa: str) -> str:
    formatted = format_placeholders(movement_out_tpl, ctx)
    base_dir = os.path.dirname(formatted) or "."
    return os.path.join(base_dir, SOURCE_MAP.get(source_tag, source_tag.upper()), sanitize_empresa_folder(empresa))

def total_out_dir_from_output_csv(total_out_tpl: str, ctx: Dict[str, Any], source_tag: str, empresa: str) -> str:
    formatted = format_placeholders(total_out_tpl, ctx)
    base_dir = os.path.dirname(formatted) or "."
    return os.path.join(base_dir, SOURCE_MAP.get(source_tag, source_tag.upper()), sanitize_empresa_folder(empresa))

# ===================== escaneo rápido para IDs =====================
def scan_ids_needed(years_list: List[str], sources: List[str], empresa: str, periodo: str) -> Set[str]:
    print("▶️  Escaneando IDs en insumos direct/movement...")
    t0 = time.perf_counter()
    ids: Set[str] = set()
    for years in years_list:
        mov_year = derive_mov_year(years, periodo)
        ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
        for src in sources:
            a_dir = alerts_dir_from_output_csv(config["OUTPUT_CSV"], ctx, src, empresa)
            a_csv = os.path.join(a_dir, f"{src}_direct_alert_{empresa}_{years}.csv")
            if os.path.isfile(a_csv):
                try:
                    df = pd.read_csv(a_csv, dtype=str, usecols=["id"])
                    ids.update(df["id"].map(_norm_id))
                except Exception:
                    pass
            m_dir = movement_dir_from_output_csv(config["MOVEMENT_RISK_OUTPUT_CSV"], ctx, src, empresa)
            m_csv = os.path.join(m_dir, f"{src}_movement_alerts_{empresa}_{years}.csv")
            if os.path.isfile(m_csv):
                try:
                    dfm = pd.read_csv(m_csv, dtype=str, usecols=["id"])
                    ids.update(dfm["id"].map(_norm_id))
                except Exception:
                    pass
    ids.discard("")
    print(f"⏱ Escaneo IDs: {time.perf_counter() - t0:.2f}s | {len(ids)} ids encontrados")
    return ids

# ===================== caché de métricas espaciales =====================
def cache_dir_for_total(total_tpl: str, empresa: str, periodo: str) -> str:
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": "", "MOV_YEAR": ""}
    base = os.path.dirname(format_placeholders(total_tpl, ctx)) or "."
    cdir = os.path.join(base, "__cache__", sanitize_empresa_folder(empresa))
    ensure_dir(cdir)
    return cdir

def compute_metrics_cache(
    farms: gpd.GeoDataFrame,
    ids_needed: Set[str],
    frontier_gdf: Optional[gpd.GeoDataFrame],
    protected_gdf: Optional[gpd.GeoDataFrame],
) -> pd.DataFrame:
    farms_sel = farms[farms["id"].astype(str).isin(ids_needed)].copy()
    rows = []
    t0 = time.perf_counter()
    print(f"▶️  Calculando métricas espaciales para {len(farms_sel)} fincas (sin unary_union global)...")
    for _, r in tqdm(farms_sel.iterrows(), total=len(farms_sel), desc="Fincas", unit="finca"):
        fid = str(r["id"]).strip()
        geom = r.geometry
        total = area_ha(geom)
        in_ha = 0.0 if frontier_gdf is None else compute_intersection_area_ha_via_sindex(geom, frontier_gdf)
        out_ha = max(total - in_ha, 0.0)
        prot_ha = 0.0 if protected_gdf is None else compute_intersection_area_ha_via_sindex(geom, protected_gdf)
        rows.append({
            "id": fid,
            "farming_in_ha": round(in_ha, 4),
            "farming_in_prop": round(0.0 if total == 0 else in_ha / total, 6),
            "farming_out_ha": round(out_ha, 4),
            "farming_out_prop": round(0.0 if total == 0 else out_ha / total, 6),
            "protected_ha": round(prot_ha, 4),
            "protected_prop": round(0.0 if total == 0 else prot_ha / total, 6),
        })
    print(f"⏱ Cálculo métricas espaciales: {time.perf_counter() - t0:.2f}s")
    return pd.DataFrame(rows)

def load_or_build_cache(empresa: str, periodo: str,
                        farms: gpd.GeoDataFrame,
                        ids_needed: Set[str],
                        frontier_gdf: Optional[gpd.GeoDataFrame],
                        protected_gdf: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    cdir = cache_dir_for_total(config["TOTAL_RISK_OUTPUT_CSV"], empresa, periodo)
    cache_path = os.path.join(cdir, f"spatial_metrics_{empresa}.csv")
    if os.path.isfile(cache_path):
        print(f"▶️  Leyendo caché espacial: {cache_path}")
        t0 = time.perf_counter()
        try:
            df_cache = pd.read_csv(cache_path, dtype=str)
            have = set(df_cache["id"].astype(str))
            missing = ids_needed - have
            print(f"⏱ Caché leída en {time.perf_counter() - t0:.2f}s | IDs en caché: {len(have)} | Faltan: {len(missing)}")
            if not missing:
                return df_cache
            add_df = compute_metrics_cache(farms, missing, frontier_gdf, protected_gdf)
            merged = pd.concat([df_cache, add_df], ignore_index=True)
            merged.to_csv(cache_path, index=False, encoding="utf-8")
            return merged
        except Exception as e:
            print(f"⚠ Caché inválida, se reconstruirá ({e})")
    df_all = compute_metrics_cache(farms, ids_needed, frontier_gdf, protected_gdf)
    print(f"▶️  Guardando caché espacial en: {cache_path}")
    t1 = time.perf_counter()
    df_all.to_csv(cache_path, index=False, encoding="utf-8")
    print(f"⏱ Caché guardada en {time.perf_counter() - t1:.2f}s")
    return df_all

# ===================== MONGO: utilidades de mapeo (CON PRINTS) =====================
def build_mongo_maps(ids_needed: Set[str]) -> pd.DataFrame:
    """
    Mapea: id (CSV normalizado) -> farm._id -> farmpolygons._id
    Usa $elemMatch sobre ext_id para asegurar que source y ext_code
    provienen del MISMO elemento del array.
    """
    cols = ["id", "farm_id", "farm_poligons_id"]
    if not ids_needed:
        print("MONGO ▶ No hay ids_needed")
        return pd.DataFrame(columns=cols)

    # ids normalizados (strings tipo "396204")
    ids_str = [_norm_id(x) for x in ids_needed if _norm_id(x) != ""]
    ids_str = list(dict.fromkeys(ids_str))  # únicos y orden estable

    # misma lista pero como enteros (por si ext_code está numérico en Mongo)
    ids_int = []
    for s in ids_str:
        if s.isdigit():
            try:
                ids_int.append(int(s))
            except Exception:
                pass
    ids_any = list(dict.fromkeys(ids_str + ids_int))

    print(f"MONGO ▶ ids_str={len(ids_str)} | ids_int={len(ids_int)} (ejemplos str: {ids_str[:8]})")

    if MongoClient is None:
        logging.warning("pymongo no está instalado; columnas vacías.")
        print("MONGO ▶ pymongo no disponible, devolviendo columnas vacías.")
        return pd.DataFrame({"id": ids_str, "farm_id": "", "farm_poligons_id": ""})

    MONGO_URI      = str(config.get("MONGO_URI", "mongodb://localhost:27017")).strip()
    MONGO_DB       = str(config.get("MONGO_DB_NAME", "ganabosques")).strip()
    MONGO_SRC_NAME = str(config.get("MONGO_EXT_SOURCE", "SIT_CODE")).strip()
    FARMS_COL      = str(config.get("MONGO_FARMS_COLLECTION", "farm")).strip()
    POLYS_COL      = str(config.get("MONGO_FARMPOLYGONS_COLLECTION", "farmpolygons")).strip()

    print(f"MONGO ▶ Conectando a {MONGO_URI} / db='{MONGO_DB}' | farms='{FARMS_COL}' | polys='{POLYS_COL}' | source='{MONGO_SRC_NAME}'")
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        farms_col = db[FARMS_COL]
        polys_col = db[POLYS_COL]
    except Exception as e:
        logging.warning(f"No se pudo conectar a Mongo ({e}); columnas vacías.")
        print(f"MONGO ▶ ERROR de conexión: {e}")
        return pd.DataFrame({"id": ids_str, "farm_id": "", "farm_poligons_id": ""})

    # ------- 1) FARMS: $elemMatch sobre ext_id
    query = {"ext_id": {"$elemMatch": {"source": MONGO_SRC_NAME, "ext_code": {"$in": ids_any}}}}
    proj  = {"_id": 1, "ext_id": 1}
    print(f"MONGO ▶ Query farms: {query}")
    farm_docs = list(farms_col.find(query, proj))
    print(f"MONGO ▶ farms encontrados: {len(farm_docs)}")

    # ext_code(normalizado) -> farm._id
    farm_map: Dict[str, Any] = {}
    for doc in farm_docs:
        for ext in doc.get("ext_id", []):
            if ext.get("source") == MONGO_SRC_NAME:
                code = _norm_id(ext.get("ext_code"))
                if code in ids_str and code not in farm_map:
                    farm_map[code] = doc.get("_id")

    # Prints de diagnóstico de mapeo
    print(f"MONGO ▶ mapeos id->farm_id: {len(farm_map)} (ejemplos: {list(farm_map.items())[:5]})")

    farm_ids = [v for v in set(farm_map.values()) if v is not None]
    print(f"MONGO ▶ farm_ids únicos: {len(farm_ids)}")

    # ------- 2) FARMPOLYGONS: farm_id -> farmpolygons._id
    if farm_ids:
        polygon_docs = list(polys_col.find({"farm_id": {"$in": farm_ids}}, {"_id": 1, "farm_id": 1}))
    else:
        polygon_docs = []
    print(f"MONGO ▶ farmpolygons encontrados: {len(polygon_docs)}")

    # Si hay múltiples polígonos por farm_id, tomamos el primero (ajustable)
    polygon_map: Dict[Any, Any] = {}
    for d in polygon_docs:
        fid = d.get("farm_id")
        if fid is not None and fid not in polygon_map:
            polygon_map[fid] = d.get("_id")
    print(f"MONGO ▶ mapeos farm_id->farmpoligons_id: {len(polygon_map)} (ejemplos: {list(polygon_map.items())[:5]})")

    # ------- 3) DF final (+ prints de matched/unmatched)
    rows = []
    matched = 0
    unmatched_ids = []
    for sid in ids_str:
        f_id = farm_map.get(sid)
        if f_id:
            matched += 1
        else:
            unmatched_ids.append(sid)
        fp_id = polygon_map.get(f_id, "") if f_id else ""
        rows.append({
            "id": sid,
            "farm_id": str(f_id) if f_id else "",
            "farm_poligons_id": str(fp_id) if fp_id else ""
        })

    print(f"MONGO ▶ ids totales={len(ids_str)} | ids con farm_id={matched} | ids sin match={len(unmatched_ids)}")
    if unmatched_ids:
        print(f"MONGO ▶ ejemplos sin match: {unmatched_ids[:10]}")

    return pd.DataFrame(rows, columns=cols)

# ===================== proceso por fuente/año =====================
def process_year_source(empresa: str, periodo: str, years: str, mov_year: str,
                        source_tag: str,
                        metrics_cache: pd.DataFrame,
                        mongo_map_df: pd.DataFrame) -> Optional[str]:
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
    alerts_dir   = alerts_dir_from_output_csv(config["OUTPUT_CSV"], ctx, source_tag, empresa)
    movement_dir = movement_dir_from_output_csv(config["MOVEMENT_RISK_OUTPUT_CSV"], ctx, source_tag, empresa)

    prefix = source_tag.lower()
    direct_alert_csv   = os.path.join(alerts_dir,   f"{prefix}_direct_alert_{empresa}_{years}.csv")
    movement_alert_csv = os.path.join(movement_dir, f"{prefix}_movement_alerts_{empresa}_{years}.csv")

    total_dir = total_out_dir_from_output_csv(config["TOTAL_RISK_OUTPUT_CSV"], ctx, source_tag, empresa)
    ensure_dir(total_dir)
    total_csv = os.path.join(total_dir, f"{prefix}_total_risk_{empresa}_{years}.csv")
    log_name  = f"{prefix}_total_log_{empresa}_{years}.txt"

    # --- Direct risk requerido
    if not os.path.isfile(direct_alert_csv):
        write_reason_log(total_dir, log_name, f"[{years}] Falta alerta directa: {direct_alert_csv}")
        return None

    # Lee Direct
    t0 = time.perf_counter()
    df_direct = pd.read_csv(direct_alert_csv, dtype=str, low_memory=False)
    print(f"⏱ Lectura DIRECT ({source_tag.upper()} {years}): {time.perf_counter() - t0:.2f}s")

    if "id" not in df_direct.columns:
        write_reason_log(total_dir, log_name, f"[{years}] DIRECT sin columna 'id'.")
        return None

    # Normaliza id
    df_direct["id"] = df_direct["id"].map(_norm_id)

    # Asegura columnas directas solicitadas
    if "direct_alert" not in df_direct.columns:
        for c in ["intersect_deforestation","intersect_early_warnings","intersect_active_hotspots"]:
            if c in df_direct.columns:
                df_direct["direct_alert"] = df_direct[c]
                break
    if "direct_alert" not in df_direct.columns:
        df_direct["direct_alert"] = "False"

    for c in ["intersect_deforestation", "deforested_ha", "deforested_prop"]:
        if c not in df_direct.columns:
            df_direct[c] = "no data"

    # --- Movimiento (si falta, se simulan columnas con "no_info")
    if os.path.isfile(movement_alert_csv):
        t1 = time.perf_counter()
        df_move = pd.read_csv(movement_alert_csv, dtype=str, low_memory=False)
        print(f"⏱ Lectura MOVEMENT ({source_tag.upper()} {years}): {time.perf_counter() - t1:.2f}s")
        if "id" not in df_move.columns:
            write_reason_log(total_dir, log_name, f"[{years}] MOVEMENT sin columna 'id'. Se generará con 'no_info'.")
            df_move = None
        else:
            df_move["id"] = df_move["id"].map(_norm_id)
    else:
        write_reason_log(total_dir, log_name, f"[{years}] Falta movimiento: {movement_alert_csv}")
        df_move = None

    mov_cols = ["n_total_mov","n_in","n_out","n_indirect_in","n_indirect_out","indirect_alert_in","indirect_alert_out"]

    if df_move is None:
        df_move = pd.DataFrame({
            "id": df_direct["id"].astype(str),
            **{c: "no_info" for c in mov_cols}
        })
    else:
        for c in mov_cols:
            if c not in df_move.columns:
                df_move[c] = "no_info"

    # --- Merge direct + movement
    t2 = time.perf_counter()
    merged = pd.merge(
        df_direct[["id","direct_alert","intersect_deforestation","deforested_ha","deforested_prop"]],
        df_move[["id"] + mov_cols],
        on="id", how="outer", copy=False
    )
    print(f"⏱ Merge direct+movement: {time.perf_counter() - t2:.2f}s")

    # --- Merge con métricas espaciales
    t3 = time.perf_counter()
    final = pd.merge(merged, metrics_cache, on="id", how="left", copy=False)
    print(f"⏱ Merge con métricas espaciales: {time.perf_counter() - t3:.2f}s")

    # --- Merge con mapeo Mongo (farm_id y farm_poligons_id)
    if mongo_map_df is not None and not mongo_map_df.empty:
        t4 = time.perf_counter()
        final = pd.merge(final, mongo_map_df, on="id", how="left", copy=False)
        if "farm_id" not in final.columns:
            final["farm_id"] = ""
        if "farm_poligons_id" not in final.columns:
            final["farm_poligons_id"] = ""
        print(f"⏱ Merge con Mongo maps: {time.perf_counter() - t4:.2f}s")
        # Prints de verificación rápida
        print("MONGO ▶ ejemplo de filas con farm_id asignado:")
        print(final[final["farm_id"].astype(str) != ""].head(5).to_string(index=False))
        print("MONGO ▶ ejemplo de filas sin farm_id:")
        print(final[final["farm_id"].astype(str) == ""].head(5).to_string(index=False))
    else:
        final["farm_id"] = ""
        final["farm_poligons_id"] = ""
        print("MONGO ▶ mongo_map_df vacío; columnas farm_id/farm_poligons_id quedarán vacías.")

    # --- Guardar
    try:
        t5 = time.perf_counter()
        final.to_csv(total_csv, index=False, encoding="utf-8")
        print(f"⏱ Guardado salida: {time.perf_counter() - t5:.2f}s → {total_csv}")
        return total_csv
    except Exception as e:
        write_reason_log(total_dir, log_name, f"[{years}] No se pudo guardar salida: {e}")
        return None

# ===================== main =====================
def main():
    setup_logging()
    T0 = time.perf_counter()

    empresa = str(config["EMPRESA"]).strip()
    periodo = str(config["PERIODO"]).strip()
    years_cfg = str(config["YEARS"]).strip()
    years_list = parse_year_periods(years_cfg) or ([years_cfg] if years_cfg else [])
    crs = str(config["CRS_METROS"]).strip()
    sources = ["smbyc","atd","nad"]

    # 1) IDs necesarios
    ids_needed = scan_ids_needed(years_list, sources, empresa, periodo)
    if not ids_needed:
        print("⚠ No hay IDs en insumos (direct/movement). Nada que hacer.")
        print(f"⏱ Tiempo total: {time.perf_counter()-T0:.2f}s")
        return

    # 1.5) Mongo: construir mapeo una sola vez (CON PRINTS)
    print("▶️  Consultando Mongo para farm_id y farm_poligons_id...")
    mongo_map_df = build_mongo_maps(ids_needed)
    print(f"MONGO ▶ DF mapeo filas: {len(mongo_map_df)}")
    print(mongo_map_df.head(10).to_string(index=False))

    # 2) Fincas (reproyección solo aquí)
    print("▶️  Cargando fincas (reproyectando si es necesario)...")
    t_farms = time.perf_counter()
    farms = load_farms_geoms_reproject(config["FOLDER_GEOJSONS"], empresa, crs)
    print(f"⏱ Fincas listas en: {time.perf_counter() - t_farms:.2f}s")

    # 3) Máscaras
    print("▶️  Cargando Frontera y PNN (sin reproyección, sin unary_union)...")
    t_masks = time.perf_counter()
    frontier_gdf  = load_mask_gdf_strict(config["FARMING_FRONTIER_SHP"], crs, "Frontera Agrícola")
    protected_gdf = load_mask_gdf_strict(config["SHP_PROTECTED"], crs, "Áreas protegidas (PNN)")
    print(f"⏱ Máscaras listas en: {time.perf_counter() - t_masks:.2f}s")

    # 4) Caché de métricas espaciales
    print(f"▶️  Calculando/leyendo caché de métricas para {len(ids_needed)} ids...")
    t_cache = time.perf_counter()
    metrics_cache = load_or_build_cache(empresa, periodo, farms, ids_needed, frontier_gdf, protected_gdf)
    print(f"⏱ Caché espacial lista en: {time.perf_counter() - t_cache:.2f}s")

    # 5) Generación por YEARS × fuente
    produced: List[str] = []
    tasks = [(y, s) for y in years_list for s in sources]
    print(f"▶️  Generando salidas para {len(tasks)} tareas (YEARS × fuente)...")
    t_gen = time.perf_counter()
    for years, src in tqdm(tasks, desc="Generando total_risk", unit="tarea"):
        mov_year = derive_mov_year(years, periodo)
        out = process_year_source(empresa, periodo, years, mov_year, src, metrics_cache, mongo_map_df)
        if out:
            produced.append(out)
    print(f"⏱ Generación total_risk: {time.perf_counter() - t_gen:.2f}s")

    # 6) Resumen
    if produced:
        print("\n🎉 Archivos generados:")
        for p in produced:
            print("  -", p)
    else:
        print("\n⚠ No se generaron archivos. Revisa logs en las carpetas por fuente.")
    print(f"⏱ Tiempo total: {time.perf_counter()-T0:.2f}s")

if __name__ == "__main__":
    main()
