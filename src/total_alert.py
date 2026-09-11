#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from config import config
from ganabosques_risk_package.total_risk import total_risk as pkg_total_risk
from utils import area_ha, compute_intersection_area_ha_via_sindex, parse_year_periods, normalize_farm_id, setup_logging

# --- ORM (preferido) o fallback sin conexión
try:
    from ganabosques_orm.collections.farm import Farm
    from ganabosques_orm.collections.farmpolygons import FarmPolygons
    HAS_ORM = True
except ImportError:
    HAS_ORM = False
    Farm = None
    FarmPolygons = None

# ===================== logging =====================

# ===================== placeholders =====================

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

def assert_crs_exact(gdf: gpd.GeoDataFrame, expected_crs: str, label: str):
    if gdf.crs is None:
        raise RuntimeError(f"{label}: capa sin CRS definido.")
    crs_str = gdf.crs.to_string()
    if crs_str != expected_crs:
        raise RuntimeError(f"{label}: CRS={crs_str} difiere de esperado {expected_crs}. Corrige/reproyecta offline.")

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
                    ids.update(df["id"].map(normalize_farm_id))
                except Exception:
                    pass
            m_dir = movement_dir_from_output_csv(config["MOVEMENT_RISK_OUTPUT_CSV"], ctx, src, empresa)
            m_csv = os.path.join(m_dir, f"{src}_movement_alerts_{empresa}_{years}.csv")
            if os.path.isfile(m_csv):
                try:
                    dfm = pd.read_csv(m_csv, dtype=str, usecols=["id"])
                    ids.update(dfm["id"].map(normalize_farm_id))
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
    Mapea: id (sitcode CSV) -> farm._id -> farmpolygons._id
    Usa el ORM de ganabosques para consultar MongoDB.
    
    Esta función es un fallback cuando no se pasa mongo_map_df desde DataManager.
    """
    cols = ["id", "farm_id", "farm_poligons_id"]
    if not ids_needed:
        print("ORM ▶ No hay ids_needed")
        return pd.DataFrame(columns=cols)

    # ids normalizados (strings tipo "396204")
    ids_str = [normalize_farm_id(x) for x in ids_needed if normalize_farm_id(x) != ""]
    ids_str = list(dict.fromkeys(ids_str))  # únicos y orden estable

    print(f"ORM ▶ IDs a buscar: {len(ids_str)} (ejemplos: {ids_str[:8]})") 

    if not HAS_ORM:
        logging.warning("ganabosques_orm no está instalado; columnas vacías.")
        print("ORM ▶ ganabosques_orm no disponible, devolviendo columnas vacías.")
        return pd.DataFrame({"id": ids_str, "farm_id": "", "farm_poligons_id": ""})

    print("ORM ▶ Consultando farms por SIT_CODE y GEOFARMER_ID...")
    
    try:
        # ------- 1) FARMS: buscar por ext_id.source=SIT_CODE o GEOFARMER_ID
        # Primero intentar SIT_CODE (livestock)
        farms_sit = Farm.objects(
            ext_id__source="SIT_CODE",
            ext_id__ext_code__in=ids_str
        ).only('id', 'ext_id')
        
        # Construir mapeo code -> farm._id
        farm_map: Dict[str, str] = {}
        farm_ids = []
        
        for farm in farms_sit:
            farm_id = str(farm.id)
            farm_ids.append(farm.id)
            
            if farm.ext_id:
                for ext in farm.ext_id:
                    try:
                        source_str = ext.source.value if hasattr(ext.source, 'value') else str(ext.source)
                        if source_str == "SIT_CODE":
                            code = normalize_farm_id(str(ext.ext_code))
                            if code in ids_str and code not in farm_map:
                                farm_map[code] = farm_id
                    except Exception:
                        continue
        
        # Buscar IDs no encontrados por GEOFARMER_ID (cacao)
        missing_ids = [sid for sid in ids_str if sid not in farm_map]
        if missing_ids:
            # Los ext_code en BD pueden tener prefijo FARM_ID_ que normalize_farm_id quita,
            # así que buscamos con ambas variantes
            search_ids = list(set(missing_ids + [f"FARM_ID_{sid}" for sid in missing_ids]))
            farms_geo = Farm.objects(
                ext_id__source="GEOFARMER_ID",
                ext_id__ext_code__in=search_ids
            ).only('id', 'ext_id')
            
            for farm in farms_geo:
                farm_id = str(farm.id)
                if farm.id not in farm_ids:
                    farm_ids.append(farm.id)
                
                if farm.ext_id:
                    for ext in farm.ext_id:
                        try:
                            source_str = ext.source.value if hasattr(ext.source, 'value') else str(ext.source)
                            if source_str == "GEOFARMER_ID":
                                code = normalize_farm_id(str(ext.ext_code))
                                if code in ids_str and code not in farm_map:
                                    farm_map[code] = farm_id
                        except Exception:
                            continue
        
        print(f"ORM ▶ farms encontrados: {len(farm_ids)}")
        print(f"ORM ▶ mapeos code->farm_id: {len(farm_map)} (ejemplos: {list(farm_map.items())[:5]})")
        
        # ------- 2) FARMPOLYGONS: farm_id -> farmpolygons._id
        polygon_map: Dict[str, str] = {}
        
        if farm_ids:
            polygons = FarmPolygons.objects(farm_id__in=farm_ids).only('id', 'farm_id')
            
            for poly in polygons:
                farm_oid = poly.farm_id.id if hasattr(poly.farm_id, 'id') else poly.farm_id
                farm_id_str = str(farm_oid)
                if farm_id_str not in polygon_map:
                    polygon_map[farm_id_str] = str(poly.id)
            
            print(f"ORM ▶ farmpolygons encontrados: {len(polygon_map)}")
        else:
            print("ORM ▶ No hay farm_ids para buscar polygons")
        
        # ------- 3) Construir DataFrame final
        rows = []
        matched = 0
        unmatched_ids = []
        
        for sid in ids_str:
            f_id = farm_map.get(sid, "")
            if f_id:
                matched += 1
                fp_id = polygon_map.get(f_id, "")
            else:
                unmatched_ids.append(sid)
                fp_id = ""
            
            rows.append({
                "id": sid,
                "farm_id": f_id,
                "farm_poligons_id": fp_id
            })
        
        print(f"ORM ▶ ids totales={len(ids_str)} | con farm_id={matched} | sin match={len(unmatched_ids)}")
        if unmatched_ids:
            print(f"ORM ▶ ejemplos sin match: {unmatched_ids[:10]}")
        
        return pd.DataFrame(rows, columns=cols)
        
    except Exception as e:
        logging.error(f"Error consultando MongoDB via ORM: {e}")
        print(f"ORM ▶ ERROR: {e}")
        return pd.DataFrame({"id": ids_str, "farm_id": "", "farm_poligons_id": ""})

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
    df_direct["id"] = df_direct["id"].map(normalize_farm_id)

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
            df_move["id"] = df_move["id"].map(normalize_farm_id)
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

        # --- Fallback usando diccionario  ---
        if "GEOFARMER_ID" in mongo_map_df.columns:
            geo_map = (
                mongo_map_df[["GEOFARMER_ID", "farm_id", "farm_poligons_id"]]
                .dropna(subset=["GEOFARMER_ID"])
            )

            geo_map["GEOFARMER_ID"] = geo_map["GEOFARMER_ID"].astype(str).str.strip()
            geo_map = geo_map[geo_map["GEOFARMER_ID"] != ""].drop_duplicates("GEOFARMER_ID")

            # Crear diccionarios para lookup rápido
            farm_id_map = dict(zip(geo_map["GEOFARMER_ID"], geo_map["farm_id"]))
            poly_map = dict(zip(geo_map["GEOFARMER_ID"], geo_map["farm_poligons_id"]))

            # Rellenar SOLO donde falte
            mask = final["farm_id"] == ""

            final.loc[mask, "farm_id"] = final.loc[mask, "id"].map(farm_id_map).fillna("")
            final.loc[mask, "farm_poligons_id"] = final.loc[mask, "id"].map(poly_map).fillna("")

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


# ===================== NUEVA API CALLABLE =====================
def calculate_total_risk(
    source: str,
    period_type: str,
    periods: List[str],
    workspace_dir: str,
    use_precalculated_metrics: bool = True,
    mongo_map_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Calcula el riesgo total consolidando alertas directas, indirectas y métricas espaciales.
    
    Esta función es la nueva API que puede llamarse desde main.py con parámetros explícitos,
    en lugar de depender de variables de entorno.
    
    Args:
        source: Fuente de deforestación ('smbyc')
        period_type: Tipo de período ('nad', 'atd', 'annual', 'cumulative')
        periods: Lista de períodos a procesar (ej: ['201701', '201702'])
        workspace_dir: Directorio base del workspace (ej: 'D:/data/alertas')
        use_precalculated_metrics: Si True, usa spatial_metrics.csv precalculado
        mongo_map_df: DataFrame precargado con mapeo id->farm_id->farm_polygon_id (opcional).
                      Si se proporciona, evita consulta adicional a MongoDB.
        
    Returns:
        Dict con estadísticas: {
            'success': bool,
            'periods_processed': int,
            'files_generated': List[str],
            'execution_time': float
        }
    """
    
    setup_logging()
    T0 = time.perf_counter()
    
    workspace = Path(workspace_dir)
    
    # Nueva estructura: results/{source}/{period_type}/{stage}/
    results_base = workspace / "results"
    direct_alerts_dir = results_base / source / period_type / "direct_alerts"
    indirect_alerts_dir = results_base / source / period_type / "indirect_alerts"
    # spatial_metrics.py guarda en workspace/metrics/ no en results/spatial_metrics/
    metrics_file = workspace / "metrics" / "spatial_metrics.csv"
    output_dir = results_base / source / period_type / "total_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar métricas precalculadas si existen
    metrics_df = None
    if use_precalculated_metrics and metrics_file.exists():
        print(f"📂 Cargando métricas precalculadas: {metrics_file}")
        try:
            metrics_df = pd.read_csv(metrics_file, dtype=str)
            metrics_df["id"] = metrics_df["id"].apply(normalize_farm_id)
            print(f"✅ Métricas cargadas: {len(metrics_df)} farms")
        except Exception as e:
            print(f"⚠ Error cargando métricas: {e}")
            metrics_df = None
    
    if metrics_df is None:
        print("⚠ No hay métricas precalculadas. Ejecuta --metrics primero para mejor rendimiento.")
        # Crear DataFrame vacío con columnas esperadas
        metrics_df = pd.DataFrame(columns=[
            'id', 'farming_in_ha', 'farming_in_prop', 
            'farming_out_ha', 'farming_out_prop',
            'protected_ha', 'protected_prop'
        ])
    
    # Construir mapeo MongoDB
    print("📊 Construyendo mapeo MongoDB (farm_id, farm_poligons_id)...")
    
    # Recopilar todos los IDs de los CSVs de alertas
    all_ids = set()
    
    for period in periods:
        print(f"🔍 Escaneando IDs en alertas para período: {period}")
        # Buscar alertas directas
        direct_csv = direct_alerts_dir / f"{source}_direct_alert_{period_type}_{period}.csv"
        if direct_csv.exists():
            try:
                df = pd.read_csv(direct_csv, dtype=str, usecols=['id'])
                all_ids.update(df['id'].apply(normalize_farm_id))
            except Exception as e:
                logging.warning(f"Error leyendo {direct_csv}: {e}")
        else:
            print(f"⚠ No existe alerta directa: {direct_csv}. Se generará con 'no_info'.")
        
        # Buscar alertas indirectas
        indirect_csv = indirect_alerts_dir / f"{source}_indirect_alert_{period_type}_{period}.csv"
        if indirect_csv.exists():
            try:
                df = pd.read_csv(indirect_csv, dtype=str, usecols=['id'])
                all_ids.update(df['id'].apply(normalize_farm_id))
            except Exception as e:
                logging.warning(f"Error leyendo {indirect_csv}: {e}")
        else:
            print(f"⚠ No existe alerta indirecta: {indirect_csv}. Se generará con 'no_info'.")
    
    all_ids.discard("")
    print(f"✅ IDs encontrados: {len(all_ids)}")
    
    # Obtener mapeo de MongoDB (usar precargado si disponible)
    if mongo_map_df is not None and not mongo_map_df.empty:
        # Usar mapeo precargado desde DataManager (evita consulta adicional a MongoDB)
        print(f"📦 Usando mapeo MongoDB precargado desde DataManager ({len(mongo_map_df)} farms)")
        # Filtrar solo los IDs que necesitamos
        mongo_map_df = mongo_map_df[mongo_map_df['id'].isin(all_ids) | mongo_map_df['GEOFARMER_ID'].isin(all_ids)].copy()
        print(f"   • IDs filtrados: {len(mongo_map_df)}")
    else:
        # Fallback: construir mapeo consultando MongoDB directamente
        print("📊 Construyendo mapeo MongoDB (consulta directa)...")
        mongo_map_df = build_mongo_maps(all_ids) if all_ids else pd.DataFrame(columns=['id', 'farm_id', 'farm_poligons_id', 'GEOFARMER_ID'])
    
    # Procesar cada período
    files_generated = []

    print("folder direct alerts:", direct_alerts_dir)
    print("folder indirect alerts:", indirect_alerts_dir)
    
    
    for period in tqdm(periods, desc="Consolidando riesgo total"):
        # Cargar alertas directas
        direct_csv = direct_alerts_dir / f"{source}_direct_alert_{period_type}_{period}.csv"
        
        if not direct_csv.exists():
            logging.warning(f"No existe alerta directa: {direct_csv}")
            continue
        
        df_direct = pd.read_csv(direct_csv, dtype=str, low_memory=False)
        if 'id' not in df_direct.columns:
            logging.warning(f"CSV sin columna 'id': {direct_csv}")
            continue
        
        df_direct['id'] = df_direct['id'].apply(normalize_farm_id)
        
        # Asegurar columna direct_alert
        if 'direct_alert' not in df_direct.columns:
            if 'intersect_deforestation' in df_direct.columns:
                df_direct['direct_alert'] = df_direct['intersect_deforestation']
            else:
                df_direct['direct_alert'] = 'False'
        
        # Cargar alertas indirectas (opcional)
        indirect_csv = indirect_alerts_dir / f"{source}_indirect_alert_{period_type}_{period}.csv"
        
        mov_cols = ['n_total_mov', 'n_in', 'n_out', 'n_indirect_in', 'n_indirect_out', 
                   'indirect_alert_in', 'indirect_alert_out']
        
        if indirect_csv.exists():
            df_indirect = pd.read_csv(indirect_csv, dtype=str, low_memory=False)
            if 'id' in df_indirect.columns:
                df_indirect['id'] = df_indirect['id'].apply(normalize_farm_id)
            else:
                df_indirect = pd.DataFrame({'id': df_direct['id'], **{c: 'no_info' for c in mov_cols}})
        else:
            print(f"⚠ No existe alerta indirecta: {indirect_csv}. Se generará con 'no_info'.")
            df_indirect = pd.DataFrame({'id': df_direct['id'], **{c: 'no_info' for c in mov_cols}})
        
        # Asegurar columnas de movimiento
        for c in mov_cols:
            if c not in df_indirect.columns:
                df_indirect[c] = 'no_info'
        
        # Usar paquete ganabosques_risk_package para consolidar riesgo total
        merged = pkg_total_risk(
            direct_df=df_direct,
            indirect_df=df_indirect,
            metrics_df=metrics_df if not metrics_df.empty else None,
            id_column='id',
            show_progress=True,
        )
        
        # Merge con mapeo MongoDB
        if not mongo_map_df.empty:
            # --- Merge principal por id ---
            merged = pd.merge(
                merged,
                mongo_map_df[["id", "farm_id", "farm_poligons_id"]],
                on="id",
                how="left"
            )

            # Asegurar columnas
            for col in ["farm_id", "farm_poligons_id"]:
                if col not in merged.columns:
                    merged[col] = ""

            # --- Fallback: usar GEOFARMER_ID cuando no hubo match ---
            if "GEOFARMER_ID" in mongo_map_df.columns:
                geo_map = mongo_map_df[["GEOFARMER_ID", "farm_id", "farm_poligons_id"]].copy()
                geo_map["GEOFARMER_ID"] = geo_map["GEOFARMER_ID"].astype(str).str.strip()
                geo_map = geo_map.drop_duplicates("GEOFARMER_ID")

                # Crear diccionarios
                farm_map = dict(zip(geo_map["GEOFARMER_ID"], geo_map["farm_id"]))
                poly_map = dict(zip(geo_map["GEOFARMER_ID"], geo_map["farm_poligons_id"]))

                # Normalizar id
                merged["id"] = merged["id"].astype(str).str.strip()

                # Rellenar SOLO donde falta
                mask = merged["farm_id"].isna() | (merged["farm_id"] == "")

                merged.loc[mask, "farm_id"] = merged.loc[mask, "id"].map(farm_map)
                merged.loc[mask, "farm_poligons_id"] = merged.loc[mask, "id"].map(poly_map)

                # Limpiar NaN finales
                merged["farm_id"] = merged["farm_id"].fillna("")
                merged["farm_poligons_id"] = merged["farm_poligons_id"].fillna("")
        else:
            merged["farm_id"] = ""
            merged["farm_poligons_id"] = ""
        # Guardar resultado
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"{source}_total_risk_{period_type}_{period}.csv"
        
        merged.to_csv(output_csv, index=False, encoding='utf-8')
        files_generated.append(str(output_csv))
        print(f"✅ Generado: {output_csv.name}")
    
    execution_time = time.perf_counter() - T0
    
    return {
        'success': len(files_generated) > 0,
        'periods_processed': len(files_generated),
        'files_generated': files_generated,
        'execution_time': execution_time
    }
