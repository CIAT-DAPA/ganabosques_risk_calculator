#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import gc
import time
import logging
import argparse
from typing import Dict, Any, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from tqdm import tqdm

from config import config

# ---------------- utilidades ----------------
def setup_logging():
    lvl = getattr(logging, config['LOG_LEVEL'].upper(), logging.WARNING)
    logging.basicConfig(filename=config['LOG_FILE'], level=lvl, format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

def parse_year_periods(years_raw: str) -> List[str]:
    if not years_raw:
        return []
    parts = [p.strip() for p in years_raw.split(",") if p.strip()]
    valids = []
    for p in parts:
        if re.match(r"^\d{4}\s*-\s*\d{4}$", p):
            valids.append(p.replace(" ", ""))
        else:
            logging.warning(f"YEARS ignorado por formato no válido: '{p}' (usa AAAA-AAAA)")
    return valids

def format_placeholders(template: Optional[str], ctx: Dict[str, Any]) -> Optional[str]:
    if not template:
        return template
    mixed = {
        "EMPRESA": ctx.get("EMPRESA"),
        "PERIODO": ctx.get("PERIODO"),
        "YEARS": ctx.get("YEARS"),
        "empresa": ctx.get("EMPRESA"),
        "periodo": ctx.get("PERIODO"),
        "years": ctx.get("YEARS"),
    }
    try:
        return template.format(**mixed)
    except KeyError as e:
        logging.warning(f"Placeholder faltante {e} en: {template}")
        return template

# ---------------- filtro por posiciones (1-based) ----------------
def apply_file_range(files: List[str], expr: str) -> List[str]:
    if not expr or not str(expr).strip():
        return files
    expr = expr.strip()
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    selected = []
    n = len(files)
    for tok in parts:
        m = re.match(r"^(\d+)\s*:\s*(\d+)$", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            a = max(1, min(a, n)); b = max(1, min(b, n))
            if a <= b:
                selected.extend(files[a-1:b])
        else:
            if tok.isdigit():
                if len(parts) == 1:
                    k = int(tok); a, b = 1, max(1, min(k, n))
                    selected.extend(files[a-1:b])
                else:
                    idx = int(tok) - 1
                    if 0 <= idx < n:
                        selected.append(files[idx])
            else:
                logging.warning(f"Token de FARM_FILE_RANGE ignorado: '{tok}'")
    seen = set()
    filtered = [f for f in selected if not (f in seen or seen.add(f))]
    if not filtered:
        logging.warning(f"FARM_FILE_RANGE='{expr}' no seleccionó archivos; se procesará todo.")
        return files
    print(f"📄 Filtro por posiciones FARM_FILE_RANGE='{expr}' → {len(filtered)} de {n} archivos.")
    return filtered

# ---------------- GIS helpers (solo lo necesario) ----------------
def ensure_raster_crs(path, expected_crs):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster no encontrado: {path}")
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError(f"{path} no tiene CRS definido.")
        if crs.to_string() != expected_crs:
            raise ValueError(f"{path} está en {crs.to_string()}, se esperaba {expected_crs}. Reproyecta con gdalwarp.")
        print(f"✔ Raster {os.path.basename(path)} CRS correcto: {crs.to_string()}")

def calculate_deforestation_metrics(geom, raster_path, deforest_value) -> Tuple[bool, float, float]:
    try:
        with rasterio.open(raster_path) as src:
            out_image, _ = mask(src, [mapping(geom)], crop=True, filled=False)
            arr = out_image[0]
            m = (arr == deforest_value)
            if np.count_nonzero(m) == 0:
                return False, 0.0, 0.0
            pixel_area = abs(src.transform.a * src.transform.e)
            defo_ha = np.count_nonzero(m) * pixel_area / 10_000.0
            prop = defo_ha / (geom.area / 10_000.0)
            return True, float(defo_ha), float(prop)
    except Exception as e:
        logging.warning(f"Error calculando deforestación: {e}")
        return False, 0.0, 0.0

# ---------------- carga y métricas de alertas/núcleos ----------------
VECTOR_EXTS = {".shp", ".geojson", ".gpkg"}
RASTER_EXTS = {".tif", ".tiff"}

def _list_files(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            out.append(os.path.join(root, f))
    return out

def _is_vector(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VECTOR_EXTS

def _is_raster(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RASTER_EXTS

def load_vector_from_paths(paths: List[str], expected_crs: str) -> Optional[gpd.GeoDataFrame]:
    geoms = []
    for p in paths:
        try:
            gdf = gpd.read_file(p)
            if gdf.empty or "geometry" not in gdf.columns:
                continue
            if gdf.crs is None:
                logging.warning(f"{p} no tiene CRS; se omite.")
                continue
            if gdf.crs.to_string() != expected_crs:
                gdf = gdf.to_crs(expected_crs)
            geoms.append(gdf[["geometry"]])
        except Exception as e:
            logging.warning(f"No se pudo leer vector {p}: {e}")
    if not geoms:
        return None
    merged = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True), crs=expected_crs)
    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()]
    return merged if not merged.empty else None

def load_bundle(folder_template: Optional[str], ctx: Dict[str, Any], expected_crs: str):
    summary = {"folder": None, "vec_files": 0, "ras_files": 0}
    if not folder_template:
        return None, [], summary
    folder = format_placeholders(folder_template, ctx)
    summary["folder"] = folder
    if not folder or not os.path.isdir(folder):
        logging.error(f"No se encontró carpeta: {folder}")
        return None, [], summary
    all_files = _list_files(folder)
    vec_paths = [p for p in all_files if _is_vector(p)]
    ras_paths = [p for p in all_files if _is_raster(p)]
    summary["vec_files"] = len(vec_paths)
    summary["ras_files"] = len(ras_paths)
    gdf_vec = load_vector_from_paths(vec_paths, expected_crs) if vec_paths else None
    return gdf_vec, ras_paths, summary

def _filter_rasters_same_crs(paths: List[str], expected_crs: str) -> List[str]:
    ok = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                crs = src.crs.to_string() if src.crs else None
                if crs == expected_crs:
                    ok.append(p)
                else:
                    logging.warning(f"Raster {os.path.basename(p)} ignorado por CRS {crs}≠{expected_crs}")
        except Exception as e:
            logging.warning(f"No se pudo abrir raster {p}: {e}")
    return ok

def area_from_vectors(geom, gdf_opt: Optional[gpd.GeoDataFrame]) -> float:
    if gdf_opt is None or gdf_opt.empty:
        return 0.0
    try:
        sidx = gdf_opt.sindex
        cand_idx = list(sidx.intersection(geom.bounds))
        if not cand_idx:
            return 0.0
        inters = gdf_opt.iloc[cand_idx].intersection(geom)
        inters = inters[~inters.is_empty]
        if inters.empty:
            return 0.0
        inter_union = unary_union(inters.to_list())
        return float(inter_union.area / 10_000.0)
    except Exception as e:
        logging.warning(f"Error área vectorial: {e}")
        return 0.0

def area_from_rasters(geom, raster_paths: List[str]) -> float:
    total_ha = 0.0
    for rp in raster_paths:
        try:
            with rasterio.open(rp) as src:
                out_image, _ = mask(src, [mapping(geom)], crop=True, filled=False)
                arr = out_image[0]
                nodata = src.nodata
                if nodata is None:
                    valid = np.isfinite(arr)
                else:
                    valid = np.isfinite(arr) & (arr != nodata)
                cnt = int(np.count_nonzero(valid))
                if cnt == 0:
                    continue
                pixel_area = abs(src.transform.a * src.transform.e)
                total_ha += cnt * pixel_area / 10_000.0
        except Exception as e:
            logging.warning(f"Error área raster {rp}: {e}")
    return float(total_ha)

def metrics_from_optional_sources(geom, gdf_vec: Optional[gpd.GeoDataFrame], ras_paths: List[str], expected_crs: str):
    total_ha_geom = float(geom.area / 10_000.0)
    ras_ok = _filter_rasters_same_crs(ras_paths, expected_crs) if ras_paths else []

    has_data = (gdf_vec is not None and not gdf_vec.empty) or (len(ras_ok) > 0)
    if not has_data:
        nan = float("nan")
        return False, nan, nan, nan

    ha_vec = area_from_vectors(geom, gdf_vec)
    ha_ras = area_from_rasters(geom, ras_ok)
    ha = ha_vec + ha_ras
    if ha <= 0.0 or total_ha_geom <= 0.0:
        return True, False, 0.0, 0.0
    prop = ha / total_ha_geom
    return True, True, float(ha), float(prop)

# ---------------- fila de finca (por opciones) ----------------
def process_row_option1(row, raster_deforest: str, deforest_value: int):
    farm_id = row["farm_id"]; geom = row["geometry"]
    inter_def, def_ha, def_prop = calculate_deforestation_metrics(geom, raster_deforest, deforest_value)
    def _r4(x): return round(float(x), 4)
    def _r6(x): return round(float(x), 6)
    return {
        "id": farm_id,
        "intersect_deforestation": bool(inter_def),
        "deforested_ha": _r4(def_ha),
        "deforested_prop": _r6(def_prop),
        "direct_alert": bool(inter_def),
    }

def process_row_option2(row, alerts_vec, alerts_ras, crs_expected):
    farm_id = row["farm_id"]; geom = row["geometry"]
    has, inter, ha, prop = metrics_from_optional_sources(geom, alerts_vec, alerts_ras, crs_expected)
    def _r4(x): return (np.nan if pd.isna(x) else round(float(x), 4))
    def _r6(x): return (np.nan if pd.isna(x) else round(float(x), 6))
    return {
        "id": farm_id,
        "intersect_early_warnings": (np.nan if not has else bool(inter)),
        "early_warnings_ha": (np.nan if not has else _r4(ha)),
        "early_warnings_prop": (np.nan if not has else _r6(prop)),
        "direct_alert": (False if not has else bool(inter)),
    }, has

def process_row_option3(row, nucleos_vec, nucleos_ras, crs_expected):
    farm_id = row["farm_id"]; geom = row["geometry"]
    has, inter, ha, prop = metrics_from_optional_sources(geom, nucleos_vec, nucleos_ras, crs_expected)
    def _r4(x): return (np.nan if pd.isna(x) else round(float(x), 4))
    def _r6(x): return (np.nan if pd.isna(x) else round(float(x), 6))
    return {
        "id": farm_id,
        "intersect_active_hotspots": (np.nan if not has else bool(inter)),
        "active_hotspots_ha": (np.nan if not has else _r4(ha)),
        "active_hotspots_prop": (np.nan if not has else _r6(prop)),
        "direct_alert": (False if not has else bool(inter)),
    }, has

def print_availability_option(name: str, found: bool, details: str = ""):
    yn = "Sí" if found else "No"
    print(f"  • {name}: {yn} {details}")

# ---------------- CLI ----------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Generación de alertas por opciones 1/2/3")
    p.add_argument("--options","-o", required=True,
                   help="Opciones a ejecutar: 1 (deforestación), 2 (alertas tempranas), 3 (núcleos). Ej: -o 1,2")
    p.add_argument("--source", choices=["smbyc","other"], default="smbyc",
                   help="Fuente para opción 1. 'smbyc' usa deforest_value=2. 'other' requiere --deforest-value.")
    p.add_argument("--deforest-value", type=int, default=None,
                   help="Valor de píxel de deforestación si --source other para opción 1.")
    return p

# ---------------- helpers de salida ----------------
def make_out_dir_and_paths(base_output_template: str, ctx: Dict[str,str], source_tag: str) -> Tuple[str,str]:
    """
    Estructura de salida:
      <carpeta-de-OUTPUT_CSV> / <OPCION> / <empresa-sanitizada> / <archivo.csv>

    donde <OPCION> ∈ {SMBYC, ATD, NAD} (u otro tag en mayúsculas).
    """
    formatted = format_placeholders(base_output_template, ctx)
    base_dir = os.path.dirname(formatted) or "."

    # Normaliza la opción (subcarpeta 1)
    option_map = {
        "smbyc": "SMBYC",
        "atd": "ATD",
        "nad": "NAD",
    }
    option_folder = option_map.get(str(source_tag).lower(), str(source_tag).upper())

    # Subcarpeta 2: empresa (normalizada)
    empresa_raw = ctx.get("EMPRESA", "empresa")
    empresa_folder = re.sub(r"[^A-Za-z0-9._-]+", "_", str(empresa_raw)).lower()

    out_dir = os.path.join(base_dir, option_folder, empresa_folder)
    os.makedirs(out_dir, exist_ok=True)

    years = ctx["YEARS"]; empresa = ctx["EMPRESA"]
    out_csv = os.path.join(out_dir, f"{source_tag}_direct_alert_{empresa}_{years}.csv")
    return out_dir, out_csv

def write_reason_log(out_dir: str, fname: str, msg: str):
    path = os.path.join(out_dir, fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(f"📝 Se escribió log: {path}")

# ---------------- main ----------------
def main():
    setup_logging()
    ap = build_arg_parser()
    args = ap.parse_args()

    # Parse opciones
    chosen: Set[int] = set()
    for tok in args.options.split(","):
        tok = tok.strip()
        if tok in {"1", "2", "3"}:
            chosen.add(int(tok))
    if not chosen:
        raise SystemExit("Debes elegir al menos una opción con -o 1,2,3")

    empresa = config['EMPRESA']
    periodo = config['PERIODO']
    crs = config['CRS_METROS']
    batch_size = config['BATCH_SIZE']

    # Opción 1: configurar deforest_value según fuente y tag de nombre
    if 1 in chosen:
        if args.source == "smbyc":
            defo_val = 2
            opt1_tag = "smbyc"
        else:
            if args.deforest_value is None:
                raise SystemExit("Para --source other debes indicar --deforest-value N")
            defo_val = int(args.deforest_value)
            opt1_tag = "otra"
        print(f"🟩 Opción 1 activa. Fuente: {args.source} | deforest_value={defo_val}")
    else:
        defo_val = None
        opt1_tag = None

    years_list = parse_year_periods(config['YEARS'])
    if not years_list:
        years_list = [str(config['YEARS']).strip()]
        logging.info(f"YEARS único detectado: {years_list[0]}")

    # Plantillas/insumos
    folder_geojsons_tpl = config['FOLDER_GEOJSONS']
    alerts_dir_tpl = config.get('ALERTAS_DIR', "")
    nucleos_dir_tpl = config.get('NUCLEOS_DIR', "")
    output_tpl = config['OUTPUT_CSV']  # solo usamos su carpeta

    ctx_base = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": ""}

    # GeoJSONs de fincas
    folder_geojsons = format_placeholders(folder_geojsons_tpl, ctx_base)
    files = sorted([f for f in os.listdir(folder_geojsons) if f.lower().endswith(".geojson")])
    if not files:
        raise RuntimeError(f"No se encontraron GeoJSONs en: {folder_geojsons}")
    print(f"⚙ FARM_FILE_RANGE='{config.get('FARM_FILE_RANGE','')}'")
    files = apply_file_range(files, config.get("FARM_FILE_RANGE", ""))

    start_all = time.time()
    for years in years_list:
        print(f"\n🟪 Periodo: {years} — empresa: {empresa} — etiqueta periodo: {periodo}")
        ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years}

        # Opción 1: raster de deforestación
        if 1 in chosen:
            raster_deforest = format_placeholders(config['RASTER_DEFOREST'], ctx)
            try:
                ensure_raster_crs(raster_deforest, crs)
                opt1_available = True
                print_availability_option("Deforestación (raster)", True, f"→ {os.path.basename(raster_deforest)}")
            except Exception as e:
                logging.error(f"[{years}] Opción 1: {e}")
                print("No se encontró/validó el raster de deforestación; no se generarán resultados para la opción 1.")
                opt1_available = False
        else:
            opt1_available = False

        # Opción 2: alertas tempranas
        if 2 in chosen:
            alerts_vec, alerts_ras, alerts_sum = load_bundle(alerts_dir_tpl, ctx, crs)
            has2 = (alerts_vec is not None and not alerts_vec.empty) or (len(alerts_ras) > 0)
            print_availability_option(
                "Alertas tempranas (carpeta)",
                has2,
                f"| vectores={alerts_sum.get('vec_files',0)} rasters={alerts_sum.get('ras_files',0)}"
            )
            if not has2:
                logging.error(f"[{years}] Opción 2: no se encontró el archivo/carpeta de alertas tempranas. No se generarán resultados.")
                # Log específico en subcarpeta ATD/empresa
                opt2_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "atd")
                write_reason_log(opt2_dir, f"atd_log_{empresa}_{years}.txt",
                                 f"[{years}] Sin datos de alertas tempranas (no se generó atd_direct_alert_{empresa}_{years}.csv)")
            opt2_available = has2
        else:
            alerts_vec = None; alerts_ras = []
            opt2_available = False

        # Opción 3: núcleos activos
        if 3 in chosen:
            nucleos_vec, nucleos_ras, nucleos_sum = load_bundle(nucleos_dir_tpl, ctx, crs)
            has3 = (nucleos_vec is not None and not nucleos_vec.empty) or (len(nucleos_ras) > 0)
            print_availability_option(
                "Núcleos activos (carpeta)",
                has3,
                f"| vectores={nucleos_sum.get('vec_files',0)} rasters={nucleos_sum.get('ras_files',0)}"
            )
            if not has3:
                logging.error(f"[{years}] Opción 3: no se encontró el archivo/carpeta de núcleos activos. No se generarán resultados.")
                # Log específico en subcarpeta NAD/empresa
                opt3_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "nad")
                write_reason_log(opt3_dir, f"nad_log_{empresa}_{years}.txt",
                                 f"[{years}] Sin datos de núcleos activos (no se generó nad_direct_alert_{empresa}_{years}.csv)")
            opt3_available = has3
        else:
            nucleos_vec = None; nucleos_ras = []
            opt3_available = False

        # Si ninguna opción está disponible para este periodo, saltar
        if not any([opt1_available, opt2_available, opt3_available]):
            print("⚠ Ninguna opción disponible para este periodo. Se omite generación de resultados.")
            continue

        # Prepara rutas de salida por opción
        if opt1_available:
            _, out_csv_o1 = make_out_dir_and_paths(output_tpl, ctx, opt1_tag)
            if os.path.exists(out_csv_o1): os.remove(out_csv_o1)
        if opt2_available:
            _, out_csv_o2 = make_out_dir_and_paths(output_tpl, ctx, "atd")
            if os.path.exists(out_csv_o2): os.remove(out_csv_o2)
        if opt3_available:
            _, out_csv_o3 = make_out_dir_and_paths(output_tpl, ctx, "nad")
            if os.path.exists(out_csv_o3): os.remove(out_csv_o3)

        # Procesamiento por lotes
        start = time.time()
        for i in tqdm(range(0, len(files), batch_size), desc=f"Procesando lotes ({years})", unit="lote"):
            batch = files[i:i+batch_size]
            gdf_list: List[gpd.GeoDataFrame] = []
            for file in batch:
                try:
                    gdf = gpd.read_file(os.path.join(folder_geojsons, file))
                    if gdf.crs is None or gdf.crs.to_string() != crs:
                        gdf = gdf.to_crs(crs)
                    m = re.search(r"[_-](\d+)\.geojson$", file, re.IGNORECASE) or re.search(r"(\d+)\.geojson$", file, re.IGNORECASE)
                    farm_id = int(m.group(1)) if m else os.path.splitext(file)[0]
                    gdf["farm_id"] = farm_id
                    gdf_list.append(gdf[["farm_id", "geometry"]])
                except Exception as e:
                    logging.warning(f"[{years}] Error leyendo {file}: {e}")

            if not gdf_list:
                continue

            merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

            # Acumuladores por opción
            results1 = [] if opt1_available else None
            results2 = [] if opt2_available else None
            results3 = [] if opt3_available else None

            for _, r in tqdm(merged.iterrows(), total=len(merged), desc="  Fincas en lote", unit="finca", leave=False):
                if opt1_available:
                    try:
                        o1 = process_row_option1(r, raster_deforest, defo_val)
                        results1.append(o1)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 1 error finca {r.get('farm_id')}: {e}")

                if opt2_available:
                    try:
                        o2, has = process_row_option2(r, alerts_vec, alerts_ras, crs)
                        results2.append(o2)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 2 error finca {r.get('farm_id')}: {e}")

                if opt3_available:
                    try:
                        o3, has = process_row_option3(r, nucleos_vec, nucleos_ras, crs)
                        results3.append(o3)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 3 error finca {r.get('farm_id')}: {e}")

            # Escritura por opción (por lotes, append)
            if opt1_available and results1:
                df1 = pd.DataFrame(results1)[["id","intersect_deforestation","deforested_ha","deforested_prop","direct_alert"]]
                df1.to_csv(out_csv_o1, mode='a', index=False, header=not os.path.exists(out_csv_o1))

            if opt2_available and results2:
                df2 = pd.DataFrame(results2)[["id","intersect_early_warnings","early_warnings_ha","early_warnings_prop","direct_alert"]]
                df2.to_csv(out_csv_o2, mode='a', index=False, header=not os.path.exists(out_csv_o2))

            if opt3_available and results3:
                df3 = pd.DataFrame(results3)[["id","intersect_active_hotspots","active_hotspots_ha","active_hotspots_prop","direct_alert"]]
                df3.to_csv(out_csv_o3, mode='a', index=False, header=not os.path.exists(out_csv_o3))

            del gdf_list, merged, results1, results2, results3
            gc.collect()

        # Mensajes finales por opción
        if opt1_available:
            print(f"✅ Opción 1 ({opt1_tag}) completada → {out_csv_o1}")
        if opt2_available:
            if os.path.exists(out_csv_o2):
                print(f"✅ Opción 2 (ATD) completada → {out_csv_o2}")
            else:
                print(f"⚠ Opción 2: no se generó CSV. Revisa el log en {out_dir}")
        if opt3_available:
            if os.path.exists(out_csv_o3):
                print(f"✅ Opción 3 (NAD) completada → {out_csv_o3}")
            else:
                print(f"⚠ Opción 3: no se generó CSV. Revisa el log en {out_dir}")

        print(f"⏱ Tiempo {years}: {time.time()-start:.2f} s")

    print(f"\n🎉 Periodos completados: {', '.join(years_list)}")
    print(f"⏱ Tiempo total: {time.time()-start_all:.2f} s")

if __name__ == "__main__":
    main()
