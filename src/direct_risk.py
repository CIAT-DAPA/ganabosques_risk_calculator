#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import gc
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.spatial import KDTree
from tqdm import tqdm

from config import config

# -------- utilidades --------
def setup_logging():
    lvl = getattr(logging, config['LOG_LEVEL'].upper(), logging.WARNING)
    logging.basicConfig(filename=config['LOG_FILE'], level=lvl, format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

def parse_year_periods(years_raw: str) -> List[str]:
    """
    Acepta "2023-2024 , 2010-2012,2018-2019" -> ["2023-2024","2010-2012","2018-2019"]
    """
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

def format_placeholders(template: str, ctx: Dict[str, Any]) -> str:
    """
    Rellena placeholders en mayúsculas y minúsculas.
    Permite {EMPRESA}/{empresa}, {PERIODO}/{periodo}, {YEARS}/{years}.
    """
    if template is None:
        return None
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

# -------- filtro por posiciones (1-based) --------
def apply_file_range(files: List[str], expr: str) -> List[str]:
    """
    FARM_FILE_RANGE admite:
      - '1:50'            -> del 1 al 50 (inclusive)
      - '10'              -> si es único token, equivale a '1:10'
      - '5, 12, 20'       -> índices sueltos
      - '1:10, 25, 40:45' -> combinado (rangos + sueltos)
    Índices 1-based. Si no selecciona nada, se deja la lista original.
    """
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
            a = max(1, min(a, n))
            b = max(1, min(b, n))
            if a <= b:
                selected.extend(files[a-1:b])  # 1-based → 0-based
        else:
            if tok.isdigit():
                if len(parts) == 1:
                    k = int(tok)
                    a, b = 1, max(1, min(k, n))
                    selected.extend(files[a-1:b])
                else:
                    idx = int(tok) - 1
                    if 0 <= idx < n:
                        selected.append(files[idx])
            else:
                logging.warning(f"Token de FARM_FILE_RANGE ignorado: '{tok}'")

    # Quitar duplicados preservando orden
    seen = set()
    filtered = [f for f in selected if not (f in seen or seen.add(f))]

    if not filtered:
        logging.warning(f"FARM_FILE_RANGE='{expr}' no seleccionó archivos; se procesará todo.")
        return files

    print(f"📄 Filtro por posiciones FARM_FILE_RANGE='{expr}' → {len(filtered)} de {n} archivos.")
    return filtered

# -------- columnas salida --------
DEFAULT_COLUMNS = [
    "id","intersects_deforestation","deforested_hectares","deforested_proportion",
    "deforestation_distance_to","intersects_protected_area","protected_ha",
    "protected_prop","protected_area_distance_to","risk_direct_level","risk_direct_level_value",
]

# -------- GIS helpers --------
def ensure_vector_crs(path, expected_crs):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} no tiene CRS definido.")
    if gdf.crs.to_string() != expected_crs:
        print(f"⚠ {os.path.basename(path)} está en {gdf.crs.to_string()}, reproyectando a {expected_crs}...")
        gdf = gdf.to_crs(expected_crs)
    else:
        print(f"✔ Vector {os.path.basename(path)} CRS correcto: {gdf.crs.to_string()}")
    return gdf

def ensure_raster_crs(path, expected_crs):
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError(f"{path} no tiene CRS definido.")
        if crs.to_string() != expected_crs:
            raise ValueError(f"{path} está en {crs.to_string()}, se esperaba {expected_crs}. Reproyecta con gdalwarp.")
        print(f"✔ Raster {os.path.basename(path)} CRS correcto: {crs.to_string()}")

def build_deforestation_kdtree(raster_path, deforest_value) -> Optional[KDTree]:
    t0 = time.time()
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        mask_def = (arr == deforest_value)
        count = int(np.count_nonzero(mask_def))
        if count == 0:
            print("⚠ No hay píxeles de deforestación con el valor indicado.")
            return None
        rows, cols = np.where(mask_def)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        pts = np.column_stack([xs, ys]).astype(np.float64)
        tree = KDTree(pts)
    print(f"⏱ KDTree construido con {count} píxeles en {time.time()-t0:.2f} s.")
    return tree

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

def nearest_deforestation_distance(geom, defo_tree: Optional[KDTree], k=3, max_dist=50_000) -> float:
    if defo_tree is None or defo_tree.n == 0:
        return float(max_dist)
    c = geom.centroid
    dists, idxs = defo_tree.query([c.x, c.y], k=min(k, defo_tree.n), distance_upper_bound=max_dist)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    finite = np.isfinite(dists)
    if not np.any(finite):
        return float(max_dist)
    pts = defo_tree.data[idxs[finite]]
    dmin = float(max_dist)
    for x, y in pts:
        d = float(geom.distance(Point(x, y)))
        if d < dmin:
            dmin = d
            if dmin == 0.0:
                break
    return float(min(max(dmin, 0.0), max_dist))

def protected_intersection_metrics(geom, gdf_prot):
    try:
        sidx = gdf_prot.sindex
        cand_idx = list(sidx.intersection(geom.bounds))
        if not cand_idx:
            return False, 0.0, 0.0
        cands = gdf_prot.iloc[cand_idx]
        inters = cands.intersection(geom)
        inters = inters[~inters.is_empty]
        if inters.empty:
            return False, 0.0, 0.0
        inter_union = unary_union(inters.to_list())
        area_ha = inter_union.area / 10_000.0
        prop = area_ha / (geom.area / 10_000.0)
        return True, float(area_ha), float(prop)
    except Exception as e:
        logging.warning(f"Error calculando intersección PNN: {e}")
        return False, 0.0, 0.0

def nearest_protected_distance(geom, gdf_prot, max_dist=50_000) -> float:
    try:
        sidx = gdf_prot.sindex
        cand_idx = list(sidx.intersection(geom.bounds))
        if cand_idx and gdf_prot.iloc[cand_idx].intersects(geom).any():
            return 0.0
        buf = geom.buffer(max_dist)
        cand_idx2 = list(sidx.intersection(buf.bounds))
        if not cand_idx2:
            return float(max_dist)
        dmin = float(max_dist)
        for g in gdf_prot.iloc[cand_idx2].geometry:
            d = geom.distance(g)
            if d < dmin:
                dmin = d
                if dmin == 0.0:
                    break
        return float(min(dmin, max_dist))
    except Exception as e:
        logging.warning(f"Error calculando distancia PNN: {e}")
        return float(max_dist)

def assign_risk(intersects_def, intersects_pnn):
    return ("HIGH", 3) if (intersects_def or intersects_pnn) else ("NO_RISK", 0)

def process_parcel_row(row, raster_deforest, defo_tree, gdf_protected, deforest_value, max_dist):
    farm_id = row["farm_id"]; geom = row["geometry"]
    inter_def, def_ha, def_prop = calculate_deforestation_metrics(geom, raster_deforest, deforest_value)
    defo_dist = nearest_deforestation_distance(geom, defo_tree, k=3, max_dist=max_dist)
    inter_pnn, pnn_ha, pnn_prop = protected_intersection_metrics(geom, gdf_protected)
    prot_dist = nearest_protected_distance(geom, gdf_prot=gdf_protected, max_dist=max_dist)
    risk_level, risk_value = assign_risk(inter_def, inter_pnn)
    return {
        "id": farm_id,
        "intersects_deforestation": inter_def,
        "deforested_hectares": round(def_ha, 4),
        "deforested_proportion": round(def_prop, 6),
        "deforestation_distance_to": float(round(defo_dist, 2)),
        "intersects_protected_area": inter_pnn,
        "protected_ha": round(pnn_ha, 4),
        "protected_prop": round(pnn_prop, 6),
        "protected_area_distance_to": float(round(prot_dist, 2)),
        "risk_direct_level": risk_level,
        "risk_direct_level_value": risk_value,
    }

# -------- main --------
def main():
    setup_logging()

    empresa = config['EMPRESA']
    periodo = config['PERIODO']
    crs = config['CRS_METROS']
    max_dist = config['MAX_DIST']
    defo_val = config['DEFOREST_VALUE']
    batch_size = config['BATCH_SIZE']

    years_list = parse_year_periods(config['YEARS'])
    if not years_list:
        years_list = [str(config['YEARS']).strip()]
        logging.info(f"YEARS único detectado: {years_list[0]}")

    # Insumos no dependientes del periodo
    shp_protected_tpl = config['SHP_PROTECTED']
    folder_geojsons_tpl = config['FOLDER_GEOJSONS']
    ctx_base = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": ""}

    shp_protected = format_placeholders(shp_protected_tpl, ctx_base)
    gdf_protected_proj = ensure_vector_crs(shp_protected, crs)

    folder_geojsons = format_placeholders(folder_geojsons_tpl, ctx_base)
    files = sorted([f for f in os.listdir(folder_geojsons) if f.lower().endswith(".geojson")])
    if not files:
        raise RuntimeError(f"No se encontraron GeoJSONs en: {folder_geojsons}")

    # --- aplicar filtro por posiciones (tipo R) ---
    print(f"⚙ FARM_FILE_RANGE='{config.get('FARM_FILE_RANGE','')}'")
    files = apply_file_range(files, config.get("FARM_FILE_RANGE", ""))

    start_all = time.time()
    for years in years_list:
        print(f"\n🟪 Periodo: {years} — empresa: {empresa} — etiqueta periodo: {periodo}")

        ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years}

        raster_deforest = format_placeholders(config['RASTER_DEFOREST'], ctx)
        output_csv = format_placeholders(config['OUTPUT_CSV'], ctx)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        ensure_raster_crs(raster_deforest, crs)

        # KDTree por periodo
        defo_tree = build_deforestation_kdtree(raster_deforest, defo_val)

        # Reset CSV por periodo
        if os.path.exists(output_csv):
            os.remove(output_csv)

        start = time.time()
        for i in tqdm(range(0, len(files), batch_size), desc=f"Procesando lotes ({years})", unit="lote"):
            batch = files[i:i+batch_size]
            gdf_list: List[gpd.GeoDataFrame] = []
            for file in batch:
                try:
                    gdf = gpd.read_file(os.path.join(folder_geojsons, file))
                    if gdf.crs is None or gdf.crs.to_string() != crs:
                        gdf = gdf.to_crs(crs)

                    # ID numérico desde el nombre si existe; de lo contrario, usar basename
                    m = re.search(r"[_-](\d+)\.geojson$", file, re.IGNORECASE) or re.search(r"(\d+)\.geojson$", file, re.IGNORECASE)
                    farm_id = int(m.group(1)) if m else os.path.splitext(file)[0]

                    gdf["farm_id"] = farm_id
                    gdf_list.append(gdf[["farm_id", "geometry"]])
                except Exception as e:
                    logging.warning(f"[{years}] Error leyendo {file}: {e}")

            if not gdf_list:
                continue

            merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

            results = []
            for _, r in tqdm(merged.iterrows(), total=len(merged), desc="  Fincas en lote", unit="finca", leave=False):
                try:
                    res = process_parcel_row(
                        r, raster_deforest, defo_tree, gdf_protected_proj,
                        deforest_value=defo_val, max_dist=max_dist
                    )
                    results.append(res)
                except Exception as e:
                    logging.warning(f"[{years}] Error procesando finca {r.get('farm_id')}: {e}")

            if results:
                df_out = pd.DataFrame(results)
                for col in DEFAULT_COLUMNS:
                    if col not in df_out.columns:
                        df_out[col] = 0
                df_out = df_out[DEFAULT_COLUMNS]
                header = (i == 0)
                df_out.to_csv(output_csv, mode='a', index=False, header=header)

            del gdf_list, merged, results
            gc.collect()

        print(f"✅ Periodo {years} completado → {output_csv}")
        print(f"⏱ Tiempo {years}: {time.time()-start:.2f} s")

    print(f"\n🎉 Periodos completados: {', '.join(years_list)}")
    print(f"⏱ Tiempo total: {time.time()-start_all:.2f} s")

if __name__ == "__main__":
    main()
