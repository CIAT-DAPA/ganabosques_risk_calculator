# -*- coding: utf-8 -*-
import os, re, gc, time, logging, sys
from typing import Dict, Any, Tuple, List, Optional
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.crs import CRS as RCRS
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping
from shapely.ops import unary_union, transform as shapely_transform
from shapely.validation import make_valid
from pyproj import Transformer
from tqdm import tqdm
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

# ===================== AJUSTES =====================
SETTINGS: Dict[str, Any] = {
    # ahora todo en 3116 (CRS proyectado en metros)
    "CRS_METROS": "EPSG:3116",
    "DEFOREST_VALUE": 2,
    "PERIODO": "cumulative",
    "YEARS": "2010-2011,2010-2012,2010-2013,2010-2014,2010-2015,2010-2016,2010-2017,2010-2018,2010-2019,2010-2020,2010-2021,2010-2022,2010-2023",
    "FARM_FILE_RANGE": "1:495383",
    "BATCH_SIZE": 1000,

    "FOLDER_GEOJSONS": "/opt/ganabosques/test_buffers/data_server/buffer/",
    "RASTER_DEFOREST": "/opt/ganabosques/test_buffers/data_server/smbyc/{PERIODO}\\smbyc_deforestation_{PERIODO}_{YEARS}.tif",
    "ALERTAS_DIR": "/opt/ganabosques/test_buffers/data_server/05_tmp_alertas_tempranas\\{YEARS}",
    "NUCLEOS_DIR": "/opt/ganabosques/test_buffers/data_server/06_tmp_nucleos_activos/{YEARS}",

    "OUTPUT_CSV": "/opt/ganabosques/test_buffers/data_server/alertas/{PERIODO}/direct_alert/direct_alert_{YEARS}.csv",

    "LOG_LEVEL": "WARNING",
    "LOG_FILE": "risk_analysis_intersections.log",
}

# ===================== utilidades básicas =====================
def norm(p: str) -> str:
    return os.path.normpath(p.replace("\\", "/")) if p else p

def setup_logging():
    lvl = getattr(logging, SETTINGS['LOG_LEVEL'].upper(), logging.WARNING)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(filename=SETTINGS['LOG_FILE'], level=lvl, format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

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
        "PERIODO": ctx.get("PERIODO"),
        "YEARS": ctx.get("YEARS"),
        "periodo": ctx.get("PERIODO"),
        "years": ctx.get("YEARS"),
        "FARM_FILE_RANGE": SETTINGS.get("FARM_FILE_RANGE", ""),
    }
    try:
        return template.format(**mixed)
    except KeyError as e:
        logging.warning(f"Placeholder faltante {e} en: {template}")
        return template

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
            a = max(1, min(a, n))
            b = max(1, min(b, n))
            if a <= b:
                selected.extend(files[a-1:b])
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
    seen = set()
    filtered = [f for f in selected if not (f in seen or seen.add(f))]
    if not filtered:
        logging.warning(f"FARM_FILE_RANGE='{expr}' no seleccionó archivos; se procesará todo.")
        return files
    print(f"📄 Filtro por posiciones FARM_FILE_RANGE='{expr}' → {len(filtered)} de {n} archivos.")
    return filtered

def _crs_eq(a: str, b: str) -> bool:
    try:
        return RCRS.from_string(a).to_epsg() == RCRS.from_string(b).to_epsg()
    except Exception:
        return a == b

def ensure_raster_crs(path, expected_crs):
    path = norm(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster no encontrado: {path}")
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError(f"{path} no tiene CRS definido.")
        if not _crs_eq(crs.to_string(), expected_crs):
            # avisamos que será aceptado pero procesado virtualmente en expected_crs (3116)
            logging.warning(f"{path} está en {crs.to_string()}, se procesará en {expected_crs} mediante WarpedVRT.")
        print(f"✔ Raster {os.path.basename(path)} detectado: {crs.to_string()}")

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

# ===================== Área en hectáreas en CRS proyectado (3116) =====================
def area_hectares_from_3116(geom) -> float:
    """
    Calcula área (hectáreas) de un Polygon/MultiPolygon en un CRS proyectado en metros
    (por defecto EPSG:3116). Se asume que la geometría YA está en ese CRS.
    """
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area) / 10000.0

# ===================== Cargar vectores (asegurando 3116) =====================
def load_vector_from_paths(paths: List[str], expected_crs: str) -> Optional[gpd.GeoDataFrame]:
    geoms = []
    exp = gpd.GeoSeries([0], crs=expected_crs).crs
    for p in paths:
        try:
            gdf = gpd.read_file(norm(p))
            if gdf.empty or "geometry" not in gdf.columns:
                continue
            if gdf.crs is None:
                logging.warning(f"{p} no tiene CRS; se omite.")
                continue
            if not _crs_eq(gdf.crs.to_string(), expected_crs):
                gdf = gdf.to_crs(exp)   # reproyectar a 3116 solo si es distinto
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
    folder = norm(format_placeholders(folder_template, ctx))
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
            with rasterio.open(norm(p)) as src:
                crs = src.crs.to_string() if src.crs else None
                if crs and _crs_eq(crs, expected_crs):
                    ok.append(p)
                else:
                    # Aceptamos, pero avisamos que será leído virtualmente en expected_crs
                    logging.warning(f"Raster {os.path.basename(p)} será leído virtualmente en {expected_crs} (orig:{crs}).")
                    ok.append(p)
        except Exception as e:
            logging.warning(f"No se pudo abrir raster {p}: {e}")
    return ok

# ===================== Rasters: abrir (si necesario, usar WarpedVRT a 3116) =====================
def open_raster(path, target_crs="EPSG:3116"):
    """
    Abre raster y si su CRS no es target_crs (p.ej. EPSG:3116),
    devuelve un WarpedVRT que lo presenta en target_crs.
    """
    p = norm(path)
    src = rasterio.open(p)
    src_crs = src.crs
    if src_crs is None:
        raise ValueError(f"{p} no tiene CRS.")
    if _crs_eq(src_crs.to_string(), target_crs):
        return src  # caller must close
    else:
        # reproyección virtual a target_crs (3116)
        vrt = WarpedVRT(src, crs=target_crs)
        return vrt

def open_rasters(paths: List[str], target_crs="EPSG:3116"):
    srcs = []
    for p in paths or []:
        try:
            srcs.append(open_raster(p, target_crs=target_crs))
        except Exception as e:
            logging.warning(f"No se pudo abrir raster {p}: {e}")
    return srcs

def close_rasters(srcs):
    for s in srcs or []:
        try:
            s.close()
        except Exception:
            pass

# ===================== Util: área aproximada por píxel cuando raster está en 4326 =====================
def pixel_area_m2_approx_for_vrt(src):
    """
    Si src está en EPSG:4326 (grados), aproximamos el área de un píxel en m^2
    usando la latitud media del raster. Para CRS proyectados (3116) se usará
    directamente la transform (a*e).
    """
    try:
        t = src.transform
        pixel_deg_x = abs(t.a)
        pixel_deg_y = abs(t.e)
        # get center latitude from bounds
        left, bottom, right, top = src.bounds
        center_lat = (top + bottom) / 2.0
        lat_rad = math.radians(center_lat)
        m_per_deg_lat = 110574.0
        m_per_deg_lon = 111320.0 * math.cos(lat_rad)
        pixel_m2 = pixel_deg_x * pixel_deg_y * m_per_deg_lon * m_per_deg_lat
        return abs(pixel_m2)
    except Exception:
        # fallback: approximate using average ~ 111000 m/deg
        return abs(pixel_deg_x * pixel_deg_y * (111000.0**2))

# ===================== Área desde vectores (usando 3116) =====================
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
        # área en hectáreas en 3116
        return float(area_hectares_from_3116(inter_union))
    except Exception as e:
        logging.warning(f"Error área vectorial: {e}")
        return 0.0

# ===================== Cálculos usando rasters ABIERTOS =====================
def calculate_deforestation_metrics_from_src(src, geom, deforest_value, use_precise_area=False, supersample_factor=5) -> Tuple[bool, float, float]:
    """
    - src: puede ser dataset o WarpedVRT; si es WarpedVRT, está en CRS_METROS (3116).
    - geom: geometría en CRS_METROS (3116).
    - use_precise_area: Si True, usa super-sampling para calcular fracciones de píxel (más preciso pero ~250x más lento)
    - supersample_factor: Factor de super-sampling (default 5 = 5×5 = 25 sub-píxeles por píxel)
    Procedimiento:
      - Si src.crs == CRS_METROS -> usar geom directamente y mask
      - Si src.crs != CRS_METROS, transformar geom a src.crs y hacer mask.
    Retorna: (intersectó?, defo_ha, proportion_of_geom_clamped_0_1)
    """
    try:
        expected_crs = SETTINGS["CRS_METROS"]
        src_crs = src.crs.to_string() if src.crs else None
        geom_for_mask = geom

        if src_crs and not _crs_eq(src_crs, expected_crs):
            # transformar geom al CRS del raster
            transformer = Transformer.from_crs(expected_crs, src_crs, always_xy=True)
            func = lambda x, y, z=None: transformer.transform(x, y)
            geom_for_mask = shapely_transform(func, geom)

        # all_touched=True cuando precise-area: incluir TODOS los píxeles que tocan
        # el polígono (no solo los cuyo centro cae dentro). El super-sampling
        # se encargará de calcular la fracción real de cobertura de cada píxel.
        out_image, out_transform = mask(
            src, [mapping(geom_for_mask)],
            crop=True, filled=False,
            all_touched=use_precise_area
        )
        arr = out_image[0]  # MaskedArray
        valid = ~np.ma.getmaskarray(arr)
        m = valid & (arr.data == deforest_value)
        cnt = int(np.count_nonzero(m))
        if cnt == 0:
            return False, 0.0, 0.0

        # determinar área de píxel en m2
        if src_crs and _crs_eq(src_crs, "EPSG:4326"):
            pixel_area = pixel_area_m2_approx_for_vrt(src)
        else:
            # en raster proyectado (metros), se puede usar transform directamente
            pixel_area = abs(src.transform.a * src.transform.e)

        # Calcular área deforestada
        if use_precise_area:
            # 🎯 MÉTODO PRECISO: Super-sampling con fracciones de píxel
            # Crear grilla de alta resolución para calcular cobertura fraccionaria
            out_image, out_transform = mask(src, [mapping(geom)], crop=True, filled=False,
            all_touched=use_precise_area)
            arr = out_image[0]

            # Crear máscara de clase deforestación
            mask_def = arr == deforest_value
            if not mask_def.any():
                return False, 0.0, 0.0

            # Vectorizar SOLO píxeles deforestados
            pixel_polygons = []
            for geom_json, value in shapes(arr, mask=mask_def, transform=out_transform):
                if value == deforest_value:
                    pixel_polygons.append(shape(geom_json))

            if not pixel_polygons:
                return False, 0.0, 0.0

            # Unir todos los píxeles en una sola geometría
            union_pixels = unary_union(pixel_polygons)

            # Intersección geométrica real
            intersection = geom.intersection(union_pixels)

            if intersection.is_empty:
                return False, 0.0, 0.0

            # Área exacta en hectáreas
            area_m2 = intersection.area
            defo_ha = area_m2 / 10000.0

            """ elif use_precise_area:
                # 🎯 MÉTODO PRECISO: Super-sampling con fracciones de píxel
                # Crear grilla de alta resolución para calcular cobertura fraccionaria
                height, width = arr.shape
                supersample_shape = (height * supersample_factor, width * supersample_factor)
                
                # Ajustar transform para la grilla fina
                from rasterio.transform import Affine
                super_transform = Affine(
                    out_transform.a / supersample_factor,
                    out_transform.b,
                    out_transform.c,
                    out_transform.d,
                    out_transform.e / supersample_factor,
                    out_transform.f
                )
                
                # Rasterizar geometría en grilla fina
                super_coverage = rasterize(
                    [(geom_for_mask, 1)],
                    out_shape=supersample_shape,
                    transform=super_transform,
                    all_touched=True,
                    dtype='uint8',
                    fill=0
                )
                
                # Reducir a resolución original contando sub-píxeles
                reshaped = super_coverage.reshape(height, supersample_factor, width, supersample_factor)
                coverage_fraction = reshaped.sum(axis=(1, 3)) / (supersample_factor * supersample_factor)
                
                # Calcular área usando fracciones
                fractional_count = float(np.sum(coverage_fraction[m]))
                defo_ha = fractional_count * pixel_area / 10_000.0 """
        
        else:
            # ⚡ MÉTODO RÁPIDO: Contar píxeles completos (método actual)
            defo_ha = cnt * pixel_area / 10_000.0

        # área de la geometría en hectáreas (CRS_METROS = 3116)
        geom_ha = area_hectares_from_3116(geom)
        if geom_ha <= 0:
            prop = 0.0
        else:
            prop = defo_ha / geom_ha

        # 🔒 Limitar proporción al rango [0, 1]
        prop = max(0.0, min(float(prop), 1.0))

        return True, float(defo_ha), prop
    except Exception as e:
        logging.warning(f"Error calculando deforestación (src): {e}")
        return False, 0.0, 0.0

def area_from_rasters_open(geom, src_list) -> float:
    total_ha = 0.0
    expected_crs = SETTINGS["CRS_METROS"]
    for src in src_list or []:
        try:
            src_crs = src.crs.to_string() if src.crs else None
            geom_for_mask = geom
            if src_crs and not _crs_eq(src_crs, expected_crs):
                transformer = Transformer.from_crs(expected_crs, src_crs, always_xy=True)
                func = lambda x, y, z=None: transformer.transform(x, y)
                geom_for_mask = shapely_transform(func, geom)

            out_image, _ = mask(src, [mapping(geom_for_mask)], crop=True, filled=False)
            arr = out_image[0]
            valid = ~np.ma.getmaskarray(arr)
            cnt = int(np.count_nonzero(valid))
            if cnt == 0:
                continue

            if src_crs and _crs_eq(src_crs, "EPSG:4326"):
                pixel_area = pixel_area_m2_approx_for_vrt(src)
            else:
                pixel_area = abs(src.transform.a * src.transform.e)

            total_ha += cnt * pixel_area / 10_000.0
        except Exception as e:
            logging.warning(f"Error área raster (abierto): {e}")
    return float(total_ha)

# ===================== Procesamiento por fila (usando rasters abiertos) =====================
def process_row_option1(row, raster_src, deforest_value: int, use_precise_area: bool = False, pixel_divisions: int = 5):
    farm_id = row["farm_id"]; geom = row["geometry"]
    inter_def, def_ha, def_prop = calculate_deforestation_metrics_from_src(raster_src, geom, deforest_value, use_precise_area=use_precise_area, supersample_factor=pixel_divisions)
    def _r4(x): return round(float(x), 4)
    def _r6(x): return round(float(x), 6)
    return {
        "id": farm_id,
        "intersect_deforestation": bool(inter_def),
        "deforested_ha": _r4(def_ha),
        "deforested_prop": _r6(def_prop),  # fracción entre 0.0 y 1.0 (no porcentaje)
        "direct_alert": bool(inter_def),
    }

def process_row_option2(row, alerts_vec, alerts_src_list, crs_expected):
    farm_id = row["farm_id"]; geom = row["geometry"]
    total_ha_geom = float(area_hectares_from_3116(geom))
    has_data = (alerts_vec is not None and not alerts_vec.empty) or (alerts_src_list and len(alerts_src_list) > 0)
    if not has_data:
        nan = float("nan")
        return {
            "id": farm_id,
            "intersect_early_warnings": nan,
            "early_warnings_ha": nan,
            "early_warnings_prop": nan,
            "direct_alert": False,
        }, False
    ha_vec = area_from_vectors(geom, alerts_vec)
    ha_ras = area_from_rasters_open(geom, alerts_src_list)
    ha = ha_vec + ha_ras
    if ha <= 0.0 or total_ha_geom <= 0.0:
        return {
            "id": farm_id,
            "intersect_early_warnings": True,
            "early_warnings_ha": 0.0,
            "early_warnings_prop": 0.0,
            "direct_alert": False,
        }, True
    prop = ha / total_ha_geom
    return {
        "id": farm_id,
        "intersect_early_warnings": True,
        "early_warnings_ha": round(float(ha), 4),
        "early_warnings_prop": round(float(prop), 6),
        "direct_alert": True,
    }, True

def process_row_option3(row, nucleos_vec, nucleos_src_list, crs_expected):
    farm_id = row["farm_id"]; geom = row["geometry"]
    total_ha_geom = float(area_hectares_from_3116(geom))
    has_data = (nucleos_vec is not None and not nucleos_vec.empty) or (nucleos_src_list and len(nucleos_src_list) > 0)
    if not has_data:
        nan = float("nan")
        return {
            "id": farm_id,
            "intersect_active_hotspots": nan,
            "active_hotspots_ha": nan,
            "active_hotspots_prop": nan,
            "direct_alert": False,
        }, False
    ha_vec = area_from_vectors(geom, nucleos_vec)
    ha_ras = area_from_rasters_open(geom, nucleos_src_list)
    ha = ha_vec + ha_ras
    if ha <= 0.0 or total_ha_geom <= 0.0:
        return {
            "id": farm_id,
            "intersect_active_hotspots": True,
            "active_hotspots_ha": 0.0,
            "active_hotspots_prop": 0.0,
            "direct_alert": False,
        }, True
    prop = ha / total_ha_geom
    return {
        "id": farm_id,
        "intersect_active_hotspots": True,
        "active_hotspots_ha": round(float(ha), 4),
        "active_hotspots_prop": round(float(prop), 6),
        "direct_alert": True,
    }, True

def print_availability_option(name: str, found: bool, details: str = ""):
    yn = "Sí" if found else "No"
    print(f"  • {name}: {yn} {details}")

def make_out_dir_and_paths(base_output_template: str, ctx: Dict[str,str], source_tag: str, deforestation_type: str = None) -> Tuple[str,str]:
    """
    Genera directorio de salida y ruta del CSV para alertas directas.
    
    Nueva estructura de carpetas:
    results/{source}/{deforestation_type}/direct_alerts/{period}/
    
    Args:
        base_output_template: Template base para la ruta de salida
        ctx: Diccionario con contexto (PERIODO, YEARS)
        source_tag: Fuente de deforestación (siempre 'smbyc')
        deforestation_type: Tipo de deforestación ('annual', 'cumulative', 'nad', 'atd')
    
    Returns:
        Tuple con (directorio_salida, ruta_csv)
    """
    formatted = norm(format_placeholders(base_output_template, ctx))
    base_dir = os.path.dirname(formatted) or "."
    
    # Subir un nivel desde el directorio base para construir la nueva estructura
    # base_dir típicamente es: .../results/direct_alerts
    # Queremos: .../results/{source}/{deforestation_type}/direct_alerts/{period}
    results_parent = os.path.dirname(base_dir)  # .../results
    
    years = ctx["YEARS"]
    periodo = ctx["PERIODO"]
    
    # Si se proporciona deforestation_type, usar nueva estructura
    if deforestation_type:
        # Estructura: results/{source}/{deforestation_type}/direct_alerts/
        out_dir = norm(os.path.join(results_parent, source_tag.lower(), deforestation_type.lower(), "direct_alerts"))
    else:
        # Fallback a estructura anterior para compatibilidad
        option_map = {"smbyc": "SMBYC", "atd": "ATD", "nad": "NAD", "otra": "OTRA"}
        option_folder = option_map.get(str(source_tag).lower(), str(source_tag).upper())
        out_dir = norm(os.path.join(base_dir, option_folder, years))
    
    os.makedirs(out_dir, exist_ok=True)
    out_csv = norm(os.path.join(out_dir, f"{source_tag}_direct_alert_{deforestation_type or periodo}_{years}.csv"))
    return out_dir, out_csv

def write_reason_log(out_dir: str, fname: str, msg: str):
    path = norm(os.path.join(out_dir, fname))
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(f"📝 Se escribió log: {path}")

# ===================== Runner =====================
def calculate_direct_alerts(
    source: str,
    period_type: str,
    years: List[str],
    farm_folder: str,
    raster_template: str,
    output_csv: str,
    batch_size: int = 1000,
    farm_range: str = "",
    crs: str = "EPSG:3116",
    deforest_value: int = 2,
    log_level: str = "WARNING",
    log_file: str = "risk_analysis_intersections.log",
    raster_paths_dict: Optional[Dict[str, str]] = None,
    use_precise_area: bool = False,
    pixel_divisions: int = 5,
    _data_manager: Optional[Any] = None,
    _farms_metadata: Optional[List[Dict]] = None,
    _farm_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calcula alertas directas de deforestación para predios ganaderos.
    
    Args:
        source: Fuente de deforestación ('smbyc')
        period_type: Tipo de período ('annual', 'cumulative', 'nad', 'atd')
        years: Lista de períodos a procesar (ej: ['2013-2014', '2014-2015'])
        farm_folder: Ruta a carpeta con GeoJSON de predios
        raster_template: Template del raster de deforestación con placeholders {PERIODO} y {YEARS}
        output_csv: Template del CSV de salida con placeholders
        batch_size: Tamaño de lote para procesamiento
        farm_range: Rango de archivos a procesar (ej: "1:1000")
        crs: Sistema de coordenadas (default: EPSG:3116)
        deforest_value: Valor de píxel de deforestación en raster
        log_level: Nivel de logging
        log_file: Archivo de log
        raster_paths_dict: Diccionario {period_name: raster_path} con rutas preparadas (opcional)
        use_precise_area: Si True, usa super-sampling para cálculo preciso de fracciones de píxel
        pixel_divisions: Divisiones por píxel para super-sampling (default 5 = 5×5 = 25 sub-píxeles)
        
    Returns:
        Dict con estadísticas: {
            'periods_processed': int,
            'farms_processed': int,
            'alerts_generated': int,
            'execution_time': float
        }
    """
    # Configurar logging
    lvl = getattr(logging, log_level.upper(), logging.WARNING)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(filename=log_file, level=lvl, format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    # Mapear source a tag y deforest_value
    if source.lower() == "smbyc":
        defo_val = deforest_value
        opt1_tag = "smbyc"
    else:
        # Fallback para fuentes no reconocidas
        defo_val = deforest_value
        opt1_tag = source.lower() if source else "smbyc"
    
    print(f"🟩 Procesando alertas directas. Fuente: {source} | Tipo: {period_type} | deforest_value={defo_val}")
    if use_precise_area:
        print(f"🎯 Modo PRECISO habilitado: Usando super-sampling {pixel_divisions}×{pixel_divisions} para fracciones de píxel")
    else:
        print("⚡ Modo RÁPIDO: Conteo de píxeles completos (puede tener error ~100% en bordes)")
    
    folder_geojsons = norm(farm_folder)
    output_tpl = output_csv
    periodo = period_type
    years_list = years

    if not os.path.isdir(folder_geojsons):
        raise RuntimeError(f"No existe la carpeta de geojsons: {folder_geojsons}")

    # Si tenemos farms_metadata, solo procesar esos farms específicos
    if _farms_metadata and len(_farms_metadata) > 0:
        print(f"🎯 Usando farms cargados desde base de datos: {len(_farms_metadata):,} farms")
        
        # Construir lista de archivos geojson basado SOLO en sitcode
        files = []
        farms_without_sitcode = []
        downloaded_count = 0
        
        for farm_meta in _farms_metadata:
            sitcode = farm_meta.get('sitcode')
            mongo_id = farm_meta.get('mongo_id')
            
            # POLÍTICA: Solo procesar farms con sitcode
            if not sitcode:
                farms_without_sitcode.append(mongo_id)
                continue
            
            # Si tenemos DataManager, usar su método para asegurar disponibilidad
            if _data_manager and hasattr(_data_manager, 'ensure_geojson_available'):
                geojson_path = _data_manager.ensure_geojson_available(farm_meta)
                if geojson_path:
                    geojson_file = os.path.basename(geojson_path)
                    files.append(geojson_file)
                    # Verificar si fue descarga reciente
                    from pathlib import Path
                    path = Path(geojson_path)
                    if path.exists():
                        age_seconds = time.time() - path.stat().st_mtime
                        if age_seconds < 5:  # Creado en los últimos 5 segundos
                            downloaded_count += 1
                else:
                    farms_without_sitcode.append(f"{sitcode} (no geojson)")
            else:
                # Modo sin DataManager: buscar por sitcode únicamente
                sitcode_file = f"{sitcode}.geojson"
                sitcode_path = os.path.join(folder_geojsons, sitcode_file)
                if os.path.exists(sitcode_path):
                    files.append(sitcode_file)
                else:
                    farms_without_sitcode.append(f"{sitcode} (no existe)")
        
        if farms_without_sitcode:
            print(f"⚠ Farms sin sitcode o sin geojson: {len(farms_without_sitcode)}/{len(_farms_metadata)}")
            if len(farms_without_sitcode) <= 10:
                for mf in farms_without_sitcode[:10]:
                    print(f"   • {mf}")
        
        if downloaded_count > 0:
            print(f"📥 Geojsons descargados desde MongoDB: {downloaded_count}")
        
        total_files_initial = len(files)
        total_files = len(files)
        
        if not files:
            raise RuntimeError(f"No se encontraron GeoJSONs para los {len(_farms_metadata)} farms especificados")
        
        print(f"✅ GeoJSONs disponibles: {total_files:,} de {len(_farms_metadata):,} farms")
        print(f"⚙ BATCH_SIZE={batch_size}")
        
    else:
        # Modo fallback: usar geojsons de carpeta (cuando falla BD o no hay farms)
        print(f"📂 Modo fallback: buscando geojsons en carpeta...")
        all_files = sorted([f for f in os.listdir(folder_geojsons) if f.lower().endswith(".geojson")])
        
        if not all_files:
            raise RuntimeError(
                f"❌ ERROR CRÍTICO: No se puede continuar\n"
                f"   • No hay conexión a base de datos\n"
                f"   • No hay geojsons en: {folder_geojsons}\n"
                f"   → Solución: Verifica la conexión a MongoDB o coloca geojsons en la carpeta"
            )
        
        print(f"📦 GeoJSONs encontrados: {len(all_files):,} en {folder_geojsons}")
        
        # Aplicar límite si se especificó
        if _farm_limit and _farm_limit > 0:
            # Respetar el límite especificado por el usuario
            if _farm_limit < len(all_files):
                files = all_files[:_farm_limit]
                print(f"⚠ Límite aplicado: Se procesarán {_farm_limit:,} de {len(all_files):,} geojsons disponibles")
            else:
                files = all_files
                print(f"✅ Se procesarán todos los {len(all_files):,} geojsons disponibles (límite {_farm_limit:,} no alcanzado)")
        elif farm_range and farm_range != "":
            # Si hay FARM_FILE_RANGE, usarlo
            files = apply_file_range(all_files, farm_range)
            print(f"⚙ FARM_FILE_RANGE='{farm_range}' aplicado: {len(files):,} geojsons")
        else:
            # Sin límites, usar todos
            files = all_files
            if _farm_limit:
                print(f"⚠ farm_limit recibido pero no aplicado: {_farm_limit} (verif icar lógica)")
            print(f"✅ Se procesarán todos los geojsons disponibles")
        
        print(f"⚙ BATCH_SIZE={batch_size}")
        total_files_initial = len(all_files)
        total_files = len(files)
    
    total_batches = (total_files + batch_size - 1) // batch_size
    print(f"🏁 Se procesarán {total_files} archivos en {total_batches} lotes (batch_size={batch_size}).")

    start_all = time.time()
    # ===================== LOOP POR YEARS =====================
    periods_progress = tqdm(
        enumerate(years_list, start=1),
        total=len(years_list),
        desc="📅 Períodos",
        unit="período",
        position=0,
        leave=True
    )
    for p_idx, years in periods_progress:
        periods_progress.set_description(f"📅 Procesando {years}")
        ctx = {"PERIODO": periodo, "YEARS": years}

        # --- Cargar raster de deforestación para este período
        # Usar ruta preparada si está disponible, sino usar template
        if raster_paths_dict and years in raster_paths_dict:
            raster_deforest = raster_paths_dict[years]
            print(f"🔎 Usando raster preparado: {os.path.basename(raster_deforest)}")
        else:
            raster_deforest = norm(format_placeholders(raster_template, ctx))
            print(f"🔎 Validando raster deforestación: {raster_deforest}")
        
        try:
            ensure_raster_crs(raster_deforest, crs)
            raster_src_o1 = open_raster(raster_deforest, target_crs=crs)
            opt1_available = True
            print_availability_option("Deforestación (raster)", True, f"→ {os.path.basename(raster_deforest)}")
        except Exception as e:
            logging.error(f"[{years}] Error cargando raster: {e}")
            print("❌ No se encontró/validó el raster de deforestación; se omite este periodo.")
            raster_src_o1 = None
            opt1_available = False

        # Verificar si hay datos disponibles
        if not opt1_available:
            print("⚠ No hay raster disponible para este periodo. Se omite generación de resultados.")
            close_rasters([raster_src_o1] if raster_src_o1 else [])
            gc.collect()
            continue

        # Salida CSV con nueva estructura de carpetas
        # results/{source}/{deforestation_type}/direct_alerts/{period}/
        _, out_csv_o1 = make_out_dir_and_paths(output_tpl, ctx, opt1_tag, deforestation_type=periodo)
        if os.path.exists(out_csv_o1):
            os.remove(out_csv_o1)
        print(f"🗂️ Salida: {out_csv_o1}")

        # ===================== LOOP POR LOTES =====================
        start_period = time.time()
        for b_idx, i in enumerate(range(0, total_files, batch_size), start=1):
            t_batch = time.time()
            batch = files[i:i+batch_size]
            print(f"\n  ➤ Lote {b_idx}/{total_batches} | archivos {i+1}-{i+len(batch)} (de {total_files})")

            gdf_list: List[gpd.GeoDataFrame] = []
            
            # 🚀 OPTIMIZACIÓN: Usar geometrías desde caché si DataManager está disponible
            if _data_manager and hasattr(_data_manager, '_geometries_cache') and _data_manager._geometries_cache:
                for file in batch:
                    try:
                        # El archivo se nombró con sitcode (SIT_CODE o GEOFARMER_ID),
                        # que es la misma clave usada en _geometries_cache.
                        # Usar el stem del archivo directamente como farm_id.
                        farm_id = os.path.splitext(file)[0]
                        
                        # Obtener geometría desde caché
                        geom = _data_manager.get_geometry(farm_id)
                        if geom is None:
                            logging.warning(f"{file} no tiene geometría en caché; se omite.")
                            continue
                        
                        # Crear GeoDataFrame desde geometría en caché
                        gdf = gpd.GeoDataFrame({'farm_id': [farm_id], 'geometry': [geom]}, crs=crs)
                        gdf_list.append(gdf)
                    except Exception as e:
                        logging.warning(f"[{years}] Error obteniendo geometría desde caché {file}: {e}")
            else:
                # Método tradicional: leer desde archivos
                for file in batch:
                    fpath = norm(os.path.join(folder_geojsons, file))
                    try:
                        gdf = gpd.read_file(fpath)
                        # si no tiene CRS o no está en 3116, convertir a 3116
                        if gdf.crs is None:
                            logging.warning(f"{file} no tiene CRS; se omite.")
                            continue
                        if not _crs_eq(gdf.crs.to_string(), crs):
                            gdf = gdf.to_crs(crs)
                        m = re.search(r"(?:[_-])(\d+)\.geojson$", file, re.IGNORECASE) or re.search(r"(\d+)\.geojson$", file, re.IGNORECASE)
                        farm_id = m.group(1) if m else os.path.splitext(file)[0]
                        gdf["farm_id"] = farm_id
                        gdf_list.append(gdf[["farm_id", "geometry"]])
                    except Exception as e:
                        logging.warning(f"[{years}] Error leyendo {file}: {e}")

            if not gdf_list:
                print("    (sin geometrías válidas en este lote)")
                continue

            merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

            # Acumulador de resultados
            results = []

            total_rows = len(merged)
            # Usar tqdm para barra de progreso dentro de cada período
            farm_progress = tqdm(
                enumerate(merged.itertuples(index=False), start=1),
                total=total_rows,
                desc=f"    [{years}] Fincas",
                unit="finca",
                leave=False,
                ncols=100
            )
            for ridx, r in farm_progress:
                row = {"farm_id": r.farm_id, "geometry": r.geometry}

                try:
                    o1 = process_row_option1(row, raster_src_o1, defo_val, use_precise_area=use_precise_area, pixel_divisions=pixel_divisions)
                    results.append(o1)
                except Exception as e:
                    logging.warning(f"[{years}] Error finca {row.get('farm_id')}: {e}")

            # Escritura de resultados
            if results:
                df1 = pd.DataFrame(results)[["id","intersect_deforestation","deforested_ha","deforested_prop","direct_alert"]]
                header_needed = not os.path.exists(out_csv_o1)
                df1.to_csv(out_csv_o1, mode='a', index=False, header=header_needed)

            # Limpieza por lote
            del gdf_list, merged, results
            gc.collect()
            print(f"  ⏱️ Tiempo lote {b_idx}: {time.time()-t_batch:.2f}s")

        # Mensaje final
        print(f"✅ {opt1_tag.upper()} completado → {out_csv_o1}")

        print(f"⏱ Tiempo periodo {years}: {time.time()-start_period:.2f} s")

        # ===================== CIERRE Y LIMPIEZA POR YEARS =====================
        try:
            if raster_src_o1:
                raster_src_o1.close()
        except Exception:
            pass
        gc.collect()

    print(f"\n🎉 Periodos completados: {', '.join(years_list)}")
    total_time = time.time() - start_all
    print(f"⏱ Tiempo total: {total_time:.2f} s")
    
    # Retornar estadísticas
    return {
        'periods_processed': len(years_list),
        'farms_processed': total_files,
        'execution_time': total_time,
        'success': True
    }


# ===================== CLI para ejecución standalone =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cálculo de Alertas Directas de Deforestación",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Leer de variables de entorno si están disponibles (para compatibilidad con main.py)
    env_source = os.getenv("DEFORESTATION_SOURCE")
    env_period = os.getenv("PERIODO")
    env_years = os.getenv("YEARS")
    
    # Argumentos CLI
    parser.add_argument("--source", "-s", default=env_source or "smbyc",
                       help="Fuente de deforestación: smbyc")
    parser.add_argument("--period-type", "-pt", default=env_period or "annual",
                       help="Tipo de período: annual, cumulative, nad, atd")
    parser.add_argument("--years", "-y", default=env_years,
                       help="Períodos separados por coma: 2013-2014,2014-2015 o 202301,202302")
    
    parser.add_argument("--farm-folder", "-f", default=SETTINGS.get("FOLDER_GEOJSONS", ""),
                       help="Carpeta con GeoJSONs de predios")
    parser.add_argument("--raster-template", "-r", default=SETTINGS.get("RASTER_DEFOREST", ""),
                       help="Template del raster con {PERIODO} y {YEARS}")
    parser.add_argument("--output-csv", "-o", default=SETTINGS.get("OUTPUT_CSV", ""),
                       help="Template del CSV de salida")
    
    parser.add_argument("--batch-size", type=int, default=SETTINGS.get("BATCH_SIZE", 1000),
                       help="Tamaño de lote para procesamiento")
    parser.add_argument("--farm-range", default=SETTINGS.get("FARM_FILE_RANGE", ""),
                       help="Rango de archivos: 1:1000")
    parser.add_argument("--crs", default=SETTINGS.get("CRS_METROS", "EPSG:3116"),
                       help="Sistema de coordenadas")
    parser.add_argument("--deforest-value", type=int, default=SETTINGS.get("DEFOREST_VALUE", 2),
                       help="Valor de píxel de deforestación")
    parser.add_argument("--log-level", default=SETTINGS.get("LOG_LEVEL", "WARNING"),
                       help="Nivel de logging: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-file", default=SETTINGS.get("LOG_FILE", "risk_analysis_intersections.log"),
                       help="Archivo de log")
    
    args = parser.parse_args()
    
    # Validar argumentos requeridos
    if not args.years:
        print("❌ Error: --years es requerido (ej: --years 2013-2014,2014-2015)")
        sys.exit(1)
    
    # Parsear years (puede venir separado por comas)
    years_list = [y.strip() for y in args.years.split(",") if y.strip()]
    
    try:
        result = calculate_direct_alerts(
            source=args.source,
            period_type=args.period_type,
            years=years_list,
            farm_folder=args.farm_folder,
            raster_template=args.raster_template,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            farm_range=args.farm_range,
            crs=args.crs,
            deforest_value=args.deforest_value,
            log_level=args.log_level,
            log_file=args.log_file
        )
        print(f"\n✅ Completado exitosamente:")
        print(f"   • Períodos procesados: {result['periods_processed']}")
        print(f"   • Predios procesados: {result['farms_processed']}")
        print(f"   • Tiempo total: {result['execution_time']:.2f}s")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.exception("Error en calculate_direct_alerts")
        sys.exit(1)
