# -*- coding: utf-8 -*-
import os, re, gc, time, logging
from typing import Dict, Any, Tuple, List, Optional, Set
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS as RCRS
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping
from shapely.ops import unary_union, transform as shapely_transform
from shapely.validation import make_valid
from pyproj import Transformer

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
def calculate_deforestation_metrics_from_src(src, geom, deforest_value) -> Tuple[bool, float, float]:
    """
    - src: puede ser dataset o WarpedVRT; si es WarpedVRT, está en CRS_METROS (3116).
    - geom: geometría en CRS_METROS (3116).
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

        out_image, _ = mask(src, [mapping(geom_for_mask)], crop=True, filled=False)
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
def process_row_option1(row, raster_src, deforest_value: int):
    farm_id = row["farm_id"]; geom = row["geometry"]
    inter_def, def_ha, def_prop = calculate_deforestation_metrics_from_src(raster_src, geom, deforest_value)
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

def make_out_dir_and_paths(base_output_template: str, ctx: Dict[str,str], source_tag: str) -> Tuple[str,str]:
    formatted = norm(format_placeholders(base_output_template, ctx))
    base_dir = os.path.dirname(formatted) or "."
    option_map = {"smbyc": "SMBYC", "atd": "ATD", "nad": "NAD", "otra": "OTRA"}
    option_folder = option_map.get(str(source_tag).lower(), str(source_tag).upper())
    years = ctx["YEARS"]
    out_dir = norm(os.path.join(base_dir, option_folder, years))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = norm(os.path.join(out_dir, f"{source_tag}_direct_alert_{ctx['PERIODO']}_{years}.csv"))
    return out_dir, out_csv

def write_reason_log(out_dir: str, fname: str, msg: str):
    path = norm(os.path.join(out_dir, fname))
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(f"📝 Se escribió log: {path}")

# ===================== Runner =====================
def run_in_notebook(options: str = "1", source: str = "smbyc", deforest_value: Optional[int] = None):
    setup_logging()

    chosen: Set[int] = set()
    for tok in options.split(","):
        tok = tok.strip()
        if tok in {"1", "2", "3"}:
            chosen.add(int(tok))
    if not chosen:
        raise SystemExit("Debes elegir al menos una opción: '1', '2', '3' (p.ej. options='1,2')")

    periodo = SETTINGS['PERIODO']
    crs = SETTINGS['CRS_METROS']  # ahora EPSG:3116
    batch_size = SETTINGS['BATCH_SIZE']

    if 1 in chosen:
        if source == "smbyc":
            defo_val = SETTINGS.get("DEFOREST_VALUE", 2)
            opt1_tag = "smbyc"
        else:
            if deforest_value is None:
                raise SystemExit("Para source='other' debes indicar deforest_value=N")
            defo_val = int(deforest_value)
            opt1_tag = "otra"
        print(f"🟩 Opción 1 activa. Fuente: {source} | deforest_value={defo_val}")
    else:
        defo_val = None
        opt1_tag = None

    years_list = parse_year_periods(SETTINGS['YEARS']) or [str(SETTINGS['YEARS']).strip()]

    folder_geojsons = norm(SETTINGS['FOLDER_GEOJSONS'])
    alerts_dir_tpl = SETTINGS.get('ALERTAS_DIR', "")
    nucleos_dir_tpl = SETTINGS.get('NUCLEOS_DIR', "")
    output_tpl = SETTINGS['OUTPUT_CSV']

    if not os.path.isdir(folder_geojsons):
        raise RuntimeError(f"No existe la carpeta de geojsons: {folder_geojsons}")

    files = sorted([f for f in os.listdir(folder_geojsons) if f.lower().endswith(".geojson")])
    total_files_initial = len(files)
    if not files:
        raise RuntimeError(f"No se encontraron GeoJSONs en: {folder_geojsons}")
    print(f"📦 GeoJSONs encontrados: {total_files_initial} en {folder_geojsons}")
    print(f"⚙ FARM_FILE_RANGE='{SETTINGS.get('FARM_FILE_RANGE','')}' | BATCH_SIZE={batch_size}")

    files = apply_file_range(files, SETTINGS.get("FARM_FILE_RANGE", ""))
    total_files = len(files)
    total_batches = (total_files + batch_size - 1) // batch_size
    print(f"🏁 Se procesarán {total_files} archivos en {total_batches} lotes (batch_size={batch_size}).")

    start_all = time.time()
    # ===================== LOOP POR YEARS =====================
    for p_idx, years in enumerate(years_list, start=1):
        print(f"\n=== [{p_idx}/{len(years_list)}] Periodo: {years} — etiqueta periodo: {periodo} ===")
        ctx = {"PERIODO": periodo, "YEARS": years}

        # --- Opción 1: deforestación (abrir UNA VEZ por YEARS)
        if 1 in chosen:
            raster_deforest = norm(format_placeholders(SETTINGS['RASTER_DEFOREST'], ctx))
            print(f"🔎 Validando raster deforestación: {raster_deforest}")
            try:
                ensure_raster_crs(raster_deforest, crs)
                raster_src_o1 = open_raster(raster_deforest, target_crs=crs)  # abierto una sola vez (WarpedVRT si necesario) en 3116
                opt1_available = True
                print_availability_option("Deforestación (raster)", True, f"→ {os.path.basename(raster_deforest)}")
            except Exception as e:
                logging.error(f"[{years}] Opción 1: {e}")
                print("❌ No se encontró/validó el raster de deforestación; se omite opción 1 en este periodo.")
                raster_src_o1 = None
                opt1_available = False
        else:
            raster_src_o1 = None
            opt1_available = False

        # --- Opción 2: alertas tempranas (abrir UNA VEZ por YEARS)
        if 2 in chosen:
            print(f"📂 Cargando bundle de alertas tempranas...")
            alerts_vec, alerts_ras, alerts_sum = load_bundle(alerts_dir_tpl, ctx, crs)
            alerts_ras_ok = _filter_rasters_same_crs(alerts_ras, crs) if alerts_ras else []
            alerts_src_list = open_rasters(alerts_ras_ok, target_crs=crs) if alerts_ras_ok else []
            has2 = (alerts_vec is not None and not alerts_vec.empty) or (len(alerts_src_list) > 0)
            print_availability_option(
                "Alertas tempranas (carpeta)", has2,
                f"| vectores={alerts_sum.get('vec_files',0)} rasters={alerts_sum.get('ras_files',0)} (usables={len(alerts_src_list)})"
            )
            if not has2:
                opt2_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "atd")
                write_reason_log(opt2_dir, f"atd_log_{periodo}_{years}.txt",
                                 f"[{years}] Sin datos de alertas tempranas (no se generó atd_direct_alert_{periodo}_{years}.csv)")
            opt2_available = has2
        else:
            alerts_vec = None
            alerts_src_list = []
            opt2_available = False

        # --- Opción 3: núcleos activos (abrir UNA VEZ por YEARS)
        if 3 in chosen:
            print(f"📂 Cargando bundle de núcleos activos...")
            nucleos_vec, nucleos_ras, nucleos_sum = load_bundle(nucleos_dir_tpl, ctx, crs)
            nucleos_ras_ok = _filter_rasters_same_crs(nucleos_ras, crs) if nucleos_ras else []
            nucleos_src_list = open_rasters(nucleos_ras_ok, target_crs=crs) if nucleos_ras_ok else []
            has3 = (nucleos_vec is not None and not nucleos_vec.empty) or (len(nucleos_src_list) > 0)
            print_availability_option(
                "Núcleos activos (carpeta)", has3,
                f"| vectores={nucleos_sum.get('vec_files',0)} rasters={nucleos_sum.get('ras_files',0)} (usables={len(nucleos_src_list)})"
            )
            if not has3:
                opt3_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "nad")
                write_reason_log(opt3_dir, f"nad_log_{periodo}_{years}.txt",
                                 f"[{years}] Sin datos de núcleos activos (no se generó nad_direct_alert_{periodo}_{years}.csv)")
            opt3_available = has3
        else:
            nucleos_vec = None
            nucleos_src_list = []
            opt3_available = False

        if not any([opt1_available, opt2_available, opt3_available]):
            print("⚠ Ninguna opción disponible para este periodo. Se omite generación de resultados.")
            # Limpieza mínima por si quedaron objetos
            close_rasters([raster_src_o1] if raster_src_o1 else [])
            close_rasters(alerts_src_list)
            close_rasters(nucleos_src_list)
            del alerts_vec, alerts_src_list, nucleos_vec, nucleos_src_list
            gc.collect()
            continue

        # Salidas por opción (se limpian si existen)
        if opt1_available:
            _, out_csv_o1 = make_out_dir_and_paths(output_tpl, ctx, opt1_tag)
            if os.path.exists(out_csv_o1):
                os.remove(out_csv_o1)
            print(f"🗂️ Salida O1: {out_csv_o1}")
        if opt2_available:
            _, out_csv_o2 = make_out_dir_and_paths(output_tpl, ctx, "atd")
            if os.path.exists(out_csv_o2):
                os.remove(out_csv_o2)
            print(f"🗂️ Salida O2: {out_csv_o2}")
        if opt3_available:
            _, out_csv_o3 = make_out_dir_and_paths(output_tpl, ctx, "nad")
            if os.path.exists(out_csv_o3):
                os.remove(out_csv_o3)
            print(f"🗂️ Salida O3: {out_csv_o3}")

        # ===================== LOOP POR LOTES =====================
        start_period = time.time()
        for b_idx, i in enumerate(range(0, total_files, batch_size), start=1):
            t_batch = time.time()
            batch = files[i:i+batch_size]
            print(f"\n  ➤ Lote {b_idx}/{total_batches} | archivos {i+1}-{i+len(batch)} (de {total_files})")

            gdf_list: List[gpd.GeoDataFrame] = []
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
                    farm_id = int(m.group(1)) if m else os.path.splitext(file)[0]
                    gdf["farm_id"] = farm_id
                    gdf_list.append(gdf[["farm_id", "geometry"]])
                except Exception as e:
                    logging.warning(f"[{years}] Error leyendo {file}: {e}")

            if not gdf_list:
                print("    (sin geometrías válidas en este lote)")
                continue

            merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

            # Acumuladores por opción
            results1 = [] if opt1_available else None
            results2 = [] if opt2_available else None
            results3 = [] if opt3_available else None

            total_rows = len(merged)
            for ridx, r in enumerate(merged.itertuples(index=False), start=1):
                row = {"farm_id": r.farm_id, "geometry": r.geometry}

                if opt1_available:
                    try:
                        o1 = process_row_option1(row, raster_src_o1, defo_val)
                        results1.append(o1)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 1 error finca {row.get('farm_id')}: {e}")

                if opt2_available:
                    try:
                        o2, _ = process_row_option2(row, alerts_vec, alerts_src_list, crs)
                        results2.append(o2)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 2 error finca {row.get('farm_id')}: {e}")

                if opt3_available:
                    try:
                        o3, _ = process_row_option3(row, nucleos_vec, nucleos_src_list, crs)
                        results3.append(o3)
                    except Exception as e:
                        logging.warning(f"[{years}] Opción 3 error finca {row.get('farm_id')}: {e}")

                if (ridx % 500 == 0) or (ridx == total_rows):
                    print(f"    - Fincas procesadas: {ridx}/{total_rows}")

            # Escritura por opción (append)
            if opt1_available and results1:
                df1 = pd.DataFrame(results1)[["id","intersect_deforestation","deforested_ha","deforested_prop","direct_alert"]]
                header_needed = not os.path.exists(out_csv_o1)
                df1.to_csv(out_csv_o1, mode='a', index=False, header=header_needed)

            if opt2_available and results2:
                df2 = pd.DataFrame(results2)[["id","intersect_early_warnings","early_warnings_ha","early_warnings_prop","direct_alert"]]
                header_needed = not os.path.exists(out_csv_o2)
                df2.to_csv(out_csv_o2, mode='a', index=False, header=header_needed)

            if opt3_available and results3:
                df3 = pd.DataFrame(results3)[["id","intersect_active_hotspots","active_hotspots_ha","active_hotspots_prop","direct_alert"]]
                header_needed = not os.path.exists(out_csv_o3)
                df3.to_csv(out_csv_o3, mode='a', index=False, header=header_needed)

            # Limpieza por lote
            del gdf_list, merged, results1, results2, results3
            gc.collect()
            print(f"  ⏱️ Tiempo lote {b_idx}: {time.time()-t_batch:.2f}s")

        # Mensajes finales por opción
        if opt1_available:
            print(f"✅ Opción 1 ({opt1_tag}) completada → {out_csv_o1}")
        if opt2_available:
            if os.path.exists(out_csv_o2):
                print(f"✅ Opción 2 (ATD) completada → {out_csv_o2}")
            else:
                opt2_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "atd")
                print(f"⚠ Opción 2: no se generó CSV. Revisa el log en {opt2_dir}")
        if opt3_available:
            if os.path.exists(out_csv_o3):
                print(f"✅ Opción 3 (NAD) completada → {out_csv_o3}")
            else:
                opt3_dir, _ = make_out_dir_and_paths(output_tpl, ctx, "nad")
                print(f"⚠ Opción 3: no se generó CSV. Revisa el log en {opt3_dir}")

        print(f"⏱ Tiempo periodo {years}: {time.time()-start_period:.2f} s")

        # ===================== CIERRE Y LIMPIEZA POR YEARS =====================
        try:
            if raster_src_o1:
                raster_src_o1.close()
            close_rasters(alerts_src_list)
            close_rasters(nucleos_src_list)
        except Exception:
            pass

        del alerts_vec, alerts_src_list, nucleos_vec, nucleos_src_list
        gc.collect()

    print(f"\n🎉 Periodos completados: {', '.join(years_list)}")
    print(f"⏱ Tiempo total: {time.time()-start_all:.2f} s")
