# Filename: etl_ganabosques2.py
# Description:
#   Ganabosques ETL with explicit CRS management and Shapefile handling:
#     - Extract via ganabosques_orm (raw / no_dereference / batch).
#     - Download 1 raster per deforestation record (WCS by target CRS).
#     - Download farming/protected as ESRI Shapefiles (.zip or .shp URLs).
#     - Reproject all geodata to --target-crs (default EPSG:3116).
#     - Run alert_direct → alert_indirect → calculate_alert per deforestation raster.
#
# Author: CIAT-DAPA

from __future__ import annotations
import argparse
import json
import os
import shutil
import zipfile
from typing import Dict, List, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import requests

# ORM
from ganabosques_orm.base import init_db
from ganabosques_orm.collections.farm import Farm
from ganabosques_orm.collections.farmpolygons import FarmPolygons
from ganabosques_orm.collections.deforestation import Deforestation
from ganabosques_orm.collections.farmingareas import FarmingAreas
from ganabosques_orm.collections.protectedareas import ProtectedAreas
from ganabosques_orm.collections.movement import Movement
from ganabosques_orm.collections.adm3 import Adm3
from ganabosques_orm.collections.enterprise import Enterprise

# Risk package
from ganabosques_risk_package.plot_alert_direct import alert_direct
from ganabosques_risk_package.plot_alert_indirect import alert_indirect
from ganabosques_risk_package.entity_alert import calculate_alert


# ============================================================================================
# PATHS
# ============================================================================================
BASE_DIR = "D:\\CIAT\\Code\\BID\\ganabosques_risk_calculator\\data"
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GEOSERVER_DIR = os.path.join(INPUTS_DIR, "geoserver")
CSV_DIR = os.path.join(INPUTS_DIR, "csv")
DEFOR_DIR = os.path.join(GEOSERVER_DIR, "deforestation")
MOVEMENT_DIR = os.path.join(CSV_DIR, "movement_by_year")

PLOTS_CSV = os.path.join(CSV_DIR, "plots.csv")
DEFORESTATION_CSV = os.path.join(CSV_DIR, "deforestation.csv")
FARMING_AREAS_CSV = os.path.join(CSV_DIR, "farmingareas.csv")
PROTECTED_AREAS_CSV = os.path.join(CSV_DIR, "protectedareas.csv")
ADM3_CSV = os.path.join(CSV_DIR, "adm3.csv")
ENTERPRISE_CSV = os.path.join(CSV_DIR, "enterprise.csv")

# Output shapefiles (reprojected) to feed the pipeline
FARMING_SHP_OUT = os.path.join(GEOSERVER_DIR, "farmingareas_target.shp")
PROTECTED_SHP_OUT = os.path.join(GEOSERVER_DIR, "protectedareas_target.shp")


# ============================================================================================
# UTILITIES
# ============================================================================================

def ensure_folders(from_scratch: bool = False):
    print("\n[SETUP] Preparing folder structure...")
    if from_scratch and os.path.exists(INPUTS_DIR):
        print(f"[SETUP] Cleaning existing inputs at {INPUTS_DIR}")
        shutil.rmtree(INPUTS_DIR)

    for folder in [INPUTS_DIR, CSV_DIR, GEOSERVER_DIR, DEFOR_DIR, OUTPUTS_DIR, MOVEMENT_DIR]:
        os.makedirs(folder, exist_ok=True)
        print(f"[SETUP] Folder ready: {folder}")


def _safe_geojson_to_geom(geojson_str: str):
    if not geojson_str:
        return None
    try:
        gj = json.loads(geojson_str)
        t = gj.get("type")
        if t in {"Polygon", "MultiPolygon"}:
            return shape(gj)
        if t == "Feature":
            return shape(gj["geometry"])
        if t == "FeatureCollection":
            feats = gj.get("features") or []
            if feats:
                return shape(feats[0]["geometry"])
    except Exception:
        return None
    return None


def _count_q(q):
    """Safe count with MongoEngine queryset (for tqdm totals)."""
    try:
        return q.count()
    except Exception:
        return None


def _download_file(url: str, out_path: str):
    """Download file with streaming."""
    print(f"[DOWNLOAD] URL → {url}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):
                if chunk:
                    f.write(chunk)
    print(f"[DOWNLOAD] Saved: {out_path}")
    return out_path


def _name_from_url(url: str, default_name: str) -> str:
    if not url:
        return default_name
    return url.split("?")[0].split("/")[-1] or default_name


def _output_dir_for_deforestation(deforestation_id: str) -> str:
    path = os.path.join(OUTPUTS_DIR, "deforestation", deforestation_id)
    os.makedirs(path, exist_ok=True)
    return path


def _delete_shapefile_set(base_path_no_ext: str):
    """Remove a shapefile set by basename (e.g. /path/layer.*)."""
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]
    for ext in exts:
        p = base_path_no_ext + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


# ============================================================================================
# EXTRACT STAGE (Approach A)
# ============================================================================================

def extract_all_with_orm() -> gpd.GeoDataFrame:
    """
    EXTRACT (Optimized A):
      - Usa .no_dereference().as_pymongo().batch_size() para evitar hidratación/desreferencia.
      - Conversiones ObjectId->str y fechas se hacen de forma vectorizada cuando es posible.
      - Selección del último polígono por farm usando comparación por _id (ObjectId es temporal).
      - Guarda CSVs en inputs/csv y devuelve un GeoDataFrame listo para alert_direct.
    """
    print("\n[EXTRACT] Connecting via ganabosques_orm.init_db() ...")
    init_db()

    # Farms map
    print("[EXTRACT] Loading Farm (id → adm3_id) ...")
    farm_q = (Farm.objects.only("id", "adm3_id").no_dereference().as_pymongo().batch_size(10000))
    farm_total = _count_q(Farm.objects)
    farms_map = {}
    for doc in tqdm(farm_q, total=farm_total, desc="Farms"):
        _id = doc.get("_id")
        farms_map[str(_id)] = str(doc.get("adm3_id")) if doc.get("adm3_id") else None

    # Latest FarmPolygons per farm
    print("[EXTRACT] Selecting latest FarmPolygons per farm ...")
    fp_q = (FarmPolygons.objects.only("id", "farm_id", "geojson").no_dereference().as_pymongo().batch_size(10000))
    fp_total = _count_q(FarmPolygons.objects)
    latest_by_farm = {} # farm_id(str) -> raw_doc
    for doc in tqdm(fp_q, total=fp_total, desc="FarmPolygons"):
        farm_id = doc.get("farm_id")
        if not farm_id:
            continue
        farm_id = str(farm_id)
        prev = latest_by_farm.get(farm_id)
        if (prev is None) or (doc["_id"] > prev["_id"]):  # ObjectId monotonic
            latest_by_farm[farm_id] = doc

    # Build plots
    print("[EXTRACT] Building plot rows (id, farm_id, adm3_id, geometry WKT) ...")
    plot_rows = []
    for farm_id, poly in tqdm(list(latest_by_farm.items()), desc="Build plots"):
        geom = _safe_geojson_to_geom(poly.get("geojson"))
        if geom is None:
            continue
        plot_rows.append({
            "id": str(poly["_id"]),
            "farm_id": farm_id,
            "adm3_id": farms_map.get(farm_id),
            "geometry": geom.wkt
        })
    plots_df = pd.DataFrame(plot_rows)
    plots_df.to_csv(PLOTS_CSV, index=False)
    print(f"[EXTRACT] Saved plots → {PLOTS_CSV} ({len(plots_df)} rows)")

    # Deforestation
    def_q = (Deforestation.objects.only("id", "name", "period_start", "period_end", "path")
             .no_dereference().as_pymongo().batch_size(10000))
    def_total = _count_q(Deforestation.objects)
    def_rows = []
    for doc in tqdm(def_q, total=def_total, desc="Deforestation"):
        def_rows.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
            "period_start": doc.get("period_start"),
            "period_end": doc.get("period_end"),
            "path": doc.get("path"),
        })
    def_df = pd.DataFrame.from_records(def_rows)
    if not def_df.empty:
        def_df["period_start"] = pd.to_datetime(def_df["period_start"], errors="coerce")
        def_df["period_end"] = pd.to_datetime(def_df["period_end"], errors="coerce")
    def_df.to_csv(DEFORESTATION_CSV, index=False)
    print(f"[EXTRACT] Saved deforestation → {DEFORESTATION_CSV} ({len(def_df)} rows)")

    # FarmingAreas
    print("[EXTRACT] FarmingAreas (raw) ...")
    fa_q = (FarmingAreas.objects.only("id", "name", "path").no_dereference().as_pymongo().batch_size(10000))
    fa_total = _count_q(FarmingAreas.objects)
    fa_rows = []
    for doc in tqdm(fa_q, total=fa_total, desc="FarmingAreas"):
        fa_rows.append({"id": str(doc.get("_id")), "name": doc.get("name"), "path": doc.get("path")})
    pd.DataFrame.from_records(fa_rows).to_csv(FARMING_AREAS_CSV, index=False)
    print(f"[EXTRACT] Saved farmingareas → {FARMING_AREAS_CSV} ({len(fa_rows)} rows)")

    # ProtectedAreas
    print("[EXTRACT] ProtectedAreas (raw) ...")
    pa_q = (ProtectedAreas.objects.only("id", "name", "path").no_dereference().as_pymongo().batch_size(10000))
    pa_total = _count_q(ProtectedAreas.objects)
    pa_rows = []
    for doc in tqdm(pa_q, total=pa_total, desc="ProtectedAreas"):
        pa_rows.append({"id": str(doc.get("_id")), "name": doc.get("name"), "path": doc.get("path")})
    pd.DataFrame.from_records(pa_rows).to_csv(PROTECTED_AREAS_CSV, index=False)
    print(f"[EXTRACT] Saved protectedareas → {PROTECTED_AREAS_CSV} ({len(pa_rows)} rows)")

    # Adm3
    print("[EXTRACT] Adm3 (raw) ...")
    adm_q = (Adm3.objects.only("id", "name").no_dereference().as_pymongo().batch_size(10000))
    adm_total = _count_q(Adm3.objects)
    adm_rows = []
    for doc in tqdm(adm_q, total=adm_total, desc="Adm3"):
        adm_rows.append({"id": str(doc.get("_id")), "name": doc.get("name")})
    pd.DataFrame.from_records(adm_rows).to_csv(ADM3_CSV, index=False)
    print(f"[EXTRACT] Saved adm3 → {ADM3_CSV} ({len(adm_rows)} rows)")

    # Enterprise
    print("[EXTRACT] Enterprise (raw) ...")
    ent_q = (Enterprise.objects.only("id", "name").no_dereference().as_pymongo().batch_size(10000))
    ent_total = _count_q(Enterprise.objects)
    ent_rows = []
    for doc in tqdm(ent_q, total=ent_total, desc="Enterprise"):
        ent_rows.append({"id": str(doc.get("_id")), "name": doc.get("name")})
    pd.DataFrame.from_records(ent_rows).to_csv(ENTERPRISE_CSV, index=False)
    print(f"[EXTRACT] Saved enterprise → {ENTERPRISE_CSV} ({len(ent_rows)} rows)")

    # Movement by year
    print("[EXTRACT] Movement (raw) ...")
    mov_q = (Movement.objects.only("id", "date", "farm_id_origin", "farm_id_destination")
             .no_dereference().as_pymongo().batch_size(10000))
    mov_total = _count_q(Movement.objects)
    mov_rows = []
    for doc in tqdm(mov_q, total=mov_total, desc="Movements"):
        mov_rows.append({
            "id": str(doc.get("_id")),
            "date": doc.get("date"),
            "origen_id": str(doc.get("farm_id_origin")) if doc.get("farm_id_origin") else None,
            "destination_id": str(doc.get("farm_id_destination")) if doc.get("farm_id_destination") else None,
        })
    df_mov = pd.DataFrame.from_records(mov_rows)
    if not df_mov.empty:
        df_mov["date"] = pd.to_datetime(df_mov["date"], errors="coerce")
        df_mov["year"] = df_mov["date"].dt.year
        # Guardar por año con tqdm
        for year, dfx in tqdm(df_mov.groupby("year"), desc="Saving movement by year"):
            dfx.drop(columns=["year"]).to_csv(os.path.join(MOVEMENT_DIR, f"movement_{int(year)}.csv"), index=False)
            #print(f"[EXTRACT] Movement → {os.path.join(MOVEMENT_DIR, f'movement_{int(year)}.csv')}")
    else:
        print("[EXTRACT] No movement rows found.")

    # Return plots as GDF (CRS se asigna y reproyecta luego)
    gdf = gpd.GeoDataFrame(
        plots_df[["id"]].copy(),
        geometry=gpd.GeoSeries.from_wkt(plots_df["geometry"]),
        crs=None
    )
    return gdf


# ============================================================================================
# DOWNLOAD RASTERS (WCS por CRS)
# ============================================================================================

def _wcs_url_for_crs(base_layer_path: str, period_start, period_end, target_crs: str) -> str:
    layer_code = base_layer_path.replace("/", ":").rstrip(":")
    t0 = pd.to_datetime(period_start, errors="coerce")
    t1 = pd.to_datetime(period_end, errors="coerce")
    time_param = f"{t0.year}-{t1.year}" if pd.notna(t0) and pd.notna(t1) else ""

    crs = target_crs.upper()
    if crs == "EPSG:3116":
        bbox = "100000,800000,1200000,1800000"; width = "4096"; height = "2048"
    elif crs == "EPSG:4326":
        bbox = "-180,-90,180,90"; width = "4096"; height = "2048"
    else:
        print(f"[WARN] CRS {target_crs} not explicitly supported for WCS BBOX; fallback to EPSG:4326.")
        crs = "EPSG:4326"; bbox = "-180,-90,180,90"; width = "4096"; height = "2048"

    url = (
        "https://ganageo.alliance.cgiar.org/geoserver/deforestation/wcs?"
        "service=WCS&version=1.0.0&request=GetCoverage"
        f"&coverage={layer_code}"
        f"&format=GeoTIFF"
        f"{'&time=' + time_param if time_param else ''}"
        f"&CRS={crs}&BBOX={bbox}&WIDTH={width}&HEIGHT={height}"
    )
    return url


def download_all_deforestation_layers(target_crs: str) -> List[Dict]:
    print("\n[DOWNLOAD] Downloading all deforestation rasters ...")
    df = pd.read_csv(DEFORESTATION_CSV)
    items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Deforestation rasters"):
        defo_id = str(row["id"])
        base_path = str(row.get("path", "")).strip()
        if not base_path:
            print(f"[SKIP] Deforestation {defo_id} has empty path.")
            continue
        url = _wcs_url_for_crs(base_path, row.get("period_start"), row.get("period_end"), target_crs)
        out_path = os.path.join(DEFOR_DIR, f"{defo_id}.tif")
        _download_file(url, out_path)
        items.append({
            "id": defo_id,
            "name": row.get("name"),
            "period_start": pd.to_datetime(row.get("period_start"), errors="coerce"),
            "period_end": pd.to_datetime(row.get("period_end"), errors="coerce"),
            "raster_path": out_path,
            "crs": target_crs
        })
    print(f"[DOWNLOAD] Total deforestation files downloaded: {len(items)}")
    return items


# ============================================================================================
# DOWNLOAD + REPROJECT SHAPEFILES (Farming & Protected)
# ============================================================================================

def _find_first_shp(root_dir: str) -> Optional[str]:
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".shp"):
                return os.path.join(root, fn)
    return None


def _try_download_sidecars(base_url: str, out_dir: str, base_name_no_ext: str):
    """If URL is a .shp, attempt downloading sidecar files from same URL base path."""
    side_exts = [".shx", ".dbf", ".prj", ".cpg"]
    for ext in side_exts:
        side_url = base_url[:-4] + ext  # replace .shp
        out_path = os.path.join(out_dir, base_name_no_ext + ext)
        try:
            _download_file(side_url, out_path)
        except Exception:
            # Not all servers expose sidecars individually; ignore failures.
            pass


def download_shapefile(csv_path: str, label: str) -> str:
    """
    Download shapefile for given label:
      - If URL is .zip → unzip; return first .shp path inside.
      - If URL is .shp → download .shp and try sidecars; return .shp path.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"[ERROR] {label} CSV is empty.")
    url = str(df.iloc[0]["path"]).strip()
    if not url:
        raise RuntimeError(f"[ERROR] {label} path is empty.")

    if url.lower().endswith(".zip"):
        zip_path = os.path.join(GEOSERVER_DIR, f"{label}.zip")
        print(f"[DOWNLOAD] {label} (ZIP) → {zip_path}")
        _download_file(url, zip_path)
        extract_dir = os.path.join(GEOSERVER_DIR, f"{label}_unzipped")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"[UNZIP] {label} extracted → {extract_dir}")
        shp = _find_first_shp(extract_dir)
        if not shp:
            raise RuntimeError(f"No .shp found in {extract_dir} for {label}.")
        return shp

    # Direct .shp
    if url.lower().endswith(".shp"):
        shp_name = _name_from_url(url, f"{label}.shp")
        shp_out = os.path.join(GEOSERVER_DIR, shp_name)
        print(f"[DOWNLOAD] {label} (SHP) → {shp_out}")
        _download_file(url, shp_out)
        base_no_ext = os.path.splitext(os.path.basename(shp_out))[0]
        _try_download_sidecars(url, GEOSERVER_DIR, base_no_ext)  # silently best-effort
        return shp_out

    raise RuntimeError(f"[ERROR] {label} URL must be .zip or .shp, got: {url}")


def reproject_shapefile(src_shp: str, dst_shp: str, target_crs: str):
    """
    Read a shapefile (or dir with shapefiles), reproject to target CRS, and save to dst_shp.
    If multiple shapefiles exist in src folder (ZIP case), they are concatenated and then written as one.
    """
    print(f"[CRS] Reprojecting {src_shp} → {dst_shp} (CRS={target_crs})")
    # If src_shp is a file, load it; if it's a dir (defensive), merge all.
    to_concat = []
    if os.path.isdir(src_shp):
        for root, _, files in os.walk(src_shp):
            for fn in files:
                if fn.lower().endswith(".shp"):
                    to_concat.append(gpd.read_file(os.path.join(root, fn)))
    else:
        to_concat.append(gpd.read_file(src_shp))

    g = gpd.GeoDataFrame(pd.concat(to_concat, ignore_index=True), crs=to_concat[0].crs if to_concat else None)
    if g.crs is None:
        print(f"[WARN] Source shapefile has no CRS; assigning {target_crs}.")
        g.set_crs(target_crs, inplace=True)
    else:
        g = g.to_crs(target_crs)

    # Ensure clean overwrite of shapefile set
    _delete_shapefile_set(os.path.splitext(dst_shp)[0])
    g.to_file(dst_shp, driver="ESRI Shapefile")
    print(f"[CRS] Wrote {dst_shp} with {len(g)} features; crs={g.crs}")


# ============================================================================================
# CRS FOR PLOTS
# ============================================================================================

def reproject_plots_gdf(plots_gdf: gpd.GeoDataFrame, plots_src_crs: str, target_crs: str) -> gpd.GeoDataFrame:
    print(f"[CRS] Plots: assigning src {plots_src_crs} and reprojecting to {target_crs}")
    g = plots_gdf.copy()
    g.set_crs(plots_src_crs, inplace=True, allow_override=True)
    return g.to_crs(target_crs)


# ============================================================================================
# ALERT PIPELINE
# ============================================================================================

def run_alerts_for_deforestation(plots_gdf_target_crs: gpd.GeoDataFrame,
                                 defo_item: Dict,
                                 farming_shp_target: str,
                                 protected_shp_target: str,
                                 n_workers: int):
    defo_id = defo_item["id"]
    print(f"\n[ALERT] Running pipeline for deforestation ID={defo_id}")
    out_dir = _output_dir_for_deforestation(defo_id)

    alert_direct_csv = os.path.join(out_dir, "alert_direct.csv")
    alert_indirect_csv = os.path.join(out_dir, "alert_indirect.csv")
    entity_alert_csv = os.path.join(out_dir, "entity_alert_adm3.csv")

    # Step 1: Direct
    print("[ALERT] Step 1: alert_direct() ...")
    df_direct = alert_direct(
        plots=plots_gdf_target_crs,
        deforestation=defo_item["raster_path"],
        protected_areas=protected_shp_target,
        farming_areas=farming_shp_target,
        deforestation_value=2,
        n_workers=n_workers
    )
    df_direct.to_csv(alert_direct_csv, index=False)
    print(f"[ALERT] alert_direct → {alert_direct_csv}")

    # Step 2: Indirect
    print("[ALERT] Step 2: alert_indirect() ...")
    movement_files = [os.path.join(MOVEMENT_DIR, f) for f in os.listdir(MOVEMENT_DIR) if f.endswith(".csv")]
    movement_parts = []
    for f in tqdm(movement_files, desc="Loading movement files"):
        dfm = pd.read_csv(f)
        if {"origen_id", "destination_id"}.issubset(dfm.columns):
            movement_parts.append(dfm[["origen_id", "destination_id"]])
    movement_df = pd.concat(movement_parts) if movement_parts else pd.DataFrame(columns=["origen_id", "destination_id"])
    df_indirect = alert_indirect(df_direct, movement_df, n_workers)
    df_indirect.to_csv(alert_indirect_csv, index=False)
    print(f"[ALERT] alert_indirect → {alert_indirect_csv}")

    # Step 3: ADM3 aggregation
    print("[ALERT] Step 3: calculate_alert() for ADM3 ...")
    adm3_df = pd.read_csv(ADM3_CSV)[["id", "name"]].rename(columns={"id": "entity_id", "name": "entity_name"})
    plots_raw = pd.read_csv(PLOTS_CSV)
    provider_df = plots_raw[["id", "adm3_id"]].dropna().rename(columns={"id": "plot_id", "adm3_id": "entity_id"})
    df_entity = calculate_alert(df_indirect, adm3_df, provider_df, n_workers)
    df_entity.to_csv(entity_alert_csv, index=False)
    print(f"[ALERT] calculate_alert → {entity_alert_csv}")


# ============================================================================================
# MAIN
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description="Ganabosques ETL (Shapefiles) with explicit CRS management")
    parser.add_argument("--from-scratch", action="store_true", help="Rebuild inputs (extract + download)")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--target-crs", type=str, default="EPSG:3116", help="Target CRS for all geodata")
    parser.add_argument("--plots-src-crs", type=str, default="EPSG:4326",
                        help="Source CRS for plots geometries parsed from GeoJSON")
    args = parser.parse_args()

    print("=" * 80)
    print("[START] Ganabosques ETL pipeline (Shapefiles)")
    print(f"[ARGS] from_scratch={args.from_scratch}, n_workers={args.n_workers}, "
          f"target_crs={args.target_crs}, plots_src_crs={args.plots_src_crs}")
    print("=" * 80)

    ensure_folders(args.from_scratch)

    if args.from_scratch:
        print("[FLOW] Full ETL from scratch")
        plots_gdf = extract_all_with_orm()
        defo_items = download_all_deforestation_layers(args.target_crs)

        # Download + reproject SHPs
        farming_src = download_shapefile(FARMING_AREAS_CSV, "farmingareas")
        protected_src = download_shapefile(PROTECTED_AREAS_CSV, "protectedareas")
        reproject_shapefile(farming_src, FARMING_SHP_OUT, args.target_crs)
        reproject_shapefile(protected_src, PROTECTED_SHP_OUT, args.target_crs)
    else:
        print("[FLOW] Using existing local inputs")
        # defo list from files
        defo_items = [{"id": f.rsplit(".", 1)[0], "raster_path": os.path.join(DEFOR_DIR, f), "crs": args.target_crs}
                      for f in os.listdir(DEFOR_DIR) if f.endswith(".tif")]
        if not defo_items:
            raise RuntimeError("No deforestation rasters found; run with --from-scratch first.")
        # shapefiles already reprojected?
        if not (os.path.exists(FARMING_SHP_OUT) and os.path.exists(PROTECTED_SHP_OUT)):
            raise RuntimeError("Target shapefiles not found; run with --from-scratch to build them.")
        # load plots CSV into GDF
        df_plots = pd.read_csv(PLOTS_CSV)
        plots_gdf = gpd.GeoDataFrame(df_plots[["id"]], geometry=gpd.GeoSeries.from_wkt(df_plots["geometry"]), crs=None)

    # Reproject plots to target CRS
    plots_target = reproject_plots_gdf(plots_gdf, args.plots_src_crs, args.target_crs)

    print(f"[FLOW] Found {len(defo_items)} deforestation rasters to process.")
    for item in tqdm(defo_items, desc="Processing deforestation rasters"):
        run_alerts_for_deforestation(
            plots_gdf_target_crs=plots_target,
            defo_item=item,
            farming_shp_target=FARMING_SHP_OUT,
            protected_shp_target=PROTECTED_SHP_OUT,
            n_workers=args.n_workers
        )

    print("\n[END] ETL completed successfully.")
    print(f"[RESULTS] All outputs saved in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
