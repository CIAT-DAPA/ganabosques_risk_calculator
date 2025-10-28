# Filename: etl_ganabosques.py
# Description:
#   Ganabosques ETL Process:
#     1. Connects to MongoDB via ganabosques_orm.
#     2. Extracts data into ./inputs (CSV + GeoJSON files).
#     3. Downloads GeoServer resources (one raster per deforestation record).
#     4. Runs alert_direct → alert_indirect → calculate_alert for each raster.
#     5. Saves outputs organized by deforestation id.
#
#   Includes tqdm progress bars and detailed console tracing.
#
# Author: CIAT-DAPA

from __future__ import annotations
import argparse
import json
import os
import shutil
from typing import Dict, List
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import requests
import zipfile

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
    """Parse GeoJSON safely to shapely geometry (Polygon/MultiPolygon/Feature/FeatureCollection)."""
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):
                if chunk:
                    f.write(chunk)
    return out_path


def _output_dir_for_deforestation(deforestation_id: str) -> str:
    path = os.path.join(OUTPUTS_DIR, "deforestation", deforestation_id)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================================
# EXTRACT STAGE
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

    # -----------------------------
    # Farms (id -> adm3_id map)
    # -----------------------------
    print("[EXTRACT] Loading Farm (id → adm3_id) as raw docs ...")
    farm_q = (Farm.objects.only("id", "adm3_id")
              .no_dereference()
              .as_pymongo()
              .batch_size(10000))
    farm_total = _count_q(Farm.objects)
    farms_map = {}
    for doc in tqdm(farm_q, total=farm_total, desc="Farms"):
        _id = doc.get("_id")
        adm3 = doc.get("adm3_id")
        farms_map[str(_id)] = str(adm3) if adm3 else None

    # -------------------------------------------
    # FarmPolygons (latest per farm) as raw docs
    # -------------------------------------------
    print("[EXTRACT] Selecting latest FarmPolygons per farm (raw) ...")
    fp_q = (FarmPolygons.objects.only("id", "farm_id", "geojson")
            .no_dereference()
            .as_pymongo()
            .batch_size(10000))
    fp_total = _count_q(FarmPolygons.objects)

    latest_by_farm = {}  # farm_id(str) -> raw_doc
    for doc in tqdm(fp_q, total=fp_total, desc="FarmPolygons"):
        farm_id = doc.get("farm_id")
        if not farm_id:
            continue
        farm_id = str(farm_id)
        prev = latest_by_farm.get(farm_id)
        # ObjectId is monotonic by time → mayor _id == más reciente
        if (prev is None) or (doc["_id"] > prev["_id"]):
            latest_by_farm[farm_id] = doc

    # Build plots rows
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

    # -------------------------------------------
    # Deforestation (raw)
    # -------------------------------------------
    print("[EXTRACT] Deforestation (raw) ...")
    def_q = (Deforestation.objects.only("id", "name", "period_start", "period_end", "path")
             .no_dereference()
             .as_pymongo()
             .batch_size(10000))
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
    # Fechas vectorizadas
    if not def_df.empty:
        def_df["period_start"] = pd.to_datetime(def_df["period_start"], errors="coerce")
        def_df["period_end"] = pd.to_datetime(def_df["period_end"], errors="coerce")
    def_df.to_csv(DEFORESTATION_CSV, index=False)
    print(f"[EXTRACT] Saved deforestation → {DEFORESTATION_CSV} ({len(def_df)} rows)")

    # -------------------------------------------
    # FarmingAreas (raw)
    # -------------------------------------------
    print("[EXTRACT] FarmingAreas (raw) ...")
    fa_q = (FarmingAreas.objects.only("id", "name", "path")
            .no_dereference()
            .as_pymongo()
            .batch_size(10000))
    fa_total = _count_q(FarmingAreas.objects)

    fa_rows = []
    for doc in tqdm(fa_q, total=fa_total, desc="FarmingAreas"):
        fa_rows.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
            "path": doc.get("path"),
        })
    fa_df = pd.DataFrame.from_records(fa_rows)
    fa_df.to_csv(FARMING_AREAS_CSV, index=False)
    print(f"[EXTRACT] Saved farmingareas → {FARMING_AREAS_CSV} ({len(fa_df)} rows)")

    # -------------------------------------------
    # ProtectedAreas (raw)
    # -------------------------------------------
    print("[EXTRACT] ProtectedAreas (raw) ...")
    pa_q = (ProtectedAreas.objects.only("id", "name", "path")
            .no_dereference()
            .as_pymongo()
            .batch_size(10000))
    pa_total = _count_q(ProtectedAreas.objects)

    pa_rows = []
    for doc in tqdm(pa_q, total=pa_total, desc="ProtectedAreas"):
        pa_rows.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
            "path": doc.get("path"),
        })
    pa_df = pd.DataFrame.from_records(pa_rows)
    pa_df.to_csv(PROTECTED_AREAS_CSV, index=False)
    print(f"[EXTRACT] Saved protectedareas → {PROTECTED_AREAS_CSV} ({len(pa_df)} rows)")

    # -------------------------------------------
    # Adm3 (raw)
    # -------------------------------------------
    print("[EXTRACT] Adm3 (raw) ...")
    adm_q = (Adm3.objects.only("id", "name")
             .no_dereference()
             .as_pymongo()
             .batch_size(10000))
    adm_total = _count_q(Adm3.objects)

    adm_rows = []
    for doc in tqdm(adm_q, total=adm_total, desc="Adm3"):
        adm_rows.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
        })
    adm_df = pd.DataFrame.from_records(adm_rows)
    adm_df.to_csv(ADM3_CSV, index=False)
    print(f"[EXTRACT] Saved adm3 → {ADM3_CSV} ({len(adm_df)} rows)")

    # -------------------------------------------
    # Enterprise (raw)
    # -------------------------------------------
    print("[EXTRACT] Enterprise (raw) ...")
    ent_q = (Enterprise.objects.only("id", "name")
             .no_dereference()
             .as_pymongo()
             .batch_size(10000))
    ent_total = _count_q(Enterprise.objects)

    ent_rows = []
    for doc in tqdm(ent_q, total=ent_total, desc="Enterprise"):
        ent_rows.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
        })
    ent_df = pd.DataFrame.from_records(ent_rows)
    ent_df.to_csv(ENTERPRISE_CSV, index=False)
    print(f"[EXTRACT] Saved enterprise → {ENTERPRISE_CSV} ({len(ent_df)} rows)")

    # -------------------------------------------
    # Movement (raw) — Enfoque A
    # -------------------------------------------
    print("[EXTRACT] Movement (raw) ...")
    mov_q = (Movement.objects.only("id", "date", "farm_id_origin", "farm_id_destination")
             .no_dereference()
             .as_pymongo()
             .batch_size(10000))
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
            # opcional: print(f"[EXTRACT] Movement → {os.path.join(MOVEMENT_DIR, f'movement_{int(year)}.csv')}")
    else:
        print("[EXTRACT] No movement rows found.")

    # -------------------------------------------
    # Return plots GeoDataFrame for alert_direct
    # -------------------------------------------
    gdf = gpd.GeoDataFrame(
        plots_df[["id"]].copy(),
        geometry=gpd.GeoSeries.from_wkt(plots_df["geometry"]),
        crs=None
    )
    return gdf

# ============================================================================================
# DOWNLOAD STAGE
# ============================================================================================

def download_all_deforestation_layers() -> List[Dict]:
    """Download each deforestation raster and return list of metadata."""
    print("\n[DOWNLOAD] Downloading all deforestation rasters ...")
    df = pd.read_csv(DEFORESTATION_CSV)
    items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Deforestation rasters"):
        defo_id = str(row["id"])
        url = str(row.get("path", "")).strip()
        if not url:
            continue
        #filename = _name_from_url(url, f"{defo_id}.tif")
        filename = f"{defo_id}.tif"
        out_path = os.path.join(DEFOR_DIR, filename)
        time = str(pd.to_datetime(row.get("period_start"), errors="coerce").year) + "-" + str(pd.to_datetime(row.get("period_end"), errors="coerce").year)
        # CRS 3116 = &CRS=EPSG:3116&BBOX=100000,800000,1200000,1800000
        # CRS 4326 = &CRS=EPSG:4326&BBOX=-180,-90,180,90
        url_final = f"https://ganageo.alliance.cgiar.org/geoserver/deforestation/wcs?service=WCS&version=1.0.0&request=GetCoverage&coverage={url.replace("/",":")[:-1]}&format=GeoTIFF&time={time}&CRS=EPSG:3116&BBOX=100000,800000,1200000,1800000&WIDTH=4096&HEIGHT=2048"
        #print(url_final)
        #_download_file(url, out_path)
        _download_file(url_final, out_path)
        items.append({
            "id": defo_id,
            "name": row.get("name"),
            "period_start": pd.to_datetime(row.get("period_start"), errors="coerce"),
            "period_end": pd.to_datetime(row.get("period_end"), errors="coerce"),
            "raster_path": out_path
        })
    print(f"[DOWNLOAD] Total deforestation files downloaded: {len(items)}")
    return items


def download_vector_layer(csv_path: str, label: str) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"[ERROR] {label} CSV is empty.")
    url = str(df.iloc[0]["path"]).strip()
    #filename = _name_from_url(url, f"{label}.zip")
    filename = f"{label}.zip"
    out_path = os.path.join(GEOSERVER_DIR, filename)
    print(f"[DOWNLOAD] Downloading {label} → {out_path}")
    _download_file(url, out_path)
    #return out_path
    # ✅ Descomprimir automáticamente el ZIP descargado
    extract_dir = os.path.join(GEOSERVER_DIR, label)
    try:
        with zipfile.ZipFile(out_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[UNZIP] Extracted {label} → {extract_dir}")
    except zipfile.BadZipFile:
        print(f"[ERROR] {label} is not a valid ZIP file or is corrupted.")
        raise

    return extract_dir


# ============================================================================================
# ALERT PIPELINE
# ============================================================================================

def run_alerts_for_deforestation(plots_gdf, defo_item, farming_path, protected_path, n_workers):
    defo_id = defo_item["id"]
    print(f"\n[ALERT] Running pipeline for deforestation ID={defo_id}")
    out_dir = _output_dir_for_deforestation(defo_id)

    alert_direct_csv = os.path.join(out_dir, "alert_direct.csv")
    alert_indirect_csv = os.path.join(out_dir, "alert_indirect.csv")
    entity_alert_csv = os.path.join(out_dir, "entity_alert_adm3.csv")

    # Step 1: Direct alerts
    print("[ALERT] Step 1: Running alert_direct() ...")
    df_direct = alert_direct(
        plots=plots_gdf,
        deforestation=defo_item["raster_path"],
        protected_areas=protected_path,
        farming_areas=farming_path,
        deforestation_value=2,
        n_workers=n_workers
    )
    df_direct.to_csv(alert_direct_csv, index=False)
    print(f"[ALERT] alert_direct completed → {alert_direct_csv}")

    # Step 2: Indirect alerts
    print("[ALERT] Step 2: Preparing movement data ...")
    movement_files = [os.path.join(MOVEMENT_DIR, f) for f in os.listdir(MOVEMENT_DIR) if f.endswith(".csv")]
    movement_parts = []
    for f in tqdm(movement_files, desc="Loading movement files"):
        df = pd.read_csv(f)
        if {"origen_id", "destination_id"}.issubset(df.columns):
            movement_parts.append(df[["origen_id", "destination_id"]])
    movement_df = pd.concat(movement_parts) if movement_parts else pd.DataFrame(columns=["origen_id", "destination_id"])

    print("[ALERT] Running alert_indirect() ...")
    df_indirect = alert_indirect(df_direct, movement_df, n_workers)
    df_indirect.to_csv(alert_indirect_csv, index=False)
    print(f"[ALERT] alert_indirect completed → {alert_indirect_csv}")

    # Step 3: Aggregate (ADM3)
    print("[ALERT] Step 3: Running calculate_alert() (ADM3) ...")
    adm3_df = pd.read_csv(ADM3_CSV)[["id", "name"]].rename(columns={"id": "entity_id", "name": "entity_name"})
    plots_raw = pd.read_csv(PLOTS_CSV)
    provider_df = plots_raw[["id", "adm3_id"]].dropna().rename(columns={"id": "plot_id", "adm3_id": "entity_id"})
    df_entity = calculate_alert(df_indirect, adm3_df, provider_df, n_workers)
    df_entity.to_csv(entity_alert_csv, index=False)
    print(f"[ALERT] calculate_alert completed → {entity_alert_csv}")


# ============================================================================================
# MAIN
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description="Ganabosques ETL with progress tracking")
    parser.add_argument("--from-scratch", action="store_true", help="Rebuild inputs (extract + download)")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of parallel workers")
    args = parser.parse_args()

    print("=" * 80)
    print("[START] Ganabosques ETL pipeline")
    print(f"[ARGS] from_scratch={args.from_scratch}, n_workers={args.n_workers}")
    print("=" * 80)

    ensure_folders(args.from_scratch)

    # Choose flow
    if args.from_scratch:
        print("[FLOW] Executing full ETL from scratch")
        plots_gdf = extract_all_with_orm()
        defo_items = download_all_deforestation_layers()
        farming_path = download_vector_layer(FARMING_AREAS_CSV, "farmingareas")
        protected_path = download_vector_layer(PROTECTED_AREAS_CSV, "protectedareas")
    else:
        print("[FLOW] Using existing local data (skip DB & downloads)")
        df_plots = pd.read_csv(PLOTS_CSV)
        plots_gdf = gpd.GeoDataFrame(df_plots[["id"]], geometry=gpd.GeoSeries.from_wkt(df_plots["geometry"]), crs=None)
        defo_items = [{"id": f.split(".")[0], "raster_path": os.path.join(DEFOR_DIR, f)}
                      for f in os.listdir(DEFOR_DIR) if f.endswith(".tif")]
        farming_path = download_vector_layer(FARMING_AREAS_CSV, "farmingareas")
        protected_path = download_vector_layer(PROTECTED_AREAS_CSV, "protectedareas")

    print(f"[FLOW] Found {len(defo_items)} deforestation rasters to process.")
    for item in tqdm(defo_items, desc="Processing deforestation rasters"):
        run_alerts_for_deforestation(plots_gdf, item, farming_path, protected_path, args.n_workers)

    print("\n[END] ETL completed successfully.")
    print(f"[RESULTS] All outputs saved in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
