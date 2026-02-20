#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚠️  ARCHIVO DEPRECADO - NO USAR ⚠️

Este archivo ha sido reemplazado por spatial_metrics.py que está integrado
con DataManager y tiene mejor manejo de descarga automática de capas.

Usar en su lugar: spatial_metrics.py
Fecha de deprecación: Enero 2026
"""

import warnings
warnings.warn(
    "metricas.py está DEPRECADO. Usar spatial_metrics.py en su lugar.",
    DeprecationWarning,
    stacklevel=2
)

import os
import re
import geopandas as gpd
from shapely.geometry import base as shapely_base
import pandas as pd

# ============================
# CONFIGURACIÓN (ajusta aquí)
# ============================
BUFFERS_DIR = "/opt/ganabosques/test_buffers/data_server/buffer/"
PNN_SHP = "/opt/ganabosques/test_buffers/data_server/PNN_3116/PNN.shp"
FRONTERA_SHP = "/opt/ganabosques/test_buffers/data_server/Frontera_Agricola_3116/Frontera_Agricola_Abr2024_3116.shp"

OUTPUT_DIR = "/opt/ganabosques/test_buffers/data_server/alertas/metrics/"
OUTPUT_NAME = "metricas.csv"

# Tamaño de lote (ajusta según RAM/IO)
BATCH_SIZE = 100

# Decimales en salida
ROUND_DECIMALS = 2
# ============================

CRS_3116 = "EPSG:3116"


def _ensure_crs_3116(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Verifica si un GeoDataFrame está en CRS EPSG:3116.
    Si no, lo reproyecta. Si no tiene CRS, se le asigna EPSG:3116.
    (Versión silenciosa: sin prints)
    """
    if gdf.crs is None:
        return gdf.set_crs(CRS_3116)

    if str(gdf.crs) != CRS_3116:
        return gdf.to_crs(CRS_3116)

    return gdf


def _safe_geom(geom):
    """Intenta reparar geometrías inválidas de forma segura."""
    try:
        if geom is None or geom.is_empty:
            return geom
        if not geom.is_valid:
            return geom.buffer(0)
        return geom
    except Exception:
        return geom.buffer(0)


def _area_ha(geom):
    return (geom.area / 10000.0) if geom is not None and not geom.is_empty else 0.0


def _union_geoms(gdf: gpd.GeoDataFrame):
    """
    Une todas las geometrías de un GeoDataFrame en una sola.
    Usa Shapely 2.x union_all(); fallback a unary_union en versiones anteriores.
    """
    try:
        geom = gdf.geometry.union_all()
    except AttributeError:
        geom = gdf.geometry.unary_union
    return _safe_geom(geom)


def _sum_intersection_area_ha(geom, layer_gdf: gpd.GeoDataFrame):
    """
    Suma el área (ha) de la intersección de 'geom' con features relevantes de 'layer_gdf'
    usando sindex para filtrar candidatos.
    """
    if geom is None or geom.is_empty:
        return 0.0

    try:
        idx = list(layer_gdf.sindex.query(geom, predicate="intersects"))
        if not idx:
            return 0.0
        candidates = layer_gdf.iloc[idx]
    except Exception:
        # Si no hay índice espacial por alguna razón
        candidates = layer_gdf

    area = 0.0
    for g in candidates.geometry:
        if g is None:
            continue
        try:
            inter = geom.intersection(g)
        except Exception:
            inter = _safe_geom(geom).intersection(_safe_geom(g))
        if isinstance(inter, shapely_base.BaseGeometry) and not inter.is_empty:
            area += _area_ha(inter)
    return area


def compute_metrics_for_file(filepath: str,
                             pnn: gpd.GeoDataFrame,
                             frontera: gpd.GeoDataFrame):
    """
    Procesa un geojson de buffer y devuelve un dict con métricas.
    Cruces directos con PNN y Frontera. Asegura CRS 3116 para el buffer.
    """
    # id desde el nombre del archivo (e.g., '125648.geojson' -> 125648)
    basename = os.path.basename(filepath)
    m = re.search(r"(\d+)", basename)
    the_id = int(m.group(1)) if m else None

    # Cargar buffer
    gdf = gpd.read_file(filepath)

    # Asegurar CRS 3116 en el buffer (silencioso)
    gdf = _ensure_crs_3116(gdf)

    # Unir features del buffer a una sola geometría
    geom = _union_geoms(gdf)
    total_ha = _area_ha(geom)

    if total_ha <= 0:
        return {
            "id": the_id,
            "farming_in_ha": 0.0,
            "farming_in_prop": 0.0,
            "farming_out_ha": 0.0,
            "farming_out_prop": 0.0,
            "protected_ha": 0.0,
            "protected_prop": 0.0,
        }

    # --- Intersección con PNN ---
    protected_ha = _sum_intersection_area_ha(geom, pnn)
    protected_prop = protected_ha / total_ha if total_ha > 0 else 0.0

    # --- Intersección con Frontera Agrícola ---
    farming_in_ha = _sum_intersection_area_ha(geom, frontera)
    farming_out_ha = max(0.0, total_ha - farming_in_ha)

    farming_in_prop = farming_in_ha / total_ha if total_ha > 0 else 0.0
    farming_out_prop = farming_out_ha / total_ha if total_ha > 0 else 0.0

    return {
        "id": the_id,
        "farming_in_ha": round(farming_in_ha, ROUND_DECIMALS),
        "farming_in_prop": round(farming_in_prop, ROUND_DECIMALS),
        "farming_out_ha": round(farming_out_ha, ROUND_DECIMALS),
        "farming_out_prop": round(farming_out_prop, ROUND_DECIMALS),
        "protected_ha": round(protected_ha, ROUND_DECIMALS),
        "protected_prop": round(protected_prop, ROUND_DECIMALS),
    }


def _write_batch_rows(rows, out_path, write_header):
    df = pd.DataFrame(rows)
    if "id" in df.columns:
        df = df.sort_values("id")
    df.to_csv(
        out_path,
        mode="w" if write_header else "a",
        index=False,
        header=write_header,
        float_format=f"%.{ROUND_DECIMALS}f",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME if OUTPUT_NAME.lower().endswith(".csv") else OUTPUT_NAME + ".csv"
    )

    # Carga PNN
    print("Cargando capa PNN desde:", PNN_SHP)
    pnn = gpd.read_file(PNN_SHP)
    print(f"  → {len(pnn)} polígonos | CRS original: {pnn.crs}")
    pnn = _ensure_crs_3116(pnn)

    # Carga Frontera Agrícola
    print("Cargando capa Frontera Agrícola desde:", FRONTERA_SHP)
    frontera = gpd.read_file(FRONTERA_SHP)
    print(f"  → {len(frontera)} polígonos | CRS original: {frontera.crs}")
    frontera = _ensure_crs_3116(frontera)

    # Forzar sindex (si el backend lo construye lazy)
    _ = pnn.sindex
    _ = frontera.sindex

    print("Capas listas en CRS EPSG:3116. Comenzando procesamiento de buffers...\n")

    # Listar geojsons
    try:
        names = [f for f in os.listdir(BUFFERS_DIR) if f.lower().endswith(".geojson")]
    except FileNotFoundError:
        print("ERROR: Carpeta de buffers no encontrada:", BUFFERS_DIR)
        return

    if not names:
        print("ERROR: No se encontraron archivos .geojson en:", BUFFERS_DIR)
        return

    names.sort()
    geojsons = [os.path.join(BUFFERS_DIR, f) for f in names]

    total = len(geojsons)
    batches = [geojsons[i: i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"Archivos detectados: {total}")
    print(f"BATCH_SIZE = {BATCH_SIZE}\n")

    # Preparar salida: limpiar archivo si ya existe
    if os.path.exists(out_path):
        os.remove(out_path)

    # Procesamiento secuencial por batches (sin paralelización)
    for bi, batch in enumerate(batches, start=1):
        start_idx = (bi - 1) * BATCH_SIZE
        end_idx = start_idx + len(batch) - 1

        # Único print de progreso por batch
        print(f"Procesando Batch {bi}/{len(batches)} ({len(batch)} archivos) [{start_idx}-{end_idx}]...")

        rows = []
        for fp in batch:
            try:
                rows.append(compute_metrics_for_file(fp, pnn, frontera))
            except Exception as e:
                # Silencioso respecto a errores por archivo;
                # si quieres log mínimo, puedes descomentar la siguiente línea:
                # print(f"  Error procesando {os.path.basename(fp)}: {e}")
                continue

        # Escritura incremental del lote al CSV
        _write_batch_rows(rows, out_path, write_header=(bi == 1))

    print(f"\nProceso completado. Archivo generado en: {out_path}")


if __name__ == "__main__":
    main()
 