#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cálculo de métricas espaciales para farms:
- farming_in: área dentro de frontera agrícola
- farming_out: área fuera de frontera agrícola  
- protected: área en zonas protegidas (PNN)

Integrado con el sistema ganabosques (DataManager, config, etc.)
Con descarga automática desde geoserver si no existe caché local.
"""

import os
import time
import logging
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from config import config

# URLs de geoserver (sin maxFeatures para descargar todo)
FRONTERA_WFS_URL = (
    "https://ganageo.alliance.cgiar.org/geoserver/administrative/ows?"
    "service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=administrative:upra_boundaries"
    "&outputFormat=SHAPE-ZIP"
)

PNN_WFS_URL = (
    "https://ganageo.alliance.cgiar.org/geoserver/administrative/ows?"
    "service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=administrative:pnn_areas"
    "&outputFormat=SHAPE-ZIP"
)


def setup_logging():
    """Configura logging para métricas espaciales."""
    lvl = getattr(logging, str(config.get("LOG_LEVEL", "WARNING")).upper(), logging.WARNING)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s:%(message)s"
    )


def area_ha(geom: BaseGeometry) -> float:
    """Calcula área de geometría en hectáreas."""
    return float(geom.area / 10_000.0)


def download_and_extract_wfs(url: str, output_dir: Path, layer_name: str, expected_crs: str) -> Optional[Path]:
    """
    Descarga capa WFS (SHAPE-ZIP), extrae y convierte a GeoPackage.
    
    Args:
        url: URL del servicio WFS
        output_dir: Directorio donde guardar (ej: reference_layers/pnn_areas)
        layer_name: Nombre de la capa (ej: "pnn_areas")
        expected_crs: CRS objetivo (ej: "EPSG:3116")
        
    Returns:
        Path al .gpkg creado o None si falla
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = output_dir / f"{layer_name}.gpkg"
    
    # Si ya existe, retornar
    if output_gpkg.exists():
        return output_gpkg
    
    print(f"📥 Descargando {layer_name} desde geoserver...")
    print(f"   URL: {url}")
    
    try:
        # Descargar ZIP
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            tmp_zip_path = tmp_zip.name
            
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Descargando") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_zip.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Descarga completa: {total_size / 1_048_576:.2f} MB")
        
        # Extraer ZIP
        print(f"📦 Extrayendo shapefile...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            
            # Buscar .shp
            shp_files = list(Path(tmp_dir).glob("**/*.shp"))
            if not shp_files:
                raise RuntimeError(f"No se encontró .shp en el ZIP descargado")
            
            shp_path = shp_files[0]
            print(f"✅ Shapefile extraído: {shp_path.name}")
            
            # Cargar shapefile
            print(f"🔄 Cargando y reproyectando a {expected_crs}...")
            gdf = gpd.read_file(shp_path)
            
            # Reproyectar si es necesario
            if gdf.crs is None:
                print(f"⚠️  Shapefile sin CRS, asignando {expected_crs}...")
                gdf.set_crs(expected_crs, inplace=True)
            elif gdf.crs.to_string() != expected_crs:
                original_crs = gdf.crs.to_string()
                print(f"🔄 Reproyectando de {original_crs} a {expected_crs}...")
                gdf = gdf.to_crs(expected_crs)
            
            # Guardar como GeoPackage
            print(f"💾 Guardando como GeoPackage: {output_gpkg}")
            gdf.to_file(output_gpkg, driver="GPKG")
            
            print(f"✅ {layer_name} listo: {len(gdf):,} features")
        
        # Limpiar ZIP temporal
        os.unlink(tmp_zip_path)
        
        return output_gpkg
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error descargando {layer_name}: {e}")
        print(f"❌ Error de red: {e}")
        return None
    except zipfile.BadZipFile as e:
        logging.error(f"ZIP corrupto para {layer_name}: {e}")
        print(f"❌ Archivo ZIP corrupto")
        return None
    except Exception as e:
        logging.error(f"Error procesando {layer_name}: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_intersection_area_ha_via_sindex(farm_geom: BaseGeometry, mask_gdf: gpd.GeoDataFrame) -> float:
    """
    Calcula área de intersección usando spatial index para eficiencia.
    
    Args:
        farm_geom: Geometría del farm
        mask_gdf: GeoDataFrame con polígonos de referencia (Frontera/PNN)
        
    Returns:
        Área de intersección en hectáreas
    """
    if mask_gdf is None or mask_gdf.empty:
        return 0.0
    
    try:
        sidx = mask_gdf.sindex
    except Exception:
        sidx = None
    
    # Filtrar candidatos usando bounds
    cand_idx = list(sidx.intersection(farm_geom.bounds)) if sidx is not None else list(range(len(mask_gdf)))
    if not cand_idx:
        return 0.0
    
    mask_sub = mask_gdf.iloc[cand_idx]
    mask_sub = mask_sub[mask_sub.intersects(farm_geom)]
    
    if mask_sub.empty:
        return 0.0
    
    # Calcular intersección acumulada
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


def load_reference_layer(path: str, expected_crs: str, label: str) -> Optional[gpd.GeoDataFrame]:
    """
    Carga capa de referencia (Frontera/PNN) y verifica CRS.
    
    Args:
        path: Ruta al shapefile/geopackage
        expected_crs: CRS esperado (ej: "EPSG:3116")
        label: Etiqueta para mensajes
        
    Returns:
        GeoDataFrame o None si falla
    """
    if not os.path.isfile(path):
        logging.warning(f"{label}: no existe archivo: {path}")
        return None
    
    print(f"📂 Cargando {label}: {path}")
    t0 = time.perf_counter()
    
    try:
        gdf = gpd.read_file(path)
        
        # Verificar CRS
        if gdf.crs is None:
            raise RuntimeError(f"{label}: capa sin CRS definido")
        
        crs_str = gdf.crs.to_string()
        if crs_str != expected_crs:
            print(f"⚠️ {label}: CRS={crs_str}, reproyectando a {expected_crs}...")
            gdf = gdf.to_crs(expected_crs)
        
        if gdf.empty or "geometry" not in gdf.columns:
            logging.warning(f"{label}: capa vacía o sin geometría")
            return None
        
        print(f"✅ {label} cargado: {len(gdf):,} features en {time.perf_counter() - t0:.2f}s")
        
        # Construir spatial index
        try:
            _ = gdf.sindex
            print(f"   Spatial index: OK")
        except Exception:
            print(f"   ⚠️ Sin spatial index (rtree no disponible)")
        
        return gdf
        
    except Exception as e:
        logging.error(f"{label}: error cargando {path}: {e}")
        return None


def load_or_download_reference_layer(
    workspace_dir: Path,
    layer_name: str,
    wfs_url: str,
    expected_crs: str,
    label: str
) -> Optional[gpd.GeoDataFrame]:
    """
    Carga capa de referencia desde caché local o descarga desde geoserver.
    
    Args:
        workspace_dir: Directorio workspace
        layer_name: Nombre de la capa (ej: "pnn_areas")
        wfs_url: URL del servicio WFS
        expected_crs: CRS objetivo
        label: Etiqueta para mensajes
        
    Returns:
        GeoDataFrame o None si falla
    """
    # Buscar en reference_layers
    reference_dir = workspace_dir / "reference_layers" / layer_name
    gpkg_path = reference_dir / f"{layer_name}.gpkg"
    
    # Si existe caché, usar
    if gpkg_path.exists():
        print(f"📂 Usando caché local: {gpkg_path}")
        return load_reference_layer(str(gpkg_path), expected_crs, label)
    
    # Si no existe, intentar descargar
    print(f"📥 Caché no encontrado, descargando {label}...")
    downloaded_path = download_and_extract_wfs(wfs_url, reference_dir, layer_name, expected_crs)
    
    if downloaded_path:
        return load_reference_layer(str(downloaded_path), expected_crs, label)
    else:
        print(f"❌ No se pudo descargar {label}")
        return None


def load_farm_geometries(farms_metadata: List[Dict], data_manager, expected_crs: str) -> gpd.GeoDataFrame:
    """
    Carga geometrías de farms desde caché de DataManager (preferido) o desde disco.
    
    Args:
        farms_metadata: Lista de dicts con metadata de farms
        data_manager: Instancia de DataManager
        expected_crs: CRS esperado
        
    Returns:
        GeoDataFrame con geometrías de farms
    """
    # 🚀 OPTIMIZACIÓN: Usar caché de DataManager si está disponible
    if data_manager and hasattr(data_manager, 'is_geometry_cache_loaded') and data_manager.is_geometry_cache_loaded():
        print(f"🚀 Usando caché de geometrías en memoria ({len(data_manager.get_all_cached_geometries()):,} farms)")
        t0 = time.perf_counter()
        
        farms_gdf = data_manager.get_farms_geodataframe(expected_crs)
        
        # Disolver por ID (por si hay múltiples polígonos)
        farms_gdf = farms_gdf.dissolve(by='id', as_index=False)
        
        print(f"✅ Farms desde caché: {len(farms_gdf):,} en {time.perf_counter() - t0:.2f}s")
        return farms_gdf
    
    # Fallback: leer desde disco (método original)
    print(f"📂 Cargando geometrías de {len(farms_metadata):,} farms desde disco...")
    t0 = time.perf_counter()
    
    frames = []
    failed = 0
    
    for farm in tqdm(farms_metadata, desc="Cargando farms", unit="farm"):
        sitcode = farm.get('sitcode')
        if not sitcode:
            failed += 1
            continue
        
        geojson_path = data_manager.geojsons_dir / f"{sitcode}.geojson"
        
        if not geojson_path.exists():
            failed += 1
            continue
        
        try:
            gdf = gpd.read_file(geojson_path)
            
            # Asegurar CRS
            if gdf.crs is None:
                gdf.set_crs(expected_crs, inplace=True)
            elif gdf.crs.to_string() != expected_crs:
                gdf = gdf.to_crs(expected_crs)
            
            if gdf.empty or 'geometry' not in gdf.columns:
                failed += 1
                continue
            
            # Agregar ID
            gdf_farm = gdf[['geometry']].copy()
            gdf_farm['id'] = sitcode
            frames.append(gdf_farm)
            
        except Exception as e:
            logging.warning(f"Error cargando {sitcode}: {e}")
            failed += 1
    
    if not frames:
        raise RuntimeError("No se pudo cargar ninguna geometría de farm")
    
    farms_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=expected_crs)
    
    # Disolver por ID (por si hay múltiples polígonos)
    farms_gdf = farms_gdf.dissolve(by='id', as_index=False)
    
    print(f"✅ Farms cargados: {len(farms_gdf):,} (fallidos: {failed}) en {time.perf_counter() - t0:.2f}s")
    
    return farms_gdf


def compute_metrics_for_farms(
    farms_gdf: gpd.GeoDataFrame,
    frontier_gdf: Optional[gpd.GeoDataFrame],
    protected_gdf: Optional[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """
    Calcula métricas espaciales para cada farm.
    
    Args:
        farms_gdf: GeoDataFrame con farms
        frontier_gdf: GeoDataFrame con frontera agrícola
        protected_gdf: GeoDataFrame con áreas protegidas
        
    Returns:
        DataFrame con métricas por farm
    """
    print(f"\n📊 Calculando métricas espaciales para {len(farms_gdf):,} farms...")
    t0 = time.perf_counter()
    
    rows = []
    
    for _, farm_row in tqdm(farms_gdf.iterrows(), total=len(farms_gdf), desc="Procesando", unit="farm"):
        farm_id = farm_row['id']
        geom = farm_row.geometry
        
        # Área total
        total_ha = area_ha(geom)
        
        # Intersección con Frontera Agrícola
        if frontier_gdf is not None:
            farming_in_ha = compute_intersection_area_ha_via_sindex(geom, frontier_gdf)
        else:
            farming_in_ha = 0.0
        
        # Área fuera de frontera (resto)
        farming_out_ha = max(total_ha - farming_in_ha, 0.0)
        
        # Intersección con Áreas Protegidas
        if protected_gdf is not None:
            protected_ha = compute_intersection_area_ha_via_sindex(geom, protected_gdf)
        else:
            protected_ha = 0.0
        
        # Proporciones
        farming_in_prop = farming_in_ha / total_ha if total_ha > 0 else 0.0
        farming_out_prop = farming_out_ha / total_ha if total_ha > 0 else 0.0
        protected_prop = protected_ha / total_ha if total_ha > 0 else 0.0
        
        rows.append({
            'id': farm_id,
            'total_ha': round(total_ha, 4),
            'farming_in_ha': round(farming_in_ha, 4),
            'farming_in_prop': round(farming_in_prop, 6),
            'farming_out_ha': round(farming_out_ha, 4),
            'farming_out_prop': round(farming_out_prop, 6),
            'protected_ha': round(protected_ha, 4),
            'protected_prop': round(protected_prop, 6)
        })
    
    print(f"✅ Métricas calculadas en {time.perf_counter() - t0:.2f}s")
    
    return pd.DataFrame(rows)


def calculate_spatial_metrics(
    farms_metadata: List[Dict],
    data_manager,
    output_dir: Optional[str] = None,
    output_name: str = "spatial_metrics.csv",
    use_parallel: bool = False,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Pipeline completo de cálculo de métricas espaciales.
    
    Args:
        farms_metadata: Lista de metadata de farms
        data_manager: Instancia de DataManager
        output_dir: Directorio de salida (default: workspace/metrics)
        output_name: Nombre del archivo CSV de salida
        use_parallel: Si True, usa procesamiento paralelo
        num_workers: Número de workers para modo paralelo (None = auto)
        
    Returns:
        Dict con estadísticas de ejecución
    """
    T0 = time.perf_counter()
    setup_logging()
    
    print("\n" + "=" * 70)
    print("📐 CÁLCULO DE MÉTRICAS ESPACIALES" + (" (PARALELO)" if use_parallel else ""))
    print("=" * 70)
    
    # CRS del proyecto
    crs = config.get('CRS_METROS', 'EPSG:3116')
    
    # Workspace dir
    workspace_dir = Path(data_manager.workspace_dir)
    
    # 1) Cargar capas de referencia (con descarga automática si no existen)
    print("\n📂 Cargando capas de referencia...")
    
    frontier_gdf = load_or_download_reference_layer(
        workspace_dir=workspace_dir,
        layer_name="upra_boundaries",
        wfs_url=FRONTERA_WFS_URL,
        expected_crs=crs,
        label="Frontera Agrícola (UPRA)"
    )
    
    protected_gdf = load_or_download_reference_layer(
        workspace_dir=workspace_dir,
        layer_name="pnn_areas",
        wfs_url=PNN_WFS_URL,
        expected_crs=crs,
        label="Áreas Protegidas (PNN)"
    )
    
    if frontier_gdf is None and protected_gdf is None:
        raise RuntimeError("No se pudo cargar ninguna capa de referencia")
    
    # 2) Modo paralelo o secuencial
    if use_parallel and data_manager.is_geometry_cache_loaded():
        # 🚀 MODO PARALELO
        from parallel_processor import run_parallel_spatial_metrics
        
        parallel_params = {
            '_geometries_cache': data_manager.get_all_cached_geometries(),
            '_frontier_gdf': frontier_gdf,
            '_protected_gdf': protected_gdf,
            'crs': crs
        }
        
        parallel_result = run_parallel_spatial_metrics(parallel_params, num_workers)
        
        if not parallel_result['success']:
            raise RuntimeError(f"Error en procesamiento paralelo: {parallel_result.get('error')}")
        
        metrics_df = parallel_result['results_df']
    else:
        # Modo secuencial (original)
        # 2) Cargar geometrías de farms
        farms_gdf = load_farm_geometries(farms_metadata, data_manager, crs)
        
        # 3) Calcular métricas
        metrics_df = compute_metrics_for_farms(farms_gdf, frontier_gdf, protected_gdf)
    
    # 4) Guardar resultados
    if output_dir is None:
        output_dir = data_manager.workspace_dir / "metrics"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / output_name
    
    print(f"\n💾 Guardando resultados en: {output_path}")
    metrics_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Estadísticas
    total_time = time.perf_counter() - T0
    
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"✅ Farms procesados: {len(metrics_df):,}")
    print(f"✅ Archivo generado: {output_path}")
    print(f"⏱️  Tiempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
    
    # Estadísticas de métricas
    print(f"\n📈 Estadísticas:")
    print(f"   • Área total promedio: {metrics_df['total_ha'].mean():.2f} ha")
    print(f"   • Farms en frontera agrícola: {(metrics_df['farming_in_ha'] > 0).sum():,}")
    print(f"   • Farms en áreas protegidas: {(metrics_df['protected_ha'] > 0).sum():,}")
    print(f"   • % promedio en frontera: {metrics_df['farming_in_prop'].mean() * 100:.1f}%")
    print(f"   • % promedio protegido: {metrics_df['protected_prop'].mean() * 100:.1f}%")
    
    return {
        'success': True,
        'farms_processed': len(metrics_df),
        'output_file': str(output_path),
        'execution_time': total_time,
        'metrics': {
            'avg_total_ha': float(metrics_df['total_ha'].mean()),
            'avg_farming_in_prop': float(metrics_df['farming_in_prop'].mean()),
            'avg_protected_prop': float(metrics_df['protected_prop'].mean()),
            'farms_in_frontier': int((metrics_df['farming_in_ha'] > 0).sum()),
            'farms_in_protected': int((metrics_df['protected_ha'] > 0).sum())
        }
    }


def main():
    """Función principal para ejecución standalone."""
    from data_manager import DataManager
    
    # Obtener farm_limit de variables de entorno (si existe)
    import sys
    farm_limit = None
    if '--farm-limit' in sys.argv:
        idx = sys.argv.index('--farm-limit')
        if idx + 1 < len(sys.argv):
            farm_limit = int(sys.argv[idx + 1])
    
    # Inicializar DataManager
    workspace_dir = config.get('WORKSPACE_DIR')
    gs_url = config.get('GEOSERVER_URL')
    gs_user = config.get('GEOSERVER_USER', 'admin')
    gs_pass = config.get('GEOSERVER_PASS', 'geoserver')
    
    data_manager = DataManager(
        workspace_dir=workspace_dir,
        geoserver_url=gs_url,
        geoserver_user=gs_user,
        geoserver_pass=gs_pass
    )
    
    # Cargar metadata de farms
    print(f"📂 Cargando metadata de farms...")
    if farm_limit:
        print(f"⚠️  MODO TESTING: Limitando a {farm_limit:,} farms")
    
    farms_metadata, db_error = data_manager.load_farms_metadata(limit=farm_limit)
    
    if db_error:
        print(f"❌ Error conectando a base de datos: {db_error}")
        sys.exit(1)
    
    if not farms_metadata:
        print(f"❌ No se encontraron farms en la base de datos")
        sys.exit(1)
    
    print(f"✅ Metadata cargada: {len(farms_metadata):,} farms")
    
    # Asegurar geojsons disponibles
    print(f"\n📂 Verificando geojsons...")
    geojsons_available = data_manager.prepare_geojsons(farms_metadata)
    print(f"✅ GeoJSONs disponibles: {geojsons_available}/{len(farms_metadata)}")
    
    # Calcular métricas
    result = calculate_spatial_metrics(farms_metadata, data_manager)
    
    if result['success']:
        print("\n✅ Proceso completado exitosamente")
    else:
        print("\n❌ Proceso finalizado con errores")
        sys.exit(1)


if __name__ == "__main__":
    main()
