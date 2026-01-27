# -*- coding: utf-8 -*-
"""
Gestor de Datos para Pipeline de Alertas de Deforestación
- Caché de farms (solo metadata en memoria)
- Caché de geojsons (archivos individuales por sitcode/id)
- Caché de rasters (descarga desde geoserver con verificación)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import geopandas as gpd
from rtree import index
import geopandas as gpd
from rtree import index

try:
    from ganabosques_orm.collections.farm import Farm
    from ganabosques_orm.collections.farmpolygons import FarmPolygons
    from ganabosques_orm.collections.deforestation import Deforestation
    HAS_ORM = True
except ImportError:
    HAS_ORM = False
    print("⚠ Warning: ganabosques_orm no disponible")


class DataManager:
    """
    Gestor centralizado de datos con caché inteligente.
    
    Estructura de workspace:
    workspace/
    ├── farms/
    │   └── geojsons/
    │       ├── [sitcode].geojson
    │       └── [mongo_id].geojson
    ├── rasters/
    │   ├── smbyc/
    │   │   ├── annual/
    │   │   └── cumulative/
    │   └── nad/
    │       └── quarterly/
    └── results/
        ├── direct_alerts/
        ├── indirect_alerts/
        └── total_risk/
    """
    
    def __init__(
        self,
        workspace_dir: str,
        geoserver_url: str,
        geoserver_user: str,
        geoserver_pass: str
    ):
        self.workspace_dir = Path(workspace_dir) / "alertas"
        self.gs_url = geoserver_url.rstrip('/')
        self.gs_auth = HTTPBasicAuth(geoserver_user, geoserver_pass)
        
        # Directorios de caché
        self.farms_dir = self.workspace_dir / "farms"
        self.geojsons_dir = self.farms_dir / "geojsons"
        self.rasters_dir = self.workspace_dir / "rasters"
        self.results_dir = self.workspace_dir / "results"
        
        # Crear estructura
        self.geojsons_dir.mkdir(parents=True, exist_ok=True)
        self.rasters_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Caché en memoria
        self._farms_metadata = None
        self._geometries_cache = {}  # {farm_id: shapely.geometry}
        self._spatial_index = None  # R-tree index para búsqueda espacial rápida
        self._index_to_farm = {}  # {rtree_idx: farm_id}
        
        logging.info(f"DataManager inicializado: {self.workspace_dir}")
    
    def load_farms_metadata(self, limit: Optional[int] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Carga metadata de farms desde caché JSON o MongoDB (solo IDs, sin geojson).
        
        Args:
            limit: Número máximo de farms a cargar (None = todos)
        
        Returns:
            Tupla (metadata, error):
            - metadata: Lista de dicts con farm info
            - error: None si OK, mensaje de error si falló conexión/carga
        """
        if self._farms_metadata:
            if limit and len(self._farms_metadata) > limit:
                print(f"⚠ Usando solo {limit:,} de {len(self._farms_metadata):,} farms cargados")
                return (self._farms_metadata[:limit], None)
            return (self._farms_metadata, None)
        
        # Intentar cargar desde caché JSON
        cache_file = self.farms_dir / "farms_metadata.json"
        
        if cache_file.exists():
            print(f"📦 Cargando farms desde caché: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if limit and len(metadata) > limit:
                    print(f"✅ Cargados {limit:,} de {len(metadata):,} farms desde caché (limitado)")
                    self._farms_metadata = metadata[:limit]
                else:
                    print(f"✅ Cargados {len(metadata):,} farms desde caché")
                    self._farms_metadata = metadata
                
                return (self._farms_metadata, None)
            except Exception as e:
                print(f"⚠ Error leyendo caché, recargando desde MongoDB: {e}")
        
        # Cargar desde MongoDB
        if not HAS_ORM:
            error_msg = "ganabosques_orm no está disponible"
            logging.error(error_msg)
            return ([], error_msg)
        
        print("📊 Cargando metadata de farms desde MongoDB...")
        
        try:
            # Obtener total count primero para la barra de progreso
            total_farms = Farm.objects.count()
            print(f"   Total de farms en BD: {total_farms:,}")
            
            if limit:
                print(f"   ⚠ MODO TESTING: Solo se cargarán {limit:,} farms")
            
            # Ordenar por ID para resultados consistentes y reproducibles
            farms = Farm.objects.only('id', 'ext_id', 'adm3_id').order_by('id')
        except Exception as e:
            error_msg = f"No se pudo conectar a MongoDB: {e}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            return ([], error_msg)
        
        metadata = []
        processed = 0
        
        # Usar tqdm para mostrar progreso
        desc = f"Cargando farms{f' (límite: {limit:,})' if limit else ''}"
        for farm in tqdm(farms, total=min(limit, total_farms) if limit else total_farms, 
                        desc=desc, unit="farm"):
            sitcode = None
            producer_id = None
            
            # Extraer ext_ids usando el ORM
            if farm.ext_id:
                for ext in farm.ext_id:
                    try:
                        # ext es un objeto ExtIdFarm del ORM
                        # ext.source es un Enum(Source) con .value
                        # ext.ext_code es un StringField
                        if ext.source and ext.ext_code:
                            source_str = ext.source.value if hasattr(ext.source, "value") else str(ext.source)
                            
                            if source_str == "SIT_CODE":
                                sitcode = str(ext.ext_code).strip()
                            elif source_str == "PRODUCER_ID":
                                producer_id = str(ext.ext_code).strip()
                    except Exception as e:
                        logging.warning(f"Error extrayendo ext_id de farm {farm.id}: {e}")
                        continue
            
            metadata.append({
                'mongo_id': str(farm.id),
                'sitcode': sitcode,
                'producer_id': producer_id,
                'adm3_id': str(farm.adm3_id.id) if farm.adm3_id else None
            })
            
            processed += 1
            if limit and processed >= limit:
                print(f"\n✋ Límite alcanzado: {limit:,} farms")
                break
        
        # Guardar en caché JSON para próximas ejecuciones
        if not limit:  # Solo guardar caché completo
            try:
                print(f"💾 Guardando caché en: {cache_file}")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f)
                print(f"✅ Caché guardado ({len(metadata):,} farms)")
            except Exception as e:
                print(f"⚠ No se pudo guardar caché: {e}")
        
        self._farms_metadata = metadata
        
        if not metadata:
            error_msg = "No se encontraron farms en la base de datos"
            logging.warning(error_msg)
            print(f"⚠ {error_msg}")
            return ([], error_msg)
        
        print(f"✅ Cargados {len(metadata):,} farms (metadata en memoria)")
        return (metadata, None)
    
    def prepare_geojsons(self, farms_metadata: List[Dict]) -> int:
        """
        Descarga geojsons desde MongoDB para los farms que no existan en caché.
        
        Args:
            farms_metadata: Lista de metadata de farms
            
        Returns:
            Número de geojsons disponibles (en caché + descargados)
        """
        print(f"📥 Preparando geojsons para {len(farms_metadata):,} farms...")
        
        available = 0
        downloaded = 0
        failed = 0
        
        for farm_meta in tqdm(farms_metadata, desc="Verificando geojsons", unit="farm"):
            geojson_path = self.ensure_geojson_available(farm_meta)
            
            if geojson_path:
                available += 1
                # Verificar si fue descarga nueva (archivo reciente)
                from pathlib import Path
                path = Path(geojson_path)
                if path.exists():
                    import time
                    age_seconds = time.time() - path.stat().st_mtime
                    if age_seconds < 5:  # Creado en los últimos 5 segundos
                        downloaded += 1
            else:
                failed += 1
        
        if downloaded > 0:
            print(f"✅ Geojsons: {available:,} disponibles ({downloaded:,} descargados, {failed:,} sin geojson)")
        else:
            print(f"✅ Geojsons: {available:,} en caché ({failed:,} sin geojson)")
        
        return available
    
    def ensure_geojson_available(self, farm_metadata: Dict) -> Optional[str]:
        """
        Verifica si geojson existe en caché.
        Si no existe, lo descarga desde MongoDB y lo guarda SIEMPRE con sitcode.
        
        IMPORTANTE: Los geojsons SIEMPRE se guardan con sitcode para evitar duplicados.
        Si un farm no tiene sitcode, se omite (no se descarga).
        
        Args:
            farm_metadata: Dict con 'mongo_id' y 'sitcode'
            
        Returns:
            Ruta al archivo geojson o None si no se pudo obtener
        """
        mongo_id = farm_metadata['mongo_id']
        sitcode = farm_metadata['sitcode']
        
        # Si no hay sitcode, no podemos procesar este farm
        if not sitcode:
            logging.warning(f"Farm {mongo_id} sin SIT_CODE, se omite")
            return None
        
        # Buscar geojson por sitcode
        sitcode_path = self.geojsons_dir / f"{sitcode}.geojson"
        if sitcode_path.exists():
            return str(sitcode_path)
        
        # No existe, descargar desde MongoDB y guardar con sitcode
        return self._download_geojson(mongo_id, sitcode)
    
    def _download_geojson(self, mongo_id: str, sitcode: Optional[str]) -> Optional[str]:
        """Descarga geojson desde MongoDB y lo guarda en caché."""
        if not HAS_ORM:
            return None
        
        try:
            from bson import ObjectId
            farm_polygon = FarmPolygons.objects(farm_id=ObjectId(mongo_id)).first()
            
            if not farm_polygon or not farm_polygon.geojson:
                logging.warning(f"Farm {mongo_id} sin geojson en MongoDB")
                return None
            
            # Parsear el geojson (viene como string)
            geojson_str = farm_polygon.geojson
            geojson_data = json.loads(geojson_str)
            
            # Guardar en archivo (preferir sitcode si existe)
            if sitcode:
                output_path = self.geojsons_dir / f"{sitcode}.geojson"
            else:
                output_path = self.geojsons_dir / f"{mongo_id}.geojson"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f)
            
            logging.debug(f"Geojson guardado: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error descargando geojson {mongo_id}: {e}")
            return None
    
    def ensure_raster_available(
        self,
        source: str,
        period_type: str,
        layer_name: str,
        period_name: str,
        time_filter: str
    ) -> Optional[str]:
        """
        Verifica si raster existe en caché local.
        Si no existe, lo descarga desde geoserver.
        
        Args:
            source: 'smbyc', 'nad', 'atd'
            period_type: 'annual', 'cumulative', 'quarterly'
            layer_name: Nombre de la capa en geoserver (del campo 'path' en MongoDB)
            period_name: Nombre completo del período para archivo local
            time_filter: Filtro temporal para WCS (ej: '2013-2014' o '2023-01')
            
        Returns:
            Ruta local al archivo .tif o None si falla
        """
        # Ruta local en caché (usar period_name para el archivo)
        raster_subdir = self.rasters_dir / source.lower() / period_type.lower()
        raster_subdir.mkdir(parents=True, exist_ok=True)
        
        local_path = raster_subdir / f"{period_name}.tif"
        
        # Verificar si ya existe
        if local_path.exists():
            print(f"✅ Raster en caché: {local_path.name}")
            return str(local_path)
        
        # Descargar desde geoserver
        print(f"⬇️  Descargando raster: {period_name}...")
        return self._download_raster_wcs(layer_name, time_filter, local_path)
    
    def _download_raster_wcs(
        self,
        layer_name: str,
        time_filter: str,
        output_path: Path
    ) -> Optional[str]:
        """
        Descarga raster desde geoserver usando WCS con filtro temporal.
        
        Args:
            layer_name: Nombre de la capa en geoserver (del campo 'path')
                       Ej: 'nad_deforestation_quarter', 'smbyc_deforestation_annual'
            time_filter: Filtro temporal
                       NAD/ATD: '2023-01', '2024-02'
                       SMBYC: '2013-2014', '2010-2017'
            output_path: Ruta donde guardar el archivo .tif
        """
        try:
            # Construir URL de WCS
            wcs_url = f"{self.gs_url}/deforestation/wcs"
            
            params = {
                'service': 'WCS',
                'version': '2.0.1',
                'request': 'GetCoverage',
                'coverageId': f'deforestation:{layer_name}',
                'format': 'image/geotiff',
                'time': time_filter
            }
            
            print(f"   URL: {wcs_url}")
            print(f"   Coverage: deforestation:{layer_name}")
            print(f"   Time: {time_filter}")
            
            # Hacer request con timeout
            response = requests.get(
                wcs_url,
                params=params,
                auth=self.gs_auth,
                stream=True,
                timeout=300  # 5 minutos
            )
            
            if response.status_code != 200:
                logging.error(
                    f"Error descargando raster {layer_name}: "
                    f"HTTP {response.status_code} - {response.text[:500]}"
                )
                return None
            
            # Verificar que sea un GeoTIFF (no un XML de error)
            content_type = response.headers.get('content-type', '')
            if 'xml' in content_type.lower():
                logging.error(f"Geoserver retornó XML (error): {response.text[:500]}")
                return None
            
            # Guardar archivo
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Descargado: {output_path.name} ({file_size_mb:.1f} MB)")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error descargando raster {layer_name}: {e}")
            # Limpiar archivo parcial si existe
            if output_path.exists():
                output_path.unlink()
            return None
    
    def extract_time_filter_from_name(self, layer_name: str) -> str:
        """
        Extrae el filtro temporal del nombre de la capa.
        
        Ejemplos:
        - smbyc_deforestation_annual_2013-2014 → '2013-2014'
        - smbyc_deforestation_cumulative_2010-2017 → '2010-2017'
        - nad_deforestation_quarter_202301 → '2023-01'
        """
        import re
        
        # Para SMBYC: buscar patron YYYY-YYYY
        match = re.search(r'(\d{4}-\d{4})$', layer_name)
        if match:
            return match.group(1)
        
        # Para NAD/ATD: buscar patron YYYYQQ
        match = re.search(r'(\d{6})$', layer_name)
        if match:
            period_code = match.group(1)
            # Convertir 202301 → 2023-01
            year = period_code[:4]
            quarter = period_code[4:6]
            return f"{year}-{quarter}"
        
        # Fallback: retornar los últimos caracteres
        logging.warning(f"No se pudo extraer filtro temporal de: {layer_name}")
        return layer_name.split('_')[-1]
    
    def get_results_dir(self, stage: str) -> Path:
        """Retorna directorio para guardar resultados de una etapa."""
        stage_dir = self.results_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    def load_geometries_to_cache(self, farms_metadata: List[Dict], show_progress: bool = True) -> Dict:
        """
        Carga todas las geometrías en memoria para acceso rápido.
        
        Args:
            farms_metadata: Lista de metadata de farms
            show_progress: Mostrar barra de progreso
        
        Returns:
            Dict con estadísticas: {
                'loaded': int,
                'failed': int,
                'cache_size_mb': float
            }
        """
        import sys
        
        print("\n📦 Cargando geometrías en memoria...")
        
        loaded = 0
        failed = 0
        
        iterator = tqdm(farms_metadata, desc="Cargando geometrías", unit="farm") if show_progress else farms_metadata
        
        for farm in iterator:
            sitcode = farm.get('sitcode')
            mongo_id = farm.get('mongo_id')
            
            # Solo procesar farms con sitcode (política de nombrado única)
            if not sitcode:
                failed += 1
                continue
            
            # Buscar archivo geojson POR SITCODE ÚNICAMENTE
            geojson_path = self.geojsons_dir / f"{sitcode}.geojson"
            
            if not geojson_path.exists():
                failed += 1
                continue
            
            farm_id = sitcode
            
            try:
                gdf = gpd.read_file(str(geojson_path))
                if gdf.empty or 'geometry' not in gdf.columns:
                    failed += 1
                    continue
                
                # Convertir a EPSG:3116 si es necesario
                if gdf.crs and gdf.crs.to_string() != "EPSG:3116":
                    gdf = gdf.to_crs("EPSG:3116")
                
                # Guardar geometría (unary_union si hay múltiples)
                from shapely.ops import unary_union
                geom = unary_union(gdf.geometry.tolist()) if len(gdf) > 1 else gdf.geometry.iloc[0]
                self._geometries_cache[farm_id] = geom
                loaded += 1
                
            except Exception as e:
                logging.warning(f"Error cargando geometría {farm_id}: {e}")
                failed += 1
        
        # Calcular tamaño aproximado en memoria
        cache_size_bytes = sys.getsizeof(self._geometries_cache)
        for geom in self._geometries_cache.values():
            cache_size_bytes += sys.getsizeof(geom)
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        stats = {
            'loaded': loaded,
            'failed': failed,
            'cache_size_mb': cache_size_mb
        }
        
        print(f"✅ Geometrías en caché: {loaded:,} cargadas, {failed:,} fallidas")
        print(f"   Tamaño en memoria: ~{cache_size_mb:.1f} MB")
        
        return stats
    
    def build_spatial_index(self, show_progress: bool = True) -> Dict:
        """
        Construye índice espacial R-tree para búsqueda rápida.
        
        Returns:
            Dict con estadísticas: {
                'indexed': int,
                'build_time': float
            }
        """
        import time
        
        if not self._geometries_cache:
            raise RuntimeError("Primero debes cargar geometrías con load_geometries_to_cache()")
        
        print("\n🔍 Construyendo índice espacial R-tree...")
        start = time.time()
        
        # Crear índice
        self._spatial_index = index.Index()
        self._index_to_farm = {}
        
        iterator = enumerate(self._geometries_cache.items())
        if show_progress:
            iterator = tqdm(iterator, total=len(self._geometries_cache), 
                          desc="Indexando farms", unit="farm")
        
        for idx, (farm_id, geom) in iterator:
            try:
                # Insertar bbox en índice
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                self._spatial_index.insert(idx, bounds)
                self._index_to_farm[idx] = farm_id
            except Exception as e:
                logging.warning(f"Error indexando {farm_id}: {e}")
        
        build_time = time.time() - start
        
        stats = {
            'indexed': len(self._index_to_farm),
            'build_time': build_time
        }
        
        print(f"✅ Índice espacial construido: {len(self._index_to_farm):,} farms en {build_time:.2f}s")
        
        return stats
    
    def get_geometry(self, farm_id: str):
        """
        Obtiene geometría desde caché o la carga.
        
        Args:
            farm_id: sitcode o mongo_id
        
        Returns:
            shapely.geometry o None si no existe
        """
        # Intentar desde caché
        if farm_id in self._geometries_cache:
            return self._geometries_cache[farm_id]
        
        # Intentar cargar desde archivo
        geojson_path = self.geojsons_dir / f"{farm_id}.geojson"
        if not geojson_path.exists():
            return None
        
        try:
            gdf = gpd.read_file(str(geojson_path))
            if gdf.empty or 'geometry' not in gdf.columns:
                return None
            
            if gdf.crs and gdf.crs.to_string() != "EPSG:3116":
                gdf = gdf.to_crs("EPSG:3116")
            
            from shapely.ops import unary_union
            geom = unary_union(gdf.geometry.tolist()) if len(gdf) > 1 else gdf.geometry.iloc[0]
            
            # Guardar en caché para siguiente uso
            self._geometries_cache[farm_id] = geom
            
            return geom
        except Exception as e:
            logging.warning(f"Error cargando geometría {farm_id}: {e}")
            return None
    
    def query_farms_by_bbox(self, bbox: Tuple[float, float, float, float]) -> List[str]:
        """
        Busca farms que intersectan con un bounding box usando índice espacial.
        
        Args:
            bbox: (minx, miny, maxx, maxy) en EPSG:3116
        
        Returns:
            Lista de farm_ids que potencialmente intersectan
        """
        if not self._spatial_index:
            raise RuntimeError("Primero debes construir índice con build_spatial_index()")
        
        # Buscar en índice
        indices = list(self._spatial_index.intersection(bbox))
        
        # Convertir índices a farm_ids
        farm_ids = [self._index_to_farm[idx] for idx in indices if idx in self._index_to_farm]
        
        return farm_ids
