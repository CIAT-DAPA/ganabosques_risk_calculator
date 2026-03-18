# -*- coding: utf-8 -*-
"""
Gestor de Datos para Pipeline de Alertas de Deforestación
- Caché de farms (solo metadata en memoria)
- Caché de geojsons (archivos individuales por sitcode/id)
- Caché de rasters (descarga desde geoserver con verificación)
"""

import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from bson import ObjectId
from rtree import index

try:
    from ganabosques_orm.collections.farm import Farm
    from ganabosques_orm.collections.farmpolygons import FarmPolygons
    from ganabosques_orm.collections.deforestation import Deforestation
    from ganabosques_orm.collections.analysis import Analysis
    from ganabosques_orm.collections.farmrisk import FarmRisk
    from ganabosques_orm.collections.enterpriserisk import EnterpriseRisk
    from ganabosques_orm.collections.enterprise import Enterprise
    from ganabosques_orm.collections.protectedareas import ProtectedAreas
    from ganabosques_orm.collections.farmingareas import FarmingAreas
    from ganabosques_orm.auxiliaries.attributes import Attributes
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
    │       └── [geofarmerid].geojson
    ├── rasters/
    │   └── smbyc/
    │       ├── annual/
    │       ├── cumulative/
    │       ├── nad/
    │       └── atd/
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
    
    def load_farms_metadata(self, limit: Optional[int] = None, offline_mode: bool = False, value_chain: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Carga metadata de farms desde caché JSON o MongoDB (solo IDs, sin geojson).
        
        Args:
            limit: Número máximo de farms a cargar (None = todos)
            offline_mode: Si True, solo usa caché local, no conecta a MongoDB
            value_chain: Cadena de valor para filtrar farms ('livestock', 'cacao'). None = todos.
        
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
        
        # Intentar cargar desde caché JSON (separado por value_chain)
        cache_suffix = f"_{value_chain}" if value_chain else ""
        cache_file = self.farms_dir / f"farms_metadata{cache_suffix}.json"
        
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
                
                # En modo offline, retornar directamente sin validar con BD
                if offline_mode:
                    print(f"🔌 Modo offline: usando caché sin validación")
                    return (self._farms_metadata, None)
                
                # Validar caché contra BD (verificar count y sample de sit_codes)
                if HAS_ORM:
                    try:
                        # Validar contra BD con mismo filtro de value_chain
                        if value_chain:
                            from ganabosques_orm.enums.valuechain import ValueChain as VC
                            bd_count = Farm.objects(value_chain=VC(value_chain.lower())).count()
                        else:
                            bd_count = Farm.objects.count()
                        cache_count = len(metadata)
                        
                        # Si difieren significativamente (>5%), invalidar caché
                        if abs(bd_count - cache_count) / max(bd_count, 1) > 0.05:
                            print(f"⚠ Caché desactualizado: BD={bd_count:,} vs caché={cache_count:,}")
                            print(f"📊 Recargando desde MongoDB...")
                            self._farms_metadata = None
                            # Continuar para recargar desde BD
                        else:
                            return (self._farms_metadata, None)
                    except Exception:
                        # Si no podemos validar, usar el caché
                        return (self._farms_metadata, None)
                else:
                    return (self._farms_metadata, None)
            except Exception as e:
                print(f"⚠ Error leyendo caché, recargando desde MongoDB: {e}")
        elif offline_mode:
            # Modo offline sin caché disponible
            error_msg = f"Modo offline: caché no encontrado en {cache_file}"
            return ([], error_msg)
        
        # Cargar desde MongoDB
        if not HAS_ORM:
            error_msg = "ganabosques_orm no está disponible"
            logging.error(error_msg)
            return ([], error_msg)
        
        print(f"📊 Cargando metadata de farms desde MongoDB...")
        if value_chain:
            print(f"   🔗 Filtrando por cadena de valor: {value_chain}")
        
        try:
            # Filtrar por value_chain si se especifica
            if value_chain:
                from ganabosques_orm.enums.valuechain import ValueChain as VC
                try:
                    vc_enum = VC(value_chain.lower())
                except ValueError:
                    return ([], f"value_chain '{value_chain}' no válido")
                total_farms = Farm.objects(value_chain=vc_enum).count()
            else:
                total_farms = Farm.objects.count()
            print(f"   Total de farms en BD: {total_farms:,}")
            
            if limit:
                print(f"   ⚠ MODO TESTING: Solo se cargarán {limit:,} farms")
            
            # Ordenar por ID para resultados consistentes y reproducibles
            if value_chain:
                farms = Farm.objects(value_chain=vc_enum).only('id', 'ext_id', 'adm3_id').order_by('id')
            else:
                farms = Farm.objects.only('id', 'ext_id', 'adm3_id').order_by('id')
        except Exception as e:
            error_msg = f"No se pudo conectar a MongoDB: {e}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            return ([], error_msg)
        
        # Recolectar metadata básica primero
        metadata = []
        farm_ids = []  # Para buscar farm_polygon_ids después
        processed = 0
        
        # Usar tqdm para mostrar progreso
        desc = f"Cargando farms{f' (límite: {limit:,})' if limit else ''}"
        for farm in tqdm(farms, total=min(limit, total_farms) if limit else total_farms, 
                        desc=desc, unit="farm"):
            sit_code = None
            producer_id = None
            geofarmer_id = None
            
            # Extraer TODOS los ext_ids usando el ORM
            if farm.ext_id:
                for ext in farm.ext_id:
                    try:
                        # ext es un objeto ExtIdFarm del ORM
                        # ext.source es un Enum(Source) con .value
                        # ext.ext_code es un StringField
                        if ext.source and ext.ext_code:
                            source_str = ext.source.value if hasattr(ext.source, "value") else str(ext.source)
                            
                            if source_str == "SIT_CODE":
                                sit_code = str(ext.ext_code).strip()
                            elif source_str == "PRODUCER_ID":
                                producer_id = str(ext.ext_code).strip()
                            elif source_str == "GEOFARMER_ID":
                                geofarmer_id = str(ext.ext_code).strip()
                    except Exception as e:
                        logging.warning(f"Error extrayendo ext_id de farm {farm.id}: {e}")
                        continue
            
            # Elegir identificador principal según cadena de valor:
            #   livestock → SIT_CODE
            #   cacao     → GEOFARMER_ID
            vc = (value_chain or '').lower()
            if vc == 'cacao':
                farm_code = geofarmer_id or sit_code
            else:  # livestock u otro
                farm_code = sit_code or geofarmer_id
            
            farm_ids.append(farm.id)
            metadata.append({
                'mongo_id': str(farm.id),
                'sitcode': farm_code,          # identificador principal para geojson
                'sit_code': sit_code,           # SIT_CODE original (puede ser None)
                'geofarmer_id': geofarmer_id,   # GEOFARMER_ID original (puede ser None)
                'producer_id': producer_id,     # PRODUCER_ID original (puede ser None)
                'adm3_id': str(farm.adm3_id.id) if farm.adm3_id else None,
                'farm_polygon_id': None  # Se llenará después
            })
            
            processed += 1
            if limit and processed >= limit:
                print(f"\n✋ Límite alcanzado: {limit:,} farms")
                break
        
        # Cargar farm_polygon_ids en batch (más eficiente)
        if farm_ids:
            print(f"📦 Cargando farm_polygon_ids para {len(farm_ids):,} farms...")
            try:
                # Consulta batch de FarmPolygons
                from bson import ObjectId
                polygon_docs = FarmPolygons.objects(farm_id__in=farm_ids).only('id', 'farm_id')
                
                # Crear mapeo farm_id -> polygon_id
                polygon_map = {}
                for poly in polygon_docs:
                    farm_oid = poly.farm_id.id if hasattr(poly.farm_id, 'id') else poly.farm_id
                    polygon_map[str(farm_oid)] = str(poly.id)
                
                # Actualizar metadata con polygon_ids
                matched = 0
                for meta in metadata:
                    poly_id = polygon_map.get(meta['mongo_id'])
                    if poly_id:
                        meta['farm_polygon_id'] = poly_id
                        matched += 1
                
                print(f"   ✅ {matched:,} farms con polygon_id, {len(metadata) - matched:,} sin polygon")
            except Exception as e:
                logging.warning(f"Error cargando farm_polygon_ids: {e}")
                print(f"   ⚠ No se pudieron cargar farm_polygon_ids: {e}")
        
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
    
    def count_available_geojsons(self, farms_metadata: List[Dict]) -> int:
        """
        Cuenta geojsons disponibles localmente (para modo offline).
        
        Args:
            farms_metadata: Lista de metadata de farms
            
        Returns:
            Número de geojsons encontrados localmente
        """
        print(f"📂 Verificando geojsons locales para {len(farms_metadata):,} farms...")
        
        available = 0
        missing = 0
        
        for farm_meta in tqdm(farms_metadata, desc="Verificando geojsons locales", unit="farm"):
            sitcode = farm_meta.get('sitcode')
            if not sitcode:
                missing += 1
                continue
            
            # Verificar si existe el archivo
            geojson_path = self.geojsons_dir / f"{sitcode}.geojson"
            if geojson_path.exists():
                available += 1
            else:
                missing += 1
        
        print(f"✅ Geojsons locales: {available:,} encontrados, {missing:,} faltantes")
        return available
    
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
        Si no existe, lo descarga desde MongoDB y lo guarda con el código externo.
        
        El campo 'sitcode' contiene el identificador externo principal:
        - SIT_CODE para farms de livestock
        - GEOFARMER_ID para farms de cacao
        
        Args:
            farm_metadata: Dict con 'mongo_id' y 'sitcode'
            
        Returns:
            Ruta al archivo geojson o None si no se pudo obtener
        """
        mongo_id = farm_metadata['mongo_id']
        sitcode = farm_metadata['sitcode']
        
        # Si no hay código externo, no podemos procesar este farm
        if not sitcode:
            logging.warning(f"Farm {mongo_id} sin código externo (SIT_CODE/GEOFARMER_ID), se omite")
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
            source: 'smbyc'
            period_type: 'annual', 'cumulative', 'nad', 'atd'
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
    
    def get_mongo_id_map_df(self, ids_filter: set = None) -> pd.DataFrame:
        """
        Retorna DataFrame con mapeo sitcode -> farm_id -> farm_polygon_id.
        Usa los datos ya cargados en _farms_metadata (sin consulta adicional a MongoDB).
        
        Args:
            ids_filter: Set de sitcodes a incluir (None = todos)
            
        Returns:
            DataFrame con columnas ['id', 'farm_id', 'farm_poligons_id']
        """
        cols = ["id", "farm_id", "farm_poligons_id"]
        
        if not self._farms_metadata:
            logging.warning("farms_metadata no cargado, retornando DataFrame vacío")
            return pd.DataFrame(columns=cols)
        
        rows = []
        # Regex para quitar prefijo FARM_ID_ (consistente con _norm_id en total_alert)
        import re
        farm_id_prefix = re.compile(r'^FARM_ID_', re.IGNORECASE)
        
        for meta in self._farms_metadata:
            sitcode = meta.get('sitcode')
            if not sitcode:
                continue
            
            # Normalizar: quitar FARM_ID_ prefix + uppercase para matching consistente
            normalized_id = farm_id_prefix.sub('', sitcode).upper()
            
            # Filtrar si se especificó
            if ids_filter and normalized_id not in ids_filter:
                continue
            
            rows.append({
                "id": normalized_id,
                "farm_id": meta.get('mongo_id', ''),
                "farm_poligons_id": meta.get('farm_polygon_id', '') or ''
            })
        
        df = pd.DataFrame(rows, columns=cols)
        logging.info(f"Mapeo MongoDB: {len(df)} farms con sitcode")
        return df
    
    def get_results_dir(self, stage: str, source: str = None, deforestation_type: str = None, period: str = None) -> Path:
        """
        Retorna directorio para guardar resultados de una etapa.
        
        Nueva estructura de carpetas:
        results/{source}/{deforestation_type}/{stage}/{period}/
        
        Args:
            stage: Etapa del proceso ('direct_alerts', 'indirect_alerts', 'metrics', etc.)
            source: Fuente de deforestación ('smbyc')
            deforestation_type: Tipo de deforestación ('annual', 'cumulative', 'nad', 'atd')
            period: Período específico ('2012-2013', '202301', etc.)
            
        Returns:
            Path al directorio de resultados
        """
        if source and deforestation_type:
            # Estructura: results/{source}/{deforestation_type}/{stage}/
            # Los archivos van directamente aquí (se identifican por nombre)
            stage_dir = self.results_dir / source / deforestation_type / stage
        else:
            # Fallback: estructura antigua para compatibilidad
            stage_dir = self.results_dir / stage
        
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    def get_geometry(self, farm_id: str):
        """
        Obtiene geometría de farm desde el caché en memoria.
        
        Args:
            farm_id: sitcode del farm
            
        Returns:
            shapely.geometry o None si no está en caché
        """
        return self._geometries_cache.get(farm_id)
    
    def get_all_cached_geometries(self) -> Dict:
        """
        Retorna todo el caché de geometrías.
        
        Returns:
            Dict {farm_id: shapely.geometry}
        """
        return self._geometries_cache
    
    def get_farms_geodataframe(self, expected_crs: str = "EPSG:3116") -> gpd.GeoDataFrame:
        """
        Crea GeoDataFrame desde el caché de geometrías (sin leer de disco).
        
        Args:
            expected_crs: CRS para el GeoDataFrame
            
        Returns:
            GeoDataFrame con columnas ['id', 'geometry']
        """
        if not self._geometries_cache:
            raise RuntimeError("Caché de geometrías vacío. Llama load_geometries_to_cache() primero.")
        
        rows = [{'id': farm_id, 'geometry': geom} for farm_id, geom in self._geometries_cache.items()]
        gdf = gpd.GeoDataFrame(rows, crs=expected_crs)
        return gdf
    
    def is_geometry_cache_loaded(self) -> bool:
        """Verifica si el caché de geometrías está cargado."""
        return len(self._geometries_cache) > 0
    
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
        
    def download_movements_to_csv(self, year: int, output_dir: Optional[Path] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Descarga movimientos de un año desde MongoDB y guarda en formato CSV ETL.
        
        Args:
            year: Año de los movimientos (ej: 2023)
            output_dir: Directorio de salida (default: workspace/movements/)
            
        Returns:
            Tupla (csv_path, error):
            - csv_path: Ruta al CSV generado o None si falló
            - error: Mensaje de error o None si OK
        """
        if not HAS_ORM:
            return (None, "ganabosques_orm no disponible")
        
        from ganabosques_orm.collections.movement import Movement
        from datetime import datetime
        
        # Configurar directorio de salida
        if output_dir is None:
            output_dir = self.workspace_dir / "movements"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / f"movement_data_base_{year}.csv"
        
        # Verificar si ya existe
        if csv_path.exists():
            print(f"📦 Movimientos {year} ya en caché: {csv_path}")
            return (str(csv_path), None)
        
        print(f"📊 Descargando movimientos {year} desde MongoDB...")
        
        try:
            # Definir rango de fechas para el año
            start_date = datetime(year, 1, 1)
            end_date = datetime(year + 1, 1, 1)
            
            # Contar total
            total_count = Movement.objects(
                date__gte=start_date,
                date__lt=end_date
            ).count()
            
            if total_count == 0:
                return (None, f"No hay movimientos para el año {year}")
            
            print(f"   Total movimientos: {total_count:,}")
            
            # Preparar lista para CSV
            rows = []
            
            # Iterar con progreso
            movements = Movement.objects(
                date__gte=start_date,
                date__lt=end_date
            ).select_related()
            
            for mov in tqdm(movements, total=total_count, desc=f"Descargando {year}", unit="mov"):
                # Extraer sit_codes de farms
                sit_code_origen = None
                sit_code_destino = None
                producer_id_origen = None
                producer_id_destino = None
                
                # Farm origen
                if mov.farm_id_origin:
                    farm = mov.farm_id_origin
                    if farm.ext_id:
                        for ext in farm.ext_id:
                            if hasattr(ext.source, 'value'):
                                source = ext.source.value
                            else:
                                source = str(ext.source)
                            if source == "SIT_CODE":
                                sit_code_origen = str(ext.ext_code).strip()
                            elif source == "PRODUCER_ID":
                                producer_id_origen = str(ext.ext_code).strip()
                
                # Farm destino
                if mov.farm_id_destination:
                    farm = mov.farm_id_destination
                    if farm.ext_id:
                        for ext in farm.ext_id:
                            if hasattr(ext.source, 'value'):
                                source = ext.source.value
                            else:
                                source = str(ext.source)
                            if source == "SIT_CODE":
                                sit_code_destino = str(ext.ext_code).strip()
                            elif source == "PRODUCER_ID":
                                producer_id_destino = str(ext.ext_code).strip()
                
                # Enterprise origen (producer_id)
                if mov.enterprise_id_origin and not producer_id_origen:
                    producer_id_origen = str(mov.enterprise_id_origin.id)
                
                # Enterprise destino (producer_id)
                if mov.enterprise_id_destination and not producer_id_destino:
                    producer_id_destino = str(mov.enterprise_id_destination.id)
                
                # Tipo origen/destino
                tipo_origen = mov.type_origin.value if mov.type_origin else None
                tipo_destino = mov.type_destination.value if mov.type_destination else None
                
                rows.append({
                    "SIT_CODE_ORIGEN": sit_code_origen or "",
                    "SIT_CODE_DESTINO": sit_code_destino or "",
                    "DATE": mov.date.strftime("%Y-%m-%d") if mov.date else "",
                    "TIPO_ORIGEN": tipo_origen or "",
                    "TIPO_DESTINO": tipo_destino or "",
                    "PRODUCER_ID_ORIGEN": producer_id_origen or "",
                    "PRODUCER_ID_DESTINO": producer_id_destino or ""
                })
            
            # Guardar CSV
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Movimientos guardados: {csv_path} ({len(rows):,} registros)")
            return (str(csv_path), None)
            
        except Exception as e:
            error_msg = f"Error descargando movimientos: {e}"
            logging.error(error_msg)
            return (None, error_msg)
    
    def save_farm_risk_to_db(
        self,
        results_df: pd.DataFrame,
        analysis_id: str,
        deforestation_id: Optional[str] = None
    ) -> Tuple[int, int, List[Dict]]:
        """
        Guarda resultados de riesgo de fincas en MongoDB (FarmRisk).
        
        Usa el CSV de total_risk que consolida todos los datos:
        - Deforestación (directas)
        - Alertas indirectas (movimientos)
        - Métricas espaciales (farming_in, farming_out, protected)
        
        Args:
            results_df: DataFrame con columnas del total_risk:
                - id, farm_id, farm_poligons_id
                - deforested_ha, deforested_prop, direct_alert
                - farming_in_ha, farming_in_prop
                - farming_out_ha, farming_out_prop
                - protected_ha, protected_prop
                - indirect_alert_in, indirect_alert_out
            analysis_id: ID del análisis (referencia a Analysis)
            deforestation_id: ID de la capa de deforestación usada (deprecated)
            
        Returns:
            Tupla (saved, failed, errors):
            - saved: Número de registros guardados
            - failed: Número de errores
            - errors: Lista de dicts con errores [{farm_id, error}]
        """
        if not HAS_ORM:
            return (0, 0, [{"error": "ganabosques_orm no disponible"}])
        
        from bson import ObjectId
        from datetime import datetime
        
        saved = 0
        failed = 0
        errors = []
        
        # Obtener o crear Analysis
        try:
            analysis = Analysis.objects(id=ObjectId(analysis_id)).first()
            if not analysis:
                return (0, 0, [{"error": f"Analysis {analysis_id} no encontrado"}])
        except Exception as e:
            return (0, 0, [{"error": f"Error obteniendo Analysis: {e}"}])
        
        print(f"📤 Guardando {len(results_df):,} FarmRisk en BD...")
        
        # Helper para convertir valores seguros a float (definido una vez fuera del loop)
        def safe_float(val, default=0.0):
            if val is None or str(val) in ['', 'nan', 'None', 'no_info']:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Helper para convertir valores seguros a bool
        def safe_bool(val, default=False):
            if val is None or str(val) in ['', 'nan', 'None', 'no_info']:
                return default
            if isinstance(val, bool):
                return val
            return str(val).lower() in ['true', '1', 'yes']
        
        for _, row in tqdm(results_df.iterrows(), total=len(results_df), 
                          desc="Guardando FarmRisk", unit="farm"):
            try:
                sitcode_id = str(row.get('id', ''))
                
                # Intentar usar farm_id de MongoDB directamente si está en el CSV
                # OPTIMIZADO: No hacer query si ya tenemos los IDs
                mongo_farm_id = row.get('farm_id', None)
                mongo_polygon_id = row.get('farm_poligons_id', None)
                
                farm_oid = None
                farm_polygon_oid = None
                
                # Si tenemos el farm_id en el CSV, usar directamente sin query
                if mongo_farm_id and str(mongo_farm_id) not in ['', 'nan', 'None']:
                    try:
                        farm_oid = ObjectId(str(mongo_farm_id))
                        if mongo_polygon_id and str(mongo_polygon_id) not in ['', 'nan', 'None']:
                            farm_polygon_oid = ObjectId(str(mongo_polygon_id))
                    except Exception:
                        pass
                
                # Fallback: buscar por sitcode SOLO si no tenemos farm_id
                if not farm_oid:
                    
                    farm = Farm.objects(ext_id__source="SIT_CODE", ext_id__ext_code=sitcode_id).first()

                    # Buscar sin filtrar por fuente si no encontramos con SIT_CODE 
                    if not farm:
                         farm = Farm.objects(ext_id__ext_code=sitcode_id).first()
                    if not farm:
                        errors.append({"farm_id": sitcode_id, "error": "Farm no encontrado"})
                        failed += 1
                        continue
                    farm_oid = farm.id
                    # Buscar FarmPolygons solo si no tenemos el ID
                    if not farm_polygon_oid:
                        farm_polygon = FarmPolygons.objects(farm_id=farm.id).first()
                        farm_polygon_oid = farm_polygon.id if farm_polygon else None
                
                # Crear atributos de deforestación
                deforestation_attrs = Attributes(
                    ha=safe_float(row.get('deforested_ha')),
                    prop=safe_float(row.get('deforested_prop'))
                )
                
                # Crear atributos de protected areas
                protected_attrs = Attributes(
                    ha=safe_float(row.get('protected_ha')),
                    prop=safe_float(row.get('protected_prop'))
                )
                
                # Crear atributos de farming_in (frontera agrícola dentro)
                farming_in_attrs = Attributes(
                    ha=safe_float(row.get('farming_in_ha')),
                    prop=safe_float(row.get('farming_in_prop'))
                )
                
                # Crear atributos de farming_out (frontera agrícola fuera)
                farming_out_attrs = Attributes(
                    ha=safe_float(row.get('farming_out_ha')),
                    prop=safe_float(row.get('farming_out_prop'))
                )
                
                # Determinar risk_input y risk_output desde alertas indirectas
                risk_input = safe_bool(row.get('indirect_alert_in'))
                risk_output = safe_bool(row.get('indirect_alert_out'))
                risk_direct = safe_bool(row.get('direct_alert'))
                
                # Buscar o crear FarmRisk (usando farm_oid directamente)
                farm_risk = FarmRisk.objects(
                    farm_id=farm_oid,
                    analysis_id=analysis.id
                ).first()
                
                if farm_risk:
                    # Actualizar existente
                    farm_risk.deforestation = deforestation_attrs
                    farm_risk.protected = protected_attrs
                    farm_risk.farming_in = farming_in_attrs
                    farm_risk.farming_out = farming_out_attrs
                    farm_risk.risk_direct = risk_direct
                    farm_risk.risk_input = risk_input
                    farm_risk.risk_output = risk_output
                    farm_risk.save()
                else:
                    # Crear nuevo
                    farm_risk = FarmRisk(
                        farm_id=farm_oid,
                        analysis_id=analysis.id,
                        farm_polygons_id=farm_polygon_oid,
                        deforestation=deforestation_attrs,
                        protected=protected_attrs,
                        farming_in=farming_in_attrs,
                        farming_out=farming_out_attrs,
                        risk_direct=risk_direct,
                        risk_input=risk_input,
                        risk_output=risk_output
                    )
                    farm_risk.save()
                
                saved += 1
                
            except Exception as e:
                errors.append({"farm_id": str(row.get('id', '')), "error": str(e)})
                failed += 1
        
        print(f"✅ FarmRisk guardados: {saved:,} OK, {failed:,} errores")
        return (saved, failed, errors)
    
    def save_farm_risk_to_db_bulk(
        self,
        results_df: pd.DataFrame,
        analysis_id: str,
        chunk_size: int = 1000
    ) -> Tuple[int, int, List[Dict]]:
        """
        Versión optimizada de save_farm_risk_to_db usando bulk inserts.
        
        IMPORTANTE: Este método hace INSERT ONLY (no upsert).
        Si ya existen FarmRisk para el analysis_id, se producirán errores de duplicado.
        
        Args:
            results_df: DataFrame con columnas del total_risk
            analysis_id: ID del análisis
            chunk_size: Tamaño de chunks para bulk insert (default: 1000)
            
        Returns:
            Tupla (saved, failed, errors)
        """
        if not HAS_ORM:
            return (0, 0, [{"error": "ganabosques_orm no disponible"}])
        
        from bson import ObjectId
        
        saved = 0
        failed = 0
        errors = []
        
        # 1. Validar analysis_id
        try:
            analysis = Analysis.objects(id=ObjectId(analysis_id)).first()
            if not analysis:
                return (0, 0, [{"error": f"Analysis {analysis_id} no encontrado"}])
            analysis_oid = analysis.id
        except Exception as e:
            return (0, 0, [{"error": f"Error validando analysis: {e}"}])
        
        # 2. Verificar que el CSV tiene las columnas necesarias
        required_cols = ['farm_id', 'farm_poligons_id']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            # Fallback: si no están las columnas, usar resolución manual
            print(f"⚠️  Columnas faltantes {missing_cols}, resolviendo sitcodes manualmente...")
            sitcode_to_farm_map = self._bulk_resolve_farms(results_df['id'].tolist())
            use_csv_ids = False
        else:
            print(f"✅ Usando farm_id y farm_poligons_id directamente del CSV")
            sitcode_to_farm_map = None
            use_csv_ids = True
        
        # 3. Procesar por chunks
        total_chunks = (len(results_df) + chunk_size - 1) // chunk_size
        print(f"📤 Guardando FarmRisk en {total_chunks} chunks de {chunk_size}...")
        
        for i in tqdm(range(0, len(results_df), chunk_size), 
                     desc="Bulk inserting FarmRisk", unit="chunk"):
            chunk_df = results_df.iloc[i:i+chunk_size]
            
            # Preparar documentos válidos del chunk
            valid_docs, chunk_errors = self._prepare_farm_risk_docs(
                chunk_df, 
                analysis_oid, 
                sitcode_to_farm_map,
                use_csv_ids
            )
            
            # Bulk insert
            if valid_docs:
                chunk_saved, chunk_failed, insert_errors = self._bulk_insert_farm_risks(valid_docs)
                saved += chunk_saved
                failed += chunk_failed
                errors.extend(insert_errors)
            
            # Agregar errores de preparación
            errors.extend(chunk_errors)
            failed += len(chunk_errors)
        
        print(f"✅ FarmRisk bulk: {saved:,} insertados, {failed:,} errores")
        return (saved, failed, errors)
    
    def _bulk_resolve_farms(self, sitcodes: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Resuelve sitcodes a farm_ids y farm_polygon_ids en batch.
        
        Args:
            sitcodes: Lista de sitcodes
            
        Returns:
            Dict {sitcode: {"farm_id": str, "farm_polygon_id": str}}
        """
        if not self._farms_metadata:
            logging.warning("farms_metadata no cargado, usando consulta MongoDB directa")
            return self._bulk_resolve_farms_from_db(sitcodes)
        
        # Usar metadata ya cargada (más eficiente)
        sitcode_map = {}
        
        for meta in tqdm(self._farms_metadata, desc="Resolviendo farms en batch", unit="farm"):
            sitcode = meta.get('sitcode')
            if sitcode and sitcode in sitcodes:
                sitcode_map[sitcode] = {
                    "farm_id": meta.get('mongo_id', ''),
                    "farm_polygon_id": meta.get('farm_polygon_id', '') or ''
                }
        
        found = len(sitcode_map)
        total = len(set(sitcodes))  # Quitar duplicados para count
        print(f"   ✅ Resueltos {found:,} de {total:,} sitcodes desde caché")
        
        return sitcode_map
    
    def _bulk_resolve_farms_from_db(self, sitcodes: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fallback: resolver sitcodes desde MongoDB directamente.
        """
        if not HAS_ORM:
            return {}
        
        print(f"   🔍 Consultando MongoDB para {len(sitcodes):,} sitcodes...")
        
        try:
            # Consulta batch de farms
            farms = Farm.objects(
                ext_id__source="SIT_CODE", 
                ext_id__ext_code__in=sitcodes
            ).only('id', 'ext_id')
            
            # Crear mapeo sitcode -> farm_id
            sitcode_to_farm = {}
            farm_ids = []
            
            for farm in farms:
                if farm.ext_id:
                    for ext in farm.ext_id:
                        if ext.source and ext.ext_code:
                            source_str = ext.source.value if hasattr(ext.source, "value") else str(ext.source)
                            if source_str == "SIT_CODE":
                                sitcode = str(ext.ext_code).strip()
                                sitcode_to_farm[sitcode] = {"farm_id": str(farm.id), "farm_polygon_id": ""}
                                farm_ids.append(farm.id)
                                break
            
            # Consulta batch de FarmPolygons
            if farm_ids:
                polygons = FarmPolygons.objects(farm_id__in=farm_ids).only('id', 'farm_id')
                farm_to_polygon = {str(poly.farm_id.id): str(poly.id) for poly in polygons}
                
                # Actualizar mapeo con polygon_ids
                for sitcode, data in sitcode_to_farm.items():
                    farm_id = data["farm_id"]
                    data["farm_polygon_id"] = farm_to_polygon.get(farm_id, "")
            
            print(f"   ✅ Resueltos {len(sitcode_to_farm):,} sitcodes desde MongoDB")
            return sitcode_to_farm
            
        except Exception as e:
            logging.error(f"Error resolviendo farms desde MongoDB: {e}")
            return {}
    
    def _prepare_farm_risk_docs(
        self, 
        chunk_df: pd.DataFrame, 
        analysis_oid: ObjectId, 
        sitcode_to_farm_map: Dict[str, Dict[str, str]] = None,
        use_csv_ids: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepara documentos FarmRisk para bulk insert.
        
        Args:
            chunk_df: DataFrame con chunk de datos
            analysis_oid: ObjectId del análisis 
            sitcode_to_farm_map: Mapeo opcional (solo si use_csv_ids=False)
            use_csv_ids: Si True, usa farm_id del CSV directamente
        
        Returns:
            Tupla (valid_docs, errors)
        """
        from bson import ObjectId
        
        valid_docs = []
        errors = []
        
        # Helper functions (mismas que el método original)
        def safe_float(val, default=0.0):
            if val is None or str(val) in ['', 'nan', 'None', 'no_info']:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        def safe_bool(val, default=False):
            if val is None or str(val) in ['', 'nan', 'None', 'no_info']:
                return default
            if isinstance(val, bool):
                return val
            return str(val).lower() in ['true', '1', 'yes']
        
        for _, row in chunk_df.iterrows():
            try:
                sitcode = str(row.get('id', ''))
                
                # Usar farm_ids directamente del CSV (más eficiente)
                if use_csv_ids:
                    farm_id = row.get('farm_id', '')
                    farm_polygon_id = row.get('farm_poligons_id', '')  # Nota: es 'farm_poligons_id' no 'farm_polygons_id'
                    
                    # Validar que tenemos los IDs
                    if not farm_id or str(farm_id) in ['', 'nan', 'None']:
                        errors.append({"farm_id": sitcode, "error": "farm_id vacío en CSV"})
                        continue
                    
                    farm_id = str(farm_id)
                    farm_polygon_id = str(farm_polygon_id) if farm_polygon_id and str(farm_polygon_id) not in ['', 'nan', 'None'] else None
                else:
                    # Fallback: usar mapeo pre-calculado
                    farm_info = sitcode_to_farm_map.get(sitcode)
                    if not farm_info:
                        errors.append({"farm_id": sitcode, "error": "Farm no encontrado en mapeo"})
                        continue
                    
                    farm_id = farm_info["farm_id"]
                    farm_polygon_id = farm_info["farm_polygon_id"]
                    
                    if not farm_id:
                        errors.append({"farm_id": sitcode, "error": "Farm sin mongo_id"})
                        continue
                
                # Convertir a ObjectIds
                try:
                    farm_oid = ObjectId(farm_id)
                    farm_polygon_oid = ObjectId(farm_polygon_id) if farm_polygon_id else None
                except Exception:
                    errors.append({"farm_id": sitcode, "error": "ObjectId inválido"})
                    continue
                
                # Preparar documento para insert
                doc = {
                    "farm_id": farm_oid,
                    "analysis_id": analysis_oid,
                    "farm_polygons_id": farm_polygon_oid,
                    
                    # Atributos de deforestación
                    "deforestation": {
                        "ha": safe_float(row.get('deforested_ha')),
                        "prop": safe_float(row.get('deforested_prop'))
                    },
                    
                    # Atributos de áreas protegidas
                    "protected": {
                        "ha": safe_float(row.get('protected_ha')),
                        "prop": safe_float(row.get('protected_prop'))
                    },
                    
                    # Atributos de frontera agrícola dentro
                    "farming_in": {
                        "ha": safe_float(row.get('farming_in_ha')),
                        "prop": safe_float(row.get('farming_in_prop'))
                    },
                    
                    # Atributos de frontera agrícola fuera
                    "farming_out": {
                        "ha": safe_float(row.get('farming_out_ha')),
                        "prop": safe_float(row.get('farming_out_prop'))
                    },
                    
                    # Flags de riesgo
                    "risk_direct": safe_bool(row.get('direct_alert')),
                    "risk_input": safe_bool(row.get('indirect_alert_in')),
                    "risk_output": safe_bool(row.get('indirect_alert_out'))
                }
                
                valid_docs.append(doc)
                
            except Exception as e:
                errors.append({"farm_id": str(row.get('id', '')), "error": f"Error preparando doc: {e}"})
        
        return (valid_docs, errors)
    
    def _bulk_insert_farm_risks(self, docs: List[Dict]) -> Tuple[int, int, List[Dict]]:
        """
        Ejecuta bulk insert de documentos FarmRisk.
        
        Returns:
            Tupla (saved, failed, errors)
        """
        if not docs:
            return (0, 0, [])
        
        try:
            # Crear objetos FarmRisk desde dicts
            farm_risks = []
            for doc in docs:
                # Convertir nested dicts a Attributes objects
                deforestation_attrs = Attributes(**doc["deforestation"])
                protected_attrs = Attributes(**doc["protected"])
                farming_in_attrs = Attributes(**doc["farming_in"])
                farming_out_attrs = Attributes(**doc["farming_out"])
                
                farm_risk = FarmRisk(
                    farm_id=doc["farm_id"],
                    analysis_id=doc["analysis_id"],
                    farm_polygons_id=doc["farm_polygons_id"],
                    deforestation=deforestation_attrs,
                    protected=protected_attrs,
                    farming_in=farming_in_attrs,
                    farming_out=farming_out_attrs,
                    risk_direct=doc["risk_direct"],
                    risk_input=doc["risk_input"],
                    risk_output=doc["risk_output"]
                )
                farm_risks.append(farm_risk)
            
            # Bulk insert
            FarmRisk.objects.insert(farm_risks, load_bulk=False)
            
            return (len(farm_risks), 0, [])
            
        except Exception as e:
            # Si hay error en bulk insert, clasificar tipo de error
            error_msg = str(e).lower()
            
            if "duplicate" in error_msg or "unique" in error_msg:
                error_type = "Documentos duplicados (analysis_id + farm_id ya existe)"
            elif "validation" in error_msg:
                error_type = "Error de validación de esquema"
            else:
                error_type = f"Error de inserción: {e}"
            
            return (0, len(docs), [{"error": error_type, "count": len(docs)}])
    
    def save_enterprise_risk_to_db(
        self,
        enterprise_df: pd.DataFrame,
        analysis_id: str
    ) -> Tuple[int, int, List[Dict]]:
        """
        Guarda resultados de riesgo de empresas en MongoDB (EnterpriseRisk).
        
        El CSV tiene columnas: idpro, id_farm, typemove, period, year, 
        farm_has_direct_alert, enterprise_type_raw
        
        La búsqueda de Enterprise se hace por:
        - ext_id.label = PRODUCTIONUNIT_ID
        - ext_id.ext_code = idpro (del CSV)
        - type_enterprise = enterprise_type_raw (del CSV)
        
        Args:
            enterprise_df: DataFrame con columnas del enterprise_alerts CSV
            analysis_id: ID del análisis (referencia a Analysis)
            
        Returns:
            Tupla (saved, failed, errors)
        """
        if not HAS_ORM:
            return (0, 0, [{"error": "ganabosques_orm no disponible"}])
        
        from bson import ObjectId
        from ganabosques_orm.enums.label import Label
        from ganabosques_orm.enums.typeenterprise import TypeEnterprise
        
        saved = 0
        failed = 0
        errors = []
        
        # Obtener Analysis
        try:
            analysis = Analysis.objects(id=ObjectId(analysis_id)).first()
            if not analysis:
                return (0, 0, [{"error": f"Analysis {analysis_id} no encontrado"}])
        except Exception as e:
            return (0, 0, [{"error": f"Error obteniendo Analysis: {e}"}])
        
        # Mapeo de nombres del CSV a TypeEnterprise enum
        type_mapping = {
            'SLAUGHTERHOUSE': TypeEnterprise.SLAUGHTERHOUSE,
            'COLLECTION_CENTER': TypeEnterprise.COLLECTION_CENTER,
            'CATTLE_FAIR': TypeEnterprise.CATTLE_FAIR,
            'ENTERPRISE': TypeEnterprise.ENTERPRISE,
            # Posibles variantes en minúsculas
            'slaughterhouse': TypeEnterprise.SLAUGHTERHOUSE,
            'collection_center': TypeEnterprise.COLLECTION_CENTER,
            'cattle_fair': TypeEnterprise.CATTLE_FAIR,
            'enterprise': TypeEnterprise.ENTERPRISE,
        }
        
        # Agrupar por (idpro + enterprise_type_raw) para manejar el caso
        # donde el mismo idpro puede existir con diferentes tipos
        if 'idpro' in enterprise_df.columns:
            enterprise_col = 'idpro'
        else:
            enterprise_col = 'enterprise_id'
        
        type_col = 'enterprise_type_raw' if 'enterprise_type_raw' in enterprise_df.columns else 'type_enterprise'
        
        # Crear columna combinada para agrupar
        enterprise_df = enterprise_df.copy()
        enterprise_df['_group_key'] = enterprise_df[enterprise_col].astype(str) + '_' + enterprise_df[type_col].astype(str)
        
        grouped = enterprise_df.groupby('_group_key')
        
        print(f"📤 Guardando {len(grouped):,} EnterpriseRisk en BD...")
        
        for group_key, group in tqdm(grouped, desc="Guardando EnterpriseRisk", unit="emp"):
            try:
                # Extraer idpro y type de la primera fila del grupo
                first_row = group.iloc[0]
                idpro = str(first_row[enterprise_col])
                enterprise_type_raw = str(first_row[type_col])
                
                # Convertir tipo a enum
                type_enum = type_mapping.get(enterprise_type_raw)
                if not type_enum:
                    errors.append({
                        "enterprise_id": idpro, 
                        "error": f"Tipo de empresa desconocido: {enterprise_type_raw}"
                    })
                    failed += 1
                    continue
                
                # Buscar empresa por PRODUCTIONUNIT_ID + type_enterprise
                enterprise = Enterprise.objects(
                    ext_id__label=Label.PRODUCTIONUNIT_ID,
                    ext_id__ext_code=idpro,
                    type_enterprise=type_enum
                ).first()
                
                if not enterprise:
                    # Intentar búsqueda solo por ext_id (puede que el tipo no coincida exactamente)
                    enterprise = Enterprise.objects(
                        ext_id__label=Label.PRODUCTIONUNIT_ID,
                        ext_id__ext_code=idpro
                    ).first()
                    
                    if not enterprise:
                        errors.append({
                            "enterprise_id": idpro, 
                            "error": f"Enterprise no encontrado (PRODUCTIONUNIT_ID={idpro}, type={enterprise_type_raw})"
                        })
                        failed += 1
                        continue
                
                # Separar movimientos de entrada y salida
                risk_input_refs = []
                risk_output_refs = []
                
                for _, row in group.iterrows():
                    farm_id = str(row.get('id_farm', ''))
                    typemove = str(row.get('typemove', '')).lower()
                    
                    # Buscar FarmRisk correspondiente
                    farm = Farm.objects(ext_id__source="SIT_CODE", ext_id__ext_code=farm_id).first()
                    if not farm:
                        continue
                    
                    farm_risk = FarmRisk.objects(
                        farm_id=farm.id,
                        analysis_id=analysis.id
                    ).first()
                    
                    if farm_risk:
                        if typemove == 'in':
                            risk_input_refs.append(farm_risk)
                        elif typemove == 'out':
                            risk_output_refs.append(farm_risk)
                
                # Buscar o crear EnterpriseRisk
                ent_risk = EnterpriseRisk.objects(
                    enterprise_id=enterprise.id,
                    analysis_id=analysis.id
                ).first()
                
                if ent_risk:
                    # Actualizar (agregar a las listas existentes sin duplicar)
                    existing_input_ids = {str(r.id) for r in ent_risk.risk_input or []}
                    existing_output_ids = {str(r.id) for r in ent_risk.risk_output or []}
                    
                    for ref in risk_input_refs:
                        if str(ref.id) not in existing_input_ids:
                            ent_risk.risk_input.append(ref)
                    
                    for ref in risk_output_refs:
                        if str(ref.id) not in existing_output_ids:
                            ent_risk.risk_output.append(ref)
                    
                    ent_risk.save()
                else:
                    # Crear nuevo
                    ent_risk = EnterpriseRisk(
                        enterprise_id=enterprise.id,
                        analysis_id=analysis.id,
                        risk_input=risk_input_refs,
                        risk_output=risk_output_refs
                    )
                    ent_risk.save()
                
                saved += 1
                
            except Exception as e:
                errors.append({"enterprise_id": str(group_key), "error": str(e)})
                failed += 1
        
        print(f"✅ EnterpriseRisk guardados: {saved:,} OK, {failed:,} errores")
        return (saved, failed, errors)
    
    def _get_latest_protected_areas(self) -> Optional[str]:
        """
        Obtiene la capa de áreas protegidas más reciente.
        
        Usa ordenamiento por _id descendente ya que el ObjectId de MongoDB
        tiene un timestamp embebido en los primeros 4 bytes.
        
        Returns:
            Objeto ProtectedAreas o None si no existe
        """
        if not HAS_ORM:
            return None
        
        try:
            # Ordenar por _id descendente (ObjectId tiene timestamp embebido)
            latest = ProtectedAreas.objects().order_by('-id').first()
            if latest:
                logging.info(f"ProtectedAreas más reciente: {latest.id} ({latest.name})")
                return latest
            else:
                logging.warning("No se encontraron ProtectedAreas en la BD")
                return None
        except Exception as e:
            logging.error(f"Error obteniendo ProtectedAreas: {e}")
            return None
    
    def _get_latest_farming_areas(self) -> Optional[str]:
        """
        Obtiene la capa de áreas agrícolas más reciente.
        
        Usa ordenamiento por _id descendente ya que el ObjectId de MongoDB
        tiene un timestamp embebido en los primeros 4 bytes.
        
        Returns:
            Objeto FarmingAreas o None si no existe
        """
        if not HAS_ORM:
            return None
        
        try:
            # Ordenar por _id descendente (ObjectId tiene timestamp embebido)
            latest = FarmingAreas.objects().order_by('-id').first()
            if latest:
                logging.info(f"FarmingAreas más reciente: {latest.id} ({latest.name})")
                return latest
            else:
                logging.warning("No se encontraron FarmingAreas en la BD")
                return None
        except Exception as e:
            logging.error(f"Error obteniendo FarmingAreas: {e}")
            return None
    
    def create_or_get_analysis(
        self,
        deforestation_id: str,
        user_id: Optional[str] = None,
        value_chain: str = "livestock"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Crea o recupera un análisis para la capa de deforestación.
        
        Automáticamente incluye las capas más recientes de:
        - ProtectedAreas (áreas protegidas)
        - FarmingAreas (frontera agrícola)
        
        Args:
            deforestation_id: ID de la capa de deforestación
            user_id: ID del usuario (opcional)
            value_chain: Cadena de valor ('livestock', 'cacao'). Default: 'livestock'
            
        Returns:
            Tupla (analysis_id, error)
        """
        if not HAS_ORM:
            return (None, "ganabosques_orm no disponible")
        
        from bson import ObjectId
        from datetime import datetime
        
        try:
            deforestation = Deforestation.objects(id=ObjectId(deforestation_id)).first()
            if not deforestation:
                return (None, f"Deforestation {deforestation_id} no encontrado")
            
            # Buscar análisis existente
            analysis = Analysis.objects(
                deforestation_id=deforestation.id,
                value_chain=value_chain.lower()
            ).first()
            
            if analysis:
                print(f"📊 Usando análisis existente: {analysis.id}")
                return (str(analysis.id), None)
            
            # Obtener capas más recientes de áreas protegidas y frontera agrícola
            protected_areas = self._get_latest_protected_areas()
            farming_areas = self._get_latest_farming_areas()
            
            if protected_areas:
                print(f"   📍 Áreas protegidas: {protected_areas.name}")
            else:
                print("   ⚠️  No se encontró capa de áreas protegidas")
                
            if farming_areas:
                print(f"   🌾 Frontera agrícola: {farming_areas.name}")
            else:
                print("   ⚠️  No se encontró capa de frontera agrícola")
            
            # Convertir value_chain string a enum
            from ganabosques_orm.enums.valuechain import ValueChain
            try:
                vc_enum = ValueChain(value_chain.lower())
            except ValueError:
                logging.warning(f"value_chain '{value_chain}' no válido, usando LIVESTOCK")
                vc_enum = ValueChain.LIVESTOCK
            
            # Crear nuevo análisis con todas las referencias
            analysis = Analysis(
                deforestation_id=deforestation.id,
                protected_areas_id=protected_areas,
                farming_areas_id=farming_areas,
                user_id=ObjectId(user_id) if user_id else None,
                date=datetime.utcnow(),
                value_chain=vc_enum
            )
            analysis.save()
            
            print(f"   🔗 Cadena de valor: {vc_enum.value}")
            
            print(f"✅ Análisis creado: {analysis.id}")
            return (str(analysis.id), None)
            
        except Exception as e:
            return (None, f"Error creando análisis: {e}")