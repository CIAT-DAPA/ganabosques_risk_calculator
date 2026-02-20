#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de Alertas de Deforestación para Ganabosques.

Etapas:
  1. Alertas Directas: Intersección de predios con rasters de deforestación
  2. Alertas Indirectas: Movimientos de ganado desde/hacia predios con alertas
  3. Métricas Espaciales: Cálculo de frontera agrícola y áreas protegidas
  4. Riesgo Total: Consolidación de todas las alertas y métricas

Autor: CIAT-DAPA
Fecha: 2026
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from config import config
from parallel_processor import run_parallel_direct_alerts
from direct_alert import calculate_direct_alerts
from indirect_alert_refactored import calculate_indirect_alerts_batch
from enterprise_alert_calculator import calculate_enterprise_alerts_batch
from adm3_risk_calculator import calculate_adm3_risk_batch, save_adm3_risk_from_csv_batch
from supplier_risk_calculator import calculate_and_save_supplier_risk

# ===================== SETUP LOGGING =====================
def setup_logging(log_level: str = "WARNING", log_file: str = "pipeline.log"):
    """Configura logging centralizado para todo el pipeline."""
    lvl = getattr(logging, log_level.upper(), logging.WARNING)
    
    # Remover handlers existentes
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    
    # Formato
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Handler para archivo
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    
    # Handler para consola (solo WARNING+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    
    root.setLevel(lvl)
    root.addHandler(fh)
    root.addHandler(ch)
    
    return logging.getLogger(__name__)

# ===================== IMPORTACIONES OPCIONALES =====================
# Importar ORM para consultar capas de deforestación
try:
    from ganabosques_orm.collections.deforestation import Deforestation
    from ganabosques_orm.enums.deforestationsource import DeforestationSource
    from ganabosques_orm.enums.deforestationtype import DeforestationType
    from ganabosques_orm.enums.valuechain import ValueChain
    from mongoengine import connect
    HAS_ORM = True
except ImportError:
    logging.warning("ganabosques_orm no está disponible. Funcionalidad limitada.")
    HAS_ORM = False
    # Crear placeholders para evitar errores
    DeforestationSource = None
    DeforestationType = None
    ValueChain = None

# Importar DataManager
try:
    from data_manager import DataManager
    HAS_DATA_MANAGER = True
except ImportError as e:
    print(f"error importing data_manager: {e}", file=sys.stderr)
    logging.warning("data_manager no disponible. Funcionalidad limitada.")
    HAS_DATA_MANAGER = False

# Importar ErrorLogger
try:
    from error_logger import init_error_logger, get_error_logger, log_error
    HAS_ERROR_LOGGER = True
except ImportError:
    HAS_ERROR_LOGGER = False
    def init_error_logger(*args, **kwargs): return None
    def get_error_logger(): return None
    def log_error(*args, **kwargs): pass

# ===================== FUNCIÓN PARA PARSEAR PASOS =====================
def parse_steps(step_args: List[str]) -> List[int]:
    """
    Parsea los argumentos de pasos y retorna lista de números de paso.
    
    Formatos soportados:
    - "1" → [1]
    - "1-3" → [1, 2, 3]
    - "1-3 5" → [1, 2, 3, 5]
    - "1 3 5" → [1, 3, 5]
    - "all" → [1, 2, 3, 4, 5, 6]
    
    Args:
        step_args: Lista de strings con los pasos (ej: ['1-3', '5'])
        
    Returns:
        Lista ordenada de números de paso únicos
    """
    if not step_args:
        return []
    
    steps = set()
    
    for arg in step_args:
        arg = arg.strip().lower()
        
        if arg == 'all':
            return list(range(1, 10))  # Pasos 1-9
        
        if '-' in arg:
            # Rango: "1-3"
            try:
                start, end = map(int, arg.split('-'))
                steps.update(range(start, end + 1))
            except ValueError:
                print(f"⚠ Formato de paso inválido: {arg}")
        else:
            # Número individual: "1"
            try:
                steps.add(int(arg))
            except ValueError:
                print(f"⚠ Paso inválido: {arg}")
    
    # Filtrar pasos válidos (1-9)
    valid_steps = sorted([s for s in steps if 1 <= s <= 9])
    
    return valid_steps


# ===================== CONEXIÓN Y VALIDACIONES =====================
def connect_mongodb() -> bool:
    """Conecta a MongoDB usando variables de entorno."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB_NAME", "ganabosques")
    try:
        connect(host=mongo_uri, db=mongo_db)
        return True
    except Exception as e:
        logging.error(f"Error conectando a MongoDB: {e}")
        return False


def validate_prerequisites(
    stage: str,
    data_manager: Optional['DataManager'],
    farms_metadata: Optional[List[Dict]],
    periods: List[str],
    workspace_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Valida que existan los prerequisitos necesarios para ejecutar una etapa.
    
    Args:
        stage: Etapa a validar ('direct', 'movement', 'metrics', 'total')
        data_manager: Instancia de DataManager
        farms_metadata: Lista de metadata de farms
        periods: Lista de períodos a procesar
        workspace_dir: Directorio workspace (opcional)
        
    Returns:
        Tupla (is_valid, error_message)
    """
    errors = []
    
    # Validaciones comunes
    if not periods:
        errors.append("No hay períodos para procesar")
    
    # Validaciones por etapa
    if stage == 'direct':
        if not data_manager:
            errors.append("DataManager no disponible - se necesita para gestionar rasters")
        if not farms_metadata:
            # No es error fatal si hay geojsons en disco
            if data_manager and data_manager.geojsons_dir.exists():
                geojsons = list(data_manager.geojsons_dir.glob("*.geojson"))
                if not geojsons:
                    errors.append("No hay geojsons disponibles (ni en BD ni en disco)")
            else:
                errors.append("No hay farms_metadata y no se encontró carpeta de geojsons")
    
    elif stage == 'movement':
        if workspace_dir:
            movements_dir = workspace_dir / "movements"
            if not movements_dir.exists():
                errors.append(f"Directorio de movimientos no existe: {movements_dir}")
            else:
                movement_files = list(movements_dir.glob("movement_data_base_*.csv"))
                if not movement_files:
                    errors.append("No hay archivos de movimientos (movement_data_base_YYYY.csv)")
        
        # Verificar que existan alertas directas
        if workspace_dir:
            direct_alerts_dir = workspace_dir / "results" / "direct_alerts"
            if not direct_alerts_dir.exists():
                errors.append("No hay alertas directas previas - ejecuta --direct primero")
    
    elif stage == 'metrics':
        if not data_manager:
            errors.append("DataManager requerido para métricas espaciales")
        if not farms_metadata:
            errors.append("farms_metadata requerido para métricas espaciales")
    
    elif stage == 'enterprise':
        if workspace_dir:
            movements_dir = workspace_dir / "movements"
            if not movements_dir.exists():
                errors.append(f"Directorio de movimientos no existe: {movements_dir}")
            else:
                movement_files = list(movements_dir.glob("movement_data_base_*.csv"))
                if not movement_files:
                    errors.append("No hay archivos de movimientos (movement_data_base_YYYY.csv)")
        
            # Verificar que existan alertas directas
            direct_alerts_dir = workspace_dir / "results" / "direct_alerts"
            if not direct_alerts_dir.exists():
                errors.append("No hay alertas directas previas - ejecuta --direct primero")
    
    elif stage == 'total':
        if workspace_dir:
            direct_alerts_dir = workspace_dir / "results" / "direct_alerts"
            if not direct_alerts_dir.exists():
                errors.append("No hay alertas directas - ejecuta --direct primero")
            
            # metrics es opcional pero recomendado
            metrics_dir = workspace_dir / "metrics"
            if not metrics_dir.exists():
                logging.warning("No hay métricas espaciales - considera ejecutar --metrics")
    
    if errors:
        return False, "; ".join(errors)
    return True, ""


def print_validation_summary(stage: str, is_valid: bool, error_msg: str):
    """Imprime resumen de validación de forma consistente."""
    if is_valid:
        print(f"✅ Validación {stage}: OK")
    else:
        print(f"❌ Validación {stage}: FALLIDA")
        print(f"   Errores: {error_msg}")

def query_available_periods(source: str, period_type: str) -> List[Tuple[datetime, datetime, str]]:
    """
    Consulta MongoDB para obtener períodos disponibles según fuente y tipo de deforestación.
    
    Con el nuevo modelo ORM:
    - source: siempre 'smbyc' (única fuente actual)
    - period_type: 'annual', 'cumulative', 'nad', 'atd' (tipos de deforestación)
    
    Retorna lista de tuplas: (period_start, period_end, name, mongo_id)
    """
    if not HAS_ORM:
        print("⚠ ORM no disponible, no se pueden consultar períodos")
        return []
    
    try:
        # Convertir strings a enums del ORM
        # source siempre es 'smbyc' en el nuevo modelo
        source_enum = DeforestationSource[source.upper()]
        
        # Mapear period_type a DeforestationType
        # Ahora NAD y ATD son tipos de deforestación, no fuentes
        type_map = {
            "annual": DeforestationType.ANNUAL,
            "cumulative": DeforestationType.CUMULATIVE,
            "nad": DeforestationType.NAD,
            "atd": DeforestationType.ATD
        }
        type_enum = type_map.get(period_type.lower())
        
        if not type_enum:
            print(f"⚠ Tipo de deforestación desconocido: {period_type}")
            return []
        
        # Query usando source + deforestation_type
        query = {
            "deforestation_source": source_enum.value,
            "deforestation_type": type_enum.value
        }
        
        periods = Deforestation.objects(**query).order_by('period_start')
        # Incluir id de MongoDB para usarlo como deforestation_id al guardar
        result = [(p.period_start, p.period_end, p.name, str(p.id)) for p in periods if p.period_start and p.period_end]
        
        print(f"📊 Períodos encontrados en BD: {len(result)} para {source_enum.value}/{type_enum.value}")
        return result
    except KeyError as e:
        print(f"⚠ Fuente o tipo no válido: {e}")
        return []
    except Exception as e:
        print(f"⚠ Error consultando períodos: {e}")
        return []

def generate_year_ranges(years_input: str, source: str, period_type: str, available_periods: List) -> List[str]:
    """
    Genera lista de períodos a procesar según input del usuario y períodos disponibles EN LA BD.
    
    IMPORTANTE: Solo se procesan períodos que existen en MongoDB (y por ende en geoserver).
    Si un período no está en la BD, NO se procesa.
    
    Con el nuevo modelo ORM:
    - source: siempre 'smbyc' (única fuente)
    - period_type: determina el formato ('annual', 'cumulative' = YYYY-YYYY, 'nad', 'atd' = YYYYQQ)
    
    Formatos de entrada:
    - "2024" → un solo año
    - "2010-2024" → rango de años
    
    Retorna lista de strings según formato esperado por cada tipo.
    """
    import re
    
    # Parsear input del usuario
    if "-" in years_input and years_input.count("-") == 1:
        start_year, end_year = map(int, years_input.split("-"))
    else:
        start_year = end_year = int(years_input)
    
    periods = []
    
    if not available_periods:
        print(f"⚠ No hay capas {source.upper()}/{period_type} en la base de datos")
        return []
    
    # Determinar formato según period_type
    if period_type == "annual":
        # SMBYC anual: buscar períodos tipo "2012-2013", "2013-2014"
        for p in available_periods:
            name = p[2]  # nombre del período
            # Extraer rango de años del nombre (ej: "smbyc_deforestation_annual_2012-2013")
            match = re.search(r'(\d{4})-(\d{4})$', name) or re.search(r'(\d{4})(\d{4})$', name)
            if match:
                year_start = int(match.group(1))
                year_end = int(match.group(2))
                # Verificar si el rango solicitado incluye este período
                if year_start >= (start_year - 1) and year_end <= (end_year + 1):
                    periods.append(f"{year_start}-{year_end}")
                    
    elif period_type == "cumulative":
        # SMBYC acumulado: buscar períodos tipo "2010-2012", "2010-2013"
        for p in available_periods:
            name = p[2]
            match = re.search(r'(\d{4})-(\d{4})$', name) or re.search(r'(\d{4})(\d{4})$', name)
            if match:
                year_start = int(match.group(1))
                year_end = int(match.group(2))
                # Para acumulado, verificar que el año final esté en el rango
                if year_end >= start_year and year_end <= end_year:
                    periods.append(f"{year_start}-{year_end}")
    
    elif period_type in ["nad", "atd"]:
        # NAD/ATD trimestral: SOLO usar períodos que existen en BD
        # Formato: YYYYQQ (ej: 201701, 201702, 202203)
        seen_quarters = set()
        for p in available_periods:
            name = p[2]  # nombre del período
            # Extraer YYYYQQ del nombre (ej: "smbyc_deforestation_nad_202203")
            # NOTA: Temporalmente también soporta formato antiguo "nad_deforestation_quarter_202203"
            match = re.search(r'(\d{6})$', name)
            if match:
                period_code = match.group(1)
                year = int(period_code[:4])
                # Solo incluir si el año está en el rango solicitado
                if year >= start_year and year <= end_year and period_code not in seen_quarters:
                    periods.append(period_code)
                    seen_quarters.add(period_code)
    
    # Ordenar períodos
    periods.sort()
    return periods

def run_direct_alerts(titulo: str, continue_on_error: bool, params: Dict[str, Any], period_metadata: List[Tuple]) -> bool:
    """Ejecuta el cálculo de alertas directas con preparación de datos."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    # Verificar si usar modo paralelo
    use_parallel = params.get('_use_parallel', False)
    num_workers = params.get('_num_workers', None)
    
    try:
        data_manager = params.get('_data_manager')
        available_periods = params.get('_available_periods', [])
        
        # Si hay DataManager Y períodos disponibles en BD, preparar rasters desde BD
        if data_manager and available_periods:
            print(f"🔄 Preparando rasters para {len(params['years'])} períodos...")
            
            raster_paths = []
            for period_name in params['years']:
                # Buscar metadata del período en available_periods
                period_info = None
                for p in available_periods:
                    if period_name in p[2]:  # p[2] = name
                        period_info = p
                        break
                
                if not period_info:
                    print(f"⚠ No se encontró metadata para período: {period_name}")
                    continue
                
                # Buscar el documento completo para obtener el 'path'
                try:
                    defo_doc = Deforestation.objects(name=period_info[2]).first()
                    
                    if not defo_doc or not defo_doc.path:
                        print(f"⚠ No se encontró path para: {period_name}")
                        continue
                    
                    layer_name = defo_doc.path  # Este es el coverageId real en geoserver
                    time_filter = data_manager.extract_time_filter_from_name(period_info[2])
                    
                    # Asegurar que el raster esté disponible (caché o descarga)
                    raster_path = data_manager.ensure_raster_available(
                        source=params['source'],
                        period_type=params['period_type'],
                        layer_name=layer_name,
                        period_name=period_info[2],  # Para el nombre del archivo local
                        time_filter=time_filter
                    )
                    
                    if raster_path:
                        raster_paths.append((period_name, raster_path))
                    else:
                        print(f"⚠ No se pudo obtener raster para: {period_name}")
                except Exception as e:
                    print(f"⚠ Error obteniendo path para {period_name}: {e}")
            
            print(f"✅ Rasters preparados: {len(raster_paths)}/{len(params['years'])}")
            
            # Crear diccionario de rutas: {period_name: raster_path}
            raster_paths_dict = {period: path for period, path in raster_paths}
            params['raster_paths_dict'] = raster_paths_dict
        
        # Modo fallback: sin BD o sin períodos, buscar rasters en carpeta
        elif data_manager:
            print(f"📂 Modo fallback: buscando rasters en carpeta...")
            raster_paths = []
            
            # Construir ruta base de rasters según fuente
            # Estructura ACTUAL (temporal): workspace/alertas/rasters/nad/quarter/nad_deforestation_quarter_201701.tif
            # Estructura FUTURA (cuando geoserver cambie): workspace/alertas/rasters/smbyc/nad/smbyc_deforestation_nad_201701.tif
            rasters_base = data_manager.workspace_dir / "rasters"
            source_lower = params['source'].lower()
            period_type = params['period_type']
            
            # TEMPORAL: Mantener estructura antigua para NAD/ATD hasta que geoserver cambie
            if period_type in ["nad", "atd"]:
                # Estructura antigua: rasters/nad/quarter/nad_deforestation_quarter_YYYYQQ.tif
                source_folder = rasters_base / period_type / "quarter"
                expected_pattern = lambda pt, pn: f"{pt}_deforestation_quarter_{pn}.tif"
                # FUTURA: source_folder = rasters_base / source_lower / period_type
                # FUTURA: expected_pattern = lambda pt, pn: f"{source_lower}_deforestation_{pt}_{pn}.tif"
            else:
                # Estructura para SMBYC annual/cumulative
                source_folder = rasters_base / source_lower / period_type
                expected_pattern = lambda pt, pn: f"{source_lower}_deforestation_{pt}_{pn}.tif"
            
            if not source_folder.exists():
                print(f"⚠ Carpeta de rasters no existe: {source_folder}")
                params['raster_paths_dict'] = None
            else:
                # Buscar rasters con patrón esperado
                for period_name in params['years']:
                    expected_name = expected_pattern(period_type, period_name)
                    raster_path = source_folder / expected_name
                    
                    if raster_path.exists():
                        raster_paths.append((period_name, str(raster_path)))
                        print(f"✅ Raster encontrado: {expected_name}")
                    else:
                        print(f"⚠ Raster no encontrado: {expected_name}")
                
                if raster_paths:
                    print(f"✅ Rasters encontrados en carpeta: {len(raster_paths)}/{len(params['years'])}")
                    raster_paths_dict = {period: path for period, path in raster_paths}
                    params['raster_paths_dict'] = raster_paths_dict
                else:
                    print(f"❌ No se encontraron rasters en: {source_folder}")
                    params['raster_paths_dict'] = None
        else:
            params['raster_paths_dict'] = None
        
        # Ejecutar en modo paralelo o secuencial
        if use_parallel:
            print(f"\n🚀 Usando procesamiento PARALELO con {num_workers or 'auto'} workers")
            
            # Remover parámetros internos antes de pasar
            clean_params = {k: v for k, v in params.items() if not k.startswith('_')}
            # Agregar de vuelta los parámetros necesarios para el modo paralelo
            clean_params['_data_manager'] = params.get('_data_manager')
            clean_params['_farms_metadata'] = params.get('_farms_metadata')
            clean_params['_farm_limit'] = params.get('_farm_limit')
            clean_params['_workspace_dir'] = params.get('_workspace_dir')
            # Importante: pasar raster_paths_dict para que cada worker tenga las rutas
            clean_params['raster_paths_dict'] = params.get('raster_paths_dict')
            
            result = run_parallel_direct_alerts(
                params=clean_params,
                num_workers=num_workers,
                cleanup=True
            )
            
            print(f"✔ Alertas directas (paralelo) completadas.")
            print(f"   • Workers: {result['num_workers']}")
            print(f"   • Chunks procesados: {result['chunks_processed']}")
            print(f"   • Períodos: {result['periods_processed']}")
            print(f"   • Tiempo: {result['execution_time']:.2f}s ({result['execution_time']/60:.1f} min)")
            return True
        else:
            # Modo secuencial (original)
            
            # Pasar parámetros internos necesarios para direct_alert
            call_params = {k: v for k, v in params.items() if not k.startswith('_')}
            call_params['_data_manager'] = params.get('_data_manager')
            call_params['_farms_metadata'] = params.get('_farms_metadata')
            call_params['_farm_limit'] = params.get('_farm_limit')
            call_params['raster_paths_dict'] = params.get('raster_paths_dict')
            
            result = calculate_direct_alerts(**call_params)
            print(f"✔ Alertas directas completadas.")
            print(f"   • Períodos: {result['periods_processed']}")
            print(f"   • Predios: {result['farms_processed']}")
            print(f"   • Tiempo: {result['execution_time']:.2f}s")
            return True
    except Exception as e:
        print(f"✖ Error ejecutando alertas directas: {e}")
        import traceback
        traceback.print_exc()
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
        return True

def run_spatial_metrics(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el cálculo de métricas espaciales."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    try:
        from spatial_metrics import calculate_spatial_metrics
        
        data_manager = params.get('_data_manager')
        farms_metadata = params.get('_farms_metadata')
        use_parallel = params.get('_use_parallel', False)
        num_workers = params.get('_num_workers')
        
        if not data_manager:
            print("❌ Error: DataManager no disponible")
            return False
        
        if not farms_metadata:
            print("❌ Error: No hay metadata de farms disponible")
            print("   Asegúrate de tener conexión a MongoDB")
            return False
        
        # Calcular métricas (con soporte paralelo)
        result = calculate_spatial_metrics(
            farms_metadata=farms_metadata,
            data_manager=data_manager,
            use_parallel=use_parallel,
            num_workers=num_workers
        )
        
        if result['success']:
            print(f"✔ Métricas espaciales completadas.")
            print(f"   • Farms procesados: {result['farms_processed']:,}")
            print(f"   • Archivo: {result['output_file']}")
            print(f"   • Tiempo: {result['execution_time']:.2f}s")
            return True
        else:
            print(f"✖ Error calculando métricas espaciales")
            return False
            
    except ImportError as e:
        print(f"✖ Error: No se pudo importar spatial_metrics: {e}")
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
        return True
    except Exception as e:
        print(f"✖ Error ejecutando métricas espaciales: {e}")
        import traceback
        traceback.print_exc()
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
        return True

def run_indirect_alerts(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el cálculo de alertas indirectas (movimiento) usando función callable.
    También genera alertas de empresa en la misma pasada."""

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    try:
        # Extraer parámetros necesarios
        periods = params.get('years', [])
        period_type = params.get('period_type', 'quarter')
        source = params.get('source', 'smbyc')
        workspace_dir = Path(params.get('_workspace_dir', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data/'))
        
        # Directorios con nueva estructura: results/{source}/{period_type}/{stage}/
        results_base = workspace_dir / "alertas" / "results"
        direct_alerts_dir = results_base / source / period_type / "direct_alerts"
        movement_csv_dir = workspace_dir / "alertas" / "movements"
        output_dir = results_base / source / period_type / "indirect_alerts"
        enterprise_output_dir = results_base / source / period_type / "enterprise_alerts"
        
        # Ejecutar batch (genera indirectos + empresas en una sola pasada)
        result = calculate_indirect_alerts_batch(
            periods=periods,
            period_type=period_type,
            source=source,
            direct_alerts_dir=str(direct_alerts_dir),
            movement_csv_dir=str(movement_csv_dir),
            output_dir=str(output_dir),
            enterprise_output_dir=str(enterprise_output_dir),
            generate_enterprise_alerts=True,  # Generar también alertas de empresa
            normalize_ids=True,
            strip_dot_zero=True,
            strip_leading_zeros=True,
            uppercase_ids=True
        )
        
        if result['success']:
            print(f"\n✔ Alertas indirectas completadas.")
            print(f"   • Períodos: {result['periods_processed']}")
            print(f"   • Alertas indirectas (IN): {result['total_indirect_in']}")
            print(f"   • Alertas indirectas (OUT): {result['total_indirect_out']}")
            
            # Mostrar estadísticas de empresa si se generaron
            if result.get('enterprise_generated'):
                print(f"\n✔ Alertas de empresa generadas (misma pasada):")
                print(f"   • Movimientos entrada (in): {result['total_enterprise_entries']}")
                print(f"   • Movimientos salida (out): {result['total_enterprise_exits']}")
                print(f"   • Total registros: {result['total_enterprise_records']}")
            
            return True
        else:
            print(f"\n⚠ Alertas indirectas finalizadas con errores:")
            print(f"   • Períodos exitosos: {result['periods_processed']}")
            print(f"   • Períodos fallidos: {result['periods_failed']}")
            return not continue_on_error
    
    except Exception as e:
        logging.error(f"Error en alertas indirectas: {e}", exc_info=True)
        print(f"✖ Error ejecutando alertas indirectas: {e}")
        return continue_on_error


def run_enterprise_alerts(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el cálculo de alertas de empresa basado en movimientos con fincas alertadas."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    try:
        # Extraer parámetros necesarios
        periods = params.get('years', [])
        period_type = params.get('period_type', 'quarter')
        source = params.get('source', 'smbyc')
        workspace_dir = Path(params.get('_workspace_dir', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data/'))
        
        # Directorios con nueva estructura: results/{source}/{period_type}/{stage}/
        results_base = workspace_dir / "alertas" / "results"
        direct_alerts_dir = results_base / source / period_type / "direct_alerts"
        movement_csv_dir = workspace_dir / "alertas" / "movements"
        output_dir = results_base / source / period_type / "enterprise_alerts"
        
        # Ejecutar batch
        result = calculate_enterprise_alerts_batch(
            periods=periods,
            period_type=period_type,
            source=source,
            direct_alerts_dir=str(direct_alerts_dir),
            movement_csv_dir=str(movement_csv_dir),
            output_dir=str(output_dir),
            enterprise_types=None  # Todas las empresas
        )
        
        if result['success']:
            print(f"✔ Alertas de empresa completadas.")
            print(f"   • Períodos: {result['periods_processed']}")
            print(f"   • Movimientos entrada (in): {result['total_entries']}")
            print(f"   • Movimientos salida (out): {result['total_exits']}")
            print(f"   • Total registros: {result['total_records']}")
            return True
        else:
            print(f"⚠ Alertas de empresa finalizadas con errores:")
            print(f"   • Períodos exitosos: {result['periods_processed']}")
            print(f"   • Períodos fallidos: {result['periods_failed']}")
            if result.get('failed_periods'):
                print(f"   • Períodos con error: {', '.join(result['failed_periods'])}")
            return not continue_on_error
    
    except Exception as e:
        logging.error(f"Error en alertas de empresa: {e}", exc_info=True)
        print(f"✖ Error ejecutando alertas de empresa: {e}")
        return continue_on_error


def run_total_risk(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el cálculo de riesgo total consolidando alertas directas, indirectas y métricas."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    try:
        from total_alert import calculate_total_risk
        
        data_manager = params.get('_data_manager')
        workspace_dir = str(data_manager.workspace_dir) if data_manager else params.get('_workspace_dir')
        
        if not workspace_dir:
            print("❌ Error: workspace_dir no disponible")
            return False
        
        # Obtener mapeo MongoDB desde DataManager (evita consulta adicional)
        mongo_map_df = None
        if data_manager:
            mongo_map_df = data_manager.get_mongo_id_map_df()
            if mongo_map_df is not None and not mongo_map_df.empty:
                print(f"📦 Mapeo MongoDB disponible desde DataManager: {len(mongo_map_df)} farms")
        
        result = calculate_total_risk(
            source=params['source'],
            period_type=params['period_type'],
            periods=params['years'],
            workspace_dir=workspace_dir,
            use_precalculated_metrics=True,
            mongo_map_df=mongo_map_df
        )
        
        if result['success']:
            print(f"✔ Riesgo total calculado correctamente.")
            print(f"   • Períodos procesados: {result['periods_processed']}")
            print(f"   • Archivos generados: {len(result['files_generated'])}")
            print(f"   • Tiempo: {result['execution_time']:.2f}s")
            return True
        else:
            print(f"⚠ No se generaron archivos de riesgo total")
            return False
        
    except Exception as e:
        logging.error(f"Error en riesgo total: {e}", exc_info=True)
        print(f"✖ Error ejecutando riesgo total: {e}")
        import traceback
        traceback.print_exc()
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
        return True


def run_save_to_db(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el guardado de resultados en MongoDB (Analysis, FarmRisk, EnterpriseRisk)."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    data_manager = params.get('data_manager')
    available_periods = params.get('available_periods', [])
    periods = params.get('periods', [])
    source = params.get('source', 'nad')
    period_type = params.get('period_type', 'quarter')
    value_chain = params.get('value_chain', 'livestock')
    
    # Nuevos parámetros para bulk insert
    bulk_insert = params.get('bulk_insert', False)
    chunk_size = params.get('chunk_size', 1000)
    args = params.get('args')  # Para compatibilidad con código existente
    
    if not data_manager:
        print("❌ Error: DataManager no disponible")
        return False
    
    if bulk_insert:
        print(f"🚀 Modo BULK INSERT activado (chunks de {chunk_size})")
        print(f"   ⚠️  IMPORTANTE: Solo hace INSERT, no UPDATE")
        print(f"   ⚠️  Si hay datos existentes, se producirán errores de duplicado")
    else:
        print(f"🔄 Modo GRANULAR activado (con UPSERT)")
        print(f"   ℹ️  Actualiza registros existentes, más lento pero más seguro")
    
    # Construir diccionario de período -> deforestation_id desde available_periods
    # available_periods tiene formato: [(start_date, end_date, name, mongo_id), ...]
    period_to_defo_id = {}
    for p in available_periods:
        if len(p) >= 4 and p[3]:
            # Extraer período del nombre (ej: "nad_deforestation_quarter_201701" -> "201701")
            name = p[2]
            import re
            match = re.search(r'(\d{6}|\d{4}-\d{4})$', name)
            if match:
                period_key = match.group(1)
                period_to_defo_id[period_key] = p[3]  # mongo_id
    
    if period_to_defo_id:
        print(f"📋 Capas de deforestación mapeadas: {len(period_to_defo_id)}")
    else:
        print("⚠ No se encontraron IDs de capas de deforestación en available_periods")
    
    import pandas as pd
    total_farm_saved = 0
    total_farm_failed = 0
    total_ent_saved = 0
    total_ent_failed = 0
    
    # Procesar cada período individualmente
    for period in periods:
        # Buscar deforestation_id para este período
        deforestation_id = period_to_defo_id.get(period)
        
        if not deforestation_id:
            # Intentar buscar directamente en MongoDB por nombre
            expected_name = f"{source}_deforestation_{period_type}_{period}"
            try:
                defo_doc = Deforestation.objects(name=expected_name).first()
                if defo_doc:
                    deforestation_id = str(defo_doc.id)
                    print(f"   🔍 Encontrado en BD: {expected_name} -> {deforestation_id}")
            except Exception as e:
                print(f"   ⚠ Error buscando capa {expected_name}: {e}")
        
        if not deforestation_id:
            print(f"   ⚠ Sin deforestation_id para período {period}, saltando...")
            continue
        
        # Crear o obtener Analysis para este período/capa (con value_chain)
        analysis_id, analysis_error = data_manager.create_or_get_analysis(
            deforestation_id, 
            value_chain=value_chain
        )
        
        if analysis_error:
            print(f"   ❌ Error creando Analysis para {period}: {analysis_error}")
            continue
        
        print(f"\n📅 Período {period} (Analysis: {analysis_id[:8]}...)")
        
        # Guardar FarmRisk (desde CSV de total_risk que consolida todo)
        # Nueva estructura: results/{source}/{deforestation_type}/total_risk/{period}/
        total_csv = data_manager.get_results_dir('total_risk', source=source, deforestation_type=period_type, period=period) / f"{source}_total_risk_{period_type}_{period}.csv"
        
        if total_csv.exists():
            try:
                df = pd.read_csv(total_csv)
                
                # Decidir qué método usar según el flag --bulk-insert
                if bulk_insert:
                    print(f"   📦 Usando bulk insert (chunks de {chunk_size})...")
                    saved, failed, errors = data_manager.save_farm_risk_to_db_bulk(
                        df, analysis_id, chunk_size=chunk_size
                    )
                else:
                    print(f"   🔄 Usando método granular con upsert...")
                    saved, failed, errors = data_manager.save_farm_risk_to_db(df, analysis_id)
                
                total_farm_saved += saved
                total_farm_failed += failed
                print(f"   💾 FarmRisk: {saved} guardados, {failed} fallidos")
                
                # Mostrar errores de manera más clara según el método
                if errors:
                    if bulk_insert:
                        # En bulk insert, errores pueden ser de chunks enteros
                        bulk_errors = [e for e in errors if "count" in e]
                        individual_errors = [e for e in errors if "count" not in e]
                        
                        for err in bulk_errors[:3]:  # Máximo 3 errores de chunks
                            print(f"      ⚠ Chunk error: {err['error']} (registros: {err.get('count', '?')})")
                        
                        if individual_errors and len(individual_errors) <= 5:
                            for err in individual_errors[:5]:  # Máximo 5 errores individuales
                                print(f"      ⚠ {err}")
                        elif individual_errors:
                            print(f"      ⚠ {len(individual_errors)} errores individuales (usar --verbose para ver detalles)")
                    else:
                        # Método granular, mostrar algunos errores
                        for err in errors[:3]:
                            print(f"      ⚠ {err}")
                        if len(errors) > 3:
                            print(f"      ... y {len(errors) - 3} errores más")
            except Exception as e:
                print(f"   ⚠ Error leyendo/guardando FarmRisk: {e}")
        else:
            print(f"   ⚠ No existe CSV de total_risk: {total_csv.name}")
            print(f"      Ejecuta el pipeline completo o paso 5 primero")
        
        # Guardar EnterpriseRisk (desde resultados de alertas de empresa)
        # Nueva estructura: results/{source}/{deforestation_type}/enterprise_alerts/{period}/
        enterprise_dir = data_manager.get_results_dir('enterprise_alerts', source=source, deforestation_type=period_type, period=period)
        ent_csv = enterprise_dir / f"{source}_enterprise_alert_{period_type}_{period}.csv"
        
        if ent_csv.exists():
            try:
                df = pd.read_csv(ent_csv)
                saved, failed, errors = data_manager.save_enterprise_risk_to_db(df, analysis_id)
                total_ent_saved += saved
                total_ent_failed += failed
                print(f"   💾 EnterpriseRisk: {saved} guardados, {failed} fallidos")
                if errors and len(errors) <= 3:
                    for err in errors:
                        print(f"      ⚠ {err}")
            except Exception as e:
                print(f"   ⚠ Error leyendo/guardando EnterpriseRisk: {e}")
        else:
            print(f"   ⚠ No existe CSV de alertas empresa: {ent_csv.name}")
    
    # Resumen de guardado
    print(f"\n📊 Resumen de guardado en BD:")
    print(f"   • FarmRisk: {total_farm_saved} guardados, {total_farm_failed} fallidos")
    print(f"   • EnterpriseRisk: {total_ent_saved} guardados, {total_ent_failed} fallidos")
    
    return total_farm_saved > 0 or total_ent_saved > 0


def run_subprocess_script(script_path: Path, titulo: str, continue_on_error: bool, env_vars: dict = None) -> bool:
    """Ejecuta un script mediante subprocess (para scripts no refactorizados)."""
    import subprocess
    
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print(f"▶ Ejecutando: {script_path}")
    
    # Preparar entorno con variables adicionales
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        print(f"  Variables: {', '.join(f'{k}={v}' for k, v in env_vars.items())}")
    
    rc = subprocess.run([sys.executable, str(script_path)], env=env, check=False).returncode
    if rc != 0:
        print(f"✖ Error ejecutando {script_path.name} (código {rc}).")
        if not continue_on_error:
            print("Pipeline detenido.")
            return False
    else:
        print(f"✔ {script_path.name} finalizado correctamente.")
    return True


# ===================== FUNCIÓN PRINCIPAL =====================
def main():
    """Punto de entrada principal del pipeline."""
    
    # Configurar logging inicial
    logger = setup_logging(
        log_level=config.get('LOG_LEVEL', 'WARNING'),
        log_file=config.get('LOG_FILE', 'pipeline.log')
    )
    
    parser = argparse.ArgumentParser(
        description="Pipeline de Alertas de Deforestación para Ganabosques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pasos disponibles:
  1: Alertas directas (intersección con raster de deforestación)
  2: Alertas indirectas (movimientos de ganado) + Alertas de empresa
  3: Métricas espaciales (frontera agrícola, áreas protegidas)
  4: Alertas de empresa standalone (solo si no se corrió paso 2)
  5: Consolidación de riesgo total
  6: Guardar FarmRisk/EnterpriseRisk en MongoDB
  7: Calcular riesgo ADM3 (genera CSVs para revisión)
  8: Guardar ADM3 Risk en MongoDB (lee CSVs del paso 7)

Ejemplos de uso:
  # Pipeline completo (pasos 1-5):
  python main.py -s smbyc -pt annual -y 2017-2024
  
  # Solo paso 1 (alertas directas):
  python main.py -s smbyc -pt nad -y 2024 -p 1
  
  # Pasos 1 al 3:
  python main.py -s smbyc -pt nad -y 2024 -p 1-3
  
  # Pipeline completo + guardar en BD:
  python main.py -s smbyc -pt nad -y 2024 -p 1-6
  
  # Calcular ADM3 (revisión):
  python main.py -s smbyc -pt annual -y 2010-2024 -p 7
  
  # Guardar ADM3 en BD (después de revisar CSVs):
  python main.py -s smbyc -pt annual -y 2010-2024 -p 8
  
  # Prueba rápida con pocos farms:
  python main.py -s smbyc -pt nad -y 2024 -p 1 --farm-limit 10
  
  # Pipeline completo en paralelo:
  python main.py -s smbyc -pt nad -y 2024 --parallel -w 8
        """
    )
    
    # Parámetros requeridos (usando valores de los enums del ORM)
    # NOTA: Con el nuevo ORM, DeforestationSource solo tiene 'smbyc', pero dejamos preparado para futuras fuentes
    source_choices = [s.value for s in DeforestationSource] if HAS_ORM else ["smbyc"]
    parser.add_argument("--source", "-s",
                       choices=source_choices,
                       default="smbyc",  # Default a smbyc (única fuente actual)
                       help="Fuente de datos de deforestación (default: smbyc)")
    
    # period-type ahora incluye nad y atd como tipos de deforestación
    type_choices = [s.value for s in DeforestationType] if HAS_ORM else ["annual", "cumulative", "nad", "atd"]
    parser.add_argument("--period-type", "-pt",
                       choices=type_choices,
                       required=True,
                       help="Tipo de deforestación: annual, cumulative (SMBYC histórico) o nad, atd (alertas trimestrales)")
    
    parser.add_argument("--years", "-y",
                       required=True,
                       help="Años a procesar: '2024' o '2010-2024'")
    
    # Parámetros opcionales
    parser.add_argument("--empresa", "-e",
                       help="Nombre de la empresa")
    
    # Sistema de pasos del pipeline
    parser.add_argument("--step", "-p", nargs='*', default=None,
                       help="Pasos a ejecutar: 1-9. Ejemplos: '1' (solo paso 1), '1-3' (pasos 1 al 3), '9' (riesgo por suppliers). Sin este flag se ejecutan pasos 1-5.")
    
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continuar con etapas siguientes aunque una falle")
    
    # Testing y debugging
    parser.add_argument("--farm-limit", type=int, default=None,
                       help="Limitar número de farms a procesar (para testing)")
    
    # Paralelización
    parser.add_argument("--parallel", action="store_true",
                       help="Usar procesamiento paralelo para mayor velocidad")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Número de workers paralelos (default: número de CPUs)")
    
    # Precisión de cálculo
    parser.add_argument("--precise-area", action="store_true",
                       help="Usar super-sampling para cálculo preciso de fracciones de píxel (default: 5×5 divisiones)")
    parser.add_argument("--pixel-divisions", type=int, default=None,
                       help="Número de divisiones por píxel (ej: 5 = 5×5 = 25 sub-píxeles). Solo aplica con --precise-area. Default: 5")
    parser.add_argument("--offline", action="store_true",
                       help="Modo offline: usar solo archivos locales, no conectar a MongoDB")
    
    # Optimización de base de datos
    parser.add_argument("--bulk-insert", action="store_true",
                       help="Usar bulk insert para guardar datos en MongoDB (más rápido pero solo INSERT, no UPDATE)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Tamaño de chunks para bulk insert (default: 1000)")
    
    # Cadena de valor para el análisis
    value_chain_choices = [v.value for v in ValueChain] if HAS_ORM and ValueChain else ["livestock", "cacao"]
    parser.add_argument("--value-chain", "-vc",
                       choices=value_chain_choices,
                       default="livestock",
                       help="Cadena de valor del análisis: livestock (ganado) o cacao (default: livestock)")

    parser.add_argument("--dry-run", action="store_true",
                       help="Realizar un dry run sin ejecutar cálculos, solo mostrando configuración y validaciones")

    args = parser.parse_args()
    
    # Validar combinaciones de fuente y tipo de período
    # Con el nuevo ORM: source siempre es 'smbyc', y period_type puede ser annual/cumulative/nad/atd
    source_value = args.source.lower()
    period_type_value = args.period_type.lower()
    
    # Validaciones: smbyc es la única fuente, todos los period_type son válidos para smbyc
    # (En el futuro, si hay otras fuentes, aquí se agregarían validaciones específicas)
    valid_period_types = ["annual", "cumulative", "nad", "atd"]
    if period_type_value not in valid_period_types:
        print(f"❌ Error: --period-type debe ser uno de: {', '.join(valid_period_types)}")
        sys.exit(1)
    
    # Conectar a MongoDB y consultar períodos disponibles
    print(f"\n{'='*70}")
    print(f"🚀 Iniciando Pipeline de Alertas de Deforestación")
    print(f"{'='*70}")
    print(f"📋 Configuración:")
    print(f"  • Fuente: {args.source.upper()}")
    print(f"  • Tipo de período: {args.period_type}")
    print(f"  • Años: {args.years}")
    if args.empresa:
        print(f"  • Empresa: {args.empresa}")
    if getattr(args, 'bulk_insert', False):
        print(f"  • Bulk Insert: ACTIVADO (chunks de {args.chunk_size})")
        print(f"    ⚠️  Nota: Solo INSERT (no UPDATE), limpiar datos anteriores si es necesario")
    else:
        print(f"  • Bulk Insert: DESACTIVADO (granular con UPSERT)")
    
    # Determinar si usar modo offline
    offline_mode = getattr(args, 'offline', False)
    
    # Inicializar logger de errores en CSV
    workspace_dir = config.get('WORKSPACE_DIR', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data')
    error_output_dir = os.path.join(workspace_dir, 'alertas', 'logs')
    error_logger = init_error_logger(error_output_dir, prefix=f"errors_{args.source}_{args.period_type}")
    if error_logger:
        print(f"📝 Logger de errores: {error_logger.filepath}")
    
    available_periods = []
    if offline_mode:
        print("\n🔌 MODO OFFLINE: Usando archivos locales, sin conexión a BD")
    elif HAS_ORM and connect_mongodb():
        available_periods = query_available_periods(args.source, args.period_type)
    
    # Generar lista de períodos a procesar (SOLO los que existen en BD)
    periods = generate_year_ranges(args.years, args.source, args.period_type, available_periods)
    
    if not periods:
        print(f"\n❌ ERROR: No hay capas de deforestación disponibles para los parámetros especificados")
        print(f"   Fuente: {args.source.upper()}")
        print(f"   Tipo: {args.period_type}")
        print(f"   Años solicitados: {args.years}")
        
        # Mostrar qué períodos SÍ están disponibles para ayudar al usuario
        if available_periods:
            print(f"\n📋 Períodos disponibles en la base de datos ({len(available_periods)}):")
            # Mostrar primeros y últimos 5 para no saturar
            display_periods = available_periods[:5] + (available_periods[-5:] if len(available_periods) > 10 else [])
            for p in display_periods[:10]:
                year_range = f"{p[0].year}-{p[1].year}" if p[0].year != p[1].year else str(p[0].year)
                print(f"   • {p[2]} ({year_range})")
            if len(available_periods) > 10:
                print(f"   ... y {len(available_periods) - 10} más")
            sys.exit(1)
        else:
            print(f"\n⚠ No hay conexión a BD o no hay períodos de {args.source.upper()} {args.period_type}")
            print(f"📂 Intentando modo fallback: usar rasters disponibles en disco...")
            # Generar períodos básicos para continuar (fallback mode)
            if args.period_type == "nad" or args.period_type == "atd":
                # Para quarter, generar los 4 trimestres del año
                if "-" in args.years:
                    start_year, end_year = map(int, args.years.split("-"))
                else:
                    start_year = end_year = int(args.years)
                
                for year in range(start_year, end_year + 1):
                    for q in range(1, 5):
                        periods.append(f"{year}0{q}")
            elif args.period_type == "annual":
                # Para annual SMBYC: año-siguiente_año
                # NOTA: No hay datos del 2011, entonces:
                # - Si empieza en 2010: primer período es 2010-2012, luego 2012-2013, 2013-2014...
                # - Si empieza en 2012 o después: normal (2012-2013, 2013-2014...)
                if "-" in args.years:
                    start_year, end_year = map(int, args.years.split("-"))
                else:
                    start_year = end_year = int(args.years)
                
                if start_year <= 2010:
                    # Primer período especial: 2010-2012 (salta el 2011)
                    periods.append("2010-2012")
                    # Continuar desde 2012 en adelante
                    for y in range(2012, end_year):
                        periods.append(f"{y}-{y+1}")
                else:
                    # Normal: año-siguiente
                    for y in range(start_year, end_year):
                        periods.append(f"{y}-{y+1}")
            
            elif args.period_type == "cumulative":
                # Para cumulative SMBYC: inicio-año_final
                # NOTA: No hay datos del 2011, entonces:
                # - Si empieza en 2010: primer período es 2010-2012, luego 2010-2013, 2010-2014...
                # - Si empieza en 2012 o después: normal (2012-2013, 2012-2014...)
                if "-" in args.years:
                    start_year, end_year = map(int, args.years.split("-"))
                else:
                    start_year = end_year = int(args.years)
                
                base_year = start_year
                
                if start_year <= 2010:
                    base_year = 2010
                    # Primer período especial: 2010-2012 (salta el 2011)
                    # Luego 2010-2013, 2010-2014, etc.
                    for y in range(2012, end_year + 1):
                        periods.append(f"{base_year}-{y}")
                else:
                    # Normal: inicio-año (incrementando)
                    for y in range(start_year + 1, end_year + 1):
                        periods.append(f"{base_year}-{y}")


            if not periods:
                print(f"\n❌ No se pueden generar períodos en modo fallback")
                sys.exit(1)
            
            print(f"✅ Modo fallback activado: {len(periods)} períodos generados")
    
    print(f"\n📅 Períodos a procesar ({len(periods)}):")
    for i, p in enumerate(periods, 1):
        print(f"  {i}. {p}")
    
    # Advertencia si se solicitó un rango pero solo se encontraron algunos períodos
    if "-" in args.years:
        start_req, end_req = map(int, args.years.split("-"))
        years_requested = end_req - start_req + 1
        
        # Calcular años esperados según tipo
        if args.period_type == "annual":
            periods_expected = years_requested  # Un período por año
        elif args.period_type == "cumulative":
            periods_expected = years_requested  # Un período acumulado por año
        elif args.period_type == "quarter":
            periods_expected = years_requested * 4  # 4 trimestres por año
        else:
            periods_expected = None
        
        if periods_expected and len(periods) < periods_expected:
            missing = periods_expected - len(periods)
            print(f"\n⚠ ADVERTENCIA: Se encontraron {len(periods)} de {periods_expected} períodos esperados")
            print(f"   Rango solicitado: {args.years}")
            print(f"   Períodos faltantes: ~{missing}")
            print(f"   Se procesarán SOLO los períodos disponibles en la base de datos")
            
            # Mostrar el rango real que se va a procesar
            if periods:
                if args.period_type in ["annual", "cumulative"]:
                    first_years = periods[0].split("-")
                    last_years = periods[-1].split("-")
                    print(f"   ✓ Rango real a procesar: {first_years[-1]}-{last_years[-1]}")
                elif args.period_type == "quarter":
                    first_year = periods[0][:4]
                    last_year = periods[-1][:4]
                    print(f"   ✓ Años con datos: {first_year}-{last_year}")
    
    # Preparar variables de entorno para pasar a los scripts
    env_vars = {
        "DEFORESTATION_SOURCE": args.source,
        "PERIODO": args.period_type,
        "YEARS": ",".join(periods),
    }
    if args.empresa:
        env_vars["EMPRESA"] = args.empresa
    
    # Cargar configuración y preparar DataManager
    try:
        # Agregar src/ al path para importar config
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        
        # Setear variables de entorno que config.py espera (para evitar errores)
        os.environ["EMPRESA"] = args.empresa or "test"
        os.environ["YEARS"] = ",".join(periods)
        os.environ["PERIODO"] = args.period_type
        
        # Obtener farm_limit antes de cualquier try-except para asegurar que esté disponible
        farm_limit = args.farm_limit  # Siempre tiene valor (None o número)

        # ===================== DETERMINAR ETAPAS A EJECUTAR (TEMPRANO) =====================
        # Necesitamos saber qué pasos se ejecutarán ANTES de cargar datos
        # para evitar cargar geometrías innecesariamente
        STEP_TO_STAGE = {
            1: 'direct',
            2: 'movement',
            3: 'metrics',
            4: 'enterprise',
            5: 'total',
            6: 'save',
            7: 'adm3',
            8: 'save_adm3',
            9: 'supplier_risk'
        }
        
        stages_to_run = []
        
        if args.step is not None:
            if len(args.step) == 0:
                # -p sin argumentos = mostrar ayuda de pasos (se manejará después)
                stages_to_run = []
            else:
                requested_steps = parse_steps(args.step)
                for step_num in requested_steps:
                    stage_key = STEP_TO_STAGE.get(step_num)
                    if stage_key:
                        if stage_key == 'enterprise' and 2 in requested_steps:
                            continue  # Omitir paso 4 si paso 2 está incluido
                        stages_to_run.append(stage_key)
        else:
            # Sin -p: pipeline completo
            stages_to_run = ['direct', 'movement', 'metrics', 'total', 'save']
        
        # Determinar si necesitamos cargar geometrías (solo para pasos 1 y 3)
        needs_geometries = 'direct' in stages_to_run or 'metrics' in stages_to_run
        
        # Inicializar DataManager si está disponible
        data_manager = None
        farms_metadata = None  # Inicializar para que esté disponible fuera del bloque
        geojsons_folder = config.get('FOLDER_GEOJSONS', '')  # Fallback por defecto
        
        if HAS_DATA_MANAGER:
            try:
                workspace_dir = config.get('WORKSPACE_DIR')
                gs_url = config.get('GEOSERVER_URL') if not offline_mode else None  # No conectar en offline
                gs_user = config.get('GEOSERVER_USER', 'admin')
                gs_pass = config.get('GEOSERVER_PASS', 'geoserver')
                
                if workspace_dir:
                    data_manager = DataManager(
                        workspace_dir=workspace_dir,
                        geoserver_url=gs_url,
                        geoserver_user=gs_user,
                        geoserver_pass=gs_pass
                    )
                    if offline_mode:
                        print(f"✅ DataManager inicializado en modo OFFLINE: {workspace_dir}")
                    else:
                        print(f"✅ DataManager inicializado: {workspace_dir}")
                    
                    # SIEMPRE cargar farms_metadata (es ligero, solo IDs)
                    # Esto incluye farm_polygon_id para evitar consultas duplicadas a MongoDB
                    if farm_limit:
                        print(f"⚠ MODO TESTING: Limitando a {farm_limit:,} farms")
                    
                    farms_metadata, db_error = data_manager.load_farms_metadata(
                        limit=farm_limit, 
                        offline_mode=offline_mode,
                        value_chain=args.value_chain
                    )
                    
                    # Si hay error de BD, advertir pero continuar
                    if db_error:
                        print(f"\n⚠ ERROR DE BASE DE DATOS: {db_error}")
                        if needs_geometries:
                            print(f"📂 Intentando modo fallback: usar geojsons existentes en carpeta\n")
                        farms_metadata = None  # Señal para usar fallback
                        geom_stats = {'loaded': 0, 'failed': 0, 'cache_size_mb': 0.0}
                    
                    # Solo cargar geojsons/geometrías si se necesitan (pasos 1 o 3)
                    if needs_geometries and farms_metadata:
                        # Preparar geojsons para los farms cargados
                        if offline_mode:
                            # Modo offline: solo contar geojsons existentes, no descargar
                            geojsons_available = data_manager.count_available_geojsons(farms_metadata)
                        else:
                            geojsons_available = data_manager.prepare_geojsons(farms_metadata)
                        
                        if geojsons_available == 0:
                            print(f"⚠ Warning: No hay geojsons disponibles para procesar")
                        
                        # 🚀 OPTIMIZACIÓN: Cargar geometrías en memoria
                        print("\n" + "="*70)
                        print("🚀 Optimizaciones de Performance")
                        print("="*70)
                        geom_stats = data_manager.load_geometries_to_cache(farms_metadata)
                        
                        # 🚀 OPTIMIZACIÓN: Construir índice espacial
                        if geom_stats['loaded'] > 0:
                            try:
                                index_stats = data_manager.build_spatial_index()
                            except Exception as idx_err:
                                print(f"⚠ No se pudo construir índice espacial (no es crítico): {idx_err}")
                                index_stats = {'indexed': 0, 'build_time': 0.0}
                            print(f"\n💾 Memoria total usada: ~{geom_stats['cache_size_mb']:.1f} MB")
                    elif not needs_geometries:
                        # No se necesitan geometrías para estos pasos
                        print(f"⏭️ Omitiendo carga de geometrías (no requeridas para pasos seleccionados)")
                        geom_stats = {'loaded': 0, 'failed': 0, 'cache_size_mb': 0.0}
                    else:
                        geom_stats = {'loaded': 0, 'failed': 0, 'cache_size_mb': 0.0}
                    
                    # Preparar geojsons directory
                    geojsons_folder = str(data_manager.geojsons_dir)
                else:
                    print(f"⚠ WORKSPACE_DIR o GEOSERVER_URL no configurados")
            except Exception as e:
                print(f"⚠ Error inicializando DataManager: {e}")
                data_manager = None
        elif offline_mode:
            # Modo offline: usar archivos locales, sin DataManager
            print(f"📂 Usando archivos locales: {geojsons_folder}")
        
        # Preparar rutas de salida
        if data_manager:
            # Usar directorios gestionados por DataManager
            output_base = str(data_manager.get_results_dir('direct_alerts'))
            output_csv = f"{output_base}/direct_alert_{{PERIODO}}_{{YEARS}}.csv"
            farm_folder = str(data_manager.geojsons_dir)
        else:
            # Fallback a config.py legacy
            output_csv = config.get('OUTPUT_CSV', '')
            farm_folder = config.get('FOLDER_GEOJSONS', '')
        
        # Preparar parámetros para direct_alert usando función directa
        direct_params = {
            'source': args.source,
            'period_type': args.period_type,
            'years': periods,  # lista de períodos
            'farm_folder': farm_folder,
            'raster_template': '',  # Se llenará dinámicamente por período
            'output_csv': output_csv,
            'batch_size': config.get('BATCH_SIZE', 1000),
            'farm_range': config.get('FARM_FILE_RANGE', ''),
            'crs': config.get('CRS_METROS', 'EPSG:3116'),
            'deforest_value': config.get('DEFOREST_VALUE', 2),
            'log_level': config.get('LOG_LEVEL', 'WARNING'),
            'log_file': config.get('LOG_FILE', 'risk_analysis_intersections.log'),
            'use_precise_area': args.precise_area,  # Flag para super-sampling
            'pixel_divisions': args.pixel_divisions if args.pixel_divisions else (5 if args.precise_area else 1),  # Divisiones por píxel
            '_workspace_dir': config.get('WORKSPACE_DIR', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data'),
            '_data_manager': data_manager,  # Pasar DataManager a direct_alert
            '_farms_metadata': farms_metadata,  # Pasar lista de farms cargados
            '_farm_limit': farm_limit,  # Límite para fallback
            '_available_periods': available_periods,  # Metadata de períodos
            '_use_parallel': args.parallel,  # Activar modo paralelo
            '_num_workers': args.workers  # Número de workers
        }
        print(f"✅ Configuración cargada desde config.py")
    except Exception as e:
        print(f"⚠ Warning: No se pudo cargar config.py: {e}")
        print("   Se usará ejecución mediante subprocess con variables de entorno")
        direct_params = None
        data_manager = None
    
    # Resolver rutas de los scripts (para subprocess como fallback)
    scripts_dir = Path(__file__).resolve().parent
    workspace_path = Path(config.get('WORKSPACE_DIR', './workspace')) / "alertas"

    # ===================== DEFINIR ETAPAS Y MOSTRAR INFO =====================
    # Definir etapas disponibles
    STAGES = {
        'direct': ("Paso 1: Calculando alertas directas de deforestación", 'run_direct'),
        'movement': ("Paso 2: Calculando alertas indirectas por movimiento", 'run_movement'),
        'metrics': ("Paso 3: Calculando métricas espaciales", 'run_metrics'),
        'enterprise': ("Paso 4: Calculando alertas de empresa (standalone)", 'run_enterprise'),
        'total': ("Paso 5: Consolidando riesgo total", 'run_total'),
        'save': ("Paso 6: Guardando FarmRisk/EnterpriseRisk en MongoDB", 'run_save'),
        'adm3': ("Paso 7: Calculando riesgo ADM3 (genera CSVs)", 'run_adm3'),
        'save_adm3': ("Paso 8: Guardando ADM3 Risk en MongoDB", 'run_save_adm3'),
        'supplier_risk': ("Paso 9: Riesgo de empresa por Suppliers (sin movimientos)", 'run_supplier_risk'),
    }
    
    # Manejar -p sin argumentos (mostrar ayuda)
    if args.step is not None and len(args.step) == 0:
        print("\n📋 Pasos disponibles:")
        for step_num, stage_key in STEP_TO_STAGE.items():
            title, _ = STAGES[stage_key]
            print(f"   {step_num}: {title}")
        print("\n   Uso: -p 1-3 5 (ejecuta pasos 1, 2, 3 y 5)")
        sys.exit(0)
    
    # Mostrar info de pasos a ejecutar
    if args.step is not None:
        requested_steps = parse_steps(args.step)
        print(f"\n📋 Ejecutando pasos seleccionados: {', '.join([str(s) for s in requested_steps])}")
        if 2 in requested_steps and 4 in requested_steps:
            print(f"   ℹ Paso 4 (enterprise standalone) omitido: ya incluido en paso 2")
    else:
        print("\n📋 Ejecutando pipeline COMPLETO (pasos 1-3, 5, 6)")
        print("   ℹ Paso 4 (enterprise standalone) omitido: ya incluido en paso 2")

    if not stages_to_run:
        print("❌ No se seleccionaron etapas válidas.")
        print("   Uso: -p 1 (paso 1), -p 1-3 (pasos 1 al 3), -p 1-3 5 (pasos 1,2,3 y 5)")
        sys.exit(1)

    # ===================== VALIDAR PREREQUISITOS =====================
    print("\n" + "=" * 70)
    print("🔍 Validando prerequisitos...")
    print("=" * 70)
    
    all_valid = True
    for stage in stages_to_run:
        # El paso 'save' no tiene validación especial
        if stage == 'save':
            print_validation_summary(stage, True, "")
            continue
            
        is_valid, error_msg = validate_prerequisites(
            stage=stage,
            data_manager=data_manager,
            farms_metadata=farms_metadata,
            periods=periods,
            workspace_dir=workspace_path
        )
        print_validation_summary(stage, is_valid, error_msg)
        
        if not is_valid and not args.continue_on_error:
            all_valid = False
            print(f"\n❌ Prerequisitos no cumplidos para '{stage}'. Usa --continue-on-error para ignorar.")
            sys.exit(1)
    
    if all_valid:
        print("\n✅ Todas las validaciones pasaron")
    
    # ===================== EJECUTAR ETAPAS =====================
    ok = True
    results_summary = {}
    
    for stage_key in stages_to_run:
        title, _ = STAGES[stage_key]
        
        if stage_key == 'direct':
            if direct_params:
                success = run_direct_alerts(title, args.continue_on_error, direct_params, available_periods)
            else:
                print(f"⚠ Sin configuración para alertas directas")
                success = False
                
        elif stage_key == 'movement':
            if direct_params:
                success = run_indirect_alerts(title, args.continue_on_error, direct_params)
            else:
                print(f"⚠ Sin configuración para alertas de movimiento")
                success = False
                
        elif stage_key == 'metrics':
            if direct_params:
                success = run_spatial_metrics(title, args.continue_on_error, direct_params)
            else:
                print(f"⚠ DataManager y farms_metadata requeridos para métricas")
                success = False
                
        elif stage_key == 'enterprise':
            if direct_params:
                success = run_enterprise_alerts(title, args.continue_on_error, direct_params)
            else:
                print(f"⚠ Sin configuración para alertas de empresa")
                success = False
                
        elif stage_key == 'total':
            if direct_params:
                success = run_total_risk(title, args.continue_on_error, direct_params)
            else:
                # Fallback: usar subprocess
                total_script = scripts_dir / "total_alert.py"
                if total_script.exists():
                    success = run_subprocess_script(total_script, title, args.continue_on_error, env_vars)
                else:
                    print(f"❌ Script total_alert.py no encontrado")
                    success = False
        
        elif stage_key == 'save':
            # Paso 6: Guardar en MongoDB
            success = run_save_to_db(title, args.continue_on_error, {
                'data_manager': data_manager,
                'available_periods': available_periods,
                'periods': periods,
                'source': args.source,
                'period_type': args.period_type,
                'value_chain': args.value_chain,
                'bulk_insert': getattr(args, 'bulk_insert', False),
                'chunk_size': getattr(args, 'chunk_size', 1000),
                'args': args  # Pasar args completo para acceso fácil
            })
        
        elif stage_key == 'adm3':
            # Paso 7: Calcular riesgo ADM3 (solo genera CSVs para revisión)
            print("\n" + "=" * 70)
            print(title)
            print("=" * 70)
            
            if not farms_metadata:
                print("❌ Error: farms_metadata no disponible")
                success = False
            elif not available_periods:
                print("❌ Error: No hay períodos disponibles")
                success = False
            else:
                try:
                    import re
                    # Construir lista de (period_name, deforestation_id) SOLO para períodos solicitados
                    # La variable 'periods' ya está filtrada por generate_year_ranges()
                    periods_with_defo_id = []
                    for p in available_periods:
                        if len(p) >= 4 and p[3]:  # (start, end, name, mongo_id)
                            period_name = p[2]
                            # Extraer código del período (ej: "201701" o "2012-2013")
                            match = re.search(r'(\d{4}-\d{4}|\d{6})$', period_name)
                            if match:
                                period_code = match.group(1)
                                # Solo incluir si está en la lista de períodos filtrados
                                if period_code in periods:
                                    periods_with_defo_id.append((period_name, p[3]))
                    
                    if not periods_with_defo_id:
                        print("⚠ No se encontraron períodos con deforestation_id")
                        success = False
                    else:
                        # Directorio de salida
                        output_dir = data_manager.results_dir if data_manager else Path('./results')
                        
                        # Solo calcular y generar CSVs (sin guardar en BD)
                        processed, failed, csv_paths = calculate_adm3_risk_batch(
                            periods_with_defo_id=periods_with_defo_id,
                            farms_metadata=farms_metadata,
                            value_chain=args.value_chain,
                            output_dir=output_dir,
                            source=args.source,
                            period_type=args.period_type
                        )
                        
                        success = processed > 0
                        
                except Exception as e:
                    print(f"❌ Error calculando ADM3 risk: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
        
        elif stage_key == 'save_adm3':
            # Paso 8: Guardar ADM3 Risk en MongoDB (lee CSVs del paso 7)
            print("\n" + "=" * 70)
            print(title)
            print("=" * 70)
            
            if not data_manager:
                print("❌ Error: DataManager no disponible")
                success = False
            elif not periods:
                print("❌ Error: No hay períodos para procesar")
                success = False
            else:
                try:
                    import re
                    # Construir mapeo período → deforestation_id
                    period_to_defo_id = {}
                    for p in available_periods:
                        if len(p) >= 4 and p[3]:
                            period_name = p[2]
                            match = re.search(r'(\d{4}-\d{4}|\d{6})$', period_name)
                            if match:
                                period_code = match.group(1)
                                if period_code in periods:
                                    period_to_defo_id[period_code] = p[3]
                    
                    bulk_insert = getattr(args, 'bulk_insert', False)
                    
                    csv_found, csv_missing, db_saved, db_failed = save_adm3_risk_from_csv_batch(
                        periods=periods,
                        value_chain=args.value_chain,
                        results_dir=data_manager.results_dir,
                        source=args.source,
                        period_type=args.period_type,
                        bulk_insert=bulk_insert,
                        period_to_defo_id=period_to_defo_id
                    )
                    
                    success = db_saved > 0
                    
                except Exception as e:
                    print(f"❌ Error guardando ADM3 Risk: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
        
        elif stage_key == 'supplier_risk':
            # Paso 9: Riesgo de empresa por Suppliers (sin movimientos)
            print("\n" + "=" * 70)
            print(title)
            print("=" * 70)
            
            empresa_name = args.empresa
            if not empresa_name:
                print("❌ Error: Se requiere --empresa para el paso 9 (supplier risk)")
                print("   Uso: -p 9 -e 'Nombre de la empresa'")
                success = False
            elif not periods:
                print("❌ Error: No hay períodos para procesar")
                success = False
            else:
                try:
                    cache_dir = None
                    if data_manager:
                        cache_dir = Path(data_manager.workspace_dir) / "cache"
                    
                    result = calculate_and_save_supplier_risk(
                        enterprise_name_or_code=empresa_name,
                        periods=periods,
                        period_type=args.period_type,
                        source=args.source,
                        value_chain=args.value_chain,
                        cache_dir=cache_dir,
                        dry_run=args.dry_run
                    )
                    
                    success = result.get('success', False)
                    
                except Exception as e:
                    print(f"❌ Error calculando supplier risk: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
        
        else:
            print(f"⚠ Etapa desconocida: {stage_key}")
            success = False
        
        results_summary[stage_key] = success
        
        if not success:
            ok = False
            if not args.continue_on_error:
                print(f"\n❌ Etapa '{stage_key}' falló. Pipeline detenido.")
                break

    # ===================== RESUMEN FINAL =====================
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE EJECUCIÓN")
    print("=" * 70)
    
    for stage_key, success in results_summary.items():
        status = "✅ OK" if success else "❌ ERROR"
        title, _ = STAGES[stage_key]
        print(f"  {status} - {title}")
    
    # Mostrar resumen de errores
    error_logger = get_error_logger()
    if error_logger:
        error_logger.print_summary()
    
    if ok:
        print("\n🎉 Pipeline completado con éxito.")
        sys.exit(0)
    else:
        print("\n⚠ Pipeline finalizado con errores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
