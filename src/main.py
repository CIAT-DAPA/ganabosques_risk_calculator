#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
from config import config
from parallel_processor import run_parallel_direct_alerts
from direct_alert import calculate_direct_alerts
from indirect_alert_refactored import calculate_indirect_alerts_batch
    

# Importar ORM para consultar capas de deforestación
try:
    from ganabosques_orm.collections.deforestation import Deforestation
    from ganabosques_orm.enums.deforestationsource import DeforestationSource
    from ganabosques_orm.enums.deforestationtype import DeforestationType
    from mongoengine import connect
    HAS_ORM = True
except ImportError:
    print("⚠ Warning: ganabosques_orm no está disponible. Funcionalidad limitada.")
    HAS_ORM = False

# Importar DataManager
try:
    from data_manager import DataManager
    HAS_DATA_MANAGER = True
except ImportError:
    print("⚠ Warning: data_manager no disponible. Funcionalidad limitada.")
    HAS_DATA_MANAGER = False

def connect_mongodb():
    """Conecta a MongoDB usando variables de entorno."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB_NAME", "ganabosques")
    try:
        connect(host=mongo_uri, db=mongo_db)
        #print(f"✔ Conectado a MongoDB: {mongo_db}")
        return True
    except Exception as e:
        print(f"✖ Error conectando a MongoDB: {e}")
        return False

def query_available_periods(source: str, period_type: str) -> List[Tuple[datetime, datetime, str]]:
    """
    Consulta MongoDB para obtener períodos disponibles según fuente y tipo.
    Retorna lista de tuplas: (period_start, period_end, name)
    """
    if not HAS_ORM:
        print("⚠ ORM no disponible, no se pueden consultar períodos")
        return []
    
    try:
        # Convertir strings a enums del ORM
        source_enum = DeforestationSource[source.upper()]
        
        # Mapear period_type a DeforestationType
        type_map = {
            "annual": DeforestationType.ANNUAL,
            "cumulative": DeforestationType.CUMULATIVE,
            "quarter": DeforestationType.QUARTERLY,
            "warning": DeforestationType.WARNING
        }
        type_enum = type_map.get(period_type)
        
        if not type_enum:
            print(f"⚠ Tipo de período desconocido: {period_type}")
            return []
        
        query = {
            "deforestation_source": source_enum.value,
            "deforestation_type": type_enum.value
        }
        
        periods = Deforestation.objects(**query).order_by('period_start')
        result = [(p.period_start, p.period_end, p.name) for p in periods if p.period_start and p.period_end]
        
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
    
    Formatos de entrada:
    - "2024" → un solo año
    - "2010-2024" → rango de años
    
    Retorna lista de strings según formato esperado por cada fuente/tipo.
    """
    # Parsear input del usuario
    if "-" in years_input and years_input.count("-") == 1:
        start_year, end_year = map(int, years_input.split("-"))
    else:
        start_year = end_year = int(years_input)
    
    periods = []
    
    if source == "smbyc":
        if not available_periods:
            print(f"⚠ No hay capas SMBYC {period_type} en la base de datos")
            return []
        
        # Extraer años disponibles desde la BD
        if period_type == "annual":
            # SMBYC anual: buscar períodos tipo "2012-2013", "2013-2014"
            for p in available_periods:
                name = p[2]  # nombre del período
                # Extraer rango de años del nombre (ej: "smbyc_deforestation_annual_2012-2013")
                import re
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
                import re
                match = re.search(r'(\d{4})-(\d{4})$', name) or re.search(r'(\d{4})(\d{4})$', name)
                if match:
                    year_start = int(match.group(1))
                    year_end = int(match.group(2))
                    # Para acumulado, verificar que el año final esté en el rango
                    if year_end >= start_year and year_end <= end_year:
                        periods.append(f"{year_start}-{year_end}")
    
    elif source in ["nad", "atd"]:
        if period_type == "quarter":
            if not available_periods:
                print(f"⚠ No hay capas {source.upper()} quarter en la base de datos")
                return []
            
            # NAD/ATD trimestral: SOLO usar períodos que existen en BD
            seen_quarters = set()
            for p in available_periods:
                name = p[2]  # nombre del período
                # Extraer YYYYQQ del nombre (ej: "nad_deforestation_quarter_202203")
                import re
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
            # Estructura: workspace/alertas/rasters/nad/quarter/nad_deforestation_quarter_201701.tif
            rasters_base = data_manager.workspace_dir / "rasters"
            source_lower = params['source'].lower()
            period_type = params['period_type']
            source_folder = rasters_base / source_lower / period_type
            
            if not source_folder.exists():
                print(f"⚠ Carpeta de rasters no existe: {source_folder}")
                params['raster_paths_dict'] = None
            else:
                # Buscar rasters con patrón esperado
                for period_name in params['years']:
                    # Patrón: nad_deforestation_quarter_201701.tif
                    expected_name = f"{source_lower}_deforestation_{period_type}_{period_name}.tif"
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

def run_indirect_alerts(titulo: str, continue_on_error: bool, params: Dict[str, Any]) -> bool:
    """Ejecuta el cálculo de alertas indirectas (movimiento) usando función callable."""

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    
    try:
        # Extraer parámetros necesarios
        periods = params.get('years', [])
        period_type = params.get('period_type', 'quarter')
        source = params.get('source', 'nad')
        workspace_dir = Path(params.get('_workspace_dir', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data/'))
        
        # Directorios
        direct_alerts_dir = workspace_dir / "alertas" / "results" / "direct_alerts"
        movement_csv_dir = workspace_dir / "alertas" / "movements"
        output_dir = workspace_dir / "alertas" / "results" / "indirect_alerts"
        
        # Ejecutar batch
        result = calculate_indirect_alerts_batch(
            periods=periods,
            period_type=period_type,
            source=source,
            direct_alerts_dir=str(direct_alerts_dir),
            movement_csv_dir=str(movement_csv_dir),
            output_dir=str(output_dir),
            normalize_ids=True,
            strip_dot_zero=True,
            strip_leading_zeros=True,
            uppercase_ids=True
        )
        
        if result['success']:
            print(f"✔ Alertas indirectas completadas.")
            print(f"   • Períodos: {result['periods_processed']}")
            print(f"   • Alertas indirectas (IN): {result['total_indirect_in']}")
            print(f"   • Alertas indirectas (OUT): {result['total_indirect_out']}")
            return True
        else:
            print(f"⚠ Alertas indirectas finalizadas con errores:")
            print(f"   • Períodos exitosos: {result['periods_processed']}")
            print(f"   • Períodos fallidos: {result['periods_failed']}")
            return not continue_on_error
    
    except Exception as e:
        logging.error(f"Error en alertas indirectas: {e}", exc_info=True)
        print(f"✖ Error ejecutando alertas indirectas: {e}")
        return continue_on_error

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

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Alertas de Deforestación: Alertas Directas -> Alertas Indirectas (Movimiento) -> Riesgo Total",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar SMBYC anual para un año:
  python main.py --source smbyc --period-type annual --years 2024
  
  # Procesar SMBYC acumulado histórico:
  python main.py -s smbyc -pt cumulative -y 2010-2024
  
  # Procesar NAD trimestral (quarter):
  python main.py -s nad -pt quarter -y 2023
  
  # Procesar ATD trimestral:
  python main.py -s atd -pt quarter -y 2024
  
  # Ejecutar solo alertas directas:
  python main.py -s smbyc -pt annual -y 2024 --direct
        """
    )
    
    # Parámetros requeridos (usando valores de los enums del ORM)
    source_choices = [s.value for s in DeforestationSource] if HAS_ORM else ["smbyc", "nad", "atd"]
    parser.add_argument("--source", "-s",
                       choices=source_choices,
                       required=True,
                       help="Fuente de datos de deforestación (del ORM: smbyc, nad, atd)")
    
    # Para period-type, filtramos solo los que usamos (sin 'warning')
    type_choices =  [s.value for s in DeforestationType if s.value != "warning"] if HAS_ORM else ["annual", "cumulative", "quarter"]
    parser.add_argument("--period-type", "-pt",
                       choices=type_choices,
                       required=True,
                       help="Tipo de período: annual/cumulative (SMBYC) o quarter (NAD/ATD)")
    
    parser.add_argument("--years", "-y",
                       required=True,
                       help="Años a procesar: '2024' o '2010-2024'")
    
    # Parámetros opcionales
    parser.add_argument("--empresa", "-e",
                       help="Nombre de la empresa")
    
    # Etapas del pipeline (unificado: nombre largo y corto en una línea)
    parser.add_argument("--direct-risk", "--direct", action="store_true",
                       help="Ejecutar solo alertas directas")
    parser.add_argument("--movement-risk", "--movement", action="store_true",
                       help="Ejecutar solo alertas indirectas (movimiento)")
    parser.add_argument("--total-risk", "--total", action="store_true",
                       help="Ejecutar solo cálculo de riesgo total")
    
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continuar con etapas siguientes aunque una falle")
    
    # Testing y debugging
    parser.add_argument("--farm-limit", type=int, default=None,
                       help="Limitar número de farms a procesar (para testing)")
    
    # Paralelización
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Usar procesamiento paralelo para mayor velocidad")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Número de workers paralelos (default: número de CPUs)")
    
    # Precisión de cálculo
    parser.add_argument("--precise-area", action="store_true",
                       help="Usar super-sampling 5×5 para cálculo preciso de fracciones de píxel (más lento pero reduce error del 100%% a ~10-20%%)")

    args = parser.parse_args()
    
    # Validar combinaciones de fuente y tipo de período (usando valores del ORM)
    source_value = args.source.lower()
    period_orm_value = args.period_type.lower()
    
    # Validaciones según enums del ORM
    if source_value == DeforestationSource.SMBYC.value:
        if period_orm_value not in [DeforestationType.ANNUAL.value, DeforestationType.CUMULATIVE.value]:
            print(f"❌ Error: SMBYC requiere --period-type annual o cumulative")
            sys.exit(1)
    elif source_value in [DeforestationSource.NAD.value, DeforestationSource.ATD.value]:
        if period_orm_value != DeforestationType.QUARTERLY.value:
            print(f"❌ Error: NAD/ATD requiere --period-type quarter")
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
    
    available_periods = []
    if HAS_ORM and connect_mongodb():
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
            if args.period_type == "quarter":
                # Para quarter, generar los 4 trimestres del año
                if "-" in args.years:
                    start_year, end_year = map(int, args.years.split("-"))
                else:
                    start_year = end_year = int(args.years)
                
                for year in range(start_year, end_year + 1):
                    for q in range(1, 5):
                        periods.append(f"{year}0{q}")
            elif args.period_type == "annual":
                # Para annual, usar el rango año-año
                if "-" in args.years:
                    start_year, end_year = map(int, args.years.split("-"))
                    for y in range(start_year, end_year + 1):
                        periods.append(f"{y}-{y+1}")
                else:
                    year = int(args.years)
                    periods.append(f"{year}-{year+1}")
            
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

        
        # Inicializar DataManager si está disponible
        data_manager = None
        farms_metadata = None  # Inicializar para que esté disponible fuera del bloque
        if HAS_DATA_MANAGER:
            try:
                workspace_dir = config.get('WORKSPACE_DIR')
                gs_url = config.get('GEOSERVER_URL')
                gs_user = config.get('GEOSERVER_USER', 'admin')
                gs_pass = config.get('GEOSERVER_PASS', 'geoserver')
                
                if workspace_dir and gs_url:
                    data_manager = DataManager(
                        workspace_dir=workspace_dir,
                        geoserver_url=gs_url,
                        geoserver_user=gs_user,
                        geoserver_pass=gs_pass
                    )
                    print(f"✅ DataManager inicializado: {workspace_dir}")
                    
                    # Cargar metadata de farms (ligero, solo IDs)
                    if farm_limit:
                        print(f"⚠ MODO TESTING: Limitando a {farm_limit:,} farms")
                    
                    farms_metadata, db_error = data_manager.load_farms_metadata(limit=farm_limit)
                    
                    # Si hay error de BD, advertir pero continuar (fallback a geojsons)
                    if db_error:
                        print(f"\n⚠ ERROR DE BASE DE DATOS: {db_error}")
                        print(f"📂 Intentando modo fallback: usar geojsons existentes en carpeta\n")
                        farms_metadata = None  # Señal para usar fallback en direct_alert
                        geojsons_available = 0
                        geom_stats = {'loaded': 0, 'failed': 0, 'cache_size_mb': 0.0}
                    else:
                        # Preparar geojsons para los farms cargados
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
                        index_stats = data_manager.build_spatial_index()
                        print(f"\n💾 Memoria total usada: ~{geom_stats['cache_size_mb']:.1f} MB")
                    
                    # Preparar geojsons directory
                    geojsons_folder = str(data_manager.geojsons_dir)
                else:
                    print(f"⚠ WORKSPACE_DIR o GEOSERVER_URL no configurados")
            except Exception as e:
                print(f"⚠ Error inicializando DataManager: {e}")
                data_manager = None
        
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
            'alerts_dir': config.get('ALERTAS_DIR'),
            'nucleos_dir': config.get('NUCLEOS_DIR'),
            'batch_size': config.get('BATCH_SIZE', 1000),
            'farm_range': config.get('FARM_FILE_RANGE', ''),
            'crs': config.get('CRS_METROS', 'EPSG:3116'),
            'deforest_value': config.get('DEFOREST_VALUE', 2),
            'log_level': config.get('LOG_LEVEL', 'WARNING'),
            'log_file': config.get('LOG_FILE', 'risk_analysis_intersections.log'),
            'use_precise_area': args.precise_area,  # Flag para super-sampling
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
    
    # Resolver rutas de los scripts (main.py está en src/ junto con los otros .py)
    scripts_dir = Path(__file__).resolve().parent
    direct_script   = scripts_dir / "direct_alert.py"
    movement_script = scripts_dir / "indirect_alert.py"
    total_script    = scripts_dir / "total_alert.py"

    # Determinar etapas a ejecutar
    any_flag = args.direct_risk or args.movement_risk or args.total_risk
    stages = []

    if not any_flag:
        # Ejecuta todo el pipeline en orden
        stages = [
            ("Paso 1: Calculando alertas directas de deforestación",       direct_script),
            ("Paso 2: Calculando alertas indirectas por movimiento de ganado", movement_script),
            ("Paso 3: Calculando riesgo total y métricas espaciales",         total_script),
        ]
    else:
        if args.direct_risk:
            stages.append(("Paso 1: Calculando alertas directas de deforestación", direct_script))
        if args.movement_risk:
            stages.append(("Paso 2: Calculando alertas indirectas por movimiento de ganado", movement_script))
        if args.total_risk:
            stages.append(("Paso 3: Calculando riesgo total y métricas espaciales", total_script))

    if not stages:
        print("No se seleccionaron etapas. Usa --direct-risk/--movement-risk/--total-risk (o sus alias).")
        sys.exit(1)

    ok = True
    for title, script in stages:
        if not script.exists():
            print(f"✖ No se encontró el script: {script}")
            ok = False
            if not args.continue_on_error:
                break
            else:
                continue

        # Usar función callable para scripts refactorizados
        if script.name == "direct_alert.py" and direct_params:
            if not run_direct_alerts(title, args.continue_on_error, direct_params, available_periods):
                ok = False
                if not args.continue_on_error:
                    break
        elif script.name == "indirect_alert.py" and direct_params:
            # Usar versión refactorizada de alertas indirectas
            if not run_indirect_alerts(title, args.continue_on_error, direct_params):
                ok = False
                if not args.continue_on_error:
                    break
        else:
            # Para scripts no refactorizados, usar subprocess
            if not run_subprocess_script(script, title, args.continue_on_error, env_vars):
                ok = False
                if not args.continue_on_error:
                    break

    if ok:
        print("\n🎉 Pipeline completado con éxito.")
        sys.exit(0)
    else:
        print("\n⚠ Pipeline finalizado con errores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
