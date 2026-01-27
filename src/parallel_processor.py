# -*- coding: utf-8 -*-
"""
Procesador paralelo para alertas de deforestación.
Divide el trabajo en chunks y procesa en paralelo usando multiprocessing.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm


def process_farm_chunk(args: tuple) -> Dict[str, Any]:
    """
    Procesa un chunk de farms en un worker.
    
    Args:
        args: Tupla con (chunk_id, farm_files, params)
        
    Returns:
        Dict con estadísticas del chunk procesado
    """
    chunk_id, farm_files, params = args
    
    # Importar aquí para evitar problemas de serialización
    from direct_alert import calculate_direct_alerts
    
    # Crear directorio temporal para este chunk
    temp_base_dir = params.get('_temp_dir')
    if not temp_base_dir:
        return {
            'chunk_id': chunk_id,
            'success': False,
            'error': 'No temp_dir provided'
        }
    
    temp_dir = Path(temp_base_dir) / f"chunk_{chunk_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Modificar output_csv para este chunk
    chunk_params = params.copy()
    chunk_params['output_csv'] = str(temp_dir / "output.csv")
    chunk_params['farm_folder'] = params['farm_folder']
    chunk_params['farm_range'] = ''  # Procesar todos los files del chunk
    
    # Remover parámetros internos (que empiezan con _)
    clean_params = {k: v for k, v in chunk_params.items() if not k.startswith('_')}
    
    # Crear symlinks/copiar solo los geojsons de este chunk
    chunk_geojson_dir = temp_dir / "geojsons"
    chunk_geojson_dir.mkdir(exist_ok=True)
    
    source_dir = Path(params['farm_folder'])
    for farm_file in farm_files:
        source = source_dir / farm_file
        target = chunk_geojson_dir / farm_file
        if source.exists() and not target.exists():
            try:
                # En Windows, crear hardlink es más rápido que copiar
                os.link(str(source), str(target))
            except:
                # Fallback: copiar archivo
                import shutil
                shutil.copy2(str(source), str(target))
    
    clean_params['farm_folder'] = str(chunk_geojson_dir)
    
    try:
        result = calculate_direct_alerts(**clean_params)
        result['chunk_id'] = chunk_id
        result['temp_dir'] = str(temp_dir)
        return result
    except Exception as e:
        logging.error(f"Error en chunk {chunk_id}: {e}")
        return {
            'chunk_id': chunk_id,
            'success': False,
            'error': str(e),
            'temp_dir': str(temp_dir)
        }


def combine_csv_results(temp_dirs: List[str], output_csv: str, period: str, source: str) -> int:
    """
    Combina los CSVs de todos los chunks en un solo archivo.
    
    Args:
        temp_dirs: Lista de directorios temporales con resultados
        output_csv: Ruta del CSV final
        period: Período procesado
        source: Fuente de deforestación
        
    Returns:
        Número total de registros combinados
    """
    all_dfs = []
    
    for temp_dir in temp_dirs:
        # Buscar CSVs en subdirectorios por período
        temp_path = Path(temp_dir)
        
        # Buscar patrón: [SOURCE]/[PERIOD]/*.csv
        source_upper = source.upper()
        csv_pattern = f"{source_upper}/{period}/*.csv"
        
        csv_files = list(temp_path.glob(csv_pattern))
        
        if not csv_files:
            # Intentar patrón alternativo si no encuentra
            csv_pattern_alt = f"**/{source_upper}/{period}/*.csv"
            csv_files = list(temp_path.glob(csv_pattern_alt))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                logging.info(f"CSV combinado: {csv_file.name} ({len(df)} registros)")
            except Exception as e:
                logging.warning(f"Error leyendo {csv_file}: {e}")
    
    if not all_dfs:
        logging.warning(f"No se encontraron CSVs para combinar en período {period}")
        return 0
    
    # Combinar todos los DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Eliminar duplicados por ID (por si acaso)
    combined = combined.drop_duplicates(subset=['id'], keep='first')
    
    # Guardar CSV final
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    logging.info(f"CSV combinado guardado: {output_path} ({len(combined)} registros)")
    return len(combined)


def cleanup_temp_dirs(temp_dirs: List[str]):
    """Limpia directorios temporales después de combinar resultados."""
    import shutil
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"No se pudo eliminar {temp_dir}: {e}")


def run_parallel_direct_alerts(
    params: Dict[str, Any],
    num_workers: Optional[int] = None,
    cleanup: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta calculate_direct_alerts en paralelo dividiendo por farms.
    
    Args:
        params: Parámetros para calculate_direct_alerts
        num_workers: Número de workers (None = usar CPUs disponibles)
        cleanup: Si True, elimina directorios temporales después
        
    Returns:
        Dict con estadísticas del procesamiento paralelo
    """
    start_time = time.time()
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n🚀 MODO PARALELO: {num_workers} workers")
    print("=" * 70)
    
    # Obtener lista de farms a procesar
    farm_folder = Path(params['farm_folder'])
    
    # Verificar si hay lista de farms desde BD (modo conectado)
    farms_metadata = params.get('_farms_metadata')
    farm_limit = params.get('_farm_limit')
    
    if farms_metadata:
        # Modo BD: usar solo farms de la metadata
        print(f"🎯 Usando farms cargados desde base de datos: {len(farms_metadata)} farms")
        all_farm_files = [f"{farm['sitcode']}.geojson" for farm in farms_metadata if farm.get('sitcode')]
    else:
        # Modo fallback: listar geojsons de carpeta
        all_farm_files = sorted([f for f in os.listdir(farm_folder) if f.lower().endswith('.geojson')])
        total_available = len(all_farm_files)
        
        # Aplicar límite si se especificó
        if farm_limit and farm_limit > 0 and farm_limit < total_available:
            all_farm_files = all_farm_files[:farm_limit]
            print(f"⚠ MODO TESTING: Limitando a {farm_limit:,} de {total_available:,} farms disponibles")
    
    total_farms = len(all_farm_files)
    print(f"📊 Total de farms a procesar: {total_farms:,}")
    
    # Dividir en chunks
    chunk_size = (total_farms + num_workers - 1) // num_workers
    chunks = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_farms)
        
        if start_idx >= total_farms:
            break
        
        chunk_files = all_farm_files[start_idx:end_idx]
        chunks.append((i, chunk_files, params))
        print(f"  • Worker {i+1}: {len(chunk_files):,} farms (índices {start_idx}-{end_idx-1})")
    
    print(f"\n⚙️ Procesando {len(chunks)} chunks en paralelo...")
    
    # Crear directorio temporal base
    workspace_dir = Path(params.get('_workspace_dir', 'D:/OneDrive - CGIAR/Proyectos/ganabosques/data'))
    temp_base = workspace_dir / "alertas" / "temp_parallel"
    temp_base.mkdir(parents=True, exist_ok=True)
    params['_temp_dir'] = str(temp_base)
    
    # Procesar chunks en paralelo
    results = []
    
    with Pool(processes=num_workers) as pool:
        # Usar imap para tener barra de progreso
        with tqdm(total=len(chunks), desc="Workers completados", unit="chunk") as pbar:
            for result in pool.imap_unordered(process_farm_chunk, chunks):
                results.append(result)
                pbar.update(1)
    
    # Verificar resultados
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\n✅ Workers exitosos: {len(successful)}/{len(chunks)}")
    if failed:
        print(f"❌ Workers fallidos: {len(failed)}")
        for f in failed:
            print(f"   • Chunk {f['chunk_id']}: {f.get('error', 'Unknown error')}")
    
    # Combinar CSVs de todos los períodos
    print(f"\n📦 Combinando resultados...")
    
    temp_dirs = [r['temp_dir'] for r in successful]
    years_list = params['years']
    source = params['source']
    output_template = params['output_csv']
    
    combined_stats = {}
    
    for period in years_list:
        # Formatear output_csv para este período
        ctx = {'PERIODO': params['period_type'], 'YEARS': period}
        
        # Determinar ruta de salida final
        from direct_alert import make_out_dir_and_paths, norm, format_placeholders
        
        formatted = norm(format_placeholders(output_template, ctx))
        base_dir = os.path.dirname(formatted) or "."
        
        option_map = {"smbyc": "SMBYC", "atd": "ATD", "nad": "NAD", "otra": "OTRA"}
        option_folder = option_map.get(str(source).lower(), str(source).upper())
        
        out_dir = norm(os.path.join(base_dir, option_folder, period))
        os.makedirs(out_dir, exist_ok=True)
        
        final_csv = norm(os.path.join(out_dir, f"{source}_direct_alert_{params['period_type']}_{period}.csv"))
        
        # Combinar CSVs de este período
        num_records = combine_csv_results(temp_dirs, final_csv, period, source)
        combined_stats[period] = num_records
        
        print(f"  ✓ {period}: {num_records:,} registros → {final_csv}")
    
    # Limpieza
    if cleanup:
        print(f"\n🧹 Limpiando directorios temporales...")
        cleanup_temp_dirs(temp_dirs)
        try:
            temp_base.rmdir()  # Eliminar directorio base si está vacío
        except:
            pass
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Procesamiento paralelo completado en {total_time:.2f}s ({total_time/60:.1f} min)")
    
    return {
        'success': True,
        'num_workers': num_workers,
        'chunks_processed': len(successful),
        'chunks_failed': len(failed),
        'periods_processed': len(years_list),
        'farms_per_period': total_farms,
        'execution_time': total_time,
        'combined_stats': combined_stats
    }
