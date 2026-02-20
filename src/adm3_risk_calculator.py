# -*- coding: utf-8 -*-
"""
ADM3 Risk Calculator - Calcula riesgo agregado a nivel de vereda/municipio.

Usa el ORM de ganabosques para:
- Obtener FarmRisks por Analysis
- Agregar métricas por ADM3
- Generar CSV con resultados

Autor: CIAT-DAPA
Fecha: 2026
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ORM imports
try:
    from ganabosques_orm.collections.adm3 import Adm3
    from ganabosques_orm.collections.adm3risk import Adm3Risk
    from ganabosques_orm.collections.analysis import Analysis
    from ganabosques_orm.collections.farmrisk import FarmRisk
    from ganabosques_orm.collections.deforestation import Deforestation
    from ganabosques_orm.enums.valuechain import ValueChain
    from bson import ObjectId
    HAS_ORM = True
except ImportError:
    HAS_ORM = False
    logging.warning("ganabosques_orm no disponible para adm3_risk_calculator")


def get_analysis_for_period(
    deforestation_id: str,
    value_chain: str = "livestock"
) -> Optional[str]:
    """
    Obtiene el Analysis ID para una capa de deforestación y cadena de valor.
    
    Args:
        deforestation_id: ID de la capa de deforestación
        value_chain: Cadena de valor ('livestock', 'cacao')
        
    Returns:
        analysis_id como string o None si no existe
    """
    if not HAS_ORM:
        return None
    
    try:
        # Convertir value_chain a enum
        vc_enum = ValueChain(value_chain.lower())
        
        # Buscar analysis por deforestation_id + value_chain
        analysis = Analysis.objects(
            deforestation_id=ObjectId(deforestation_id),
            value_chain=vc_enum
        ).first()
        
        if analysis:
            print(f"   ✓ Encontrado analysis para deforestation_id={deforestation_id}, value_chain={value_chain}: {analysis.id}")
            return str(analysis.id)
        else:
            logging.warning(f"No se encontró Analysis para deforestation={deforestation_id}, value_chain={value_chain}")
            return None
            
    except Exception as e:
        logging.error(f"Error buscando Analysis: {e}")
        return None


def get_all_adm3_ids() -> List[str]:
    """
    Obtiene todos los IDs de ADM3 existentes en la BD.
    
    Returns:
        Lista de adm3_id como strings
    """
    if not HAS_ORM:
        return []
    
    try:
        adm3_list = Adm3.objects().only('id')
        return [str(adm3.id) for adm3 in adm3_list]
    except Exception as e:
        logging.error(f"Error obteniendo ADM3s: {e}")
        return []


def build_farm_to_adm3_map(farms_metadata: List[Dict]) -> Dict[str, str]:
    """
    Construye mapeo farm_id → adm3_id desde metadata precargada.
    
    Args:
        farms_metadata: Lista de dicts con farm info (incluye mongo_id y adm3_id)
        
    Returns:
        Dict {farm_mongo_id: adm3_id}
    """
    farm_to_adm3 = {}
    for farm in farms_metadata:
        mongo_id = farm.get('mongo_id')
        adm3_id = farm.get('adm3_id')
        if mongo_id and adm3_id:
            farm_to_adm3[mongo_id] = adm3_id
    return farm_to_adm3


def build_adm3_total_farms(farms_metadata: List[Dict]) -> Dict[str, int]:
    """
    Cuenta el total de fincas por ADM3 (para farm_amount_total).
    
    Args:
        farms_metadata: Lista de dicts con farm info
        
    Returns:
        Dict {adm3_id: total_farms}
    """
    adm3_totals = defaultdict(int)
    for farm in farms_metadata:
        adm3_id = farm.get('adm3_id')
        if adm3_id:
            adm3_totals[adm3_id] += 1
    return dict(adm3_totals)


def calculate_adm3_risk_for_period(
    analysis_id: str,
    farm_to_adm3: Dict[str, str],
    all_adm3_ids: List[str],
    adm3_total_farms: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Calcula métricas de riesgo por ADM3 para un período/analysis específico.
    
    Args:
        analysis_id: ID del Analysis
        farm_to_adm3: Mapeo farm_id → adm3_id
        all_adm3_ids: Lista de todos los ADM3 IDs
        adm3_total_farms: Total de fincas por ADM3
        
    Returns:
        Lista de dicts con métricas por ADM3
    """
    if not HAS_ORM:
        return []
    
    try:
        analysis_oid = ObjectId(analysis_id)
        
        # Query FarmRisks con algún tipo de riesgo
        farm_risks = FarmRisk.objects(
            analysis_id=analysis_oid,
            __raw__={
                "$or": [
                    {"risk_direct": True},
                    {"risk_input": True},
                    {"risk_output": True}
                ]
            }
        )

        print(f"  ✓ Encontrados {farm_risks.count():,} FarmRisks con riesgo para analysis {analysis_id}")
        
        # Agregar por ADM3
        adm3_metrics = defaultdict(lambda: {
            'farm_count': 0,
            'def_ha': 0.0,
            'farm_ids': set()
        })
        
        for fr in tqdm(farm_risks, desc="Agregando FarmRisks por ADM3", unit="farmrisk"):

            farm_id = str(fr.farm_id.id) if fr.farm_id else None
            if not farm_id:
                continue
                
            # Obtener adm3_id del mapeo precargado
            adm3_id = farm_to_adm3.get(farm_id)
            if not adm3_id:
                continue
            
            # Evitar contar duplicados
            if farm_id in adm3_metrics[adm3_id]['farm_ids']:
                continue
            adm3_metrics[adm3_id]['farm_ids'].add(farm_id)
            
            # Contar farm
            adm3_metrics[adm3_id]['farm_count'] += 1
            
            # Sumar hectáreas solo si tiene riesgo directo
            if fr.risk_direct and fr.deforestation:
                ha = fr.deforestation.ha or 0.0
                adm3_metrics[adm3_id]['def_ha'] += ha
        
        # Construir resultado para TODOS los ADM3
        rows = []
        for adm3_id in all_adm3_ids:
            metrics = adm3_metrics.get(adm3_id)
            
            if metrics:
                row = {
                    'adm3_id': adm3_id,
                    'analysis_id': analysis_id,
                    'def_ha': round(metrics['def_ha'], 4),
                    'risk_total': True,  # Si está en metrics, tiene riesgo
                    'farm_amount': metrics['farm_count'],
                    'farm_amount_total': adm3_total_farms.get(adm3_id, 0)
                }
            else:
                row = {
                    'adm3_id': adm3_id,
                    'analysis_id': analysis_id,
                    'def_ha': 0.0,
                    'risk_total': False,
                    'farm_amount': 0,
                    'farm_amount_total': adm3_total_farms.get(adm3_id, 0)
                }
            
            rows.append(row)
        
        return rows
        
    except Exception as e:
        logging.error(f"Error calculando ADM3 risk para analysis {analysis_id}: {e}")
        return []


def calculate_adm3_risk_batch(
    periods_with_defo_id: List[Tuple[str, str]],
    farms_metadata: List[Dict],
    value_chain: str = "livestock",
    output_dir: Optional[Path] = None,
    source: str = "smbyc",
    period_type: str = "nad"
) -> Tuple[int, int, List[str]]:
    """
    Calcula ADM3 risk para múltiples períodos y guarda CSVs.
    
    Args:
        periods_with_defo_id: Lista de tuplas (period_name, deforestation_id)
        farms_metadata: Lista de metadata de farms precargada
        value_chain: Cadena de valor ('livestock', 'cacao')
        output_dir: Directorio base para guardar resultados
        source: Fuente de deforestación
        period_type: Tipo de período
        
    Returns:
        Tupla (períodos_procesados, períodos_fallidos, lista_de_csvs)
    """
    if not HAS_ORM:
        print("❌ ORM no disponible")
        return (0, 0, [])
    
    print(f"\n{'='*70}")
    print(f"📊 Calculando ADM3 Risk")
    print(f"{'='*70}")
    print(f"   • Períodos: {len(periods_with_defo_id)}")
    print(f"   • Farms: {len(farms_metadata):,}")
    print(f"   • Value Chain: {value_chain}")
    
    # Pre-construir mapeos (una sola vez)
    print("\n📋 Preparando mapeos...")
    farm_to_adm3 = build_farm_to_adm3_map(farms_metadata)
    print(f"   ✓ Farm → ADM3: {len(farm_to_adm3):,} mapeos")
    
    adm3_total_farms = build_adm3_total_farms(farms_metadata)
    print(f"   ✓ ADM3 totales: {len(adm3_total_farms):,} con fincas")
    
    all_adm3_ids = get_all_adm3_ids()
    print(f"   ✓ ADM3 en BD: {len(all_adm3_ids):,}")
    
    processed = 0
    failed = 0
    csv_paths = []
    
    print(f"\n🔄 Procesando períodos...")
    
    for period_name, deforestation_id in tqdm(periods_with_defo_id, desc="ADM3 Risk", unit="período"):
        try:
            # Obtener Analysis para este período + value_chain
            analysis_id = get_analysis_for_period(deforestation_id, value_chain)
            
            if not analysis_id:
                print(f"\n   ⚠ Sin Analysis para {period_name} (value_chain={value_chain})")
                failed += 1
                continue
            
            # Calcular métricas
            rows = calculate_adm3_risk_for_period(
                analysis_id,
                farm_to_adm3,
                all_adm3_ids,
                adm3_total_farms
            )
            
            if not rows:
                print(f"\n   ⚠ Sin resultados para {period_name}")
                failed += 1
                continue
            
            # Crear DataFrame
            df = pd.DataFrame(rows, columns=[
                'adm3_id', 'analysis_id', 'def_ha', 'risk_total', 
                'farm_amount', 'farm_amount_total'
            ])
            
            # Determinar ruta de salida
            if output_dir:
                # Extraer período del nombre (ej: "smbyc_deforestation_nad_201701" → "201701")
                import re
                match = re.search(r'(\d{6}|\d{4}-\d{4})$', period_name)
                period_key = match.group(1) if match else period_name
                
                adm3_dir = output_dir / source / period_type / "adm3_risk"
                adm3_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = adm3_dir / f"{source}_adm3_risk_{period_type}_{period_key}.csv"
            else:
                csv_path = Path(f"adm3_risk_{period_name}.csv")
            
            # Guardar CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            csv_paths.append(str(csv_path))
            processed += 1
            
        except Exception as e:
            logging.error(f"Error procesando {period_name}: {e}")
            print(f"\n   ❌ Error en {period_name}: {e}")
            failed += 1
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"📊 Resumen ADM3 Risk:")
    print(f"   • Procesados: {processed}")
    print(f"   • Fallidos: {failed}")
    print(f"   • ADM3 por archivo: {len(all_adm3_ids):,}")
    
    if csv_paths:
        print(f"\n📁 Archivos generados:")
        for path in csv_paths[:5]:
            print(f"   • {path}")
        if len(csv_paths) > 5:
            print(f"   ... y {len(csv_paths) - 5} más")
    
    return (processed, failed, csv_paths)


def save_adm3_risk_to_db(
    adm3_risks: List[Dict[str, Any]],
    analysis_id: str
) -> Tuple[int, int, List[Dict]]:
    """
    Guarda resultados de ADM3 Risk en MongoDB.
    
    Hace upsert basado en (adm3_id, analysis_id) para evitar duplicados.
    
    Args:
        adm3_risks: Lista de dicts con métricas por ADM3
        analysis_id: ID del Analysis
        
    Returns:
        Tupla (saved, failed, errors)
    """
    if not HAS_ORM:
        return (0, 0, [{"error": "ORM no disponible"}])
    
    saved = 0
    failed = 0
    errors = []
    
    try:
        analysis_oid = ObjectId(analysis_id)
    except Exception as e:
        return (0, 0, [{"error": f"analysis_id inválido: {e}"}])
    
    for row in tqdm(adm3_risks, desc="Guardando Adm3Risk", unit="adm3"):
        try:
            adm3_id = row.get('adm3_id')
            if not adm3_id:
                failed += 1
                continue
            
            adm3_oid = ObjectId(adm3_id)
            
            # Buscar si ya existe (upsert manual)
            existing = Adm3Risk.objects(
                adm3_id=adm3_oid,
                analysis_id=analysis_oid
            ).first()
            
            if existing:
                # Actualizar
                existing.def_ha = row.get('def_ha', 0.0)
                existing.farm_amount = row.get('farm_amount', 0)
                existing.risk_total = row.get('risk_total', False)
                existing.save()
            else:
                # Crear nuevo
                adm3_risk = Adm3Risk(
                    adm3_id=adm3_oid,
                    analysis_id=analysis_oid,
                    def_ha=row.get('def_ha', 0.0),
                    farm_amount=row.get('farm_amount', 0),
                    risk_total=row.get('risk_total', False)
                )
                adm3_risk.save()
            
            saved += 1
            
        except Exception as e:
            failed += 1
            errors.append({
                'adm3_id': row.get('adm3_id'),
                'error': str(e)
            })
    
    return (saved, failed, errors)


def save_adm3_risk_to_db_bulk(
    adm3_risks: List[Dict[str, Any]],
    analysis_id: str,
    chunk_size: int = 1000
) -> Tuple[int, int, List[Dict]]:
    """
    Guarda ADM3 Risk usando bulk operations (más rápido).
    
    Primero elimina registros existentes para el analysis_id, 
    luego hace bulk insert.
    
    Args:
        adm3_risks: Lista de dicts con métricas por ADM3
        analysis_id: ID del Analysis
        chunk_size: Tamaño de chunks para bulk
        
    Returns:
        Tupla (saved, failed, errors)
    """
    if not HAS_ORM:
        return (0, 0, [{"error": "ORM no disponible"}])
    
    try:
        analysis_oid = ObjectId(analysis_id)
    except Exception as e:
        return (0, 0, [{"error": f"analysis_id inválido: {e}"}])
    
    # 1. Eliminar existentes para este analysis
    try:
        deleted = Adm3Risk.objects(analysis_id=analysis_oid).delete()
        if deleted > 0:
            print(f"   🗑️ Eliminados {deleted:,} Adm3Risk existentes para este analysis")
    except Exception as e:
        logging.warning(f"Error eliminando Adm3Risk existentes: {e}")
    
    # 2. Preparar documentos para insert
    docs_to_insert = []
    errors = []
    
    for row in adm3_risks:
        try:
            adm3_id = row.get('adm3_id')
            if not adm3_id:
                continue
            
            doc = Adm3Risk(
                adm3_id=ObjectId(adm3_id),
                analysis_id=analysis_oid,
                def_ha=row.get('def_ha', 0.0),
                farm_amount=row.get('farm_amount', 0),
                risk_total=row.get('risk_total', False)
            )
            docs_to_insert.append(doc)
            
        except Exception as e:
            errors.append({
                'adm3_id': row.get('adm3_id'),
                'error': str(e)
            })
    
    # 3. Bulk insert por chunks
    saved = 0
    total_chunks = (len(docs_to_insert) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(docs_to_insert), chunk_size):
        chunk = docs_to_insert[i:i + chunk_size]
        try:
            Adm3Risk.objects.insert(chunk)
            saved += len(chunk)
        except Exception as e:
            logging.error(f"Error en bulk insert chunk {i//chunk_size + 1}: {e}")
            errors.append({
                'chunk': i // chunk_size + 1,
                'error': str(e)
            })
    
    failed = len(adm3_risks) - saved
    return (saved, failed, errors)


def calculate_and_save_adm3_risk_batch(
    periods_with_defo_id: List[Tuple[str, str]],
    farms_metadata: List[Dict],
    value_chain: str = "livestock",
    output_dir: Optional[Path] = None,
    source: str = "smbyc",
    period_type: str = "nad",
    save_to_db: bool = True,
    bulk_insert: bool = True
) -> Tuple[int, int, List[str], int, int]:
    """
    Calcula ADM3 risk, guarda CSVs y opcionalmente guarda en BD.
    
    Args:
        periods_with_defo_id: Lista de tuplas (period_name, deforestation_id)
        farms_metadata: Lista de metadata de farms
        value_chain: Cadena de valor
        output_dir: Directorio para CSVs
        source: Fuente de deforestación
        period_type: Tipo de período
        save_to_db: Si True, guarda en MongoDB
        bulk_insert: Si True, usa bulk insert (más rápido)
        
    Returns:
        Tupla (csv_processed, csv_failed, csv_paths, db_saved, db_failed)
    """
    if not HAS_ORM:
        print("❌ ORM no disponible")
        return (0, 0, [], 0, 0)
    
    print(f"\n{'='*70}")
    print(f"📊 Calculando y Guardando ADM3 Risk")
    print(f"{'='*70}")
    print(f"   • Períodos: {len(periods_with_defo_id)}")
    print(f"   • Farms: {len(farms_metadata):,}")
    print(f"   • Value Chain: {value_chain}")
    print(f"   • Guardar en BD: {'Sí (bulk)' if save_to_db and bulk_insert else 'Sí' if save_to_db else 'No'}")
    
    # Pre-construir mapeos (una sola vez)
    print("\n📋 Preparando mapeos...")
    farm_to_adm3 = build_farm_to_adm3_map(farms_metadata)
    print(f"   ✓ Farm → ADM3: {len(farm_to_adm3):,} mapeos")
    
    adm3_total_farms = build_adm3_total_farms(farms_metadata)
    print(f"   ✓ ADM3 totales: {len(adm3_total_farms):,} con fincas")
    
    all_adm3_ids = get_all_adm3_ids()
    print(f"   ✓ ADM3 en BD: {len(all_adm3_ids):,}")
    
    csv_processed = 0
    csv_failed = 0
    csv_paths = []
    total_db_saved = 0
    total_db_failed = 0
    
    print(f"\n🔄 Procesando períodos...")
    
    for period_name, deforestation_id in tqdm(periods_with_defo_id, desc="ADM3 Risk", unit="período"):
        try:
            # Obtener Analysis para este período + value_chain
            analysis_id = get_analysis_for_period(deforestation_id, value_chain)
            
            if not analysis_id:
                print(f"\n   ⚠ Sin Analysis para {period_name} (value_chain={value_chain})")
                csv_failed += 1
                continue
            
            # Calcular métricas
            rows = calculate_adm3_risk_for_period(
                analysis_id,
                farm_to_adm3,
                all_adm3_ids,
                adm3_total_farms
            )
            
            if not rows:
                print(f"\n   ⚠ Sin resultados para {period_name}")
                csv_failed += 1
                continue
            
            # Crear DataFrame (con farm_amount_total para CSV)
            df = pd.DataFrame(rows, columns=[
                'adm3_id', 'analysis_id', 'def_ha', 'risk_total', 
                'farm_amount', 'farm_amount_total'
            ])
            
            # Determinar ruta de salida para CSV
            if output_dir:
                import re
                match = re.search(r'(\d{6}|\d{4}-\d{4})$', period_name)
                period_key = match.group(1) if match else period_name
                
                adm3_dir = output_dir / source / period_type / "adm3_risk"
                adm3_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = adm3_dir / f"{source}_adm3_risk_{period_type}_{period_key}.csv"
            else:
                csv_path = Path(f"adm3_risk_{period_name}.csv")
            
            # Guardar CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            csv_paths.append(str(csv_path))
            csv_processed += 1
            
            # Guardar en BD si está habilitado
            if save_to_db:
                if bulk_insert:
                    db_saved, db_failed, _ = save_adm3_risk_to_db_bulk(rows, analysis_id)
                else:
                    db_saved, db_failed, _ = save_adm3_risk_to_db(rows, analysis_id)
                
                total_db_saved += db_saved
                total_db_failed += db_failed
                print(f"   💾 {period_name}: {db_saved:,} guardados en BD")
            
        except Exception as e:
            logging.error(f"Error procesando {period_name}: {e}")
            print(f"\n   ❌ Error en {period_name}: {e}")
            csv_failed += 1
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"📊 Resumen ADM3 Risk:")
    print(f"   • CSV procesados: {csv_processed}")
    print(f"   • CSV fallidos: {csv_failed}")
    if save_to_db:
        print(f"   • BD guardados: {total_db_saved:,}")
        print(f"   • BD fallidos: {total_db_failed:,}")
    
    if csv_paths:
        print(f"\n📁 Archivos generados:")
        for path in csv_paths[:5]:
            print(f"   • {path}")
        if len(csv_paths) > 5:
            print(f"   ... y {len(csv_paths) - 5} más")
    
    return (csv_processed, csv_failed, csv_paths, total_db_saved, total_db_failed)


def save_adm3_risk_from_csv_batch(
    periods: List[str],
    value_chain: str = "livestock",
    results_dir: Optional[Path] = None,
    source: str = "smbyc",
    period_type: str = "nad",
    bulk_insert: bool = True,
    period_to_defo_id: Dict[str, str] = None
) -> Tuple[int, int, int, int]:
    """
    Lee CSVs de ADM3 Risk existentes y los guarda en MongoDB.
    
    No recalcula nada, solo lee los archivos generados por el paso 7.
    
    Args:
        periods: Lista de períodos a procesar (ej: ['201701', '201702'])
        value_chain: Cadena de valor
        results_dir: Directorio base de resultados
        source: Fuente de deforestación
        period_type: Tipo de período
        bulk_insert: Si True, usa bulk insert
        period_to_defo_id: Mapeo período → deforestation_id
        
    Returns:
        Tupla (csv_found, csv_missing, db_saved, db_failed)
    """
    if not HAS_ORM:
        print("❌ ORM no disponible")
        return (0, 0, 0, 0)
    
    print(f"\n{'='*70}")
    print(f"💾 Guardando ADM3 Risk desde CSVs")
    print(f"{'='*70}")
    print(f"   • Períodos: {len(periods)}")
    print(f"   • Value Chain: {value_chain}")
    print(f"   • Modo: {'Bulk insert' if bulk_insert else 'Individual'}")
    
    if not results_dir:
        print("❌ Error: results_dir no especificado")
        return (0, 0, 0, 0)
    
    csv_found = 0
    csv_missing = 0
    total_db_saved = 0
    total_db_failed = 0
    
    print(f"\n🔄 Procesando CSVs...")
    
    for period in tqdm(periods, desc="Guardando ADM3 Risk", unit="período"):
        try:
            # Construir ruta del CSV
            csv_path = results_dir / source / period_type / "adm3_risk" / f"{source}_adm3_risk_{period_type}_{period}.csv"
            
            if not csv_path.exists():
                print(f"\n   ⚠ CSV no encontrado: {csv_path}")
                csv_missing += 1
                continue
            
            csv_found += 1
            
            # Leer CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print(f"\n   ⚠ CSV vacío: {period}")
                continue
            
            # Obtener analysis_id del CSV o buscarlo
            if 'analysis_id' in df.columns and df['analysis_id'].notna().any():
                analysis_id = str(df['analysis_id'].iloc[0])
            else:
                # Buscar usando deforestation_id
                defo_id = period_to_defo_id.get(period) if period_to_defo_id else None
                if not defo_id:
                    print(f"\n   ⚠ Sin deforestation_id para {period}")
                    csv_missing += 1
                    continue
                    
                analysis_id = get_analysis_for_period(defo_id, value_chain)
                if not analysis_id:
                    print(f"\n   ⚠ Sin Analysis para {period}")
                    csv_missing += 1
                    continue
            
            # Convertir DataFrame a lista de dicts
            rows = df.to_dict('records')
            
            # Guardar en BD
            if bulk_insert:
                db_saved, db_failed, _ = save_adm3_risk_to_db_bulk(rows, analysis_id)
            else:
                db_saved, db_failed, _ = save_adm3_risk_to_db(rows, analysis_id)
            
            total_db_saved += db_saved
            total_db_failed += db_failed
            
            print(f"   ✓ {period}: {db_saved:,} guardados")
            
        except Exception as e:
            logging.error(f"Error procesando CSV {period}: {e}")
            print(f"\n   ❌ Error en {period}: {e}")
            csv_missing += 1
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"📊 Resumen guardado ADM3 Risk:")
    print(f"   • CSVs encontrados: {csv_found}")
    print(f"   • CSVs faltantes: {csv_missing}")
    print(f"   • BD guardados: {total_db_saved:,}")
    print(f"   • BD fallidos: {total_db_failed:,}")
    
    return (csv_found, csv_missing, total_db_saved, total_db_failed)
