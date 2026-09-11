#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cálculo de riesgo de empresa basado en Suppliers (relación empresa-fincas).

A diferencia del enterprise_alert (que usa movimientos de ganado),
este módulo calcula el riesgo de empresas que NO tienen movimientos, pero
SÍ tienen relación con fincas a través de la colección Suppliers.

Flujo:
  1. Buscar empresa por nombre o ext_id.ext_code
  2. Cargar Suppliers (empresa→fincas, con años de relación)
  3. Para cada período (annual/cumulative/nad/atd):
     - Determinar qué suppliers aplican según el año del período
     - Buscar FarmRisk de esas fincas en el Analysis correspondiente
     - Guardar EnterpriseRisk con risk_input = lista de FarmRisk encontrados
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from bson import ObjectId
from tqdm import tqdm
from ganabosques_risk_package.supplier_risk import supplier_risk as pkg_supplier_risk

# ===================== IMPORTACIONES ORM =====================
try:
    from ganabosques_orm.collections.enterprise import Enterprise
    from ganabosques_orm.collections.suppliers import Suppliers
    from ganabosques_orm.collections.farm import Farm
    from ganabosques_orm.collections.farmrisk import FarmRisk
    from ganabosques_orm.collections.enterpriserisk import EnterpriseRisk
    from ganabosques_orm.collections.analysis import Analysis
    from ganabosques_orm.collections.deforestation import Deforestation
    from ganabosques_orm.enums.valuechain import ValueChain
    from ganabosques_orm.enums.label import Label
    HAS_ORM = True
except ImportError:
    HAS_ORM = False
    logging.warning("ganabosques_orm no disponible para supplier_risk_calculator")


# ===================== BÚSQUEDA DE EMPRESA =====================

def find_enterprise(
    name_or_code: str,
    value_chain: Optional[str] = None
) -> Optional[Any]:
    """
    Busca una empresa por nombre o por ext_id.ext_code.
    
    Primero intenta match exacto por ext_id.ext_code (en todos los ext_id),
    luego por nombre (case-insensitive, contiene).
    
    Args:
        name_or_code: Nombre de la empresa o código externo
        value_chain: Cadena de valor para filtrar ('livestock', 'cacao', None=todas)
        
    Returns:
        Objeto Enterprise o None
    """
    if not HAS_ORM:
        print("❌ ganabosques_orm no disponible")
        return None
    
    if not name_or_code or not name_or_code.strip():
        print("❌ Debe proporcionar nombre o código de empresa")
        return None
    
    query_val = name_or_code.strip()
    
    # Construir filtro base de value_chain
    vc_filter = {}
    if value_chain:
        try:
            vc_enum = ValueChain(value_chain.lower())
            vc_filter['value_chain'] = vc_enum
        except ValueError:
            print(f"⚠ value_chain '{value_chain}' no válido, buscando sin filtro")
    
    # 1) Buscar por ext_id.ext_code (match exacto en cualquier ext_id)
    enterprise = Enterprise.objects(
        ext_id__ext_code=query_val,
        **vc_filter
    ).first()
    
    if enterprise:
        print(f"✅ Empresa encontrada por ext_code: {enterprise.name} (id={enterprise.id})")
        return enterprise
    
    # 2) Buscar por nombre exacto (case-insensitive)
    enterprise = Enterprise.objects(
        name__iexact=query_val,
        **vc_filter
    ).first()
    
    if enterprise:
        print(f"✅ Empresa encontrada por nombre exacto: {enterprise.name} (id={enterprise.id})")
        return enterprise
    
    # 3) Buscar por nombre parcial (contiene, case-insensitive)
    matches = Enterprise.objects(
        name__icontains=query_val,
        **vc_filter
    )
    
    count = matches.count()
    if count == 1:
        enterprise = matches.first()
        print(f"✅ Empresa encontrada por nombre parcial: {enterprise.name} (id={enterprise.id})")
        return enterprise
    elif count > 1:
        print(f"⚠ Se encontraron {count} empresas con nombre que contiene '{query_val}':")
        for i, ent in enumerate(matches[:10], 1):
            ext_codes = [e.ext_code for e in (ent.ext_id or []) if e.ext_code]
            ext_str = f" (ext: {', '.join(ext_codes)})" if ext_codes else ""
            print(f"   {i}. {ent.name}{ext_str} [{ent.type_enterprise}]")
        if count > 10:
            print(f"   ... y {count - 10} más")
        print(f"\n💡 Usa un nombre o código más específico")
        return None
    
    print(f"❌ No se encontró empresa: '{query_val}'")
    return None


# ===================== CARGA DE SUPPLIERS =====================

def load_suppliers_for_enterprise(
    enterprise_id: str,
    cache_dir: Optional[Path] = None,
    force_reload: bool = False
) -> List[Dict[str, Any]]:
    """
    Carga todos los Suppliers de una empresa (con caché).
    
    Un Supplier tiene: enterprise_id, farm_id, years (list[int]).
    
    Args:
        enterprise_id: ID (ObjectId str) de la empresa
        cache_dir: Directorio para guardar caché (None = sin caché)
        force_reload: Si True, ignorar caché y recargar desde BD
        
    Returns:
        Lista de dicts: [{'farm_id': str, 'years': [int]}]
    """
    if not HAS_ORM:
        return []
    
    # Intentar caché
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"suppliers_{enterprise_id}.json"
        
        if not force_reload and cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📦 Suppliers desde caché: {len(data)} relaciones")
                return data
            except Exception as e:
                print(f"⚠ Error leyendo caché de suppliers: {e}")
    
    # Cargar desde BD
    print(f"📊 Cargando suppliers desde MongoDB...")
    try:
        suppliers = Suppliers.objects(enterprise_id=ObjectId(enterprise_id))
        
        result = []
        for sup in suppliers:
            farm_oid = sup.farm_id.id if hasattr(sup.farm_id, 'id') else sup.farm_id
            result.append({
                'farm_id': str(farm_oid),
                'years': list(sup.years) if sup.years else []
            })
        
        print(f"✅ Suppliers cargados: {len(result)} relaciones finca-empresa")
        
        
        if result:
            # Estadísticas
            all_years = set()
            for s in result:
                all_years.update(s['years'])
            if all_years:
                print(f"   • Años cubiertos: {min(all_years)}-{max(all_years)}")
                print(f"   • Fincas únicas: {len(set(s['farm_id'] for s in result))}")
        
        # Guardar en caché
        if cache_file and result:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f)
                print(f"💾 Caché guardado: {cache_file.name}")
            except Exception as e:
                logging.warning(f"Error guardando caché de suppliers: {e}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error cargando suppliers: {e}")
        print(f"❌ Error cargando suppliers: {e}")
        return []


# ===================== UTILIDADES DE PERÍODOS =====================

def get_years_for_period(period: str, period_type: str) -> Set[int]:
    """
    Extrae los años que aplican para un período dado.
    
    Args:
        period: Código del período (ej: '2017', '2010-2017', '201701')
        period_type: Tipo ('annual', 'cumulative', 'nad', 'atd')
        
    Returns:
        Set de años (int) que aplican
    """
    if period_type == "cumulative" and "-" in period:
        # '2010-2017' → {2010, 2011, ..., 2017}
        parts = period.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    
    elif period_type in ("nad", "atd") and len(period) == 6:
        # '201701' → {2017}
        return {int(period[:4])}
    
    elif period_type == "annual" and "-" in period:
        # '2017-2018' → {2017, 2018}
        parts = period.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    
    else:
        # '2017' → {2017}
        try:
            return {int(period[:4])}
        except ValueError:
            return set()


def expand_year_to_quarters(year: int) -> List[str]:
    """
    Expande un año a sus 4 períodos trimestrales (para nad/atd).
    
    Args:
        year: Año (ej: 2017)
        
    Returns:
        Lista de códigos trimestrales: ['201701', '201702', '201703', '201704']
    """
    return [f"{year}{q:02d}" for q in range(1, 5)]


def get_suppliers_for_period(
    suppliers: List[Dict[str, Any]],
    period: str,
    period_type: str
) -> List[str]:
    """
    Filtra suppliers que aplican para un período según sus años de relación.
    
    Args:
        suppliers: Lista de suppliers [{'farm_id': str, 'years': [int]}]
        period: Código del período
        period_type: Tipo de período
        
    Returns:
        Lista de farm_id (str) que aplican
    """
    target_years = get_years_for_period(period, period_type)
    
    if not target_years:
        return []
    
    farm_ids = []
    for sup in suppliers:
        sup_years = set(sup.get('years', []))
        # Si hay intersección entre los años del supplier y los años del período
        # if sup_years & target_years: Comentado para correr todos los años incialmente, Descomentar mas adelante 
        farm_ids.append(sup['farm_id'])
    
    return farm_ids


# ===================== BÚSQUEDA DE ANALYSIS Y FARMRISK =====================

def get_analysis_for_period(
    period: str,
    period_type: str,
    source: str,
    value_chain: str
) -> Optional[str]:
    """
    Obtiene el Analysis ID para un período dado.
    
    Busca la capa de deforestación por nombre y luego el Analysis asociado.
    
    Args:
        period: Código del período
        period_type: Tipo de período
        source: Fuente ('smbyc')
        value_chain: Cadena de valor
        
    Returns:
        analysis_id (str) o None
    """
    if not HAS_ORM:
        return None
    
    # Construir nombre esperado de la capa
    expected_name = f"{source}_deforestation_{period_type}_{period}"
    
    try:
        defo = Deforestation.objects(name=expected_name).first()
        if not defo:
            logging.warning(f"Capa de deforestación no encontrada: {expected_name}")
            return None
        
        analysis = Analysis.objects(
            deforestation_id=defo.id,
            value_chain=value_chain.lower()
        ).first()
        
        if not analysis:
            logging.warning(f"Analysis no encontrado para {expected_name} (vc={value_chain})")
            return None
        
        return str(analysis.id)
        
    except Exception as e:
        logging.error(f"Error buscando analysis para {period}: {e}")
        return None


def find_farm_risks_for_farms(
    farm_ids: List[str],
    analysis_id: str
) -> List[Any]:
    """
    Busca los FarmRisk en BD para una lista de farms y un analysis.
    
    Args:
        farm_ids: Lista de farm ObjectId strings
        analysis_id: ID del Analysis
        
    Returns:
        Lista de objetos FarmRisk encontrados
    """
    if not HAS_ORM or not farm_ids:
        return []
    
    try:
        farm_oids = [ObjectId(fid) for fid in farm_ids]
        
        farm_risks = FarmRisk.objects(
            farm_id__in=farm_oids,
            analysis_id=ObjectId(analysis_id)
        )
        
        return list(farm_risks)
        
    except Exception as e:
        logging.error(f"Error buscando FarmRisk: {e}")
        return []


# ===================== CÁLCULO Y GUARDADO =====================

def calculate_and_save_supplier_risk(
    enterprise_name_or_code: str,
    periods: List[str],
    period_type: str,
    source: str = "smbyc",
    value_chain: str = "livestock",
    cache_dir: Optional[Path] = None,
    total_risk_dir: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Calcula y guarda el riesgo de una empresa basado en sus Suppliers.
    
    Flujo completo:
    1. Buscar empresa por nombre o ext_code
    2. Cargar suppliers (con caché)
    3. Para cada período:
       a. Filtrar suppliers por año
       b. (Opcional) Usar paquete ganabosques_risk_package para identificar fincas con riesgo desde CSV
       c. Buscar FarmRisk de las fincas relacionadas
       d. Guardar EnterpriseRisk con risk_input
    
    Para nad/atd: si el supplier tiene año 2017, aplica a
    201701, 201702, 201703, 201704.
    
    Args:
        enterprise_name_or_code: Nombre o ext_code de la empresa
        periods: Lista de períodos a procesar
        period_type: Tipo de período
        source: Fuente de deforestación
        value_chain: Cadena de valor
        cache_dir: Directorio de caché para suppliers
        total_risk_dir: Directorio con CSVs de total_risk (opcional). Si se proporciona,
                        usa ganabosques_risk_package para pre-identificar fincas con riesgo.
        dry_run: Si True, no guardar en BD (solo mostrar)
        
    Returns:
        Dict con estadísticas del procesamiento
    """
    if not HAS_ORM:
        return {"success": False, "error": "ganabosques_orm no disponible"}
    
    # 1. Buscar empresa
    print(f"\n🔍 Buscando empresa: '{enterprise_name_or_code}'...")
    enterprise = find_enterprise(enterprise_name_or_code, value_chain)
    
    if not enterprise:
        return {"success": False, "error": f"Empresa no encontrada: {enterprise_name_or_code}"}
    
    enterprise_id = str(enterprise.id)
    print(f"   • Nombre: {enterprise.name}")
    print(f"   • Tipo: {enterprise.type_enterprise}")
    print(f"   • ID: {enterprise_id}")
    
    # 2. Cargar suppliers
    suppliers = load_suppliers_for_enterprise(enterprise_id, cache_dir)
    
    if not suppliers:
        return {
            "success": False,
            "error": f"No se encontraron suppliers para {enterprise.name}"
        }
    
    # Construir suppliers_df para uso con el paquete ganabosques_risk_package
    suppliers_df = pd.DataFrame(suppliers)
    suppliers_df['enterprise_id'] = enterprise_id
    
    # 3. Procesar cada período
    total_saved = 0
    total_skipped = 0
    total_farm_risks = 0
    total_filter_farm_risks = 0
    period_results = []
    
    print(f"\n📋 Procesando {len(periods)} períodos (tipo: {period_type})...")
    
    for period in tqdm(periods, desc="Procesando períodos", unit="per"):
        # Filtrar suppliers por año del período
        farm_ids = get_suppliers_for_period(suppliers, period, period_type)
        
        if not farm_ids:
            logging.info(f"Período {period}: sin suppliers activos")
            total_skipped += 1
            period_results.append({
                "period": period, "status": "skipped",
                "reason": "sin suppliers activos para este año"
            })
            continue
        
        # Buscar analysis
        analysis_id = get_analysis_for_period(period, period_type, source, value_chain)
        
        if not analysis_id:
            logging.warning(f"Período {period}: sin Analysis en BD")
            total_skipped += 1
            period_results.append({
                "period": period, "status": "skipped",
                "reason": "Analysis no encontrado"
            })
            continue
        
        # Buscar FarmRisk de las fincas relacionadas
        farm_risks = find_farm_risks_for_farms(farm_ids, analysis_id)
        
        if not farm_risks:
            logging.info(f"Período {period}: {len(farm_ids)} suppliers pero sin FarmRisk")
            total_skipped += 1
            period_results.append({
                "period": period, "status": "skipped",
                "reason": f"{len(farm_ids)} suppliers pero sin FarmRisk en BD"
            })
            continue

        filter_farmrisk = [fr for fr in farm_risks if fr.risk_input or fr.risk_direct or fr.risk_output ]
        
        total_farm_risks += len(farm_risks)
        total_filter_farm_risks += len(filter_farmrisk)
        
        # Usar paquete ganabosques_risk_package para análisis de riesgo por supplier
        try:
            # Construir total_risk_df a partir de FarmRisk de MongoDB
            risk_rows = []
            for fr in farm_risks:
                farm_oid = str(fr.farm_id.id) if hasattr(fr.farm_id, 'id') else str(fr.farm_id)
                risk_rows.append({
                    "id": farm_oid,
                    "direct_alert": bool(fr.risk_direct) if fr.risk_direct else False,
                    "indirect_alert_in": bool(fr.risk_input) if fr.risk_input else False,
                    "indirect_alert_out": bool(fr.risk_output) if fr.risk_output else False,
                })
            total_risk_df = pd.DataFrame(risk_rows)
            
            pkg_result = pkg_supplier_risk(
                total_risk_df=total_risk_df,
                suppliers_df=suppliers_df,
                period=period,
                period_type=period_type,
                id_column="id",
                farm_id_column="farm_id",
                enterprise_id_column="enterprise_id",
                normalize_ids=False,
                show_progress=True,
            )
            
            # Guardar CSV con análisis del paquete si hay directorio de salida
            if total_risk_dir and not pkg_result.empty:
                out_csv_dir = Path(total_risk_dir)
                out_csv_dir.mkdir(parents=True, exist_ok=True)
                pkg_csv = out_csv_dir / f"{source}_supplier_risk_{period_type}_{period}_{enterprise.name.replace(' ', '_')}.csv"
                pkg_result.to_csv(pkg_csv, index=False, encoding="utf-8-sig")
                logging.info(f"Período {period}: análisis paquete guardado en {pkg_csv.name}")
        except Exception as e:
            logging.warning(f"Error usando paquete supplier_risk para {period}: {e}")
        
        # Guardar EnterpriseRisk
        if not filter_farmrisk:
            logging.info(f"Período {period}: FarmRisk encontrados pero ninguno con riesgo activo")
            total_skipped += 1
            period_results.append({
                "period": period, "status": "skipped",
                "reason": f"{len(farm_risks)} FarmRisk encontrados pero ninguno con riesgo activo"
            })
            continue

        if dry_run:
            print(f"   [DRY-RUN] {period}: {len(filter_farmrisk)} FarmRisk de {len(farm_ids)} suppliers")
            period_results.append({
                "period": period, "status": "dry_run",
                "suppliers": len(farm_ids), "farm_risks": len(farm_risks), "filtered": len(filter_farmrisk)
            })
            continue
        
        try:
            analysis = Analysis.objects(id=ObjectId(analysis_id)).first()
            
            # Buscar si ya existe
            ent_risk = EnterpriseRisk.objects(
                enterprise_id=enterprise.id,
                analysis_id=analysis.id
            ).first()
            
            if ent_risk:
                # Actualizar: agregar farm_risks sin duplicar
                existing_ids = {str(r.id) for r in (ent_risk.risk_input or [])}
                new_refs = [fr for fr in filter_farmrisk if str(fr.id) not in existing_ids]
                
                if new_refs:
                    if ent_risk.risk_input is None:
                        ent_risk.risk_input = []
                    ent_risk.risk_input.extend(new_refs)
                    ent_risk.save()
                    total_saved += 1
                    period_results.append({
                        "period": period, "status": "updated",
                        "new_refs": len(new_refs), "total_refs": len(ent_risk.risk_input)
                    })
                else:
                    period_results.append({
                        "period": period, "status": "no_change",
                        "reason": "todos los FarmRisk ya estaban"
                    })
            else:
                # Crear nuevo
                ent_risk = EnterpriseRisk(
                    enterprise_id=enterprise.id,
                    analysis_id=analysis.id,
                    risk_input=filter_farmrisk,
                    risk_output=[]
                )
                ent_risk.save()
                total_saved += 1
                period_results.append({
                    "period": period, "status": "created",
                    "farm_risks": len(farm_risks), "filtered": len(filter_farmrisk)
                })
            
        except Exception as e:
            logging.error(f"Error guardando EnterpriseRisk para {period}: {e}")
            period_results.append({
                "period": period, "status": "error", "error": str(e)
            })
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"📊 Resumen - Riesgo por Suppliers para: {enterprise.name}")
    print(f"{'='*60}")
    print(f"   • Períodos procesados: {len(periods)}")
    print(f"   • EnterpriseRisk guardados/actualizados: {total_saved}")
    print(f"   • Períodos omitidos: {total_skipped}")
    print(f"   • Total FarmRisk vinculados: {total_filter_farm_risks} (de {total_farm_risks} encontrados)")
    
    # Mostrar detalle de períodos con resultado
    created = [r for r in period_results if r['status'] == 'created']
    updated = [r for r in period_results if r['status'] == 'updated']
    skipped = [r for r in period_results if r['status'] == 'skipped']
    errors = [r for r in period_results if r['status'] == 'error']
    
    if created:
        print(f"   • Creados nuevos: {len(created)}")
    if updated:
        print(f"   • Actualizados: {len(updated)}")
    if skipped:
        print(f"   • Omitidos: {len(skipped)}")
        for s in skipped[:5]:
            print(f"     - {s['period']}: {s['reason']}")
    if errors:
        print(f"   • Con error: {len(errors)}")
        for e in errors[:5]:
            print(f"     - {e['period']}: {e['error']}")
    
    return {
        "success": True or dry_run,
        "enterprise_name": enterprise.name,
        "enterprise_id": enterprise_id,
        "periods_processed": len(periods),
        "saved": total_saved,
        "skipped": total_skipped,
        "total_farm_risks": total_farm_risks,
        "period_results": period_results
    }
