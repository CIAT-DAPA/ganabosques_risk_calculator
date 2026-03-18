#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cálculo de alertas de empresa basado en movimientos de ganado.

Este módulo identifica las empresas (mataderos, ferias, procesadores) que tuvieron
movimientos con fincas que presentaron alertas directas de deforestación.

Formato de salida:
    - idpro: ID de la empresa (code del documento enterprise)
    - id_farm: ID de la finca involucrada en el movimiento
    - typemove: Tipo de movimiento ('in' = entrada a empresa, 'out' = salida de empresa)
    - period: Período del análisis
    - farm_has_direct_alert: Si la finca tiene alerta directa
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm
from ganabosques_risk_package.alert_enterprise import alert_enterprise as pkg_alert_enterprise


# ===================== CONFIGURACIÓN =====================

DEFAULT_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_MONGO_DB = os.getenv("MONGO_DB_NAME", "ganabosques")

# Colecciones MongoDB
COL_ENTERPRISE = "enterprise"
COL_MOVEMENT = "movement"
COL_FARM = "farm"


# ===================== UTILIDADES =====================

def normalize_id(value: Any) -> str:
    """Normaliza un ID a string limpio."""
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    # Quitar .0 de números flotantes
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]
    # Quitar ceros a la izquierda
    if re.match(r"^0*\d+$", s):
        s = str(int(s))
    return s.upper()


def parse_year_from_period(period: str, period_type: str) -> int:
    """
    Extrae el año de un período para buscar movimientos.
    
    Para cumulative (ej: '2010-2024'), retorna el último año (2024).
    Para nad/atd (ej: '201701'), retorna el año (2017).
    Para annual (ej: '2017'), retorna el año directamente.
    """
    if period_type == "cumulative" and "-" in period:
        # Formato: 2010-2024 -> usar último año para movimientos
        return int(period.split("-")[1])
    elif period_type in ("nad", "atd") and len(period) == 6:
        # Formato: YYYYQQ -> extraer año
        return int(period[:4])
    return int(period[:4])


def parse_quarter_from_period(period: str, period_type: str) -> Optional[int]:
    """
    Extrae el trimestre de un período (1-4) o None si no aplica.
    
    Solo aplica para nad/atd con formato YYYYQQ.
    Para annual y cumulative retorna None.
    """
    if period_type in ("nad", "atd") and len(period) == 6:
        return int(period[-2:])
    return None


# ===================== CARGA DE DATOS =====================

def load_direct_alerts_with_alert(csv_path: str) -> Set[str]:
    """
    Carga CSV de alertas directas y retorna IDs de fincas CON alerta.
    
    Args:
        csv_path: Ruta al CSV de alertas directas
        
    Returns:
        Set de IDs de fincas con alerta directa = True
    """
    if not os.path.isfile(csv_path):
        logging.warning(f"No existe CSV de alertas directas: {csv_path}")
        return set()
    
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        
        if "id" not in df.columns:
            logging.warning(f"CSV sin columna 'id': {csv_path}")
            return set()
        
        # Buscar columna de alerta
        col_alert = None
        for col_name in ["direct_alert", "intersect_deforestation", "intersect_early_warnings"]:
            if col_name in df.columns:
                col_alert = col_name
                break
        
        if not col_alert:
            logging.warning(f"CSV sin columna de alerta: {csv_path}")
            return set()
        
        # Filtrar solo los que tienen alerta True
        df["id_normalized"] = df["id"].apply(normalize_id)
        
        def str_to_bool(x):
            s = str(x).strip().lower()
            return s in {"true", "t", "1", "yes", "y", "si", "sí"}
        
        df["has_alert"] = df[col_alert].apply(str_to_bool)
        
        farms_with_alert = set(df[df["has_alert"]]["id_normalized"].unique())
        
        logging.info(f"Fincas con alerta directa: {len(farms_with_alert)} / {len(df)}")
        return farms_with_alert
    
    except Exception as e:
        logging.error(f"Error cargando alertas directas: {e}")
        return set()


def load_movements_for_year(
    csv_path: str,
    quarter: Optional[int] = None
) -> pd.DataFrame:
    """
    Carga movimientos desde CSV del año.
    
    Args:
        csv_path: Ruta al CSV de movimientos
        quarter: Trimestre a filtrar (1-4) o None para todo el año
        
    Returns:
        DataFrame con columnas: origen_id, destination_id, tipo_origen, tipo_destino,
                                producer_id_origen, producer_id_destino, date
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe archivo de movimientos: {csv_path}")
    
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    
    # Normalizar nombres de columnas
    column_map = {
        "SIT_CODE_ORIGEN": "origen_id",
        "SIT_CODE_DESTINO": "destination_id",
        "TIPO_ORIGEN": "tipo_origen",
        "TIPO_DESTINO": "tipo_destino",
        "PRODUCER_ID_ORIGEN": "producer_id_origen",
        "PRODUCER_ID_DESTINO": "producer_id_destino",
        "DATE": "date",
        "IDPRO": "idpro",
        "ID_PRO": "idpro"
    }
    
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Normalizar IDs de farms
    if "origen_id" in df.columns:
        df["origen_id"] = df["origen_id"].apply(normalize_id)
    if "destination_id" in df.columns:
        df["destination_id"] = df["destination_id"].apply(normalize_id)
    
    # Normalizar IDs de productores/empresas
    if "producer_id_origen" in df.columns:
        df["producer_id_origen"] = df["producer_id_origen"].apply(normalize_id)
    if "producer_id_destino" in df.columns:
        df["producer_id_destino"] = df["producer_id_destino"].apply(normalize_id)
    
    # Filtrar por trimestre si aplica
    if quarter is not None and "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["quarter"] = df["date"].dt.quarter
            df = df[df["quarter"] == quarter].copy()
            logging.info(f"Filtrado a trimestre {quarter}: {len(df):,} movimientos")
        except Exception as e:
            logging.warning(f"Error filtrando por trimestre: {e}")
    
    return df


def load_enterprise_mapping(
    mongo_uri: str = DEFAULT_MONGO_URI,
    mongo_db: str = DEFAULT_MONGO_DB,
    enterprise_types: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Carga mapeo de empresas desde MongoDB.
    
    Args:
        mongo_uri: URI de MongoDB
        mongo_db: Nombre de la base de datos
        enterprise_types: Lista de tipos a filtrar (ej: ["SLAUGHTERHOUSE", "CATTLE_FAIR"])
        
    Returns:
        Dict {code: {name, type_enterprise, _id}}
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        
        query = {}
        if enterprise_types:
            query["type_enterprise"] = {"$in": enterprise_types}
        
        enterprises = {}
        for doc in db[COL_ENTERPRISE].find(query):
            code = doc.get("code", "")
            if code:
                enterprises[str(code)] = {
                    "name": doc.get("name", ""),
                    "type_enterprise": doc.get("type_enterprise", ""),
                    "_id": str(doc["_id"])
                }
        
        client.close()
        logging.info(f"Empresas cargadas: {len(enterprises)}")
        return enterprises
    
    except Exception as e:
        logging.error(f"Error cargando empresas de MongoDB: {e}")
        return {}


# ===================== CÁLCULO DE ALERTAS =====================

def calculate_enterprise_alerts_for_period(
    period: str,
    period_type: str,
    source: str,
    direct_alerts_dir: str,
    movement_csv_dir: str,
    output_dir: str,
    enterprise_types: Optional[List[str]] = None,
    mongo_uri: str = DEFAULT_MONGO_URI,
    mongo_db: str = DEFAULT_MONGO_DB
) -> Dict[str, Any]:
    """
    Calcula alertas de empresa para un período.
    
    Args:
        period: Período a procesar (ej: "201701" para nad/atd, "2017" para annual, "2010-2024" para cumulative)
        period_type: Tipo de período ("nad", "atd", "annual", "cumulative")
        source: Fuente de deforestación ("smbyc")
        direct_alerts_dir: Directorio con CSVs de alertas directas
        movement_csv_dir: Directorio con CSVs de movimientos
        output_dir: Directorio de salida
        enterprise_types: Tipos de empresa a incluir
        mongo_uri: URI de MongoDB
        mongo_db: Nombre de la base de datos
        
    Returns:
        Dict con estadísticas del procesamiento
    """
    try:
        # 1. Determinar año (para cumulative usa el último año del rango)
        year = parse_year_from_period(period, period_type)
        quarter = parse_quarter_from_period(period, period_type)
        
        # 2. Cargar alertas directas (direct_alerts_dir ya incluye source/period_type)
        direct_csv = Path(direct_alerts_dir) / f"{source}_direct_alert_{period_type}_{period}.csv"
        
        logging.info(f"Cargando alertas directas: {direct_csv}")
        farms_with_alert = load_direct_alerts_with_alert(str(direct_csv))
        
        if not farms_with_alert:
            logging.warning(f"No hay fincas con alerta directa para {period}")
            return {
                "success": False,
                "period": period,
                "error": "No farms with direct alert"
            }
        
        # 3. Cargar movimientos del año
        movement_csv = Path(movement_csv_dir) / f"movement_data_base_{year}.csv"
        
        if not movement_csv.exists():
            logging.warning(f"No existe CSV de movimientos: {movement_csv}")
            return {
                "success": False,
                "period": period,
                "error": f"Movement data not found for year {year}"
            }
        
        logging.info(f"Cargando movimientos: {movement_csv}")
        movements_df = load_movements_for_year(str(movement_csv), quarter)
        
        logging.info(f"Movimientos cargados: {len(movements_df):,}")
        
        # 4. Cargar mapeo de empresas (opcional, para enriquecer)
        enterprise_map = load_enterprise_mapping(mongo_uri, mongo_db, enterprise_types)
        
        # 5-11. Usar paquete ganabosques_risk_package para cálculo de alertas de empresa
        # Construir total_risk_df mínimo a partir de fincas con alerta
        total_risk_df = pd.DataFrame({
            "id": list(farms_with_alert),
            "direct_alert": True
        })
        
        result_df = pkg_alert_enterprise(
            total_risk_df=total_risk_df,
            movements_df=movements_df,
            id_column="id",
            normalize_ids=True,
            show_progress=True,
        )
        
        if result_df.empty:
            logging.warning(f"No se encontraron movimientos con fincas alertadas para {period}")
            return {
                "success": False,
                "period": period,
                "error": "No movements with alerted farms"
            }
        
        # Agregar metadata del período
        result_df["period"] = period
        result_df["year"] = year
        if quarter:
            result_df["quarter"] = quarter
        result_df["source"] = source
        
        # Renombrar enterprise_type para compatibilidad
        if "enterprise_type" in result_df.columns:
            result_df["enterprise_type_raw"] = result_df["enterprise_type"]
        
        # 12. Agregar info de empresa si está disponible
        if enterprise_map:
            result_df["enterprise_name"] = result_df["idpro"].map(
                lambda x: enterprise_map.get(x, {}).get("name", "")
            )
            result_df["enterprise_type"] = result_df["idpro"].map(
                lambda x: enterprise_map.get(x, {}).get("type_enterprise", "")
            )
        
        # 13. Guardar resultado (output_dir ya incluye source/period_type)
        output_path = Path(output_dir) / f"{source}_enterprise_alert_{period_type}_{period}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        # Estadísticas
        n_entries = (result_df["typemove"] == "in").sum()
        n_exits = (result_df["typemove"] == "out").sum()
        unique_enterprises = result_df["idpro"].nunique()
        unique_farms = result_df["id_farm"].nunique()
        
        logging.info(f"Alertas de empresa calculadas para {period}")
        logging.info(f"  • Movimientos entrada (in): {n_entries}")
        logging.info(f"  • Movimientos salida (out): {n_exits}")
        logging.info(f"  • Empresas únicas: {unique_enterprises}")
        logging.info(f"  • Fincas únicas: {unique_farms}")
        logging.info(f"  • Guardado en: {output_path}")
        
        return {
            "success": True,
            "period": period,
            "year": year,
            "quarter": quarter,
            "entries_count": n_entries,
            "exits_count": n_exits,
            "unique_enterprises": unique_enterprises,
            "unique_farms": unique_farms,
            "total_records": len(result_df),
            "output_file": str(output_path)
        }
    
    except Exception as e:
        logging.error(f"Error procesando período {period}: {e}", exc_info=True)
        return {
            "success": False,
            "period": period,
            "error": str(e)
        }


def calculate_enterprise_alerts_batch(
    periods: List[str],
    period_type: str,
    source: str,
    direct_alerts_dir: str,
    movement_csv_dir: str,
    output_dir: str,
    enterprise_types: Optional[List[str]] = None,
    mongo_uri: str = DEFAULT_MONGO_URI,
    mongo_db: str = DEFAULT_MONGO_DB
) -> Dict[str, Any]:
    """
    Procesa alertas de empresa para múltiples períodos.
    
    Args:
        periods: Lista de períodos a procesar
        period_type: Tipo de período ("nad", "atd", "annual", "cumulative")
        source: Fuente de deforestación ("smbyc")
        direct_alerts_dir: Directorio con CSVs de alertas directas
        movement_csv_dir: Directorio con CSVs de movimientos
        output_dir: Directorio de salida
        enterprise_types: Tipos de empresa a incluir
        mongo_uri: URI de MongoDB
        mongo_db: Nombre de la base de datos
        
    Returns:
        Dict con estadísticas agregadas
    """
    results = []
    
    for period in tqdm(periods, desc="Procesando alertas empresa", unit="período"):
        result = calculate_enterprise_alerts_for_period(
            period=period,
            period_type=period_type,
            source=source,
            direct_alerts_dir=direct_alerts_dir,
            movement_csv_dir=movement_csv_dir,
            output_dir=output_dir,
            enterprise_types=enterprise_types,
            mongo_uri=mongo_uri,
            mongo_db=mongo_db
        )
        results.append(result)
    
    # Agregar estadísticas
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    total_entries = sum(r.get("entries_count", 0) for r in successful)
    total_exits = sum(r.get("exits_count", 0) for r in successful)
    total_records = sum(r.get("total_records", 0) for r in successful)
    
    return {
        "success": len(failed) == 0,
        "periods_processed": len(successful),
        "periods_failed": len(failed),
        "failed_periods": [r.get("period") for r in failed],
        "total_entries": total_entries,
        "total_exits": total_exits,
        "total_records": total_records,
        "results": results
    }


# ===================== CLI =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calcular alertas de empresa")
    parser.add_argument("-s", "--source", default="smbyc", help="Fuente (smbyc)")
    parser.add_argument("-pt", "--period-type", default="nad", 
                        choices=["nad", "atd", "annual", "cumulative"], help="Tipo de período")
    parser.add_argument("-p", "--periods", required=True, nargs="+", help="Períodos a procesar")
    parser.add_argument("--direct-alerts-dir", required=True, help="Directorio de alertas directas")
    parser.add_argument("--movement-dir", required=True, help="Directorio de movimientos")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument("--enterprise-types", nargs="*", help="Tipos de empresa (SLAUGHTERHOUSE, CATTLE_FAIR)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    result = calculate_enterprise_alerts_batch(
        periods=args.periods,
        period_type=args.period_type,
        source=args.source,
        direct_alerts_dir=args.direct_alerts_dir,
        movement_csv_dir=args.movement_dir,
        output_dir=args.output_dir,
        enterprise_types=args.enterprise_types
    )
    
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"Períodos exitosos: {result['periods_processed']}")
    print(f"Períodos fallidos: {result['periods_failed']}")
    print(f"Total movimientos entrada: {result['total_entries']}")
    print(f"Total movimientos salida: {result['total_exits']}")
    print(f"Total registros: {result['total_records']}")
