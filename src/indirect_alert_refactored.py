#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cálculo de alertas indirectas por movimiento de ganado.
Refactorizado como función callable para integración con main.py.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import pandas as pd
from tqdm import tqdm


# ===================== UTILIDADES =====================

def normalize_id_series(sr: pd.Series, config: Dict[str, bool]) -> pd.Series:
    """
    Normaliza serie de IDs según configuración.
    
    Args:
        sr: Serie de pandas con IDs
        config: Dict con flags de normalización:
            - strip_dot_zero: "123.0" -> "123"
            - strip_leading_zeros: "00123" -> "123"
            - uppercase: "ab-01" -> "AB-01"
    """
    def normalize(x):
        s = "" if x is None or pd.isna(x) else str(x).strip()
        if s == "" or s.lower() == "nan":
            return ""
        
        # Quitar sufijo ".0" si aplica
        if config.get("strip_dot_zero", True) and re.match(r"^\d+\.0$", s):
            s = s[:-2]
        
        # Quitar ceros a la izquierda (solo si queda numérico puro)
        if config.get("strip_leading_zeros", True) and re.match(r"^0*\d+$", s):
            s = str(int(s))
        
        # Mayúsculas
        if config.get("uppercase", True):
            s = s.upper()
        
        return s
    
    return sr.map(normalize)


def str_bool(x: Any) -> bool:
    """Convierte string a booleano."""
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "si", "sí"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return bool(s and s != "nan")


def parse_quarter_from_period(period: str) -> Optional[int]:
    """
    Extrae trimestre de período.
    '201701' -> 1, '201702' -> 2, etc.
    """
    if len(period) == 6 and period.isdigit():
        quarter = int(period[-2:])
        if 1 <= quarter <= 4:
            return quarter
    return None


def filter_movements_by_quarter(df: pd.DataFrame, quarter: int, date_column: str = 'DATE') -> pd.DataFrame:
    """
    Filtra movimientos por trimestre.
    
    Args:
        df: DataFrame con movimientos
        quarter: Trimestre (1-4)
        date_column: Nombre de columna con fecha
    """
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df['quarter'] = df[date_column].dt.quarter
        filtered = df[df['quarter'] == quarter].copy()
        return filtered.drop(columns=['quarter'])
    except Exception as e:
        logging.warning(f"Error filtrando por trimestre: {e}")
        return df


# ===================== CARGA DE DATOS =====================

def load_direct_alerts(csv_path: str, normalize_config: Dict[str, bool]) -> pd.DataFrame:
    """
    Carga alertas directas desde CSV.
    
    Returns:
        DataFrame con columnas: id, direct_alert (bool)
    """
    if not os.path.isfile(csv_path):
        logging.warning(f"No se encontró CSV de alertas directas: {csv_path}")
        return pd.DataFrame(columns=["id", "direct_alert"])
    
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        
        if "id" not in df.columns:
            logging.warning(f"CSV sin columna 'id': {csv_path}")
            return pd.DataFrame(columns=["id", "direct_alert"])
        
        # Buscar columna de alerta
        col_alert = None
        for col_name in ["direct_alert", "intersect_deforestation", "intersect_early_warnings"]:
            if col_name in df.columns:
                col_alert = col_name
                break
        
        if not col_alert:
            logging.warning(f"CSV sin columna de alerta: {csv_path}")
            return pd.DataFrame(columns=["id", "direct_alert"])
        
        # Extraer y normalizar IDs
        out = df[["id", col_alert]].copy()
        out["id"] = normalize_id_series(out["id"], normalize_config)
        out["direct_alert"] = out[col_alert].map(str_bool)
        
        return out[["id", "direct_alert"]]
    
    except Exception as e:
        logging.error(f"Error cargando alertas directas: {e}")
        return pd.DataFrame(columns=["id", "direct_alert"])


def load_movements(csv_path: str, normalize_config: Dict[str, bool], 
                   quarter: Optional[int] = None,
                   include_enterprise_fields: bool = False) -> pd.DataFrame:
    """
    Carga movimientos desde CSV del año.
    
    Args:
        csv_path: Ruta al CSV (movement_data_base_YYYY.csv)
        normalize_config: Config de normalización de IDs
        quarter: Si se especifica, filtra por trimestre (1-4)
        include_enterprise_fields: Si True, incluye campos para alertas de empresa
    
    Returns:
        DataFrame con columnas: origen_id, destination_id, date, y opcionalmente
        tipo_origen, tipo_destino, producer_id_origen, producer_id_destino
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe archivo de movimientos: {csv_path}")
    
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    
    # Normalizar nombres de columnas
    column_mapping = {
        "SIT_CODE_ORIGEN": "origen_id",
        "SIT_CODE_DESTINO": "destination_id",
        "DATE": "date",
        "TIPO_ORIGEN": "tipo_origen",
        "TIPO_DESTINO": "tipo_destino",
        "PRODUCER_ID_ORIGEN": "producer_id_origen",
        "PRODUCER_ID_DESTINO": "producer_id_destino"
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    if not {"origen_id", "destination_id"}.issubset(df.columns):
        raise ValueError(f"CSV debe tener columnas SIT_CODE_ORIGEN y SIT_CODE_DESTINO")
    
    # Normalizar IDs
    df["origen_id"] = normalize_id_series(df["origen_id"], normalize_config)
    df["destination_id"] = normalize_id_series(df["destination_id"], normalize_config)
    
    # Normalizar IDs de empresa si se incluyen
    if include_enterprise_fields:
        if "producer_id_origen" in df.columns:
            df["producer_id_origen"] = normalize_id_series(df["producer_id_origen"], normalize_config)
        if "producer_id_destino" in df.columns:
            df["producer_id_destino"] = normalize_id_series(df["producer_id_destino"], normalize_config)
    
    # Filtrar por trimestre si es necesario
    if quarter is not None and "date" in df.columns:
        df = filter_movements_by_quarter(df, quarter, "date")
    
    # Seleccionar columnas de salida
    output_cols = ["origen_id", "destination_id"]
    if include_enterprise_fields:
        for col in ["tipo_origen", "tipo_destino", "producer_id_origen", "producer_id_destino"]:
            if col in df.columns:
                output_cols.append(col)
    
    return df[output_cols]


# ===================== CÁLCULO DE MÉTRICAS =====================

def calculate_indirect_metrics(
    alerts_df: pd.DataFrame,
    movements_df: pd.DataFrame,
    period: str = ""
) -> pd.DataFrame:
    """
    Calcula métricas de alertas indirectas.
    
    Args:
        alerts_df: DataFrame con id, direct_alert
        movements_df: DataFrame con origen_id, destination_id
        period: Identificador del período (para barra de progreso)
    
    Returns:
        DataFrame con métricas por farm_id
    """
    # Crear dict de IDs con alerta
    alert_ids_bool = {
        row["id"]: bool(row["direct_alert"]) 
        for _, row in alerts_df.iterrows()
    }
    
    # Diagnóstico de intersección
    ids_alerta = set(alert_ids_bool.keys())
    origenes = set(movements_df["origen_id"].unique())
    destinos = set(movements_df["destination_id"].unique())
    universo_mov = origenes | destinos
    inter = ids_alerta & universo_mov
    
    logging.info(f"IDs con alerta directa TRUE: {sum(alert_ids_bool.values())} / {len(alert_ids_bool)}")
    logging.info(f"Movimientos: orígenes={len(origenes)}, destinos={len(destinos)}, total={len(universo_mov)}")
    logging.info(f"Intersección alerta ∩ movimientos: {len(inter)}")
    
    if len(inter) == 0:
        logging.warning("No hay cruce de IDs entre alertas y movimientos. Revisa normalización.")
    
    # Mostrar progreso de procesamiento
    total_farms = len(alert_ids_bool)
    total_movements = len(movements_df)
    tqdm.write(f"    [{period}] {total_farms:,} farms × {total_movements:,} movimientos")
    
    # Marcar movimientos con alerta
    def has_alert(series):
        return series.map(lambda k: bool(alert_ids_bool.get(k, False)))
    
    movements_df["origin_has_alert"] = has_alert(movements_df["origen_id"])
    movements_df["dest_has_alert"] = has_alert(movements_df["destination_id"])
    
    # Calcular métricas solo para IDs presentes en alertas
    ids = pd.Index(sorted(alert_ids_bool.keys()))
    
    # Total de movimientos entrantes/salientes
    n_in = movements_df.groupby("destination_id").size().reindex(ids).fillna(0).astype(int)
    n_out = movements_df.groupby("origen_id").size().reindex(ids).fillna(0).astype(int)
    
    # Movimientos indirectos (desde/hacia farms con alerta)
    tmp_in = movements_df[movements_df["origin_has_alert"]].groupby("destination_id").size()
    tmp_out = movements_df[movements_df["dest_has_alert"]].groupby("origen_id").size()
    
    # Construir resultado
    result = pd.DataFrame({
        "id": ids,
        "n_in": n_in.values,
        "n_out": n_out.values,
        "n_indirect_in": tmp_in.reindex(ids).fillna(0).astype(int).values,
        "n_indirect_out": tmp_out.reindex(ids).fillna(0).astype(int).values,
    })
    
    result["n_total_mov"] = result["n_in"] + result["n_out"]
    result["indirect_alert_in"] = result["n_indirect_in"] > 0
    result["indirect_alert_out"] = result["n_indirect_out"] > 0
    
    return result


def extract_enterprise_alerts(
    movements_df: pd.DataFrame,
    farms_with_alert: Set[str],
    period: str,
    year: str,
    quarter: Optional[int] = None
) -> pd.DataFrame:
    """
    Extrae alertas de empresa del DataFrame de movimientos.
    
    Identifica movimientos donde:
    - FARM con alerta → EMPRESA (tipo_destino != FARM): typemove = 'in'
    - EMPRESA → FARM con alerta (tipo_origen != FARM): typemove = 'out'
    
    Args:
        movements_df: DataFrame con movimientos (debe incluir campos de empresa)
        farms_with_alert: Set de IDs de farms con alerta directa
        period: Período (ej: "201701")
        year: Año (ej: "2017")
        quarter: Trimestre (1-4) o None
        
    Returns:
        DataFrame con alertas de empresa: idpro, id_farm, typemove, period, year, etc.
    """
    enterprise_types = {"SLAUGHTERHOUSE", "CATTLE_FAIR", "PROCESSOR", "ENTERPRISE"}
    
    # Verificar que tenemos los campos necesarios
    has_tipo_destino = "tipo_destino" in movements_df.columns
    has_tipo_origen = "tipo_origen" in movements_df.columns
    has_producer_destino = "producer_id_destino" in movements_df.columns
    has_producer_origen = "producer_id_origen" in movements_df.columns
    
    if not (has_tipo_destino or has_tipo_origen):
        logging.warning("No hay campos de tipo origen/destino para extraer alertas de empresa")
        return pd.DataFrame()
    
    results = []
    
    # 1. Movimientos FARM con alerta → EMPRESA (entrada a empresa)
    if has_tipo_destino and has_producer_destino:
        mask_to_enterprise = (
            movements_df["origen_id"].isin(farms_with_alert) &
            movements_df["tipo_destino"].isin(enterprise_types)
        )
        
        entries = movements_df[mask_to_enterprise].copy()
        if not entries.empty:
            entries["typemove"] = "in"
            entries["id_farm"] = entries["origen_id"]
            entries["idpro"] = entries["producer_id_destino"]
            entries["type_enterprise"] = entries["tipo_destino"]
            entries["farm_has_direct_alert"] = True
            results.append(entries[["idpro", "id_farm", "typemove", "type_enterprise", "farm_has_direct_alert"]])
    
    # 2. Movimientos EMPRESA → FARM con alerta (salida de empresa)
    if has_tipo_origen and has_producer_origen:
        mask_from_enterprise = (
            movements_df["destination_id"].isin(farms_with_alert) &
            movements_df["tipo_origen"].isin(enterprise_types)
        )
        
        exits = movements_df[mask_from_enterprise].copy()
        if not exits.empty:
            exits["typemove"] = "out"
            exits["id_farm"] = exits["destination_id"]
            exits["idpro"] = exits["producer_id_origen"]
            exits["type_enterprise"] = exits["tipo_origen"]
            exits["farm_has_direct_alert"] = True
            results.append(exits[["idpro", "id_farm", "typemove", "enterprise_type", "farm_has_direct_alert"]])
    
    if not results:
        return pd.DataFrame()
    
    # Combinar resultados
    enterprise_df = pd.concat(results, ignore_index=True)
    
    # Agregar metadata
    enterprise_df["period"] = period
    enterprise_df["year"] = year
    if quarter:
        enterprise_df["quarter"] = quarter
    
    # Eliminar duplicados
    enterprise_df = enterprise_df.drop_duplicates(subset=["idpro", "id_farm", "typemove"])
    
    return enterprise_df


# ===================== FUNCIÓN PRINCIPAL =====================

def calculate_indirect_alerts(
    period: str,
    period_type: str,
    source: str,
    direct_alerts_csv: str,
    movement_csv_dir: str,
    output_csv: str,
    enterprise_output_csv: Optional[str] = None,
    generate_enterprise_alerts: bool = True,
    normalize_ids: bool = True,
    strip_dot_zero: bool = True,
    strip_leading_zeros: bool = True,
    uppercase_ids: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Calcula alertas indirectas por movimiento de ganado para un período.
    También genera alertas de empresa en la misma pasada si se solicita.
    
    Args:
        period: Período a procesar (ej: "201701" para nad/atd, "2017" para annual, "2010-2024" para cumulative)
        period_type: Tipo de período ("nad", "atd", "annual", "cumulative")
        source: Fuente de deforestación ("smbyc")
        direct_alerts_csv: Ruta al CSV de alertas directas del período
        movement_csv_dir: Directorio con CSVs de movimientos por año
        output_csv: Ruta del CSV de salida para alertas indirectas
        enterprise_output_csv: Ruta del CSV de salida para alertas de empresa (opcional)
        generate_enterprise_alerts: Si True, genera también alertas de empresa
        normalize_ids: Si True, normaliza IDs
        strip_dot_zero: Quitar ".0" de IDs numéricos
        strip_leading_zeros: Quitar ceros a la izquierda
        uppercase_ids: Convertir IDs a mayúsculas
    
    Returns:
        Dict con estadísticas del procesamiento (incluye enterprise si se genera)
    """
    try:
        # Configuración de normalización
        normalize_config = {
            "strip_dot_zero": strip_dot_zero and normalize_ids,
            "strip_leading_zeros": strip_leading_zeros and normalize_ids,
            "uppercase": uppercase_ids and normalize_ids
        }
        
        # 1. Cargar alertas directas
        logging.info(f"Cargando alertas directas: {direct_alerts_csv}")
        alerts_df = load_direct_alerts(direct_alerts_csv, normalize_config)
        
        if alerts_df.empty:
            logging.warning(f"No se encontraron alertas directas para {period}")
            return {
                "success": False,
                "period": period,
                "error": "No direct alerts found"
            }
        
        farms_with_alert_count = int(alerts_df["direct_alert"].sum())
        farms_with_alert_ids = set(alerts_df[alerts_df["direct_alert"]]["id"].unique())
        logging.info(f"Farms con alerta directa: {farms_with_alert_count} / {len(alerts_df)}")
        
        # 2. Determinar año y trimestre según tipo de período
        if period_type == "cumulative" and "-" in period:
            # Formato: 2010-2024 -> usar último año para movimientos
            year = period.split("-")[1]
            quarter = None
        elif period_type in ("nad", "atd") and len(period) == 6:
            # Formato: YYYYQQ -> extraer año y trimestre
            year = period[:4]
            quarter = parse_quarter_from_period(period)
        else:
            # annual u otro formato simple
            year = period[:4]
            quarter = None
        
        # 3. Cargar movimientos del año (con campos de empresa si se requiere)
        movement_csv = Path(movement_csv_dir) / f"movement_data_base_{year}.csv"
        
        if not movement_csv.exists():
            logging.warning(f"No existe CSV de movimientos para {year}: {movement_csv}")
            return {
                "success": False,
                "period": period,
                "error": f"Movement data not found for year {year}"
            }
        
        logging.info(f"Cargando movimientos: {movement_csv} (trimestre {quarter if quarter else 'completo'})")
        movements_df = load_movements(
            str(movement_csv), 
            normalize_config, 
            quarter,
            include_enterprise_fields=generate_enterprise_alerts
        )
        
        logging.info(f"Movimientos cargados: {len(movements_df):,}")
        
        # 4. Calcular métricas de alertas indirectas
        result_df = calculate_indirect_metrics(alerts_df, movements_df, period)
        
        # 5. Agregar metadata
        result_df.insert(0, "period", period)
        result_df.insert(1, "year", year)
        if quarter:
            result_df.insert(2, "quarter", quarter)
        
        # 6. Guardar resultado de alertas indirectas
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        # Estadísticas de indirectas
        indirect_in_count = int(result_df["indirect_alert_in"].sum())
        indirect_out_count = int(result_df["indirect_alert_out"].sum())
        
        logging.info(f"Alertas indirectas calculadas para {period}")
        logging.info(f"  • indirect_alert_in: {indirect_in_count} / {len(result_df)}")
        logging.info(f"  • indirect_alert_out: {indirect_out_count} / {len(result_df)}")
        logging.info(f"  • Guardado en: {output_path}")
        
        # 7. Generar alertas de empresa si se solicita
        enterprise_stats = {
            "enterprise_generated": False,
            "enterprise_entries": 0,
            "enterprise_exits": 0,
            "enterprise_total": 0
        }
        
        if generate_enterprise_alerts and farms_with_alert_ids:
            logging.info("Generando alertas de empresa desde los mismos movimientos...")
            
            enterprise_df = extract_enterprise_alerts(
                movements_df=movements_df,
                farms_with_alert=farms_with_alert_ids,
                period=period,
                year=year,
                quarter=quarter
            )
            
            if not enterprise_df.empty:
                # Determinar ruta de salida
                if enterprise_output_csv:
                    ent_output_path = Path(enterprise_output_csv)
                else:
                    # Generar ruta automáticamente
                    ent_output_path = output_path.parent.parent / "enterprise_alerts" / f"{source}_enterprise_alert_{period_type}_{period}.csv"
                
                ent_output_path.parent.mkdir(parents=True, exist_ok=True)
                enterprise_df.to_csv(ent_output_path, index=False, encoding="utf-8-sig")
                
                # Estadísticas
                n_entries = len(enterprise_df[enterprise_df["typemove"] == "in"])
                n_exits = len(enterprise_df[enterprise_df["typemove"] == "out"])
                
                enterprise_stats = {
                    "enterprise_generated": True,
                    "enterprise_entries": n_entries,
                    "enterprise_exits": n_exits,
                    "enterprise_total": len(enterprise_df),
                    "enterprise_output_file": str(ent_output_path)
                }
                
                logging.info(f"  • Alertas empresa entrada (in): {n_entries}")
                logging.info(f"  • Alertas empresa salida (out): {n_exits}")
                logging.info(f"  • Guardado en: {ent_output_path}")
            else:
                logging.info("  • No se encontraron movimientos FARM↔EMPRESA con fincas alertadas")
        
        return {
            "success": True,
            "period": period,
            "year": year,
            "quarter": quarter,
            "farms_processed": len(result_df),
            "farms_with_direct_alert": farms_with_alert_count,
            "indirect_alert_in_count": indirect_in_count,
            "indirect_alert_out_count": indirect_out_count,
            "movements_total": len(movements_df),
            "output_file": str(output_path),
            **enterprise_stats
        }
    
    except Exception as e:
        logging.error(f"Error procesando período {period}: {e}", exc_info=True)
        return {
            "success": False,
            "period": period,
            "error": str(e)
        }


def calculate_indirect_alerts_batch(
    periods: List[str],
    period_type: str,
    source: str,
    direct_alerts_dir: str,
    movement_csv_dir: str,
    output_dir: str,
    enterprise_output_dir: Optional[str] = None,
    generate_enterprise_alerts: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Procesa múltiples períodos en batch.
    También genera alertas de empresa en la misma pasada.
    
    Args:
        periods: Lista de períodos a procesar
        period_type: Tipo de período
        source: Fuente de deforestación
        direct_alerts_dir: Directorio base con CSVs de alertas directas
        movement_csv_dir: Directorio con CSVs de movimientos
        output_dir: Directorio base de salida para alertas indirectas
        enterprise_output_dir: Directorio base para alertas de empresa (opcional)
        generate_enterprise_alerts: Si True, genera también alertas de empresa
        **kwargs: Argumentos adicionales para calculate_indirect_alerts
    
    Returns:
        Dict con estadísticas agregadas (incluye enterprise si se genera)
    """
    results = []
    
    # Barra de progreso principal por períodos
    periods_progress = tqdm(
        periods,
        desc="📅 Períodos",
        unit="período",
        position=0,
        leave=True
    )
    
    for period in periods_progress:
        periods_progress.set_description(f"📅 Procesando {period}")
        
        # Construir rutas (archivos directo en el stage dir, sin subcarpeta de período)
        # direct_alerts_dir ya es: results/{source}/{period_type}/direct_alerts
        # output_dir ya es: results/{source}/{period_type}/indirect_alerts
        direct_csv = Path(direct_alerts_dir) / f"{source}_direct_alert_{period_type}_{period}.csv"
        output_csv = Path(output_dir) / f"{source}_indirect_alert_{period_type}_{period}.csv"
        
        # Ruta para enterprise alerts
        enterprise_csv = None
        if generate_enterprise_alerts:
            ent_base = enterprise_output_dir or str(Path(output_dir).parent / "enterprise_alerts")
            enterprise_csv = str(Path(ent_base) / f"{source}_enterprise_alert_{period_type}_{period}.csv")
        
        result = calculate_indirect_alerts(
            period=period,
            period_type=period_type,
            source=source,
            direct_alerts_csv=str(direct_csv),
            movement_csv_dir=movement_csv_dir,
            output_csv=str(output_csv),
            enterprise_output_csv=enterprise_csv,
            generate_enterprise_alerts=generate_enterprise_alerts,
            **kwargs
        )
        
        results.append(result)
    
    # Agregar estadísticas
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    total_indirect_in = sum(r.get("indirect_alert_in_count", 0) for r in successful)
    total_indirect_out = sum(r.get("indirect_alert_out_count", 0) for r in successful)
    
    # Estadísticas de empresa
    total_enterprise_entries = sum(r.get("enterprise_entries", 0) for r in successful)
    total_enterprise_exits = sum(r.get("enterprise_exits", 0) for r in successful)
    total_enterprise_records = sum(r.get("enterprise_total", 0) for r in successful)
    enterprise_generated = any(r.get("enterprise_generated", False) for r in successful)
    
    return {
        "success": len(failed) == 0,
        "periods_processed": len(successful),
        "periods_failed": len(failed),
        "total_indirect_in": total_indirect_in,
        "total_indirect_out": total_indirect_out,
        "enterprise_generated": enterprise_generated,
        "total_enterprise_entries": total_enterprise_entries,
        "total_enterprise_exits": total_enterprise_exits,
        "total_enterprise_records": total_enterprise_records,
        "results": results
    }
