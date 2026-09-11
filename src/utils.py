#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades compartidas entre módulos de la ETL de Ganabosques.
Este archivo centraliza funciones duplicadas para evitar replicación de código.
"""

import os
import re
import logging
from typing import List, Optional, Any
from pathlib import Path

from shapely.geometry.base import BaseGeometry
import geopandas as gpd


# ===================== GEOMETRÍA =====================

def area_ha(geom: BaseGeometry) -> float:
    """
    Calcula área de geometría en hectáreas.
    
    Args:
        geom: Geometría de Shapely
        
    Returns:
        Área en hectáreas
    """
    return float(geom.area / 10_000.0)


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


# ===================== PERÍODOS =====================

def parse_year_periods(years_raw: str) -> List[str]:
    """
    Parsea años a períodos.
    Acepta: "2017" o "2017-2024" o "2017, 2018, 2019-2024"
    
    Args:
        years_raw: String con años/rangos separados por coma
        
    Returns:
        Lista de períodos válidos (YYYY-YYYY)
    """
    if not years_raw:
        return []
    parts = [p.strip() for p in years_raw.split(",") if p.strip()]
    valids = []
    for p in parts:
        if re.match(r"^\d{4}\s*-\s*\d{4}$", p):
            valids.append(p.replace(" ", ""))
        else:
            logging.warning(f"YEARS ignorado por formato no válido: '{p}' (usa AAAA-AAAA)")
    return valids


def parse_quarter_from_period(period: str, period_type: str = None) -> Optional[int]:
    """
    Extrae el trimestre de un período (1-4) o None si no aplica.
    
    Intenta extraer del período si tiene formato YYYYQQ (6 dígitos).
    Validación: solo retorna si es 1-4.
    
    Args:
        period: String con período (ej: '202401', '2024-2025', '2024')
        period_type: Tipo de período (nad, atd, annual, etc.) - opcional
        
    Returns:
        Trimestre (1-4) si se puede extraer, None en cualquier otro caso
    """
    # Si es formato YYYYQQ (6 dígitos), extraer último dígito como trimestre
    if len(period) == 6 and period.isdigit():
        quarter = int(period[-2:])
        if 1 <= quarter <= 4:
            return quarter
    return None


# ===================== LOGGING =====================

def setup_logging(log_level: str = None, log_file: str = None):
    """
    Configura logging centralizado.
    
    Args:
        log_level: Nivel de logging ('WARNING', 'INFO', 'DEBUG'). 
                   Si None, usa 'WARNING' como default.
        log_file: Ruta del archivo log. Si None, usa 'app.log' como default.
    """
    if log_level is None:
        log_level = "WARNING"
    if log_file is None:
        log_file = "app.log"
    
    lvl = getattr(logging, log_level.upper(), logging.WARNING)
    
    # Remover handlers existentes para evitar duplicación
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    
    # Formato estándar
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


# ===================== Normalización de IDs =====================
_FARM_ID_PREFIX = re.compile(r'^FARM_ID_', re.IGNORECASE)

def normalize_farm_id(value, strip_leading_zeros: bool = True, strip_dot_zero: bool = True) -> str:
    """
    Normaliza IDs de fincas/empresas a formato estándar.
    
    Operaciones:
    1. Si None o pd.isna() → ""
    2. Si "nan" (lowercase) → ""
    3. Strip spaces
    4. Quitar prefijo FARM_ID_ (de geojsons GeoFarmer)
    5. Si strip_dot_zero: Quitar ".0" de números flotantes (ej: "123.0" → "123")
    6. Si strip_leading_zeros: Quitar ceros a izquierda de dígitos puros (ej: "00123" → "123")
    
    Args:
        value: Valor a normalizar (int, str, float o None)
        strip_leading_zeros: Si True, "000123" → "123"
        strip_dot_zero: Si True, "123.0" → "123"
    
    Returns:
        String normalizado
    
    Examples:
        >>> normalize_farm_id("FARM_ID_abc123")
        'abc123'
        >>> normalize_farm_id("123.0")
        '123'
        >>> normalize_farm_id("00456")
        '456'
    """
    import pandas as pd
    
    # Manejo de None/NaN
    if value is None or pd.isna(value):
        return ""
    
    s = str(value).strip()
    
    if s.lower() == "nan":
        return ""
    
    # Quitar prefijo FARM_ID_
    s = _FARM_ID_PREFIX.sub('', s)
    
    # Quitar ".0" de números flotantes
    if strip_dot_zero and s.endswith(".0"):
        try:
            s = str(int(float(s)))
        except (ValueError, TypeError):
            pass
    
    # Quitar ceros a la izquierda (si es dígito puro)
    if strip_leading_zeros and s.isdigit():
        try:
            s = str(int(s))
        except (ValueError, TypeError):
            pass
    
    return s
