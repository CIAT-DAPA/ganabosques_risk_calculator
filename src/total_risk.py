#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

from config import config, RiskLevel, classify_risk  # <-- importamos desde config

# ---------------- utilidades base ----------------
def setup_logging():
    lvl = getattr(logging, config['LOG_LEVEL'].upper(), logging.WARNING)
    logging.basicConfig(filename=config['LOG_FILE'], level=lvl,
                        format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

def parse_year_periods(years_raw: str) -> List[str]:
    """'2023-2024 , 2010-2012,2018-2019' -> ['2023-2024','2010-2012','2018-2019']"""
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

def format_placeholders(template: str, ctx: Dict[str, Any]) -> str:
    """Permite {EMPRESA}/{empresa}, {PERIODO}/{periodo}, {YEARS}/{years}."""
    if template is None:
        return None
    mixed = {
        "EMPRESA": ctx.get("EMPRESA"),
        "PERIODO": ctx.get("PERIODO"),
        "YEARS": ctx.get("YEARS"),
        "empresa": ctx.get("EMPRESA"),
        "periodo": ctx.get("PERIODO"),
        "years": ctx.get("YEARS"),
    }
    try:
        return template.format(**mixed)
    except KeyError as e:
        logging.warning(f"Placeholder faltante {e} en: {template}")
        return template

def safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None

def limpiar_id(x):
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x

# ---------------- cálculo por periodo ----------------
def calculate_risk_total(df_direct: pd.DataFrame, df_movement: pd.DataFrame) -> pd.DataFrame:
    # Validar columnas mínimas
    if not {'id', 'risk_direct_level_value'}.issubset(df_direct.columns):
        raise ValueError("df_direct debe incluir 'id' y 'risk_direct_level_value'")
    if not {'plot_id', 'risk_in_score', 'risk_out_score'}.issubset(df_movement.columns):
        raise ValueError("df_movement debe incluir 'plot_id', 'risk_in_score' y 'risk_out_score'")

    df_movement_indexed = df_movement.set_index('plot_id')

    results = []
    for _, row in tqdm(df_direct.iterrows(), total=len(df_direct), desc=f"Calculando riesgo total para {len(df_direct)} fincas"):
        plot_id = str(row['id'])
        risk_direct_value = safe_float(row['risk_direct_level_value'])
        if risk_direct_value is None:
            # fila inválida -> omitir
            continue

        # Riesgo de movimiento (in/out)
        risk_in_score = 0.0
        risk_out_score = 0.0
        if plot_id in df_movement_indexed.index:
            mov_row = df_movement_indexed.loc[plot_id]
            risk_in_score = safe_float(mov_row['risk_in_score']) or 0.0
            risk_out_score = safe_float(mov_row['risk_out_score']) or 0.0

        # Ponderación
        total_score = (0.5 * risk_direct_value) + (0.4 * risk_in_score) + (0.1 * risk_out_score)
        total_enum = classify_risk(total_score)  # <-- desde config

        results.append({
            "plot_id": plot_id,
            "risk_direct_value": risk_direct_value,
            "risk_in_score": risk_in_score,
            "risk_out_score": risk_out_score,
            "risk_total": total_enum,
            "risk_total_value": total_enum.value,
            "risk_total_label": total_enum.name,
            "risk_total_score": total_score,
        })

    df_risk_total = pd.DataFrame(results)

    # Enriquecer con columnas extra del directo (si existen)
    extra_cols = [
        'id', 'deforested_hectares', 'deforested_proportion',
        'protected_ha', 'protected_prop',
        'deforestation_distance_to', 'protected_area_distance_to'
    ]
    extra_cols = [c for c in extra_cols if c in df_direct.columns]
    if extra_cols:
        df_direct_extra = df_direct[extra_cols].copy()
        df_direct_extra['id'] = df_direct_extra['id'].astype(str).apply(limpiar_id)
        df_risk_total['plot_id'] = df_risk_total['plot_id'].astype(str).apply(limpiar_id)
        df_final = df_risk_total.merge(df_direct_extra, left_on="plot_id", right_on="id", how="left")
        df_final = df_final.drop(columns=["id"], errors="ignore")
    else:
        df_final = df_risk_total

    return df_final

def run_for_period(empresa: str, periodo: str, years: str) -> str | None:
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years}

    direct_risk_path   = format_placeholders(config['DIRECT_RISK_CSV'], ctx)
    movement_risk_path = format_placeholders(config['MOVEMENT_RISK_OUTPUT_CSV'], ctx)
    output_total_path  = format_placeholders(config['TOTAL_RISK_OUTPUT_CSV'], ctx)

    print(f"\n🟪 Periodo {years}")
    print(f"  DIRECT_RISK_CSV        => {direct_risk_path}")
    print(f"  MOVEMENT_RISK_OUTPUT   => {movement_risk_path}")
    print(f"  TOTAL_RISK_OUTPUT_CSV  => {output_total_path}")

    # Validaciones
    if not os.path.exists(direct_risk_path):
        logging.warning(f"[{years}] No existe DIRECT_RISK_CSV: {direct_risk_path}. Se omite periodo.")
        return None
    if not os.path.exists(movement_risk_path):
        logging.warning(f"[{years}] No existe MOVEMENT_RISK_OUTPUT_CSV: {movement_risk_path}. Se omite periodo.")
        return None

    # Carga
    df_direct = pd.read_csv(direct_risk_path, dtype=str, low_memory=False)
    df_movement = pd.read_csv(movement_risk_path, dtype=str, low_memory=False)

    # Normalizar ID
    if 'id' in df_direct.columns:
        df_direct['id'] = df_direct['id'].astype(str).apply(limpiar_id)
    if 'plot_id' in df_movement.columns:
        df_movement['plot_id'] = df_movement['plot_id'].astype(str).apply(limpiar_id)

    # Cálculo
    out_df = calculate_risk_total(df_direct, df_movement)

    # Guardado
    os.makedirs(os.path.dirname(output_total_path), exist_ok=True)
    out_df.to_csv(output_total_path, index=False)
    print(f"✅ total_risk exportado: {output_total_path}")
    return output_total_path

# ---------------- main ----------------
def main():
    setup_logging()

    empresa = config['EMPRESA']
    periodo = config['PERIODO']
    years_list = parse_year_periods(config['YEARS']) or [str(config['YEARS']).strip()]

    produced = []
    for years in years_list:
        out = run_for_period(empresa, periodo, years)
        if out:
            produced.append(out)

    # Combinado opcional
    if config['MERGE_TOTAL_OUTPUT'] and config['TOTAL_RISK_OUTPUT_ALL']:
        dfs = []
        for years in years_list:
            ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years}
            path = format_placeholders(config['TOTAL_RISK_OUTPUT_CSV'], ctx)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.insert(0, "years_period", years)
                dfs.append(df)
        if dfs:
            os.makedirs(os.path.dirname(config['TOTAL_RISK_OUTPUT_ALL']), exist_ok=True)
            pd.concat(dfs, ignore_index=True).to_csv(config['TOTAL_RISK_OUTPUT_ALL'], index=False)
            print(f"📦 CSV maestro combinado → {config['TOTAL_RISK_OUTPUT_ALL']}")
        else:
            logging.warning("No hay archivos de periodos para combinar.")

    print("🎉 total_risk finalizado.")

if __name__ == "__main__":
    main()
