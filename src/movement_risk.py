#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, List
from config import config, RiskLevel, classify_risk  # <-- importamos desde config

# ------------------ utilidades base ------------------
def setup_logging():
    lvl = getattr(logging, config['LOG_LEVEL'].upper(), logging.WARNING)
    logging.basicConfig(filename=config['LOG_FILE'], level=lvl, format="%(asctime)s %(levelname)s:%(message)s")
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
    """
    Rellena placeholders en mayúsculas y minúsculas:
    {EMPRESA}/{empresa}, {PERIODO}/{periodo}, {YEARS}/{years}, {MOV_YEAR}/{mov_year}
    """
    if template is None:
        return None
    mixed = {
        "EMPRESA": ctx.get("EMPRESA"),
        "PERIODO": ctx.get("PERIODO"),
        "YEARS": ctx.get("YEARS"),
        "MOV_YEAR": ctx.get("MOV_YEAR"),
        "empresa": ctx.get("EMPRESA"),
        "periodo": ctx.get("PERIODO"),
        "years": ctx.get("YEARS"),
        "mov_year": ctx.get("MOV_YEAR"),
    }
    try:
        return template.format(**mixed)
    except KeyError as e:
        logging.warning(f"Placeholder faltante {e} en: {template}")
        return template

def derive_mov_year(years_range: str, periodo: str) -> str:
    """
    Devuelve el año a usar para MOV_YEAR según PERIODO:
      - 'anual'        -> primer año del rango 'YYYY-YYYY'
      - 'cumulative'   -> segundo año del rango
      - también acepto alias: 'cum', 'acumulado'
    Si no hay guion, retorna tal cual.
    """
    if "-" not in years_range:
        return years_range

    first_year, last_year = years_range.split("-", 1)
    p = (periodo or "").strip().lower()

    if p == "anual":
        return first_year
    if p in {"cumulative", "cum", "acumulado"}:
        return last_year

    # default prudente: primer año
    return first_year

# ------------------ helpers de datos ------------------
def limpiar_id(x):
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x

def compute_average(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

def to_enum(x):
    if isinstance(x, RiskLevel):
        return x
    if isinstance(x, str):
        try:
            return RiskLevel[x]
        except Exception:
            pass
    try:
        return RiskLevel(int(float(x)))
    except Exception:
        return RiskLevel.NO_RISK

# ------------------ core por periodo ------------------
def run_for_period(empresa: str, periodo: str, years: str, max_workers: int):
    # MOV_YEAR condicional según PERIODO
    mov_year = derive_mov_year(years, periodo)
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}

    # Input de riesgo directo: usa DIRECT_RISK_CSV si existe en config; si no, OUTPUT_CSV
    direct_risk_tpl = config.get('DIRECT_RISK_CSV', config['OUTPUT_CSV'])
    direct_risk_path = format_placeholders(direct_risk_tpl, ctx)

    movement_path    = format_placeholders(config['MOVEMENT_INPUT_CSV'], ctx)
    output_path      = format_placeholders(config['MOVEMENT_RISK_OUTPUT_CSV'], ctx)

    # Debug de rutas
    print(f"\n🟪 Periodo {years} (PERIODO={periodo}, MOV_YEAR={mov_year})")
    print(f"  DIRECT_RISK => {direct_risk_path}")
    print(f"  MOV_INPUT   => {movement_path}")
    print(f"  OUTPUT      => {output_path}")

    # Validaciones/IO
    if not os.path.exists(direct_risk_path):
        logging.warning(f"[{years}] No existe DIRECT_RISK_CSV/OUTPUT_CSV: {direct_risk_path}. Se omite periodo.")
        return None
    if not os.path.exists(movement_path):
        logging.warning(f"[{years}] No existe MOVEMENT_INPUT_CSV: {movement_path}. Se omite periodo.")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Carga
    df_plots_risk = pd.read_csv(direct_risk_path, dtype={"id": str}, low_memory=False)
    df_movement   = pd.read_csv(movement_path, dtype=str, low_memory=False)

    # --- Limpieza IDs
    df_plots_risk['id'] = df_plots_risk['id'].apply(limpiar_id)

    # Asegurar columnas esperadas y renombrar si hace falta
    if 'SIT_CODE_ORIGEN' in df_movement.columns:
        df_movement['SIT_CODE_ORIGEN'] = df_movement['SIT_CODE_ORIGEN'].apply(limpiar_id)
        df_movement['SIT_CODE_DESTINO'] = df_movement['SIT_CODE_DESTINO'].apply(limpiar_id)
        df_movement = df_movement.rename(columns={
            'SIT_CODE_ORIGEN': 'origen_id',
            'SIT_CODE_DESTINO': 'destination_id'
        })
    elif {'origen_id','destination_id'}.issubset(df_movement.columns):
        df_movement['origen_id'] = df_movement['origen_id'].apply(limpiar_id)
        df_movement['destination_id'] = df_movement['destination_id'].apply(limpiar_id)
    else:
        raise ValueError("El CSV de movimiento debe tener columnas 'SIT_CODE_ORIGEN'/'SIT_CODE_DESTINO' o 'origen_id'/'destination_id'.")

    # --- Diagnóstico de matches (opcional)
    origen_matches = set(df_movement['origen_id']) & set(df_plots_risk['id'])
    destino_matches = set(df_movement['destination_id']) & set(df_plots_risk['id'])
    print('Coincidencias origen:', len(origen_matches))
    print('Coincidencias destino:', len(destino_matches))
    if len(origen_matches) == 0 and len(destino_matches) == 0:
        logging.warning(f"[{years}] ADVERTENCIA: No hay coincidencias de IDs entre movimiento y riesgo directo.")

    # --- Riesgo directo a Enum
    if 'risk_direct_level' in df_plots_risk.columns:
        df_plots_risk['risk_direct_level'] = df_plots_risk['risk_direct_level'].apply(to_enum)
    elif 'risk_direct_level_value' in df_plots_risk.columns:
        df_plots_risk['risk_direct_level'] = df_plots_risk['risk_direct_level_value'].apply(lambda x: to_enum(int(float(x))))
    else:
        raise ValueError("No se encontró columna de nivel de riesgo en el CSV de riesgo directo.")

    plot_risk_dict = dict(zip(df_plots_risk['id'], df_plots_risk['risk_direct_level']))

    # --- Preproceso de movimientos
    movements_by_destination = defaultdict(list)
    movements_by_origin = defaultdict(list)
    for o, d in zip(df_movement['origen_id'], df_movement['destination_id']):
        movements_by_origin[o].append(d)
        movements_by_destination[d].append(o)

    # --- Función de parcela
    def process_plot(plot_id: str) -> Dict[str, Any]:
        in_raw = movements_by_destination.get(plot_id, [])
        out_raw = movements_by_origin.get(plot_id, [])
        in_scores = [plot_risk_dict.get(ent, RiskLevel.NO_RISK).value for ent in in_raw]
        out_scores = [plot_risk_dict.get(sal, RiskLevel.NO_RISK).value for sal in out_raw]
        avg_in = compute_average([float(v) for v in in_scores])
        avg_out = compute_average([float(v) for v in out_scores])
        risk_in_enum = classify_risk(avg_in)
        risk_out_enum = classify_risk(avg_out)
        return {
            "plot_id": plot_id,
            "risk_in_score": avg_in,
            "risk_in_enum": risk_in_enum,
            "risk_in_enum_value": risk_in_enum.value,
            "risk_in_enum_label": risk_in_enum.name,
            "risk_out_score": avg_out,
            "risk_out_enum": risk_out_enum,
            "risk_out_enum_value": risk_out_enum.value,
            "risk_out_enum_label": risk_out_enum.name,
        }

    # --- Paralelismo con barra
    ids = list(df_plots_risk['id'])
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_plot, pid) for pid in ids]
        for fut in tqdm(as_completed(futures), total=len(ids), desc=f"Procesando parcelas ({years})"):
            results.append(fut.result())

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"✅ movement_risk exportado: {output_path}")
    return output_path

# ------------------ main ------------------
def main():
    setup_logging()

    empresa = config['EMPRESA']
    periodo = config['PERIODO']
    years_list = parse_year_periods(config['YEARS']) or [str(config['YEARS']).strip()]
    max_workers = config.get('TASK_MAX_WORKERS', 8)

    produced = []
    for years in years_list:
        out = run_for_period(empresa, periodo, years, max_workers=max_workers)
        if out:
            produced.append(out)

    # Combinar si procede
    if config.get('MERGE_OUTPUT', False) and config.get('OUTPUT_CSV_ALL'):
        dfs = []
        for years in years_list:
            mov_year = derive_mov_year(years, periodo)
            ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
            path = format_placeholders(config['MOVEMENT_RISK_OUTPUT_CSV'], ctx)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.insert(0, "years_period", years)
                dfs.append(df)
        if dfs:
            os.makedirs(os.path.dirname(config['OUTPUT_CSV_ALL']), exist_ok=True)
            pd.concat(dfs, ignore_index=True).to_csv(config['OUTPUT_CSV_ALL'], index=False)
            print(f"📦 CSV maestro combinado → {config['OUTPUT_CSV_ALL']}")
        else:
            logging.warning("No hay archivos de periodos para combinar.")

    print("🎉 movement_risk finalizado.")

if __name__ == "__main__":
    main()
