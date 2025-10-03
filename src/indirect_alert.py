#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from config import config

# ===================== utilidades =====================
def setup_logging():
    lvl = getattr(logging, str(config.get('LOG_LEVEL', 'WARNING')).upper(), logging.WARNING)
    logging.basicConfig(filename=config.get('LOG_FILE', 'movement.log'),
                        level=lvl,
                        format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

def parse_year_periods(years_raw: str) -> List[str]:
    if not years_raw:
        return []
    parts = [p.strip() for p in str(years_raw).split(",") if p.strip()]
    valids = []
    for p in parts:
        if re.match(r"^\d{4}\s*-\s*\d{4}$", p):
            valids.append(p.replace(" ", ""))
        else:
            logging.warning(f"YEARS ignorado por formato no válido: '{p}' (usa AAAA-AAAA)")
    return valids

def derive_mov_year(years_range: str, periodo: str) -> str:
    """
    MOV_YEAR según PERIODO:
      - 'anual'                      -> primer año del rango 'YYYY-YYYY'
      - 'cumulative'/'cum'/'acumulado' -> segundo año
    Si no hay guion, retorna tal cual.
    """
    if not years_range or "-" not in years_range:
        return str(years_range or "")
    first_year, last_year = years_range.split("-", 1)
    p = (periodo or "").strip().lower()
    if p == "anual":
        return first_year
    if p in {"cumulative", "cum", "acumulado"}:
        return last_year
    return first_year  # default prudente

def format_placeholders(template: Optional[str], ctx: Dict[str, Any]) -> Optional[str]:
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

def normalize_id(x: Any) -> str:
    # Todo como cadena; no quitamos ".0" ni casteamos
    return "" if x is None else str(x).strip()

def str_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "si", "sí"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return bool(s and s != "nan")

def sanitize_empresa_folder(empresa: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(empresa)).lower()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_reason_log(folder: str, fname: str, msg: str):
    ensure_dir(folder)
    path = os.path.join(folder, fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(f"📝 Log: {path}")

# ===================== rutas y descubrimiento =====================
SOURCE_MAP = {
    "smbyc": "SMBYC",
    "atd":   "ATD",
    "nad":   "NAD",
}

def base_alerts_dir_for_source(output_csv_template: str, empresa: str, periodo: str, years: str, mov_year: str, source_tag: str) -> str:
    """
    Deriva el directorio base donde quedaron las alertas directas del script anterior.
    Se apoya en la MISMA plantilla de OUTPUT_CSV, aplicando placeholders para quedar en:
      <dirname(OUTPUT_CSV con placeholders)>/<SOURCE>/<empresa-normalizada>
    """
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
    formatted = format_placeholders(output_csv_template, ctx)
    base_dir = os.path.dirname(formatted) or "."
    empresa_folder = sanitize_empresa_folder(empresa)
    source_folder = SOURCE_MAP.get(source_tag.lower(), source_tag.upper())
    return os.path.join(base_dir, source_folder, empresa_folder)

def discover_alerts_exact_year(output_csv_template: str, empresa: str, periodo: str, years: str, mov_year: str, source_tag: str) -> pd.DataFrame:
    """
    Busca el CSV EXACTO para {years}:
      <base_alerts_dir>/<source>/<empresa>/{source}_direct_alert_{empresa}_{years}.csv
    """
    alerts_dir = base_alerts_dir_for_source(output_csv_template, empresa, periodo, years, mov_year, source_tag)
    prefix = source_tag.lower()
    exact_name = f"{prefix}_direct_alert_{empresa}_{years}.csv"
    exact_path = os.path.join(alerts_dir, exact_name)

    if not os.path.isdir(alerts_dir):
        logging.warning(f"No existe carpeta {SOURCE_MAP.get(source_tag, source_tag)} para la empresa: {alerts_dir}")
        return pd.DataFrame(columns=["id","direct_alert"])

    if not os.path.isfile(exact_path):
        logging.warning(f"No se encontró CSV {SOURCE_MAP.get(source_tag, source_tag)} exacto para years='{years}': {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    try:
        df = pd.read_csv(exact_path, dtype=str, low_memory=False)
    except Exception as e:
        logging.warning(f"No se pudo leer {exact_path}: {e}")
        return pd.DataFrame(columns=["id","direct_alert"])

    if "id" not in df.columns:
        logging.warning(f"CSV sin columna 'id' (se omite): {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    col_alert = "direct_alert" if "direct_alert" in df.columns else None
    # compatibilidad: si no trae direct_alert, toma bandera típica por fuente
    fallback_cols = ["intersect_deforestation", "intersect_early_warnings", "intersect_active_hotspots"]
    if col_alert is None:
        for c in fallback_cols:
            if c in df.columns:
                col_alert = c
                break
    if col_alert is None:
        logging.warning(f"CSV sin columna de alerta usable (se omite): {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    out = df[["id", col_alert]].copy()
    out["id"] = out["id"].map(normalize_id)
    out["direct_alert"] = out[col_alert].map(str_bool)
    return out[["id","direct_alert"]]

def discover_alerts_any_year(output_csv_template: str, empresa: str, periodo: str, years: str, mov_year: str, source_tag: str) -> Tuple[pd.DataFrame, bool]:
    """
    Fallback: combina TODOS los CSV disponibles en la carpeta de la fuente para esa empresa (cualquier years)
    con OR lógico por id. Retorna (df, had_any).
    """
    alerts_dir = base_alerts_dir_for_source(output_csv_template, empresa, periodo, years, mov_year, source_tag)
    if not os.path.isdir(alerts_dir):
        logging.warning(f"No existe carpeta {SOURCE_MAP.get(source_tag, source_tag)} para la empresa (fallback): {alerts_dir}")
        return pd.DataFrame(columns=["id","direct_alert"]), False

    csvs = [os.path.join(alerts_dir, f) for f in os.listdir(alerts_dir) if f.lower().endswith(".csv")]
    if not csvs:
        logging.warning(f"Sin CSVs de alerta directa en (fallback): {alerts_dir}")
        return pd.DataFrame(columns=["id","direct_alert"]), False

    rows = []
    for p in csvs:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
            if "id" not in df.columns:
                logging.warning(f"CSV sin columna 'id' (se omite): {p}")
                continue
            # elegir la mejor columna de alerta disponible
            col_alert = "direct_alert" if "direct_alert" in df.columns else None
            for c in ["intersect_deforestation", "intersect_early_warnings", "intersect_active_hotspots"]:
                if col_alert is None and c in df.columns:
                    col_alert = c
            if col_alert is None:
                logging.warning(f"CSV sin columna de alerta usable (se omite): {p}")
                continue
            tmp = df[["id", col_alert]].copy()
            tmp["id"] = tmp["id"].map(normalize_id)
            tmp["direct_alert"] = tmp[col_alert].map(str_bool)
            rows.append(tmp[["id","direct_alert"]])
        except Exception as e:
            logging.warning(f"No se pudo leer {p}: {e}")

    if not rows:
        return pd.DataFrame(columns=["id","direct_alert"]), False

    merged = pd.concat(rows, ignore_index=True)
    agg = merged.groupby("id", as_index=False)["direct_alert"].max()
    return agg, True

# ===================== movimientos =====================
def build_movement_counts(mov_path: str, alert_ids_bool: Dict[str, bool]) -> pd.DataFrame:
    """
    Lee movimientos (todo string) y calcula conteos por cada id presente en alert_ids_bool.
      - n_in  = # filas con destination_id == id
      - n_out = # filas con origen_id == id
      - n_indirect_in  = # orígenes con direct_alert TRUE para destino==id
      - n_indirect_out = # destinos con direct_alert TRUE para origen==id
      - n_total_mov = n_in + n_out
    """
    if not os.path.isfile(mov_path):
        raise FileNotFoundError(f"No existe MOVEMENT_INPUT_CSV: {mov_path}")

    df_mov = pd.read_csv(mov_path, dtype=str, low_memory=False)

    # Normalizar nombres de columnas aceptadas
    if {"SIT_CODE_ORIGEN","SIT_CODE_DESTINO"}.issubset(df_mov.columns):
        df_mov = df_mov.rename(columns={
            "SIT_CODE_ORIGEN": "origen_id",
            "SIT_CODE_DESTINO": "destination_id"
        })
    elif not {"origen_id","destination_id"}.issubset(df_mov.columns):
        raise ValueError("El CSV de movimiento debe tener columnas 'SIT_CODE_ORIGEN'/'SIT_CODE_DESTINO' o 'origen_id'/'destination_id'.")

    # Normaliza IDs (solo strip)
    df_mov["origen_id"] = df_mov["origen_id"].map(normalize_id)
    df_mov["destination_id"] = df_mov["destination_id"].map(normalize_id)

    # Mapeos de alerta directa (True/False) para orígenes y destinos
    def map_alert(series: pd.Series) -> pd.Series:
        return series.map(lambda k: bool(alert_ids_bool.get(k, False)))

    df_mov["origin_has_alert"] = map_alert(df_mov["origen_id"])
    df_mov["dest_has_alert"]   = map_alert(df_mov["destination_id"])

    # Ids a computar (los presentes en las alertas)
    ids = pd.Index(alert_ids_bool.keys())

    # n_in / n_out
    n_in  = df_mov.groupby("destination_id").size().reindex(ids).fillna(0).astype(int)
    n_out = df_mov.groupby("origen_id").size().reindex(ids).fillna(0).astype(int)

    # n_indirect_in: orígenes con alerta → por destino
    tmp_in = df_mov[df_mov["origin_has_alert"]].groupby("destination_id").size()
    n_indirect_in = tmp_in.reindex(ids).fillna(0).astype(int)

    # n_indirect_out: destinos con alerta → por origen
    tmp_out = df_mov[df_mov["dest_has_alert"]].groupby("origen_id").size()
    n_indirect_out = tmp_out.reindex(ids).fillna(0).astype(int)

    out = pd.DataFrame({
        "id": ids,
        "n_in": n_in.values,
        "n_out": n_out.values,
        "n_indirect_in": n_indirect_in.values,
        "n_indirect_out": n_indirect_out.values,
    })
    out["n_total_mov"] = out["n_in"] + out["n_out"]
    out["indirect_alert_in"]  = out["n_indirect_in"]  > 0
    out["indirect_alert_out"] = out["n_indirect_out"] > 0

    # Orden solicitado
    out = out[[
        "id",
        "n_total_mov",
        "n_in",
        "n_out",
        "n_indirect_in",
        "n_indirect_out",
        "indirect_alert_in",
        "indirect_alert_out",
    ]]
    return out

# ===================== helpers de salida (movimiento) =====================
def movement_out_dir_and_path(base_output_template: str, empresa: str, periodo: str, years: str, mov_year: str, source_tag: str) -> Tuple[str, str]:
    """
    Estructura de salida para movimientos:
      <dirname(MOVEMENT_RISK_OUTPUT_CSV con placeholders)>/<SOURCE>/<empresa>/{source}_movement_alerts_{empresa}_{years}.csv
    """
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
    formatted = format_placeholders(base_output_template, ctx)
    base_dir = os.path.dirname(formatted) or "."
    empresa_folder = sanitize_empresa_folder(empresa)
    source_folder = SOURCE_MAP.get(source_tag.lower(), source_tag.upper())
    out_dir = os.path.join(base_dir, source_folder, empresa_folder)
    ensure_dir(out_dir)
    fname = f"{source_tag.lower()}_movement_alerts_{empresa}_{years}.csv"
    return out_dir, os.path.join(out_dir, fname)

# ===================== ejecución por periodo/fuente =====================
def run_for_period_and_source(empresa: str, periodo: str, years: str, source_tag: str) -> Optional[str]:
    mov_year = derive_mov_year(years, periodo)

    # 1) Intentar ALERTA DIRECTA exacta por years
    alerts_df = discover_alerts_exact_year(config["OUTPUT_CSV"], empresa, periodo, years, mov_year, source_tag)

    used_fallback = False
    if alerts_df.empty:
        # 2) Fallback: combinar cualquier years disponible
        alerts_df, had_any = discover_alerts_any_year(config["OUTPUT_CSV"], empresa, periodo, years, mov_year, source_tag)
        used_fallback = had_any

    # 3) Preparar carpeta/logs de salida del movimiento para esta fuente
    out_dir_mov, out_csv_mov = movement_out_dir_and_path(config["MOVEMENT_RISK_OUTPUT_CSV"], empresa, periodo, years, mov_year, source_tag)

    # 4) Si no hay alertas de ninguna forma → log y salir
    if alerts_df.empty:
        write_reason_log(out_dir_mov, f"{source_tag.lower()}_log_{empresa}_{years}.txt",
                         f"[{years}] No se encontraron alertas directas {SOURCE_MAP.get(source_tag, source_tag)} para empresa='{empresa}'.")
        return None

    # 5) Mapa id -> alerta
    alert_ids_bool = {normalize_id(r["id"]): bool(r["direct_alert"]) for _, r in alerts_df.iterrows()}

    # 6) Cargar movimientos del periodo (con MOV_YEAR en plantilla si aplica)
    ctx = {"EMPRESA": empresa, "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year}
    movement_path = format_placeholders(config["MOVEMENT_INPUT_CSV"], ctx)
    if not movement_path or not os.path.isfile(movement_path):
        write_reason_log(out_dir_mov, f"{source_tag.lower()}_log_{empresa}_{years}.txt",
                         f"[{years}] No existe MOVEMENT_INPUT_CSV: {movement_path}.")
        return None

    # 7) Calcular conteos
    try:
        result_df = build_movement_counts(movement_path, alert_ids_bool)
    except Exception as e:
        write_reason_log(out_dir_mov, f"{source_tag.lower()}_log_{empresa}_{years}.txt",
                         f"[{years}] Error calculando conteos: {e}")
        return None

    # 8) Guardar CSV final por fuente
    try:
        result_df.to_csv(out_csv_mov, index=False)
        if used_fallback:
            write_reason_log(out_dir_mov, f"{source_tag.lower()}_log_{empresa}_{years}.txt",
                             f"[{years}] Archivo exacto de alertas no encontrado; se usó combinación de años disponibles.")
        print(f"✅ {SOURCE_MAP.get(source_tag, source_tag)} → {out_csv_mov}")
        return out_csv_mov
    except Exception as e:
        write_reason_log(out_dir_mov, f"{source_tag.lower()}_log_{empresa}_{years}.txt",
                         f"[{years}] No se pudo guardar salida: {e}")
        return None

# ===================== main =====================
def main():
    setup_logging()

    empresa = str(config["EMPRESA"]).strip()                 # respeta empresa del .env
    periodo = str(config.get("PERIODO","")).strip()
    years_cfg = str(config.get("YEARS","")).strip()
    years_list = parse_year_periods(years_cfg) or ([years_cfg] if years_cfg else [])

    sources = ["smbyc", "atd", "nad"]  # siempre consulta las tres, leerá sólo lo disponible

    produced: List[str] = []
    for years in years_list:
        if not years:
            continue
        print(f"\n🟪 Periodo: {years} — empresa: {empresa} — PERIODO={periodo}")
        any_source = False
        for src in sources:
            out = run_for_period_and_source(empresa, periodo, years, src)
            if out:
                produced.append(out)
                any_source = True
        if not any_source:
            # Si ninguna fuente produjo salida, deja un log general en cada fuente para ese año
            mov_year = derive_mov_year(years, periodo)
            for src in sources:
                out_dir_mov, _ = movement_out_dir_and_path(config["MOVEMENT_RISK_OUTPUT_CSV"], empresa, periodo, years, mov_year, src)
                write_reason_log(out_dir_mov, f"{src}_log_{empresa}_{years}.txt",
                                 f"[{years}] No hubo datos suficientes para generar movimientos para la fuente {SOURCE_MAP.get(src, src)}.")
    if produced:
        print("\n🎉 Archivos generados:")
        for p in produced:
            print("  -", p)
    else:
        print("\n⚠ No se generaron archivos. Revisa los logs por fuente y año.")

if __name__ == "__main__":
    main()
