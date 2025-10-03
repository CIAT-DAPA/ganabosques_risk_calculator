#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== CONFIG =====================
config = {
    "PERIODO": "annual",
    "YEARS":   "2010-2012",   # puedes poner varios separados por coma
    "YEAR_MOV": "2010",

    # Rutas (tu estructura actual)
    "OUTPUT_CSV": "/opt/ganabosques/test_buffers/data_server/alertas/{PERIODO}/direct_alert/SMBYC/{PERIODO}/{YEARS}",
    "MOVEMENT_INPUT_CSV": "/opt/ganabosques/test_buffers/data_server/movement/movement_data_base_{YEAR_MOV}.csv",
    "MOVEMENT_RISK_OUTPUT_CSV": "/opt/ganabosques/test_buffers/data_server/alertas_indirect/{PERIODO}/{YEARS}",

    # Normalización de IDs (ajústalo si tu data lo requiere)
    "ID_STRIP_DOT_ZERO": True,      # convierte "123.0" -> "123"
    "ID_STRIP_LEADING_ZEROS": True, # "00123" -> "123" (solo si es numérico)
    "ID_UPPERCASE": True,           # "ab-01" -> "AB-01"

    "LOG_LEVEL": "INFO",
    "LOG_FILE":  "./movement.log",
    "MAX_WORKERS": 10
}

# ===================== LOGGING =====================
def setup_logging():
    lvl = getattr(logging, str(config.get('LOG_LEVEL', 'WARNING')).upper(), logging.WARNING)
    logging.basicConfig(filename=config.get('LOG_FILE', 'movement.log'),
                        level=lvl,
                        format="%(asctime)s %(levelname)s:%(message)s")
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

# ===================== UTILIDADES =====================
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
    if not years_range or "-" not in years_range:
        return str(years_range or "")
    first_year, last_year = years_range.split("-", 1)
    p = (periodo or "").strip().lower()
    if p in {"anual", "annual"}:
        return first_year
    if p in {"cumulative", "cum", "acumulado"}:
        return last_year
    return first_year

def format_placeholders(template: str, ctx: Dict[str, Any]) -> str:
    return template.format(**ctx)

# ---------- Normalización de IDs robusta ----------
def _normalize_id_core(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    # quitar sufijo ".0" si aplica
    if config.get("ID_STRIP_DOT_ZERO", True) and re.match(r"^\d+\.0$", s):
        s = s[:-2]
    # quitar ceros a la izquierda (solo si queda numérico puro)
    if config.get("ID_STRIP_LEADING_ZEROS", True) and re.match(r"^0*\d+$", s):
        s = str(int(s))  # int("000") -> 0; luego str -> "0"
    # mayúsculas
    if config.get("ID_UPPERCASE", True):
        s = s.upper()
    return s

def normalize_id_series(sr: pd.Series) -> pd.Series:
    return sr.map(_normalize_id_core)

def str_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in {"true","t","1","yes","y","si","sí"}: return True
    if s in {"false","f","0","no","n"}:           return False
    return bool(s and s != "nan")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ===================== ALERTAS (SMBYC) =====================
def load_smbyc_alerts(output_base_template: str,
                      periodo: str, years: str,
                      mov_year: str, year_mov: str) -> pd.DataFrame:
    alerts_dir = format_placeholders(output_base_template, {
        "PERIODO": periodo, "YEARS": years,
        "MOV_YEAR": mov_year, "YEAR_MOV": year_mov
    })
    exact_name = f"smbyc_direct_alert_{periodo}_{years}.csv"
    exact_path = os.path.join(alerts_dir, exact_name)

    if not os.path.isfile(exact_path):
        logging.warning(f"No se encontró CSV exacto: {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    df = pd.read_csv(exact_path, dtype=str, low_memory=False)

    if "id" not in df.columns:
        logging.warning(f"CSV SMBYC sin columna 'id': {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    col_alert = "direct_alert" if "direct_alert" in df.columns else None
    if not col_alert:
        for c in ["intersect_deforestation","intersect_early_warnings","intersect_active_hotspots"]:
            if c in df.columns:
                col_alert = c
                break
    if not col_alert:
        logging.warning(f"CSV SMBYC sin columna de alerta usable: {exact_path}")
        return pd.DataFrame(columns=["id","direct_alert"])

    out = df[["id", col_alert]].copy()
    out["id"] = normalize_id_series(out["id"])
    out["direct_alert"] = out[col_alert].map(str_bool)
    return out[["id","direct_alert"]]

# ===================== MOVIMIENTOS =====================
def build_movement_counts(mov_path: str, alert_ids_bool: Dict[str, bool]) -> pd.DataFrame:
    df_mov = pd.read_csv(mov_path, dtype=str, low_memory=False)

    # Columnas aceptadas
    if {"SIT_CODE_ORIGEN","SIT_CODE_DESTINO"}.issubset(df_mov.columns):
        df_mov = df_mov.rename(columns={
            "SIT_CODE_ORIGEN": "origen_id",
            "SIT_CODE_DESTINO": "destination_id"
        })
    elif not {"origen_id","destination_id"}.issubset(df_mov.columns):
        raise ValueError("El CSV de movimiento debe tener 'SIT_CODE_ORIGEN'/'SIT_CODE_DESTINO' o 'origen_id'/'destination_id'.")

    # Normaliza IDs de movimientos con la MISMA lógica
    df_mov["origen_id"] = normalize_id_series(df_mov["origen_id"])
    df_mov["destination_id"] = normalize_id_series(df_mov["destination_id"])

    # Diagnóstico de intersección de IDs
    ids_alerta = set(alert_ids_bool.keys())
    origenes = set(df_mov["origen_id"].unique())
    destinos = set(df_mov["destination_id"].unique())
    universo = origenes | destinos
    inter = ids_alerta & universo

    print(f"   🔎 SMBYC: IDs con alerta directa TRUE: {sum(alert_ids_bool.values())} / {len(alert_ids_bool)}")
    print(f"   🔎 Movimientos: orígenes únicos={len(origenes)}, destinos únicos={len(destinos)}, universo={len(universo)}")
    print(f"   🔗 Intersección IDs alerta vs movimientos: {len(inter)}")

    if len(inter) == 0:
        print("   ⚠ No hay cruce de IDs entre alertas y movimientos. Revisa normalización ('.0', ceros, mayúsculas).")

    # Mapeos booleanos
    def map_alert(series: pd.Series) -> pd.Series:
        return series.map(lambda k: bool(alert_ids_bool.get(k, False)))

    df_mov["origin_has_alert"] = map_alert(df_mov["origen_id"])
    df_mov["dest_has_alert"]   = map_alert(df_mov["destination_id"])

    # Solo calculamos para los IDs presentes en alertas (como en tu lógica original)
    ids = pd.Index(sorted(alert_ids_bool.keys()))

    n_in  = df_mov.groupby("destination_id").size().reindex(ids).fillna(0).astype(int)
    n_out = df_mov.groupby("origen_id").size().reindex(ids).fillna(0).astype(int)

    tmp_in  = df_mov[df_mov["origin_has_alert"]].groupby("destination_id").size()
    tmp_out = df_mov[df_mov["dest_has_alert"]].groupby("origen_id").size()

    out = pd.DataFrame({
        "id": ids,
        "n_in": n_in.values,
        "n_out": n_out.values,
        "n_indirect_in": tmp_in.reindex(ids).fillna(0).astype(int).values,
        "n_indirect_out": tmp_out.reindex(ids).fillna(0).astype(int).values,
    })
    out["n_total_mov"] = out["n_in"] + out["n_out"]
    out["indirect_alert_in"]  = out["n_indirect_in"]  > 0
    out["indirect_alert_out"] = out["n_indirect_out"] > 0
    return out

# ===================== PIPELINE (solo SMBYC) =====================
def process_year(years: str, periodo: str, year_mov_override: Optional[str]) -> Optional[str]:
    print(f"▶️ Iniciando SMBYC {years} ({periodo})")

    mov_year = derive_mov_year(years, periodo)
    year_mov = (year_mov_override or "").strip() or mov_year

    # 1) Cargar alertas directas
    alerts_df = load_smbyc_alerts(config["OUTPUT_CSV"], periodo, years, mov_year, year_mov)
    if alerts_df.empty:
        print(f"⚠ No se encontraron alertas SMBYC {years}")
        return None

    alert_ids_bool = {row["id"]: bool(row["direct_alert"]) for _, row in alerts_df.iterrows()}

    # 2) Cargar movimientos
    movement_path = format_placeholders(config["MOVEMENT_INPUT_CSV"], {
        "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year, "YEAR_MOV": year_mov
    })
    if not os.path.isfile(movement_path):
        print(f"❌ No existe archivo de movimientos: {movement_path}")
        return None

    # 3) Calcular
    result_df = build_movement_counts(movement_path, alert_ids_bool)

    # 4) Guardar salida
    out_dir = format_placeholders(config["MOVEMENT_RISK_OUTPUT_CSV"], {
        "PERIODO": periodo, "YEARS": years, "MOV_YEAR": mov_year, "YEAR_MOV": year_mov
    })
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, f"smbyc_movement_alerts_{years}.csv")
    result_df.to_csv(out_file, index=False)

    print(f"✅ Terminado SMBYC {years} → {out_file}")
    print(f"   ✅ indirect_alert_in TRUE: {int(result_df['indirect_alert_in'].sum())} de {len(result_df)}")
    print(f"   ✅ indirect_alert_out TRUE: {int(result_df['indirect_alert_out'].sum())} de {len(result_df)}")
    return out_file

# ===================== MAIN =====================
def main():
    setup_logging()
    periodo = str(config.get("PERIODO","")).strip()
    years_cfg = str(config.get("YEARS","")).strip()
    years_list = parse_year_periods(years_cfg)
    year_mov_override = str(config.get("YEAR_MOV","")).strip() or None

    # Paraleliza por años (solo SMBYC)
    tasks = []
    with ThreadPoolExecutor(max_workers=config["MAX_WORKERS"]) as executor:
        for years in years_list:
            tasks.append(executor.submit(process_year, years, periodo, year_mov_override))

        results = []
        for future in as_completed(tasks):
            result = future.result()
            if result:
                results.append(result)

    print("\n🎉 Archivos SMBYC generados:")
    for r in results:
        print("  -", r)

if __name__ == "__main__":
    main()
