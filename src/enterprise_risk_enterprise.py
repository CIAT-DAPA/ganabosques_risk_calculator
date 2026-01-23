# -*- coding: utf-8 -*-
r"""
enterprise_risk_enterprises.py

Genera alertas por empresa (p. ej., COLACTEOS, CARNATURAL) a partir de listas
de códigos extraídos de archivos *.geojson en carpetas locales.

Pasos:
  1) deforestation.name == name_def -> _id_deforestation
  2) analysis.deforestation_id == _id_deforestation -> analysis_id(s)
  3) enterprise: filtrar por type_enterprise == tipo_empresa
  4) Para cada empresa mapeada a una carpeta:
      - leer *.geojson → extraer dígitos del nombre como "ext_code"
      - farm: buscar por ext_id.ext_code ∈ lista → obtener _id (farm_id)
      - farmrisk: (farm_id ∈ lista) AND (analysis_id ∈ analysis_ids) AND (cualquier riesgo TRUE)
      - agrupar por analysis_id y exportar CSV
Salida:
  - colacteos:  D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\enterprise_risk\colacteos\alert_enterprises_colacteos_{analysis_id}.csv
  - carnatural: D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\enterprise_risk\carnatural\alert_enterprise_carnatural_{analysis_id}.csv
"""

import os
import re
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

# --- Conexión Mongo ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ganabosques"

# --- Salidas base (carpetas donde escribiremos CSVs) ---
BASE_OUT = r"D:\OneDrive - CGIAR\Desktop\ganabosques\alertas\enterprise_risk"
OUT_COLACTEOS = os.path.join(BASE_OUT, "colacteos")
OUT_CARNATURAL = os.path.join(BASE_OUT, "carnatural")

# --- Nombres de colecciones (según tu base real) ---
COL_DEFORESTATION = "deforestation"
COL_ANALYSIS = "analysis"
COL_ENTERPRISE = "enterprise"
COL_FARM = "farm"
COL_FARMRISK = "farmrisk"  # ¡ojo! no "farm_risk"


@dataclass
class Buckets:
    direct: Set[str] = field(default_factory=set)
    indirect_in: Set[str] = field(default_factory=set)
    indirect_out: Set[str] = field(default_factory=set)


DIGITS_RE = re.compile(r"(\d+)")


def _extract_codes_from_folder(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"  !! Carpeta no encontrada: {folder}")
        return []
    codes: List[str] = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".geojson"):
            continue
        m = DIGITS_RE.search(os.path.splitext(name)[0])
        if m:
            codes.append(m.group(1))
    return sorted(set(codes))


def _find_analysis_ids(db, name_def: str) -> List[ObjectId]:
    def_doc = db[COL_DEFORESTATION].find_one({"name": name_def}, {"_id": 1})
    if not def_doc:
        print(f"  !! No se encontró en 'deforestation' el name='{name_def}'")
        return []
    def_id = def_doc["_id"]
    print(f"  -> _id_deforestation = {def_id}")
    analysis_ids = [doc["_id"] for doc in db[COL_ANALYSIS].find({"deforestation_id": def_id}, {"_id": 1})]
    print(f"  -> analysis_id(s) encontrados: {len(analysis_ids)}")
    return analysis_ids


def _rows_for_enterprise(db, enterprise_name: str, analysis_ids: List[ObjectId], codes_from_folder: List[str]) -> Dict[str, Any]:
    print(f"\n[EMPRESA] {enterprise_name}")
    print(f"  - Códigos extraídos de carpeta: {len(codes_from_folder)}")

    if not codes_from_folder:
        print("  -> Sin códigos; se omite.")
        return {"rows": [], "farms_count": 0, "farms_ids": []}

    farm_cursor = db[COL_FARM].find({"ext_id.ext_code": {"$in": codes_from_folder}}, {"_id": 1})
    farm_ids: List[ObjectId] = [doc["_id"] for doc in farm_cursor]
    print(f"  - Farms encontrados por ext_code: {len(farm_ids)}")

    if not farm_ids:
        return {"rows": [], "farms_count": 0, "farms_ids": []}

    fr_query = {
        "farm_id": {"$in": farm_ids},
        "analysis_id": {"$in": analysis_ids},
        "$or": [{"risk_direct": True}, {"risk_input": True}, {"risk_output": True}],
    }
    fr_fields = {"farm_id": 1, "analysis_id": 1, "risk_direct": 1, "risk_input": 1, "risk_output": 1}
    fr_cursor = db[COL_FARMRISK].find(fr_query, fr_fields)

    grouped: Dict[str, Buckets] = {}
    fr_count = 0
    for fr in fr_cursor:
        fr_count += 1
        farm_id = fr.get("farm_id")
        analysis_id = fr.get("analysis_id")
        if farm_id is None or analysis_id is None:
            continue
        key = str(analysis_id)
        b = grouped.setdefault(key, Buckets())
        s_farm = str(farm_id)
        if fr.get("risk_direct", False) is True:
            b.direct.add(s_farm)
        if fr.get("risk_input", False) is True:
            b.indirect_in.add(s_farm)
        if fr.get("risk_output", False) is True:
            b.indirect_out.add(s_farm)

    print(f"  - Registros farmrisk con riesgo TRUE: {fr_count}")
    print(f"  - Análisis con riesgos: {len(grouped)}")

    rows: List[Dict[str, Any]] = []
    for analysis_key, b in grouped.items():
        rows.append({
            "name_enterprise": enterprise_name,
            "analysis_id": analysis_key,
            "ids_directa": ",".join(sorted(b.direct)),
            "ids_indirect_in": ",".join(sorted(b.indirect_in)),
            "ids_indirect_out": ",".join(sorted(b.indirect_out)),
        })
    return {"rows": rows, "farms_count": len(farm_ids), "farms_ids": farm_ids}


def enterprise_risk_for_enterprises(name_def: str, tipo_empresa: str, enterprise_dirs: Dict[str, str]) -> Dict[str, List[str]]:
    print("\n================ ENTERPRISE RISK POR EMPRESA =================")
    print(f"Conectando a MongoDB: {MONGO_URI} | DB: {MONGO_DB_NAME}")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]

    print(f"\n[1/4] Resolviendo analysis_id(s) desde deforestation.name = '{name_def}' ...")
    analysis_ids = _find_analysis_ids(db, name_def)
    if not analysis_ids:
        print("  !! No se encontraron analysis_id para ese name_def. Proceso detenido.")
        return {}

    print(f"\n[2/4] Buscando enterprises con type_enterprise = '{tipo_empresa}' ...")
    enterprises = list(db[COL_ENTERPRISE].find({"type_enterprise": tipo_empresa}, {"name": 1}))
    names_in_db = {e.get("name", ""): e["_id"] for e in enterprises}
    print(f"  -> Empresas en DB con ese tipo: {len(names_in_db)}")

    target_names = [n for n in enterprise_dirs.keys() if n in names_in_db]
    missing = [n for n in enterprise_dirs.keys() if n not in names_in_db]
    if missing:
        print(f"  !! Advertencia: estas empresas tienen carpeta pero no aparecen en DB: {missing}")
    if not target_names:
        print("  !! Ninguna empresa de enterprise_dirs aparece en DB. Proceso detenido.")
        return {}

    print(f"  -> Empresas objetivo (con carpeta + en DB): {target_names}")

    os.makedirs(OUT_COLACTEOS, exist_ok=True)
    os.makedirs(OUT_CARNATURAL, exist_ok=True)

    csv_paths_by_enterprise: Dict[str, List[str]] = {}
    print("\n[3/4] Procesando empresas ...")
    for idx, ent_name in enumerate(target_names, start=1):
        print(f"\n--- Empresa ({idx}/{len(target_names)}): {ent_name} ---")
        folder = enterprise_dirs[ent_name]
        print(f"  Carpeta de polígonos: {folder}")

        codes = _extract_codes_from_folder(folder)
        print(f"  -> Códigos detectados: {len(codes)}")

        res = _rows_for_enterprise(db, ent_name, analysis_ids, codes)
        rows = res["rows"]

        if not rows:
            print("  -> No hay filas para exportar (sin riesgos TRUE).")
            csv_paths_by_enterprise[ent_name] = []
            continue

        by_analysis: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_analysis.setdefault(r["analysis_id"], []).append(r)

        generated: List[str] = []
        for analysis_id, group_rows in by_analysis.items():
            df = pd.DataFrame(group_rows, columns=["name_enterprise", "analysis_id", "ids_directa", "ids_indirect_in", "ids_indirect_out"])

            if ent_name.upper() == "COLACTEOS":
                outdir, fname = OUT_COLACTEOS, f"alert_enterprises_colacteos_{analysis_id}.csv"
            elif ent_name.upper() == "CARNATURAL":
                outdir, fname = OUT_CARNATURAL, f"alert_enterprise_carnatural_{analysis_id}.csv"
            else:
                safe = ent_name.lower().replace(" ", "_")
                outdir = os.path.join(BASE_OUT, safe)
                os.makedirs(outdir, exist_ok=True)
                fname = f"alert_enterprise_{safe}_{analysis_id}.csv"

            out_path = os.path.join(outdir, fname)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            generated.append(out_path)
            print(f"  -> CSV escrito: {out_path}")

        csv_paths_by_enterprise[ent_name] = generated

    print("\n[4/4] Finalizado.")
    for ent_name, paths in csv_paths_by_enterprise.items():
        print(f"  {ent_name}:")
        if paths:
            for p in paths:
                print(f"    - {p}")
        else:
            print("    - (sin resultados)")
    return csv_paths_by_enterprise


# === NUEVO: correr para múltiples name_def ===
def run_many(name_defs: List[str], tipo_empresa: str, enterprise_dirs: Dict[str, str]) -> None:
    for nd in name_defs:
        print("\n" + "=" * 80)
        print(f"== Procesando deforestación: {nd}")
        print("=" * 80)
        enterprise_risk_for_enterprises(nd, tipo_empresa, enterprise_dirs)


if __name__ == "__main__":
    # Lista de deforestaciones
    name_defs = [
        "deforestation_cumulative_2010-2011",
        "deforestation_cumulative_2010-2012",
        "deforestation_cumulative_2010-2013",
        "deforestation_cumulative_2010-2014",
        "deforestation_cumulative_2010-2015",
        "deforestation_cumulative_2010-2016",
        "deforestation_cumulative_2010-2017",
        "deforestation_cumulative_2010-2018",
        "deforestation_cumulative_2010-2019",
        "deforestation_cumulative_2010-2020",
        "deforestation_cumulative_2010-2021",
        "deforestation_cumulative_2010-2022",
        "deforestation_cumulative_2010-2023",
        
    ]

    tipo_empresa = "ENTERPRISE"
    enterprise_dirs = {
        "COLACTEOS": r"D:\OneDrive - CGIAR\Desktop\ganabosques\riesgo_empresas\input\limpios\colacteos",
        "CARNATURAL": r"D:\OneDrive - CGIAR\Desktop\ganabosques\riesgo_empresas\input\limpios\carnatural",
    }

    # Llama a la versión multi:
    run_many(name_defs, tipo_empresa, enterprise_dirs)
