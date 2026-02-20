from pymongo import MongoClient
from bson import ObjectId
import os, re

MONGO_URI = "mongodb://localhost:27017"
DB = "ganabosques"
COL_DEFORESTATION = "deforestation"
COL_ANALYSIS = "analysis"
COL_FARM = "farm"
COL_FARMRISK = "farmrisk"

client = MongoClient(MONGO_URI)
db = client[DB]

name_def = "smbyc_deforestation_annual_2015-2016"  # prueba con uno
def_doc = db[COL_DEFORESTATION].find_one({"name": name_def}, {"_id":1})
def_id = def_doc["_id"]
analysis_ids = [d["_id"] for d in db[COL_ANALYSIS].find({"deforestation_id": def_id}, {"_id":1})]
aid = analysis_ids[0]
print("def_id:", def_id, "| analysis_id:", aid)

# (A) ¿Ese analysis_id tiene algún TRUE (sin empresas)?
cnt_any_true = db[COL_FARMRISK].count_documents({
    "analysis_id": aid,
    "$or":[{"risk_direct": True},{"risk_input": True},{"risk_output": True}]
})
print("farmrisk con TRUE (global) para analysis_id:", cnt_any_true)

# (B) Códigos desde carpeta CARNATURAL
FOLDER = r"D:\OneDrive - CGIAR\Desktop\ganabosques\riesgo_empresas\input\limpios\carnatural"
codes = []
for n in os.listdir(FOLDER):
    if n.lower().endswith(".geojson"):
        m = re.search(r"(\d+)", os.path.splitext(n)[0])
        if m: codes.append(m.group(1))
codes = sorted(set(codes))
print("codigos:", len(codes))

# (C) Buscar farms por source=SIT_CODE y esos códigos
farm_q = {"ext_id": {"$elemMatch": {"source": "SIT_CODE", "ext_code": {"$in": codes}}}}
farm_ids = [d["_id"] for d in db[COL_FARM].find(farm_q, {"_id":1})]
print("farms encontrados (SIT_CODE):", len(farm_ids))

# (D) ¿Qué analysis_id tienen TRUE para esas farms? (¡clave!)
pipeline = [
    {"$match": {
        "farm_id": {"$in": farm_ids},
        "$or":[{"risk_direct": True},{"risk_input": True},{"risk_output": True}]
    }},
    {"$group": {"_id": "$analysis_id", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
print("analysis_id con TRUE para estas farms:")
for d in db[COL_FARMRISK].aggregate(pipeline):
    print("  ", d["_id"], "=>", d["count"])

# (E) ¿Y con tu analysis_id?
cnt_for_aid = db[COL_FARMRISK].count_documents({
    "farm_id": {"$in": farm_ids},
    "analysis_id": aid,
    "$or":[{"risk_direct": True},{"risk_input": True},{"risk_output": True}]
})
print("con tu analysis_id y esas farms → TRUE =", cnt_for_aid)
