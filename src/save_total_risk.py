import os
from pathlib import Path
from conf import config
from mongoengine import connect
from bson import ObjectId
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from ganabosques_orm.collections.analysis import Analysis
from ganabosques_orm.auxiliaries.attributes import Attributes

from ganabosques_orm.collections.protectedareas import ProtectedAreas
from ganabosques_orm.collections.farmingareas import FarmingAreas
from ganabosques_orm.collections.deforestation import Deforestation

from ganabosques_orm.collections.farmrisk import FarmRisk
from ganabosques_orm.collections.adm3risk import Adm3Risk

def create_analysis():
    protected = ProtectedAreas.objects.first()
    farming = FarmingAreas.objects.first()

    if not protected:
        raise ValueError("No ProtectedAreas found in the database.")

    if not farming:
        raise ValueError("No FarmingAreas found in the database.")

    deforestation_records = Deforestation.objects()
    print(f"Creating one Analysis per Deforestation record: {len(deforestation_records)} found")

    user_id = ObjectId()
    for defo in tqdm(deforestation_records, desc="Creating Analysis records"):
        analysis = Analysis(
            protected_areas_id = protected.id,
            farming_areas_id = farming.id,
            deforestation_id = defo.id,
            user_id = user_id,
            date = datetime.now()
        )
        analysis.save()
    print("Done!")

def save_farm_risk(input_file, analysis, chunk_size = 1000):
    df = pd.read_csv(input_file)

    # Validate analysis ObjectId
    try:
        analysis_id = ObjectId(analysis)
    except Exception as e:
        raise ValueError(f"Invalid analysis ObjectId: {analysis}") from e

    df['analysis_id'] = analysis
    """
    total = len(df)
    print(f"Processing {total} farm risk records in chunks of {chunk_size}...")

    for i in tqdm(range(0, total, chunk_size), desc="Bulk inserting FarmRisk"):
        chunk = df.iloc[i:i+chunk_size].to_dict(orient="records")
        docs = [FarmRisk(**record) for record in chunk]
        FarmRisk.objects.insert(docs, load_bulk=False)
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving FarmRisk records"):
        farm_risk = FarmRisk(
            analysis_id=analysis_id,
            farm_id=row["farm_id"],
            farm_polygons_id=row["farm_polygons_id"],
            risk_direct=row["risk_direct"],
            risk_input=row["risk_input"],
            risk_output=row["risk_output"],
            risk_total=row["risk_total"],
            deforestation = Attributes(ha = row["deforestation_ha"],prop=row["deforestation_prop"],distance=row["deforestation_distance"]),
            farming=row["farming"],
            protected = Attributes(ha = row["protected_ha"],prop=row["protected_prop"],distance=row["protected_distance"])
        )
        farm_risk.save()
    

    print(f"Saved {len(df)} FarmRisk records with analysis_id={analysis}")

def save_adm3_risk(input_file, analysis, chunk_size = 1000):
    # Load CSV into DataFrame
    df = pd.read_csv(input_file)
 
    # Validate ObjectId
    try:
        analysis_id = ObjectId(analysis)
    except Exception as e:
        raise ValueError(f"Invalid ObjectId for analysis: {analysis}") from e
    """
    total = len(df)
    print(f"Processing {total} adm3 risk records in chunks of {chunk_size}...")
 
    for i in tqdm(range(0, total, chunk_size), desc="Bulk inserting Adm3Risk"):
        chunk = df.iloc[i:i+chunk_size].to_dict(orient="records")
        docs = [Adm3Risk(**record) for record in chunk]
        Adm3Risk.objects.insert(docs, load_bulk=False)
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving Adm3Risk records"):
        adm3_risk = Adm3Risk(
            analysis_id=analysis_id,
            adm3_id=row["adm3_id"],
            def_ha=row["def_ha"],
            farm_amount=row["farm_amount"],
            risk_total=row["risk_total"]
        )
        adm3_risk.save()
    

    print(f"Saved {len(df)} Adm3Risk records with analysis_id={analysis}")

if __name__ == "__main__":
    # Connect with database
    connect(host=config['CONNECTION_URI'], db=config['CONNECTION_DB'])

    print("=== Ganabosques ETL Menu ===")
    print("1) Create records of Analysis")
    print("2) Create records of FarmRisk")
    print("3) Create records of Adm3Risk")

    option = input("Choose one option (1, 2, 3): ").strip()

    if option == "1":
        create_analysis()

    elif option == "2":
        input_file = input("Path for CSV file of FarmRisk: ").strip()
        analysis_id = input("Analysis ID (string): ").strip()

        if not Path(input_file).is_file():
            print(f"File not exists: {input_file}")
        else:
            save_farm_risk(input_file, analysis_id)

    elif option == "3":
        input_file = input("Path for CSV file of Adm3Risk: ").strip()
        analysis_id = input("Analysis ID (string): ").strip()

        if not Path(input_file).is_file():
            print(f"File not exists: {input_file}")
        else:
            save_adm3_risk(input_file, analysis_id)

    else:
        print("Wrong option. Try 1, 2 or 3.")