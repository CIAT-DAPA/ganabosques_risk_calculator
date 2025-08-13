import os
from pathlib import Path
from conf import config
from mongoengine import connect
from bson import ObjectId
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from ganabosques_orm.collections.analysis import Analysis

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

def save_farm_risk(input_file, analysis):
    df = pd.read_csv(input_file)

    # Validate analysis ObjectId
    try:
        analysis_id = ObjectId(analysis)
    except Exception as e:
        raise ValueError(f"Invalid analysis ObjectId: {analysis}") from e

    df['analysis_id'] = analysis

    # Convert each row to FarmRisk and save
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving FarmRisk records"):
        farm_risk = FarmRisk(
            analysis_id=analysis_id,
            farm_id=row["farm_id"],
            farm_polygons_id=row["farm_polygons_id"],
            risk_direct=row["risk_direct"],
            risk_input=row["risk_input"],
            risk_output=row["risk_output"],
            risk_total=row["risk_total"],
            deforestation_ha=row["deforestation_ha"],
            deforestation_prop=row["deforestation_prop"],
            deforestation_distance=row["deforestation_distance"],
            protected_distance=row["protected_distance"],
            protected_ha=row["protected_ha"],
            protected_prop=row["protected_prop"]
        )
        farm_risk.save()

    print(f"Saved {len(df)} FarmRisk records with analysis_id={analysis}")

def save_adm3_risk(input_file, analysis):
    # Load CSV into DataFrame
    df = pd.read_csv(input_file)

    # Validate ObjectId
    try:
        analysis_id = ObjectId(analysis)
    except Exception as e:
        raise ValueError(f"Invalid ObjectId for analysis: {analysis}") from e

    # Iterate through rows and save as Adm3Risk
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving Adm3Risk records"):
        adm3_risk = Adm3Risk(
            analysis_id=analysis_id,
            adm1_id=row["adm1_id"],
            adm1_name=row["adm1_name"],
            adm2_id=row["adm2_id"],
            adm2_name=row["adm2_name"],
            adm3_id=row["adm3_id"],
            adm3_name=row["adm3_name"],
            adm3_score=row["adm3_score"],
            adm3_risk=row["adm3_risk"],
            adm2_score=row["adm2_score"],
            adm2_risk=row["adm2_risk"],
            adm1_score=row["adm1_score"],
            adm1_risk=row["adm1_risk"]
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
