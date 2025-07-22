import pandas as pd
import sys
import os

# Ajoute le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from etl.etl_laverie2 import load_laverie2_data
from etl.etl_weather2 import load_weather
from etl.etl_calendar import load_calendar

def merge_all_laverie2():
    df_laverie = load_laverie2_data()
    df_meteo = load_weather(ville="boulogne")  # ⚠️ Modifie le nom de fichier météo si besoin
    df_cal = load_calendar()

    df = pd.merge(df_laverie, df_meteo, on="date", how="left")
    df = pd.merge(df, df_cal, on="date", how="left")

    df.fillna({
        "tempmax": 0,
        "tempmin": 0,
        "conditions": "inconnu",
        "precip": 0,
        "ferie": False,
        "vacances": False
    }, inplace=True)

    return df

if __name__ == "__main__":
    df_final = merge_all_laverie2()
    
    output_dir = "data_final"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "merged_laverie2.csv")
    df_final.to_csv(output_path, index=False)

    print("✅ Fusion terminée pour laverie2")
