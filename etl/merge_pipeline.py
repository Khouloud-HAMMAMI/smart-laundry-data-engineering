
import pandas as pd
import sys
import os

# Ajoute le dossier parent au path (racine du projet)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from etl.etl_laverie1 import load_laverie1_data
from etl.etl_weather import load_weather
from etl.etl_calendar import load_calendar



def merge_all_laverie1():
    df_laverie = load_laverie1_data()
    df_meteo = load_weather()
    df_cal = load_calendar()

    # Fusion par date
    df = pd.merge(df_laverie, df_meteo, on="date", how="left")
    df = pd.merge(df, df_cal, on="date", how="left")

    # Remplir les colonnes manquantes
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
    df_final = merge_all_laverie1()
    df_final.to_csv("data_final/merged_laverie1.csv", index=False)
    print("✅ Fusion terminée pour laverie1")
