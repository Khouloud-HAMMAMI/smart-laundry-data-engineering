import pandas as pd

def load_calendar():
    df = pd.read_csv("data_external/calendrier-scolaire-Ferie.csv", parse_dates=["date"])
    
    # Création des colonnes booléennes à partir de la colonne 'type'
    df["ferie"] = df["type"] == "jour_férié"
    df["vacances"] = df["type"] == "vacances_scolaires"
    
    # On conserve une ligne par date, avec les indicateurs
    df = df.groupby("date").agg({
        "ferie": "max",
        "vacances": "max"
    }).reset_index()
    
    return df
