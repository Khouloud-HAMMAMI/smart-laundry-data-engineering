import pandas as pd
import os

def load_laverie1_data(base_path="data_cleaned/laverie1"):
    # Charger les transactions
    df_trans = pd.read_csv(os.path.join(base_path, "transactions_jour_cleaned.csv"))
    
    # Vérifier et renommer la colonne date si nécessaire
    if "datetime" in df_trans.columns:
        df_trans["date"] = pd.to_datetime(df_trans["datetime"])
    elif "jour" in df_trans.columns:
        df_trans["date"] = pd.to_datetime(df_trans["jour"])
    elif "date" in df_trans.columns:
        df_trans["date"] = pd.to_datetime(df_trans["date"])
    else:
        raise ValueError("Aucune colonne 'date' trouvée dans transactions.")

    # Charger la conso élec
    df_conso = pd.read_csv("data_cleaned/conso_elec_laverie.csv")
    if "date" not in df_conso.columns:
        raise ValueError("Aucune colonne 'date' trouvée dans consommation élec.")
    df_conso["date"] = pd.to_datetime(df_conso["date"])

    # Fusion
    df = pd.merge(df_trans, df_conso, on="date", how="left")

    # Estimation CO2
    if "kWh" in df.columns:
        df["co2_estime"] = df["kWh"] * 0.09

    return df
