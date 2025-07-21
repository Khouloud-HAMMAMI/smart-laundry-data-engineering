import pandas as pd

def load_weather(ville="bailleul"):
    path = f"data_external/donnees_API_meteo_{ville}.csv"
    
    # Lire le CSV sans parse_dates d’abord
    df = pd.read_csv(path)
    
    # Identifier dynamiquement la colonne date
    possible_date_cols = ["datetime", "date", "time"]

    date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Aucune colonne de date trouvée dans le fichier météo : {path}")

    df["date"] = pd.to_datetime(df[date_col])
    
    # On garde uniquement les colonnes utiles
    keep_cols = ["date", "tempmax", "tempmin", "conditions", "precip"]
    available_cols = [col for col in keep_cols if col in df.columns]

    return df[available_cols]
