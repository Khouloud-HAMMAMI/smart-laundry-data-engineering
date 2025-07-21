import pandas as pd
def load_calendar():
    df = pd.read_csv("data_external/calendrier-scolaire-Ferie.csv", parse_dates=["date"])
    
    # Exemple : colonnes 'ferie' et 'vacances' déjà présentes ou à dériver
    df["ferie"] = df["ferie"].fillna(0).astype(bool)
    df["vacances"] = df["vacances"].fillna(0).astype(bool)
    return df
