import pandas as pd

def load_enriched_alertes(path="data_cleaned/laverie1/alertes_cleaned.csv"):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.replace('"', '')
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["alerte"] = df["alerte"].astype(str).str.lower()

    df["alerte_critique"] = df["type"].str.lower() == "critique"
    df["alerte_importante"] = df["type"].str.lower() == "important"
    df["alerte_tube"] = df["alerte"].str.contains("tube")
    df["alerte_choc"] = df["alerte"].str.contains("choc")
    df["alerte_lecteur"] = df["alerte"].str.contains("lecteur billet")
    df["alerte_monnaie"] = df["alerte"].str.contains("cassette|monnayeur")
    df["alerte_trop_plein"] = df["alerte"].str.contains("trop.*plein")

    daily = df.groupby("date").agg({
        "alerte": "count",
        "alerte_critique": "sum",
        "alerte_importante": "sum",
        "alerte_tube": "sum",
        "alerte_choc": "sum",
        "alerte_lecteur": "sum",
        "alerte_monnaie": "sum",
        "alerte_trop_plein": "sum"
    }).reset_index()

    daily.rename(columns={
        "alerte": "nb_alertes_total",
        "alerte_critique": "nb_alertes_critiques",
        "alerte_importante": "nb_alertes_importantes",
        "alerte_tube": "nb_alertes_tube",
        "alerte_choc": "nb_alertes_choc",
        "alerte_lecteur": "nb_alertes_lecteur",
        "alerte_monnaie": "nb_alertes_defaut_monnaie",
        "alerte_trop_plein": "nb_alertes_trop_plein"
    }, inplace=True)

    return daily
