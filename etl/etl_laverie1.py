import pandas as pd
import os

def enrichir_alertes(alertes_path):
    df = pd.read_csv(alertes_path)
    df.columns = df.columns.str.strip().str.replace('"', '')

    if "date" in df.columns and "heure" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["heure"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
    else:
        return None

    df["alerte"] = df["alerte"].astype(str).str.lower()
    df["type"] = df["type"].astype(str).str.lower()

    df["alerte_critique"] = df["type"] == "critique"
    df["alerte_importante"] = df["type"] == "important"
    df["alerte_info"] = df["type"] == "info"
    df["alerte_tube"] = df["alerte"].str.contains("tube")
    df["alerte_choc"] = df["alerte"].str.contains("choc")
    df["alerte_lecteur"] = df["alerte"].str.contains("lecteur billet")
    df["alerte_monnaie"] = df["alerte"].str.contains("cassette|monnayeur")
    df["alerte_erreur_0105"] = df["alerte"].str.contains("erreur 0105")

    daily = df.groupby("date").agg(
        nb_alertes_total=("alerte", "count"),
        nb_alertes_critique=("alerte_critique", "sum"),
        nb_alertes_importante=("alerte_importante", "sum"),
        nb_alertes_info=("alerte_info", "sum"),
        nb_alertes_tube=("alerte_tube", "sum"),
        nb_alertes_choc=("alerte_choc", "sum"),
        nb_alertes_lecteur=("alerte_lecteur", "sum"),
        nb_alertes_monnaie=("alerte_monnaie", "sum"),
        nb_alertes_erreur_0105=("alerte_erreur_0105", "sum")
    ).reset_index()

    return daily

def load_laverie_data(base_path, conso_path=None):
    df_trans = pd.read_csv(os.path.join(base_path, "transactions_jour_cleaned.csv"), parse_dates=["date"])

    if conso_path and os.path.exists(conso_path):
        df_conso = pd.read_csv(conso_path, parse_dates=["date"])
        df_trans = pd.merge(df_trans, df_conso, on="date", how="left")
        df_trans["co2_estime"] = df_trans["kWh"] * 0.09
    else:
        df_trans["co2_estime"] = None

    # ➕ Alertes enrichies
    alertes_path = os.path.join(base_path, "alertes_cleaned.csv")
    if os.path.exists(alertes_path):
        df_alertes = enrichir_alertes(alertes_path)
        if df_alertes is not None:
            df_trans = pd.merge(df_trans, df_alertes, on="date", how="left")
            df_trans.fillna(0, inplace=True)

    # ➕ Remplissage
    remplissage_path = os.path.join(base_path, "remplissages_cleaned.csv")
    if os.path.exists(remplissage_path):
        df_remp = pd.read_csv(remplissage_path, parse_dates=["datetime"])
        df_remp["date"] = df_remp["datetime"].dt.date
        remplissage_group = df_remp.groupby("date").agg(
            nb_remplissages=("id", "count"),
            total_rempli=("total", "sum")
        ).reset_index()
        remplissage_group["date"] = pd.to_datetime(remplissage_group["date"])
        df_trans = pd.merge(df_trans, remplissage_group, on="date", how="left")
        df_trans.fillna({"nb_remplissages": 0, "total_rempli": 0}, inplace=True)

    # ➕ Transactions par heure
    trans_detail_path = os.path.join(base_path, "transactions_transaction_cleaned.csv")
    if os.path.exists(trans_detail_path):
        df_det = pd.read_csv(trans_detail_path, parse_dates=["date"])
        df_det.rename(columns={"date": "datetime"}, inplace=True)
        df_det["date"] = df_det["datetime"].dt.date
        df_det["heure"] = df_det["datetime"].dt.hour

        trans_daily = df_det.groupby("date").size().reset_index(name="nb_transactions")
        trans_daily["date"] = pd.to_datetime(trans_daily["date"])
        df_trans = pd.merge(df_trans, trans_daily, on="date", how="left")
        df_trans["nb_transactions"] = df_trans["nb_transactions"].fillna(0)

    return df_trans

def load_laverie1_data():
    return load_laverie_data("data_cleaned/laverie1", conso_path="data_cleaned/conso_elec_laverie.csv")

def load_laverie2_data():
    return load_laverie_data("data_cleaned/laverie2")
