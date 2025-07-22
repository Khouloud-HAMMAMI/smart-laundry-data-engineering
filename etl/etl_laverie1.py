# ✅ Code complet et mis à jour pour laverie 1 et laverie 2

import pandas as pd
import os

def load_laverie_data(base_path, conso_path=None):
    """
    Charge et fusionne toutes les données nécessaires pour une laverie (1 ou 2).
    """
    # Chargement des transactions journalières
    df_trans = pd.read_csv(os.path.join(base_path, "transactions_jour_cleaned.csv"), parse_dates=["date"])

    # Consommation électrique (optionnelle)
    if conso_path and os.path.exists(conso_path):
        df_conso = pd.read_csv(conso_path, parse_dates=["date"])
        df_trans = pd.merge(df_trans, df_conso, on="date", how="left")
        df_trans["co2_estime"] = df_trans["kWh"] * 0.09
    else:
        df_trans["co2_estime"] = None

    # Chargement des alertes
    alertes_path = os.path.join(base_path, "alertes_cleaned.csv")
    if os.path.exists(alertes_path):
        df_alertes = pd.read_csv(alertes_path)
        df_alertes.columns = df_alertes.columns.str.strip().str.replace('"', '')
        if "Date" in df_alertes.columns and "Heure" in df_alertes.columns:
            df_alertes.rename(columns={"Date": "date", "Heure": "heure"}, inplace=True)
            df_alertes["datetime"] = pd.to_datetime(df_alertes["date"] + " " + df_alertes["heure"], errors="coerce")
            df_alertes["date"] = pd.to_datetime(df_alertes["date"], errors="coerce")
            df_alertes.dropna(subset=["datetime", "date"], inplace=True)

            # Nombre total d'alertes
            alertes_group = df_alertes.groupby("date").size().reset_index(name="nb_alertes")
            df_trans = pd.merge(df_trans, alertes_group, on="date", how="left")
            df_trans["nb_alertes"] = df_trans["nb_alertes"].fillna(0)

            # Alertes trop-plein
            df_trans["alerte_trop_plein"] = df_alertes["alerte"].str.contains("trop.*plein", case=False, na=False)
            trop_plein_daily = df_alertes[df_trans["alerte_trop_plein"]].groupby("date").size().reset_index(name="nb_trop_plein")
            df_trans = pd.merge(df_trans, trop_plein_daily, on="date", how="left")
            df_trans["nb_trop_plein"] = df_trans["nb_trop_plein"].fillna(0)

    # Chargement des remplissages
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

    # Transactions détaillées (pour affluence horaire)
    trans_detail_path = os.path.join(base_path, "transactions_transaction_cleaned.csv")
    if os.path.exists(trans_detail_path):
        df_det = pd.read_csv(trans_detail_path, parse_dates=["date"])
        df_det.rename(columns={"date": "datetime"}, inplace=True)
        df_det["date"] = df_det["datetime"].dt.date
        df_det["heure"] = df_det["datetime"].dt.hour

        # Transactions par heure
        trans_hourly = df_det.groupby(["date", "heure"]).size().reset_index(name="nb_demandes")
        trans_daily = df_det.groupby("date").size().reset_index(name="nb_transactions")
        trans_daily["date"] = pd.to_datetime(trans_daily["date"])
        df_trans = pd.merge(df_trans, trans_daily, on="date", how="left")
        df_trans["nb_transactions"] = df_trans["nb_transactions"].fillna(0)

    return df_trans


def load_laverie1_data():
    return load_laverie_data("data_cleaned/laverie1", conso_path="data_cleaned/conso_elec_laverie.csv")

def load_laverie2_data():
    return load_laverie_data("data_cleaned/laverie2")
