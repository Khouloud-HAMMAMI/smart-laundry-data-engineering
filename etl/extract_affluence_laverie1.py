import pandas as pd
import os

# 📁 Chemins vers les fichiers
base_path = "data_cleaned/laverie1"
transaction_file = os.path.join(base_path, "transactions_transaction_cleaned.csv")
conso_file = "data_cleaned/conso_elec_laverie.csv"
output_file = "data_final/affluence_laverie1.csv"

# ✅ Chargement des transactions détaillées
df_trans = pd.read_csv(transaction_file, parse_dates=["date"])
df_trans.rename(columns={"date": "datetime"}, inplace=True)

# 🧮 Extraction des infos horaires
df_trans["date"] = df_trans["datetime"].dt.date
df_trans["heure"] = df_trans["datetime"].dt.hour

# 🔢 Nombre de demandes par heure
affluence_hourly = df_trans.groupby(["date", "heure"]).size().reset_index(name="nb_demandes")
affluence_hourly["date"] = pd.to_datetime(affluence_hourly["date"])

# ⚡️ Ajout estimation de consommation électrique horaire (facultatif)
if os.path.exists(conso_file):
    df_conso = pd.read_csv(conso_file, parse_dates=["date"])
    # ⚠️ Hypothèse : kWh réparti uniformément sur 24h
    df_conso["kWh_heure"] = df_conso["kWh"] / 24
    affluence_hourly = pd.merge(affluence_hourly, df_conso[["date", "kWh_heure"]], on="date", how="left")

# 💾 Sauvegarde
os.makedirs("data_final", exist_ok=True)
affluence_hourly.to_csv(output_file, index=False)
print("✅ Fichier 'affluence_laverie1.csv' généré avec succès.")
