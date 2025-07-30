import pandas as pd
import os
from pandas.tseries.offsets import MonthEnd

# 📁 Chemins
base_path = "data_cleaned/laverie1"
transaction_file = os.path.join(base_path, "transactions_transaction_cleaned.csv")
conso_file = "data_cleaned/conso_elec_laverie.csv"
output_file = "data_final/affluence_laverie1.csv"

# ✅ Chargement des transactions
df_trans = pd.read_csv(transaction_file, parse_dates=["date"])
df_trans.rename(columns={"date": "datetime"}, inplace=True)

# ➕ Extraction jour/heure
df_trans["date"] = df_trans["datetime"].dt.normalize()
df_trans["heure"] = df_trans["datetime"].dt.hour

# 🧮 Nombre de demandes par heure
affluence_hourly = df_trans.groupby(["date", "heure"]).size().reset_index(name="nb_demandes")

# ⚡️ Ajout consommation si dispo
if os.path.exists(conso_file):
    df_conso = pd.read_csv(conso_file, parse_dates=["date"])

    # Étendre la consommation mensuelle en journalier
    daily_conso_rows = []

    for _, row in df_conso.iterrows():
        start_date = row["date"]
        end_date = start_date + MonthEnd(0)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        daily_kWh = row["kWh"] / len(date_range)

        for day in date_range:
            daily_conso_rows.append({"date": day.normalize(), "kWh": daily_kWh})

    df_daily_conso = pd.DataFrame(daily_conso_rows)

    # 🔗 Fusion avec affluence horaire
    merged = pd.merge(affluence_hourly, df_daily_conso, on="date", how="left")

    # 🔁 Somme journalière des demandes
    total_demandes_per_day = merged.groupby("date")["nb_demandes"].transform("sum")

    # ⚖️ Répartition proportionnelle
    merged["kWh_heure"] = merged["kWh"] * (merged["nb_demandes"] / total_demandes_per_day)

    # ❌ Supprimer lignes sans estimation d'énergie
    merged = merged.dropna(subset=["kWh", "kWh_heure"])

    affluence_hourly = merged

# 💾 Sauvegarde
os.makedirs("data_final", exist_ok=True)
affluence_hourly.to_csv(output_file, index=False)
print("✅ Fichier 'affluence_laverie1.csv' mis à jour avec estimation horaire filtrée.")
