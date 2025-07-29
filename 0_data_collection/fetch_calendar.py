import pandas as pd
import holidays
from datetime import datetime, timedelta

# Paramètres
annee = 2024, 2030
zone = "B"  # Ex : Boulogne est en zone C
start_date = datetime(2024, 1, 1)
end_date = datetime(2030, 6, 1)

# Jours fériés
jours_feries = holidays.France(years=annee)
df_feries = pd.DataFrame(jours_feries.items(), columns=["date", "holiday_name"])
df_feries["type"] = "jour_férié"

# Vacances scolaires (API data.gouv.fr)
vacances_url = "https://data.education.gouv.fr/api/v2/catalog/datasets/fr-en-calendrier-scolaire/exports/json"
df_vacances = pd.read_json(vacances_url)
df_vacances = df_vacances[df_vacances["location"] == f"Zone {zone}"]
df_vacances = df_vacances[["start_date", "end_date", "description"]].dropna()

vacance_dates = []
for _, row in df_vacances.iterrows():
    current = pd.to_datetime(row["start_date"])
    end = pd.to_datetime(row["end_date"])
    while current <= end:
        vacance_dates.append({"date": current, "type": "vacances", "label": row["description"]})
        current += timedelta(days=1)

df_vacances_clean = pd.DataFrame(vacance_dates)
df_feries["label"] = df_feries["holiday_name"]

# Fusion
df_all = pd.concat([df_feries[["date", "type", "label"]], df_vacances_clean])
df_all = df_all.drop_duplicates(subset=["date"])

# Sauvegarde
df_all["date"] = pd.to_datetime(df_all["date"])
df_all.to_csv("data_external/calendrier-scolaire-Ferie.csv", index=False)
print("✅ Données calendrier enregistrées dans : calendrier-scolaire-Ferie.csv")
