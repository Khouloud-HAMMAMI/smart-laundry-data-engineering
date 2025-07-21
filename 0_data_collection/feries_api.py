import requests
import pandas as pd
# API pour les jours fériés en France
url = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"
r = requests.get(url)
jours_feries = r.json()

df_feries = pd.DataFrame(list(jours_feries.items()), columns=["date", "nom"])
df_feries["date"] = pd.to_datetime(df_feries["date"])

df_feries.to_csv("jours_feries_fr.csv", index=False)
print(df_feries.head())
