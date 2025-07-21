import requests
import pandas as pd

# 🔑 Ta clé API
API_KEY = "WD9BYX5ETJR9LBA96HGRRKGCP"
ville = "Bailleul,FR"
start_date = "2024-06-01"
end_date = "2024-07-01"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ville}/{start_date}/{end_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=json"

response = requests.get(url)
data = response.json()

# Extraction des données journalières
jours = data["days"]
df_meteo = pd.DataFrame(jours)

# Colonnes utiles : datetime, tempmax, tempmin, conditions météo, etc.
df_meteo = df_meteo[["datetime", "tempmax", "tempmin", "conditions", "precip"]]
df_meteo["datetime"] = pd.to_datetime(df_meteo["datetime"])

# 🔽 Export en CSV
df_meteo.to_csv("donnees_meteo_bailleul.csv", index=False)
print(df_meteo.head())
