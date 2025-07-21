import requests
import pandas as pd
import os

# 🔑 Clé API Visual Crossing
API_KEY = "WD9BYX5ETJR9LBA96HGRRKGCP"
ville = "Boulogne-sur-Mer,FR"

# 📆 Période souhaitée
start_date = "2024-01-01"
end_date = "2025-06-01"

# 🔗 Construction de l'URL API
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ville}/{start_date}/{end_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=json"

# 📥 Requête
response = requests.get(url)
data = response.json()

# 📊 Transformation en DataFrame
jours = data["days"]
df_meteo = pd.DataFrame(jours)

# 📌 Sélection des colonnes utiles
df_meteo = df_meteo[["datetime", "tempmax", "tempmin", "conditions", "precip"]]
df_meteo["datetime"] = pd.to_datetime(df_meteo["datetime"])

# 📁 Chemin de sortie
output_path = "data_external/donnees_API_meteo_boulogne.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 💾 Sauvegarde CSV
df_meteo.to_csv(output_path, index=False)
print("✅ Données météo Boulogne exportées :", output_path)
print(df_meteo.head())
