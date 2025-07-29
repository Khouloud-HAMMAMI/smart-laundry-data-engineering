import requests
import pandas as pd

API_KEY = "WD9BYX5ETJR9LBA96HGRRKGCP"
ville = "Bailleul,FR"
start_date = "2024-01-01"
end_date = "2030-06-01"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ville}/{start_date}/{end_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=json"

print(f"Requête envoyée à : {url}")

response = requests.get(url)

if response.status_code != 200:
    print("Erreur lors de l'appel API :")
    print("Code HTTP :", response.status_code)
    print("Message :", response.text)
    exit()

data = response.json()

jours = data["days"]
df_meteo = pd.DataFrame(jours)

df_meteo = df_meteo[["date", "tempmax", "tempmin", "conditions", "precip"]]
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

df_meteo.to_csv("data_external/donnees_API_meteo_bailleul.csv", index=False)
print(df_meteo.head())
