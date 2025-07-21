import requests
import pandas as pd

url = "https://data.education.gouv.fr/api/records/1.0/search/?dataset=fr-en-calendrier-scolaire&q=&rows=1000"

r = requests.get(url)
data = r.json()

vacances = []
for record in data["records"]:
    vacances.append({
        "description": record["fields"].get("description"),
        "start_date": record["fields"].get("start_date"),
        "end_date": record["fields"].get("end_date"),
        "zones": record["fields"].get("zones")
    })

df_vacances = pd.DataFrame(vacances)
df_vacances["start_date"] = pd.to_datetime(df_vacances["start_date"])
df_vacances["end_date"] = pd.to_datetime(df_vacances["end_date"])

df_vacances.to_csv("vacances_france.csv", index=False)
print(df_vacances.head())
