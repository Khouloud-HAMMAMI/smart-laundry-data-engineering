import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
import os

# Chargement des données CA
df_ca = pd.read_csv("./data_cleaned/laverie1/transactions_jour_cleaned.csv", parse_dates=["date"])
df_ca = df_ca.rename(columns={"date": "ds", "ca_tot": "y"}).dropna()

# Remplissage (on suppose que la moyenne quotidienne reflète le flux client)
df_remplissage = pd.read_csv("./data_cleaned/laverie1/remplissage_cleaned.csv", parse_dates=["datetime"])
df_remplissage["date"] = df_remplissage["datetime"].dt.date
df_remplissage_grouped = df_remplissage.groupby("date")["total"].mean().reset_index()
df_remplissage_grouped["ds"] = pd.to_datetime(df_remplissage_grouped["date"])
df_remplissage_grouped = df_remplissage_grouped[["ds", "total"]].rename(columns={"total": "remplissage_moyen"})

# Météo simulée : 30% des jours sont pluvieux
np.random.seed(42)
df_weather = pd.DataFrame({
    "ds": df_ca["ds"],
    "pluie": np.random.choice([0, 1], size=len(df_ca), p=[0.7, 0.3])
})

# Fusion des données
df = df_ca.merge(df_remplissage_grouped, on="ds", how="left")
df = df.merge(df_weather, on="ds", how="left")

# Variables temporelles
df["jour_semaine"] = df["ds"].dt.weekday
df["is_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
df["mois"] = df["ds"].dt.month

# Jours fériés
years = df["ds"].dt.year.unique()
fr_holidays = holidays.France(years=years)
holidays_df = pd.DataFrame({
    "ds": pd.to_datetime(list(fr_holidays.keys())),
    "holiday": list(fr_holidays.values())
})

# Split 80/20
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Initialisation du modèle
model = Prophet(holidays=holidays_df)
model.add_regressor("remplissage_moyen")
model.add_regressor("pluie")
model.add_regressor("is_weekend")

model.fit(train[["ds", "y", "remplissage_moyen", "pluie", "is_weekend"]])

# Préparation du futur
future = model.make_future_dataframe(periods=len(test))
features_future = df[["ds", "remplissage_moyen", "pluie", "is_weekend"]].set_index("ds")
future = future.set_index("ds").join(features_future).reset_index()

# Prédictions
forecast = model.predict(future)

# Évaluation
forecast_eval = forecast.set_index("ds").loc[test["ds"]]
y_true = test.set_index("ds")["y"]
y_pred = forecast_eval["yhat"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)

print("=== Évaluation du modèle enrichi ===")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"MAPE : {mape*100:.2f} %")

# Enregistrer graphique et CSV
os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
fig = model.plot(forecast)
plt.title("Prévision CA avec météo et remplissage")
plt.tight_layout()
fig.savefig("./3_eda_visualisation/graphs/forecast_ca_enrichi.png")

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("./outputs/forecast_ca_enrichi.csv", index=False)
