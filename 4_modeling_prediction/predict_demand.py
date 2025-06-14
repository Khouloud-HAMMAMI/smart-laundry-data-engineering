import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import holidays
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# Chargement des données
df = pd.read_csv("./data_cleaned/laverie1/transactions_jour_cleaned.csv", parse_dates=["date"])
df = df.rename(columns={"date": "ds", "ca_tot": "y"})
df = df.dropna()

# Ajout de variables temporelles
df["jour_semaine"] = df["ds"].dt.weekday
df["is_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
df["mois"] = df["ds"].dt.month

# Jours fériés France
years = df["ds"].dt.year.unique()
fr_holidays = holidays.France(years=years)
holidays_df = pd.DataFrame({
    "ds": pd.to_datetime(list(fr_holidays.keys())),
    "holiday": list(fr_holidays.values())
})

# Initialiser le modèle avec holidays et régressors
model = Prophet(holidays=holidays_df)
model.add_regressor("is_weekend")
model.add_regressor("jour_semaine")
model.add_regressor("mois")

# Entraîner le modèle
model.fit(df[["ds", "y", "is_weekend", "jour_semaine", "mois"]])

# Générer les futures dates (2 ans)
future = model.make_future_dataframe(periods=730)
future["jour_semaine"] = future["ds"].dt.weekday
future["is_weekend"] = future["jour_semaine"].isin([5, 6]).astype(int)
future["mois"] = future["ds"].dt.month
# === 2. Split train/test pour évaluation ===
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Prédictions
forecast = model.predict(future)
# Extraire les valeurs correspondantes à la période de test
predicted_test = forecast.set_index("ds").loc[test["ds"]]
y_true = test.set_index("ds")["y"]
y_pred = predicted_test["yhat"]

# === 5. Évaluation ===
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)

print("=== Évaluation sur les données test ===")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"MAPE : {mape*100:.2f} %")
# Sauvegarder les résultats
os.makedirs("./outputs", exist_ok=True)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("./outputs/forecast_ca_2y_advanced.csv", index=False)

# Graphique
fig = model.plot(forecast)
plt.title("Prévision du CA avec variables externes")
plt.tight_layout()
os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
fig.savefig("./3_eda_visualisation/graphs/forecast_ca_advanced.png")

print("[✓] Modèle enrichi entraîné et prévisions enregistrées.")
