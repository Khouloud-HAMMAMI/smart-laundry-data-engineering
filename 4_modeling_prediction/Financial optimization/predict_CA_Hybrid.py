import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
import os
import matplotlib.pyplot as plt

# ========== 1. Chargement des données ==========
df_ca = pd.read_csv("./data_cleaned/laverie1/transactions_jour_cleaned.csv", parse_dates=["date"])
df_ca = df_ca.rename(columns={"date": "ds", "ca_tot": "y"}).dropna()

df_remplissage = pd.read_csv("./data_cleaned/laverie1/remplissages_cleaned.csv", parse_dates=["datetime"])
df_remplissage["date"] = df_remplissage["datetime"].dt.date
df_remplissage_grouped = df_remplissage.groupby("date")["total"].mean().reset_index()
df_remplissage_grouped["ds"] = pd.to_datetime(df_remplissage_grouped["date"])
df_remplissage_grouped = df_remplissage_grouped[["ds", "total"]].rename(columns={"total": "remplissage_moyen"})

# Météo simulée
np.random.seed(42)
df_weather = pd.DataFrame({
    "ds": df_ca["ds"],
    "pluie": np.random.choice([0, 1], size=len(df_ca), p=[0.7, 0.3])
})

# Fusion
df = df_ca.merge(df_remplissage_grouped, on="ds", how="left")
df = df.merge(df_weather, on="ds", how="left")

# ========== 2. Nettoyage des NaN ==========
df["remplissage_moyen"].fillna(df["remplissage_moyen"].mean(), inplace=True)
df["pluie"].fillna(0, inplace=True)

# ========== 3. Features temporelles ==========
df["jour_semaine"] = df["ds"].dt.weekday
df["is_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)

# Vacances scolaires
vacances = pd.read_csv("./data_external/vacances_scolaires.csv", parse_dates=["date"])
vacances["vacances"] = 1
df = df.merge(vacances[["date", "vacances"]], left_on="ds", right_on="date", how="left")
df["vacances"].fillna(0, inplace=True)
df.drop(columns=["date"], inplace=True)

# Jours fériés
years = df["ds"].dt.year.unique()
fr_holidays = holidays.France(years=years)
holidays_df = pd.DataFrame({
    "ds": pd.to_datetime(list(fr_holidays.keys())),
    "holiday": list(fr_holidays.values())
})

# ========== 4. Split ==========
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ========== 5. Prophet pour tendance globale ==========
model_prophet = Prophet(holidays=holidays_df)
model_prophet.add_regressor("remplissage_moyen")
model_prophet.add_regressor("pluie")
model_prophet.add_regressor("vacances")
model_prophet.add_regressor("is_weekend")

model_prophet.fit(train_df[["ds", "y", "remplissage_moyen", "pluie", "vacances", "is_weekend"]])

# ========== 6. Prédiction Prophet ==========
future = model_prophet.make_future_dataframe(periods=len(test_df))
features_future = df.set_index("ds").loc[future["ds"]][["remplissage_moyen", "pluie", "vacances", "is_weekend"]]
future = future.set_index("ds").join(features_future).reset_index()

forecast_prophet = model_prophet.predict(future)

# ========== 7. Résidus pour RandomForest ==========
df_forecast = forecast_prophet[["ds", "yhat"]].merge(df, on="ds", how="left")
df_forecast["residual"] = df_forecast["y"] - df_forecast["yhat"]

# Données pour RF
features = ["remplissage_moyen", "pluie", "vacances", "is_weekend"]
df_rf = df_forecast.dropna(subset=["residual"])

X_train = df_rf.iloc[:split_index][features]
y_train = df_rf.iloc[:split_index]["residual"]
X_test = df_rf.iloc[split_index:][features]
y_true = df_rf.iloc[split_index:]["y"]

# ========== 8. Modèle RandomForest sur résidus ==========
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
residual_preds = rf.predict(X_test)

# ========== 9. Prédiction hybride ==========
yhat_prophet = df_forecast.iloc[split_index:]["yhat"]
yhat_hybrid = yhat_prophet + residual_preds

# ========== 10. Évaluation ==========
mae = mean_absolute_error(y_true, yhat_hybrid)
rmse = np.sqrt(mean_squared_error(y_true, yhat_hybrid))
mape = mean_absolute_percentage_error(y_true, yhat_hybrid)

print("\n=== Évaluation du modèle hybride (Prophet + RF) ===")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"MAPE : {mape*100:.2f} %")

# ========== 11. Sauvegarde ==========
os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
plt.figure(figsize=(10,5))
plt.plot(df_rf.iloc[split_index:]["ds"], y_true, label="Réel")
plt.plot(df_rf.iloc[split_index:]["ds"], yhat_hybrid, label="Hybride")
plt.legend()
plt.title("Prévision Hybride (Prophet + RF)")
plt.tight_layout()
plt.savefig("./3_eda_visualisation/graphs/forecast_hybride.png")

# Export CSV
output = pd.DataFrame({
    "ds": df_rf.iloc[split_index:]["ds"],
    "y_true": y_true,
    "yhat_hybrid": yhat_hybrid
})
output.to_csv("./outputs/forecast_hybride.csv", index=False)
