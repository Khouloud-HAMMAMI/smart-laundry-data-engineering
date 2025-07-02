import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays

# 1. Chargement des données
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
df["remplissage_moyen"].fillna(df["remplissage_moyen"].mean(), inplace=True)

# Features temporelles
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
df["ferie"] = df["ds"].isin(pd.to_datetime(list(fr_holidays.keys()))).astype(int)

# 2. Split semaine/weekend
df_week = df[df["is_weekend"] == 0].copy()
df_end = df[df["is_weekend"] == 1].copy()

def train_rf(df_segment, label):
    # Split train/test
    split_index = int(len(df_segment) * 0.8)
    train = df_segment.iloc[:split_index]
    test = df_segment.iloc[split_index:]

    # Features
    features = ["remplissage_moyen", "pluie", "vacances", "ferie"]
    X_train = train[features]
    y_train = train["y"]
    X_test = test[features]
    y_test = test["y"]

    # Modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n=== Évaluation RandomForest - {label} ===")
    print(f"MAE  : {mae:.2f} €")
    print(f"RMSE : {rmse:.2f} €")
    print(f"MAPE : {mape*100:.2f} %")

    # Sauvegarde prédictions
    results = test[["ds", "y"]].copy()
    results["y_pred"] = y_pred
    os.makedirs("./outputs", exist_ok=True)
    results.to_csv(f"./outputs/forecast_rf_{label}.csv", index=False)

    # Graphique
    plt.figure(figsize=(12, 5))
    plt.plot(results["ds"], results["y"], label="Vrai CA")
    plt.plot(results["ds"], results["y_pred"], label="Prévision RF", linestyle="--")
    plt.title(f"Prévision CA - RandomForest ({label})")
    plt.legend()
    plt.grid()
    os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
    plt.savefig(f"./3_eda_visualisation/graphs/forecast_rf_{label}.png")
    plt.close()

# 3. Exécution
train_rf(df_week, label="semaine")
train_rf(df_end, label="weekend")
