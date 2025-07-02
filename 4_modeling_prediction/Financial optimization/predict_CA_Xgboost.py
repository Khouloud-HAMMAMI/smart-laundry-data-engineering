import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
import os
import matplotlib.pyplot as plt

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

# Variables temporelles
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
df["is_holiday"] = df["ds"].isin(fr_holidays).astype(int)

# 2. Séparation des segments
df_semaine = df[df["is_weekend"] == 0].copy()
df_weekend = df[df["is_weekend"] == 1].copy()

# 3. Fonction d'entraînement et d’évaluation
def train_xgb_segment(df_segment, label):
    split = int(len(df_segment) * 0.8)
    train = df_segment.iloc[:split]
    test = df_segment.iloc[split:]

    features = ["jour_semaine", "remplissage_moyen", "pluie", "vacances", "is_holiday"]
    X_train, y_train = train[features], train["y"]
    X_test, y_test = test[features], test["y"]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n=== Évaluation XGBoost - {label} ===")
    print(f"MAE  : {mae:.2f} €")
    print(f"RMSE : {rmse:.2f} €")
    print(f"MAPE : {mape * 100:.2f} %")

    # Visualisation
    os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(test["ds"], y_test, label="Vrai", color="blue")
    plt.plot(test["ds"], y_pred, label="Prédit", color="orange")
    plt.title(f"Prévision CA - XGBoost - {label}")
    plt.xlabel("Date")
    plt.ylabel("CA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./3_eda_visualisation/graphs/xgb_forecast_{label}.png")

    # Export CSV
    os.makedirs("./outputs", exist_ok=True)
    pd.DataFrame({
        "ds": test["ds"],
        "y_true": y_test,
        "y_pred": y_pred
    }).to_csv(f"./outputs/xgb_forecast_{label}.csv", index=False)

# 4. Entraînement et évaluation
train_xgb_segment(df_semaine, "semaine")
train_xgb_segment(df_weekend, "weekend")
