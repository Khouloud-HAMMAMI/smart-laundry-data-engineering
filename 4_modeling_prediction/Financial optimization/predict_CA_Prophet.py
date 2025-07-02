import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
import os

# ========== 1. Chargement des données ==========
df_ca = pd.read_csv("./data_cleaned/laverie1/transactions_jour_cleaned.csv", parse_dates=["date"])
df_ca = df_ca.rename(columns={"date": "ds", "ca_tot": "y"}).dropna()

df_remplissage = pd.read_csv("./data_cleaned/laverie1/remplissages_cleaned.csv", parse_dates=["datetime"])
df_remplissage["date"] = df_remplissage["datetime"].dt.date
df_remplissage_grouped = df_remplissage.groupby("date")["total"].mean().reset_index()
df_remplissage_grouped["ds"] = pd.to_datetime(df_remplissage_grouped["date"])
df_remplissage_grouped = df_remplissage_grouped[["ds", "total"]].rename(columns={"total": "remplissage_moyen"})

np.random.seed(42)
df_weather = pd.DataFrame({
    "ds": df_ca["ds"],
    "pluie": np.random.choice([0, 1], size=len(df_ca), p=[0.7, 0.3])
})

# ========== 2. Fusion des données ==========
df = df_ca.merge(df_remplissage_grouped, on="ds", how="left")
df = df.merge(df_weather, on="ds", how="left")
df["remplissage_moyen"].fillna(df["remplissage_moyen"].mean(), inplace=True)

df["jour_semaine"] = df["ds"].dt.weekday
df["is_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)

# ========== 3. Ajout vacances scolaires ==========
vacances = pd.read_csv("./data_external/vacances_scolaires.csv", parse_dates=["date"])
vacances["vacances"] = 1
df = df.merge(vacances[["date", "vacances"]], left_on="ds", right_on="date", how="left")
df["vacances"].fillna(0, inplace=True)
df.drop(columns=["date"], inplace=True)

# ========== 4. Jours fériés ==========
years = df["ds"].dt.year.unique()
fr_holidays = holidays.France(years=years)
holidays_df = pd.DataFrame({
    "ds": pd.to_datetime(list(fr_holidays.keys())),
    "holiday": list(fr_holidays.values())
})

# ========== 5. Split train/test ==========
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ========== 6. Séparer semaine vs weekend ==========
train_semaine = train_df[train_df["is_weekend"] == 0]
test_semaine = test_df[test_df["is_weekend"] == 0]

train_weekend = train_df[train_df["is_weekend"] == 1]
test_weekend = test_df[test_df["is_weekend"] == 1]

# ========== 7. Fonction de prédiction ==========
def train_and_predict(train, test, label):
    model = Prophet(holidays=holidays_df)
    model.add_regressor("remplissage_moyen")
    model.add_regressor("pluie")
    model.add_regressor("vacances")

    model.fit(train[["ds", "y", "remplissage_moyen", "pluie", "vacances"]])

    # Génération future dates
    future = model.make_future_dataframe(periods=len(test), freq="D")

    # Ajout des colonnes futures manuellement
    remplissage_mean = df["remplissage_moyen"].mean()
    future["jour_semaine"] = future["ds"].dt.weekday
    future["is_weekend"] = future["jour_semaine"].isin([5, 6]).astype(int)
    future["remplissage_moyen"] = remplissage_mean
    future["pluie"] = np.random.choice([0, 1], size=len(future), p=[0.7, 0.3])

    vacances["date"] = pd.to_datetime(vacances["date"])
    future = future.merge(vacances[["date", "vacances"]], left_on="ds", right_on="date", how="left")
    future["vacances"].fillna(0, inplace=True)
    future.drop(columns=["date"], inplace=True)

    # Prédiction
    forecast = model.predict(future)

    # Évaluation
    common_dates = forecast["ds"].isin(test["ds"])
    forecast_eval = forecast[common_dates].set_index("ds")
    y_true = test.set_index("ds").loc[forecast_eval.index]["y"]
    y_pred = forecast_eval["yhat"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n=== Évaluation {label} ===")
    print(f"MAE  : {mae:.2f} €")
    print(f"RMSE : {rmse:.2f} €")
    print(f"MAPE : {mape * 100:.2f} %")

    # Enregistrement
    os.makedirs("./3_eda_visualisation/graphs", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    fig = model.plot(forecast)
    plt.title(f"Prévision CA - {label}")
    plt.tight_layout()
    fig.savefig(f"./3_eda_visualisation/graphs/forecast_ca_{label}.png")

    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(f"./outputs/forecast_ca_{label}_prophet.csv", index=False)

# ========== 8. Entraînement ==========
train_and_predict(train_semaine, test_semaine, label="semaine")
train_and_predict(train_weekend, test_weekend, label="weekend")
