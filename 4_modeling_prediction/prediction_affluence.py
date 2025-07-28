
# ✅ Modèle Affluence amélioré avec features météo + calendrier

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 🔹 Chargement des données horaires
df = pd.read_csv("data_final/affluence_laverie.csv", parse_dates=["date"])

# 🔹 Ajout de variables temporelles
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 🔹 Chargement du calendrier
df_cal = pd.read_csv("data_external/calendrier-scolaire-Ferie.csv", parse_dates=["date"])
df_cal["ferie"] = df_cal["type"] == "jour_férié"
df_cal["vacances"] = df_cal["type"] == "vacances_scolaires"
df_cal = df_cal.groupby("date")[["ferie", "vacances"]].max().reset_index()

# 🔹 Chargement des données météo
df_weather = pd.read_csv("data_external/donnees_API_meteo_bailleul.csv", parse_dates=["date"])
df_weather["precip"] = pd.to_numeric(df_weather["precip"], errors="coerce")
df_weather = df_weather[["date", "tempmax", "tempmin", "precip"]]

# 🔹 Fusion avec calendrier et météo
df = df.merge(df_cal, on="date", how="left")
df = df.merge(df_weather, on="date", how="left")
df.fillna({"ferie": False, "vacances": False, "precip": 0}, inplace=True)

# 🔹 Définition des variables d'entrée
features = ["heure", "day_of_week", "month", "is_weekend", "ferie", "vacances", "tempmax", "tempmin", "precip"]
df = df.dropna(subset=features + ["nb_demandes"])

X = df[features]
y = df["nb_demandes"]

# 🔹 Entraînement du modèle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 🔹 Évaluation
y_pred = model.predict(X_test)
print("=== Modèle Affluence Amélioré ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# 🔹 Sauvegarde du modèle
import joblib
joblib.dump(model, "4_modeling_prediction/models/model_affluence_v2.pkl")



# ✅ Modèle Affluence avec XGBoost

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 🔹 Chargement des données horaires
df = pd.read_csv("data_final/affluence_laverie.csv", parse_dates=["date"])

# 🔹 Ajout de variables temporelles
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 🔹 Chargement du calendrier
df_cal = pd.read_csv("data_external/calendrier-scolaire-Ferie.csv", parse_dates=["date"])
df_cal["ferie"] = df_cal["type"] == "jour_férié"
df_cal["vacances"] = df_cal["type"] == "vacances_scolaires"
df_cal = df_cal.groupby("date")[["ferie", "vacances"]].max().reset_index()

# 🔹 Chargement des données météo
df_weather = pd.read_csv("data_external/donnees_API_meteo_bailleul.csv", parse_dates=["date"])
df_weather["precip"] = pd.to_numeric(df_weather["precip"], errors="coerce")
df_weather = df_weather[["date", "tempmax", "tempmin", "precip"]]

# 🔹 Fusion avec calendrier et météo
df = df.merge(df_cal, on="date", how="left")
df = df.merge(df_weather, on="date", how="left")
df.fillna({"ferie": False, "vacances": False, "precip": 0}, inplace=True)

# 🔹 Définition des variables d'entrée
features = ["heure", "day_of_week", "month", "is_weekend", "ferie", "vacances", "tempmax", "tempmin", "precip"]
df = df.dropna(subset=features + ["nb_demandes"])

X = df[features]
y = df["nb_demandes"]

# 🔹 Entraînement du modèle XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 🔹 Évaluation
y_pred = model.predict(X_test)
print("=== Modèle Affluence - XGBoost ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# 🔹 Sauvegarde du modèle
import joblib
joblib.dump(model, "4_modeling_prediction/models/model_affluence_xgb.pkl")
