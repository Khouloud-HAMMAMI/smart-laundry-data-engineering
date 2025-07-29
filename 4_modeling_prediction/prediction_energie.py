# ✅ prediction_energie.py - version améliorée
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Chargement des données
df = pd.read_csv("data_final/merged_laverie1.csv", parse_dates=["date"])

# ➕ Feature engineering temporel
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# ➕ Sélection de variables explicatives enrichies
features = [
    "tempmax", "tempmin", "precip",
    "ca_tot", "nb_transactions", "total_rempli",
    "nb_alertes_total", "co2_estime",
    "dayofweek", "month", "is_weekend"
]

# Nettoyage des colonnes
for col in features + ["kWh"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=features + ["kWh"], inplace=True)

# Données
X = df[features]
y = df["kWh"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔁 Modèle XGBoost
model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ Sauvegarde
joblib.dump(model, "4_modeling_prediction/models/model_energie_final.pkl")

# 📊 Prédiction
y_pred = model.predict(X_test)

# 📈 Évaluation
print("=== Modèle Énergie - Amélioré (XGBoost) ===")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²  : {r2_score(y_test, y_pred):.2f}")
