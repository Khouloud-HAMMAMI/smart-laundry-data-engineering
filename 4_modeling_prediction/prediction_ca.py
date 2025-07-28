import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# 🔹 Chargement des données
df = pd.read_csv("data_final/merged_laverie1.csv", parse_dates=["date"])

# 🔹 Variables temporelles
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 🔹 Encodage météo qualitatif
if "conditions" in df.columns:
    df["conditions"] = df["conditions"].astype(str)
    df = pd.get_dummies(df, columns=["conditions"])

# 🔁 Ajout de lag + rolling std
df = df.sort_values("date")
df["lag_ca_tot"] = df["ca_tot"].shift(1)
df["std_ca_3"] = df["ca_tot"].shift(1).rolling(window=3).std()

# 🔸 Indicateurs spéciaux
df["after_vacances"] = df["vacances"].shift(1).fillna(False).astype(int)
df["before_weekend"] = (df["day_of_week"] == 4).astype(int)

# 🔹 Sélection des features
features = [
    "tempmax", "tempmin", "precip", "ferie", "vacances",
    "day_of_week", "month", "is_weekend",
    "nb_transactions", "nb_remplissages", "kWh",
    "lag_ca_tot", "std_ca_3", "after_vacances", "before_weekend"
] + [col for col in df.columns if col.startswith("conditions_")]

# 🔹 Nettoyage
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=features + ["ca_tot"], inplace=True)

# 🔹 Split et entraînement
X = df[features]
y = df["ca_tot"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 🔹 Évaluation
y_pred = model.predict(X_test)
print("=== Modèle CA - XGBoost (rolling std + indicateurs) ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# 🔹 Sauvegarde
joblib.dump(model, "4_modeling_prediction/models/model_ca_xgb_final.pkl")
