# === prediction_energie.py ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# === Chargement des données
df = pd.read_csv("data_final/affluence_laverie1.csv", parse_dates=["date"])

# === Nettoyage : suppression des lignes sans kWh estimé
df = df.dropna(subset=["kWh_heure"])

# === Extraction des features temporelles
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# === Données d'entrée et cible
features = ["heure", "nb_demandes", "dayofweek", "month"]
target = "kWh_heure"

X = df[features]
y = df[target]

# === Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Modélisation
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# === Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("=== ✅ Modèle Énergie à partir de l'affluence ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# === Sauvegarde
os.makedirs("4_modeling_prediction/models", exist_ok=True)
joblib.dump(model, "4_modeling_prediction/models/model_energie_from_affluence.pkl")
print("✅ Modèle sauvegardé !")
