import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

df = pd.read_csv("data_final/merged_laverie1.csv")
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

features = ["tempmax", "tempmin", "precip", "ferie", "vacances", "day_of_week", "month", "is_weekend"]
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=features + ["ca_tot"], inplace=True)

X = df[features]
y = df["ca_tot"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
joblib.dump(model, "4_modeling_prediction/models/model_ca.pkl")

y_pred = model.predict(X_test)
print("=== Modèle Chiffre d'affaires ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
