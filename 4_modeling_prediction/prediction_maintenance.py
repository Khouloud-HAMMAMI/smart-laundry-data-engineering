import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data_final/merged_laverie1.csv")
df["date"] = pd.to_datetime(df["date"])
df["anomalie"] = df["nb_remplissages"].apply(lambda x: 1 if x > 0 else 0)

features = ["ca_tot", "nb_transactions", "total_rempli", "kWh", "euros"]
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=features + ["anomalie"], inplace=True)

X = df[features]
y = df["anomalie"]

if y.nunique() < 2:
    print("⚠️ Erreur : Il n'y a qu'une seule classe présente dans la variable cible.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "4_modeling_prediction/models/model_maintenance.pkl")

    y_pred = model.predict(X_test)

    print("=== Modèle Maintenance ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
