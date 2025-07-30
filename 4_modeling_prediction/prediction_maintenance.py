# ✅ prediction_maintenance.py - Version améliorée
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# === Chargement des données ===
df = pd.read_csv("data_final/merged_laverie1.csv", parse_dates=["date"])

# === Création de la cible : anomalie (si remplissage > 0)
df["anomalie_remplissage"] = (df["nb_remplissages"] > 0).astype(int)

# === Sélection des variables explicatives ===
features = [
    "ca_tot", "nb_transactions", "total_rempli", "kWh",
    "nb_alertes_total",
    "nb_alertes_tube", "nb_alertes_choc",
    "nb_alertes_lecteur", "nb_alertes_monnaie",
    "dayofweek", "month", "is_weekend"
]


# Nettoyage
df.dropna(subset=features + ["anomalie_remplissage"], inplace=True)

X = df[features]
y = df["anomalie_remplissage"]

# === Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Équilibrage
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# === Modèle
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# === Évaluation
print("F1-score cross-validation :", cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring="f1").mean())
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# === Sauvegarde
joblib.dump(model, "4_modeling_prediction/models/model_maintenance.pkl")

# === Importance des variables
importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Importance des variables (modèle maintenance)")
plt.tight_layout()
plt.show()
