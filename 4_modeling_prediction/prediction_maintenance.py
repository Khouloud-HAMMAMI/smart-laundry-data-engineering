# ✅ prediction_maintenance.py - Version finale améliorée
import pandas as pd
import joblib
#from etl_laverie1 import load_laverie1_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Chargement des données enrichies
df = pd.read_csv("data_final/merged_laverie1.csv", parse_dates=["date"])

# Création de la cible : anomalie si remplissage > 0
df["anomalie"] = (df["nb_remplissages"] > 0).astype(int)

# Affichage distribution des classes
print("Distribution des classes :")
print(df["anomalie"].value_counts())

# Définition des variables explicatives
features = [
    "ca_tot", "nb_transactions", "total_rempli", "kWh",
    "nb_alertes_total", "nb_alertes_critiques", "nb_alertes_importantes",
    "nb_alertes_tube", "nb_alertes_choc",
    "nb_alertes_lecteur", "nb_alertes_defaut_monnaie", "nb_alertes_trop_plein"
]

# Vérification des colonnes manquantes
missing = [col for col in features if col not in df.columns]
if missing:
    raise KeyError(f"Colonnes manquantes dans le DataFrame : {missing}")

# Nettoyage
df.dropna(subset=features + ["anomalie"], inplace=True)

# Séparation
X = df[features]
y = df["anomalie"]

# Split des données AVANT oversampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 🔁 SMOTE pour équilibrer
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 🔍 Modèle XGBoost avec pondération
ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# ⚙️ Entraînement
model.fit(X_train_bal, y_train_bal)

# 📊 Validation croisée
scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring="f1")
print(f"\nF1-score (moyenne validation croisée 5-folds) : {scores.mean():.2f}")

# 🎯 Prédictions
y_pred = model.predict(X_test)

print("\n=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred))

# ✅ Sauvegarde du modèle
joblib.dump(model, "4_modeling_prediction/models/model_maintenance.pkl")

# 📈 Importance des variables
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Importance des variables dans le modèle de maintenance")
plt.tight_layout()
plt.show()
