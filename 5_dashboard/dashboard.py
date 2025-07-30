# ✅ dashboard.py - Version enrichie avec heatmap, courbe horaire et recommandations

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import json
from datetime import datetime

# === Chargement des données et modèles ===
df = pd.read_csv("data_final/merged_laverie1.csv", parse_dates=["date"])
model_affluence = joblib.load("4_modeling_prediction/models/model_affluence_final.pkl")
model_ca = joblib.load("4_modeling_prediction/models/model_ca_xgb_final.pkl")
model_energie = joblib.load("4_modeling_prediction/models/model_energie_final.pkl")

# Liste des features utilisées pour la prédiction énergétique
energie_features = [
    "tempmax", "tempmin", "precip",
    "ca_tot", "nb_transactions", "total_rempli",
    "nb_alertes_total", "co2_estime",
    "dayofweek", "month", "is_weekend", "heure"
]

with open("4_modeling_prediction/models/affluence_features.json") as f:
    affluence_features = json.load(f)

# === Chargement météo et calendrier ===
cal = pd.read_csv("data_external/calendrier-scolaire-Ferie.csv", parse_dates=["date"])
cal["ferie"] = cal["type"] == "jour_férié"
cal["vacances"] = cal["type"] == "vacances_scolaires"
cal = cal.groupby("date")[["ferie", "vacances"]].max().reset_index()

weather = pd.read_csv("data_external/donnees_API_meteo_bailleul.csv", parse_dates=["date"])
weather["precip"] = pd.to_numeric(weather["precip"], errors="coerce")
weather = weather[["date", "tempmax", "tempmin", "precip"]]

st.set_page_config(page_title="Smart Laundry Dashboard", layout="wide")

# === Navigation ===
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Choisir une section :", ["📊 Optimisation Machines", "💰 Optimisation CA", "⚡ Énergie & Durabilité"])

# === 📊 Optimisation Machines ===
if section == "📊 Optimisation Machines":
    st.title("📊 Optimisation de l’usage des machines")
    st.subheader("Analyse de l'affluence (historique & prédiction)")

    df["jour"] = df["date"].dt.day_name()
    df["day_of_week"] = df["date"].dt.dayofweek
    if "heure" not in df.columns:
        df["heure"] = 12

    # 📊 Bar chart affluence par jour
    st.write("### Courbe de l'affluence journalière")
    aff = df.groupby("jour")["nb_transactions"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    st.bar_chart(aff)

    # 🔥 Heatmap jour vs heure
    st.write("### 🔥 Carte de chaleur : Affluence moyenne par heure et jour")
    heatmap_data = df.groupby(["day_of_week", "heure"])["nb_transactions"].mean().unstack()
    day_names = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
    heatmap_data.index = heatmap_data.index.map(day_names)
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Heure", y="Jour", color="Affluence moyenne"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="📅 Affluence moyenne par heure et jour de la semaine"
    )
    st.plotly_chart(fig)

    # 📈 Courbe affluence horaire
    st.write("### 📈 Affluence moyenne par heure (tous jours confondus)")
    hourly_avg = df.groupby("heure")["nb_transactions"].mean()
    fig_hour = px.line(x=hourly_avg.index, y=hourly_avg.values, labels={"x": "Heure", "y": "Affluence"}, title="📊 Courbe moyenne journalière")
    st.plotly_chart(fig_hour)

    # ✅ Recommandations horaires
    top_hours = hourly_avg.sort_values(ascending=False).head(3)
    suggestions = ", ".join([f"{int(h)}h" for h in top_hours.index])
    st.success(f"⏰ Heures avec forte affluence : {suggestions}")

    # 🔮 Prédiction affluence
    st.write("### Prédiction d'affluence")
    jour = st.selectbox("Jour de la semaine (0=lundi)", range(7))
    mois = st.selectbox("Mois", range(1, 13))
    heure = st.selectbox("Heure", range(0, 24))
    annee = st.number_input("Année", value=datetime.today().year, min_value=datetime.today().year, max_value=2026)

    date_pred = pd.to_datetime(f"{annee}-{mois:02d}-01") + pd.to_timedelta(jour, unit="D")
    meteo_row = weather[weather["date"] == date_pred]
    if meteo_row.empty:
        st.info(f"ℹ️ Aucune donnée météo pour le {date_pred.date()}. Valeurs par défaut utilisées.")
        meteo_vals = {"tempmax": 20, "tempmin": 10, "precip": 0.0}
    else:
        meteo_vals = meteo_row.iloc[0].to_dict()

    cal_row = cal[cal["date"] == date_pred]
    if cal_row.empty:
        st.info(f"ℹ️ Aucun événement férié ou vacances enregistré le {date_pred.date()}.")
        cal_vals = {"ferie": 0, "vacances": 0}
    else:
        cal_data = cal_row.iloc[0]
        cal_vals = {
            "ferie": int(cal_data["ferie"]),
            "vacances": int(cal_data["vacances"])
        }
        if cal_vals["ferie"]:
            st.warning(f"⚠️ Le {date_pred.date()} est un jour férié.")
        elif cal_vals["vacances"]:
            st.info(f"📅 Le {date_pred.date()} est pendant les vacances scolaires.")

    input_data = {
        "heure": heure,
        "day_of_week": jour,
        "month": mois,
        "is_weekend": int(jour in [5, 6]),
        "ferie": cal_vals["ferie"],
        "vacances": cal_vals["vacances"],
        "tempmax": meteo_vals["tempmax"],
        "tempmin": meteo_vals["tempmin"],
        "precip": meteo_vals["precip"]
    }
    pred_input = pd.DataFrame([input_data])[affluence_features]
    pred_affluence = model_affluence.predict(pred_input)[0]
    st.metric("🔮 Prédiction affluence", f"{int(pred_affluence)} clients")


    # === 💰 Optimisation CA ===
if section == "💰 Optimisation CA":
    st.title("💰 Optimisation du chiffre d'affaires")
    st.write("### Évolution historique du chiffre d’affaires")
    st.line_chart(df.set_index("date")["ca_tot"])

    st.write("### Prédiction du CA")
    jour = st.selectbox("Jour de la semaine (0=lundi)", range(7), key="ca_jour")
    mois = st.selectbox("Mois", range(1, 13), key="ca_mois")
    annee = st.number_input("Année", value=datetime.today().year, min_value=datetime.today().year, max_value=2026)

    date_pred = pd.to_datetime(f"{annee}-{mois:02d}-01") + pd.to_timedelta(jour, unit="D")

    meteo_row = weather[weather["date"] == date_pred]
    meteo_vals = meteo_row.iloc[0].to_dict() if not meteo_row.empty else {"tempmax": 20, "tempmin": 10, "precip": 0.0}

    cal_row = cal[cal["date"] == date_pred]
    cal_vals = {"ferie": 0, "vacances": 0}
    if not cal_row.empty:
        cal_vals = {
            "ferie": int(cal_row.iloc[0]["ferie"]),
            "vacances": int(cal_row.iloc[0]["vacances"])
        }

    input_ca = pd.DataFrame([{
        "tempmax": meteo_vals["tempmax"],
        "tempmin": meteo_vals["tempmin"],
        "precip": meteo_vals["precip"],
        "day_of_week": jour,
        "month": mois,
        "ferie": cal_vals["ferie"],
        "vacances": cal_vals["vacances"],
        "is_weekend": int(jour in [5, 6])
    }])

    # Ajout des colonnes encodées si manquantes (conditions météo)
    conditions_cols = [col for col in model_ca.get_booster().feature_names if col.startswith("conditions_")]
    for cond in conditions_cols:
        if cond not in input_ca.columns:
            input_ca[cond] = 0

    # Colonnes manquantes par défaut (features non disponibles lors de la prédiction)
    for col in model_ca.get_booster().feature_names:
        if col not in input_ca.columns:
            input_ca[col] = 0

    input_ca = input_ca[model_ca.get_booster().feature_names]
    ca_pred = model_ca.predict(input_ca)[0]
    st.metric("💰 CA prédit", f"{ca_pred:.2f} €")

    # 💳 Moyens de paiement
    if "type_paiement" in df.columns:
        st.write("### Moyens de paiement")
        st.plotly_chart(px.pie(df, names="type_paiement", title="Répartition des paiements"))

    # 📈 CA moyen par jour de semaine
    st.write("### CA moyen par jour de la semaine")
    df["jour"] = df["date"].dt.day_name()
    jour_ca = df.groupby("jour")["ca_tot"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    st.bar_chart(jour_ca)

    # 💡 Recommandation heures creuses
    st.write("### 💡 Heures creuses pour promotions")
    if "heure" not in df.columns:
        df["heure"] = 12
    heure_moy = df.groupby("heure")["nb_transactions"].mean().sort_values()
    heures_creuses = heure_moy.head(3).index.tolist()
    creux = ", ".join([f"{h}h" for h in heures_creuses])
    st.info(f"Proposez des tarifs réduits à : {creux} (heures de faible affluence)")



# === ⚡ Énergie & Durabilité ===
if section == "⚡ Énergie & Durabilité":
    st.title("⚡️ Analyse énergétique")

    st.subheader("✅ Historique de consommation (horaire)")
    try:
        df_energy = pd.read_csv("data_final/affluence_laverie1.csv", parse_dates=["date"])
        st.line_chart(df_energy.groupby("date")["kWh"].sum())
    except Exception as e:
        st.error(f"Erreur de chargement des données énergétiques horaires : {e}")

    st.subheader("🔮 Prédiction horaire de consommation énergétique")

    heure = st.selectbox("Heure", list(range(24)), key="energie_heure")
    jour = st.selectbox("Jour de la semaine (0=lundi)", list(range(7)), key="energie_jour")
    mois = st.selectbox("Mois", list(range(1, 13)), key="energie_mois")
    annee = st.number_input("Année", min_value=2023, max_value=2026, value=datetime.today().year, key="energie_annee")

    date_pred = pd.to_datetime(f"{annee}-{mois:02d}-01") + pd.to_timedelta(jour, unit="D")

    input_energy = pd.DataFrame([{
        "date": date_pred,
        "heure": heure,
        "dayofweek": jour,
        "month": mois,
        "is_weekend": int(jour in [5, 6])
    }])

    # Charger le nombre de demandes depuis le CSV
    affluence_df = pd.read_csv("data_final/affluence_laverie1.csv", parse_dates=["date"])
    match = affluence_df[(affluence_df["date"] == date_pred) & (affluence_df["heure"] == heure)]
    if not match.empty:
        input_energy["nb_demandes"] = match["nb_demandes"].values[0]
    else:
        input_energy["nb_demandes"] = affluence_df["nb_demandes"].mean()

    # Charger le modèle
    model_energy_aff = joblib.load("4_modeling_prediction/models/model_energie_from_affluence.pkl")
    input_energy = input_energy[model_energy_aff.feature_names_in_]

    # Prédiction
    kWh_pred = model_energy_aff.predict(input_energy)[0]
    co2 = kWh_pred * 0.1  # 💡 Exemple : 0.1 kg CO₂ par kWh

    st.metric("🔋 kWh prédits", f"{kWh_pred:.2f} kWh")
    st.metric("♻️ Empreinte CO₂ estimée", f"{co2:.2f} kg")
        # === 📈 Corrélation entre affluence et consommation énergétique ===
    st.subheader("📈 Corrélation : Affluence vs Consommation")
    try:
        fig_corr = px.scatter(
            affluence_df,
            x="nb_demandes",
            y="kWh_heure",
            trendline="ols",
            title="Nombre de demandes vs. Consommation horaire (kWh)"
        )
        st.plotly_chart(fig_corr)
    except Exception as e:
        st.error(f"Erreur dans la génération du graphe de corrélation : {e}")

    # === 🔍 Heures les plus énergivores (en moyenne) ===
    st.subheader("💸 Heures les plus énergivores")
    try:
        conso_par_heure = affluence_df.groupby("heure")["kWh_heure"].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            conso_par_heure,
            labels={"value": "kWh moyen", "heure": "Heure"},
            title="Consommation moyenne par heure"
        )
        st.plotly_chart(fig_bar)

        # Estimation coût (facultatif, basé sur 0.15 €/kWh)
        affluence_df["coût_estime"] = affluence_df["kWh_heure"] * 0.15
        total_cost = affluence_df["coût_estime"].sum()
        st.info(f"💰 Coût énergétique estimé total : {total_cost:.2f} €")
    except Exception as e:
        st.error(f"Erreur dans l'analyse horaire : {e}")

    # === ✅ Recommandations durables ===
    st.subheader("♻️ Recommandations durables")
    try:
        heures_faible_conso = conso_par_heure.sort_values().head(3).index.tolist()
        heures_forte_conso = conso_par_heure.sort_values(ascending=False).head(3).index.tolist()
        faibles = ", ".join([f"{h}h" for h in heures_faible_conso])
        fortes = ", ".join([f"{h}h" for h in heures_forte_conso])

        st.success(f"✅ Planifiez les cycles non urgents à : {faibles} (heures de faible consommation)")
        st.warning(f"⚠️ Heures à éviter pour limiter la charge énergétique : {fortes}")
    except Exception as e:
        st.error(f"Erreur dans les recommandations : {e}")

