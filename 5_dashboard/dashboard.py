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
    st.success(f"⏰ Heures recommandées d'ouverture avec forte affluence : {suggestions}")

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
elif section == "💰 Optimisation CA":
    st.title("💰 Optimisation du chiffre d'affaires")
    st.write("### Évolution historique du chiffre d’affaires")
    st.line_chart(df.set_index("date")["ca_tot"])

    st.write("### Prédiction du CA")
    tempmax = st.number_input("Temp max", value=20)
    tempmin = st.number_input("Temp min", value=10)
    precip = st.number_input("Précipitations", value=0.0)
    jour = st.selectbox("Jour de la semaine", range(7), key="ca_jour")
    mois = st.selectbox("Mois", range(1, 13), key="ca_mois")
    ferie = st.checkbox("Jour férié", key="ca_ferie")
    vacances = st.checkbox("Vacances", key="ca_vacances")

    input_ca = pd.DataFrame([{
        "tempmax": tempmax, "tempmin": tempmin, "precip": precip,
        "day_of_week": jour, "month": mois,
        "ferie": int(ferie), "vacances": int(vacances),
        "is_weekend": int(jour in [5, 6])
    }])

    ca_pred = model_ca.predict(input_ca)[0]
    st.metric("💰 CA prédit", f"{ca_pred:.2f} €")

    st.write("### Moyens de paiement (réel)")
    if "type_paiement" in df.columns:
        st.plotly_chart(px.pie(df, names="type_paiement", title="Répartition des paiements"))

# === ⚡ Énergie & Durabilité ===
elif section == "⚡ Énergie & Durabilité":
    st.title("⚡ Analyse énergétique")
    st.write("### Consommation énergétique historique")
    st.line_chart(df.set_index("date")["kWh"])

    st.write("### Prédiction énergie consommée")
    tempmax = st.number_input("Temp max", value=20, key="e_tempmax")
    tempmin = st.number_input("Temp min", value=10, key="e_tempmin")
    ca = st.number_input("Chiffre d'affaires", value=150.0)
    nb_trans = st.number_input("Nb transactions", value=30)
    precip = st.number_input("Précipitations", value=0.0)

    input_energy = pd.DataFrame([{
        "tempmax": tempmax,
        "tempmin": tempmin,
        "ca_tot": ca,
        "nb_transactions": nb_trans,
        "precip": precip
    }])

    kWh_pred = model_energie.predict(input_energy)[0]
    st.metric("⚡ kWh prédit", f"{kWh_pred:.2f} kWh")

    st.write("### Corrélation CA / Énergie")
    st.plotly_chart(
        px.scatter(df, x="ca_tot", y="kWh", trendline="ols", title="CA vs. Consommation électrique")
    )
