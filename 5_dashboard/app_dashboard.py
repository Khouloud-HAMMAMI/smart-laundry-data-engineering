import streamlit as st
import pandas as pd

st.title("Dashboard Laverie 🧺")

# Charger les fichiers
affluence = pd.read_csv("outputs/affluence_preds.csv")
ca = pd.read_csv("outputs/ca_preds.csv")
anomalies = pd.read_csv("outputs/anomalies.csv")

# Visualisation
st.subheader("Prédiction Affluence")
st.line_chart(affluence[['heure', 'nb_demandes']].set_index('heure'))

st.subheader("Chiffre d'affaires")
st.line_chart(ca[['date', 'ca_tot_lav1']].set_index('date'))

st.subheader("Anomalies Remboursements")
st.dataframe(anomalies)
