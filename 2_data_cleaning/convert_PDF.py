import pdfplumber
import pandas as pd
import re

# Charger le PDF
pdf_path = "Bailleul-Alertes.pdf"
data = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        lines = page.extract_text().split("\n")
        for line in lines:
            # Filtrer les lignes d’alerte avec une date au début
            match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+)\s+(.*)", line)
            if match:
                date, heure, type_, alerte = match.groups()
                data.append([date, heure, type_, alerte])

# Convertir en DataFrame
df_alertes = pd.DataFrame(data, columns=["date", "heure", "type", "alerte"])

# Sauvegarder en CSV
df_alertes.to_csv("./data_cleaned/laverie1/alertes_cleaned.csv", index=False)
print("✅ Export terminé vers alertes_cleaned.csv")
