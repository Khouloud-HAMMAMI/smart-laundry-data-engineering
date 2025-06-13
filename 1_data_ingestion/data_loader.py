

import pandas as pd
import os

# 1_data_ingestion/data_loader.py

def load_csv_safe(filepath, encodings=['utf-8', 'latin1'], separators=[',', ';', '\t']):
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                print(f"[✓] Fichier chargé : {os.path.basename(filepath)} — {df.shape[0]} lignes | encodage: {encoding} | sep: '{sep}'")
                return df
            except pd.errors.ParserError:
                continue
            except Exception as e:
                print(f"[!] Problème générique pour {filepath} : {e}")
    print(f"[✗] Impossible de parser {filepath}")
    return None


def load_all_data(base_path):
    data = {}
    fichiers = ['transactions_transaction.csv','transactions_jour.csv','remplissages.csv', 'releves.csv', 'alertes.csv']
    for fichier in fichiers:
        full_path = os.path.join(base_path, fichier)
        data[fichier.replace('.csv', '')] = load_csv_safe(full_path)
    return data
