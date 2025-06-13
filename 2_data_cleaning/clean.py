import pandas as pd
import os

def clean_transactions_transaction(filepath, output_path):
    df = pd.read_csv(filepath, sep=None, engine="python")

    # Afficher les colonnes originales (pour debug)
    print("[DEBUG] Colonnes d'origine :", df.columns.tolist())

    # 1. Renommer les colonnes
    df.columns = [col.replace('\ufeff', '').strip().lower().replace(" ", "_") for col in df.columns]

    print("[DEBUG] Colonnes nettoyées :", df.columns.tolist())

    # 2. Supprimer lignes vides
    df = df.dropna(how='all')

    # 3. Conversion de la date
    if 'date/heure(europe/paris)' in df.columns:
        date_col = 'date/heure(europe/paris)'
    elif 'date_heure(europe/paris)' in df.columns:
        date_col = 'date_heure(europe/paris)'
    else:
        raise KeyError("La colonne de date est introuvable après nettoyage.")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 4. Nettoyage des montants
    for col in ['carte_bancaire', 'prix', 'insérée', 'rendue']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Nettoyage des types (pièces, billets)
    for col in ['pièce', 'billet', 'fidélitée']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 6. Supprimer les lignes sans date ou prix
    df = df.dropna(subset=[date_col, 'prix'])

    # 7. Supprimer les lignes avec Description ou Selection manquantes
    df = df.dropna(subset=['description', 'selection'])

    # 8. Nettoyage du type
    df['type'] = df['type'].astype(str).str.strip().str.lower()

    # 9. S'assurer que le dossier existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 10. Sauvegarde
    df.to_csv(output_path, index=False)
    print(f"[✓] Fichier nettoyé et sauvegardé dans {output_path}")

    return df

# Exemple d'utilisation
if __name__ == "__main__":
    df_cleaned = clean_transactions_transaction(
        "data/laverie1/transactions_transaction.csv",
        "data_cleaned/laverie1/transactions_transaction_cleaned.csv"
    )
