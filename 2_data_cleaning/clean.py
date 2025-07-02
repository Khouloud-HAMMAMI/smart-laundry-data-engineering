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

def clean_transactions_jour(input_path, output_path):
    # Lire le fichier CSV en détectant le bon séparateur
    df = pd.read_csv(input_path, sep=None, engine="python")

    # Nettoyer les noms de colonnes
    df.columns = [col.replace('\ufeff', '').strip().lower().replace(" ", "_")
                     .replace("(€)", "").replace("(\x80)", "")
                     .replace(".", "").replace("é", "e")
                     for col in df.columns]

    print("[DEBUG] Colonnes nettoyées :", df.columns.tolist())

    # Renommer pour consistance
    rename_map = {
        'date(europe/paris)': 'date',
        'ca_esp_': 'ca_esp',
        'ca_cb_': 'ca_cb',
        'jeton_': 'jeton',
        'fidelite_': 'fidelite',
        'remb_': 'remb',
        'ca_tot_': 'ca_tot'
    }
    df = df.rename(columns=rename_map)

    # Supprimer les lignes entièrement vides
    df = df.dropna(how='all')

    # Supprimer les lignes où toutes les colonnes sauf date sont manquantes
    df = df.dropna(subset=['ca_esp', 'ca_cb', 'jeton', 'fidelite', 'remb', 'ca_tot'], how='all')

    # Conversion des colonnes numériques (virgule → point)
    montant_cols = ['ca_esp', 'ca_cb', 'jeton', 'fidelite', 'remb', 'ca_tot']
    for col in montant_cols:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Supprimer les lignes sans date valide
    df = df.dropna(subset=['date'])

    # Répertoire de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder
    df.to_csv(output_path, index=False)
    print(f"[✓] Données nettoyées sauvegardées dans : {output_path}")

# Remplissage

from collections import Counter
def clean_remplissage(input_path, output_path):
    df_raw = pd.read_csv(input_path, header=None, sep=';', encoding='utf-8-sig')

    # Utiliser la première ligne comme noms de colonnes
    new_columns = df_raw.iloc[0].tolist()
    df = df_raw[1:].copy()
    df.columns = new_columns

    print("[DEBUG] Colonnes d'origine récupérées depuis la ligne 1 :", df.columns.tolist())

    # Nettoyage des noms de colonnes
    cleaned_columns = [str(col)
                       .replace('\ufeff', '')
                       .replace('(€)', '')
                       .replace('(\x80)', '')
                       .replace('*¹', '')
                       .replace('(', '')
                       .replace(')', '')
                       .strip()
                       .lower()
                       .replace(" ", "_")
                       .replace('.', '')
                       for col in df.columns]

    # Rendre les noms uniques
    counter = Counter()
    final_columns = []
    for col in cleaned_columns:
        if counter[col]:
            final_columns.append(f"{col}_{counter[col]}")
        else:
            final_columns.append(col)
        counter[col] += 1

    df.columns = final_columns
    print("[DEBUG] Colonnes après nettoyage et renommage unique :", df.columns.tolist())

    # Renommage lisible
    rename_map = {
        'infos_europe/paris': 'id',
        'infos_europe/paris_1': 'date',
        'infos_europe/paris_2': 'heure',
        'remplissage': 'total',
        'remplissage_1': 'trop_plein',
        'details_remplissage_nombre': 'deux_euros',
        'details_remplissage_nombre_1': 'un_euro',
        'details_remplissage_nombre_2': 'cinquante_centimes',
        'details_remplissage_nombre_3': 'vingt_centimes',
        'details_remplissage_nombre_4': 'dix_centimes',
        'etat_fdc': 'etat_avant',
        'etat_fdc_1': 'etat_apres',
    }

    df = df.rename(columns=rename_map)

   

    # Nettoyage : gestion des NaN
    # Option 1 : remplir par la moyenne globale
    df["remplissage_moyen"].fillna(df["remplissage_moyen"].mean(), inplace=True)
    

    # Conversion des montants
    montant_cols = ['total', 'trop_plein', 'etat_avant', 'etat_apres']
    for col in montant_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"[WARNING] Colonne manquante : {col}")

    # Conversion des pièces
    piece_cols = ['deux_euros', 'un_euro', 'cinquante_centimes', 'vingt_centimes', 'dix_centimes']
    for col in piece_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"[WARNING] Colonne manquante : {col}")

    # Fusion date + heure
    if 'heure' in df.columns and 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['heure'], errors='coerce')
        df.drop(columns=['date', 'heure'], inplace=True)
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        df.drop(columns=['date'], inplace=True)
    else:
        print("[ERREUR] Les colonnes 'date' et 'heure' sont absentes, impossible de créer 'datetime'.")
        return

    # Nettoyage des IDs
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')

    # Supprimer les lignes sans datetime
    if 'datetime' in df.columns:
        df = df.dropna(subset=['datetime'])
    else:
        print("[ERREUR] La colonne 'datetime' est absente, abandon du nettoyage.")
        return

    # Réorganisation des colonnes
    ordered_cols = ['id', 'datetime', 'total', 'trop_plein',
                    'deux_euros', 'un_euro', 'cinquante_centimes',
                    'vingt_centimes', 'dix_centimes',
                    'etat_avant', 'etat_apres']
    df = df[[col for col in ordered_cols if col in df.columns]]

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Données nettoyées sauvegardées dans : {output_path}")





def clean_releves(input_path, output_path):
    # Lecture brute
    df_raw = pd.read_csv(input_path, header=None, sep=';', encoding='utf-8-sig')

    # Extraire les vraies colonnes depuis la deuxième ligne
    new_columns = df_raw.iloc[1].tolist()
    df = df_raw[2:].copy()
    df.columns = new_columns

    print("[DEBUG] Colonnes brutes :", df.columns.tolist())

    # Nettoyage des noms de colonnes
    cleaned_columns = [str(col)
        .replace('\ufeff', '')
        .replace('(€)', '')
        .replace('(\x80)', '')
        .replace('*¹', '')
        .replace('*²', '')
        .replace('*³', '')
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .strip()
        .lower()
        .replace(" ", "_")
        for col in df.columns
    ]
    df.columns = cleaned_columns

    print("[DEBUG] Colonnes nettoyées :", df.columns.tolist())

    # Renommage clair
    rename_map = {
        'n°': 'id',  # <- corrigé ici
        'date': 'date',
        'heure': 'heure',
        'total': 'total_pieces',
        'pieces': 'pieces',
        'billets': 'billets',
        'chiffre_affaire': 'espèces',
        'chiffre_affaire_1': 'cb',
        'chiffre_affaire_2': 'jeton',
        'chiffre_affaire_3': 'rendu',
        'chiffre_affaire_4': 'total_ca',
        'fidelite': 'fidelite_util',
        'fidelite_1': 'fidelite_charge',
        'fdc': 'fdc'
    }

    df = df.rename(columns=rename_map)

    # Conversion numérique
    montant_cols = ['pieces', 'billets', 'espèces', 'cb', 'jeton', 'rendu',
                    'total_ca', 'fidelite_util', 'fidelite_charge', 'fdc']
    for col in montant_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"[WARNING] Colonne manquante : {col}")

    # Fusion date + heure
    if 'date' in df.columns and 'heure' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['heure'], errors='coerce')
        df.drop(columns=['date', 'heure'], inplace=True)
    else:
        print("[ERREUR] 'date' et/ou 'heure' manquante(s).")
        return

    # Nettoyage ID
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')

    # Suppression des lignes sans datetime
    df = df.dropna(subset=['datetime'])

    # Réorganisation des colonnes
    ordered = ['id', 'datetime'] + [col for col in montant_cols if col in df.columns]
    ordered = [col for col in ordered if col in df.columns]  # sécurité ajoutée
    print("[DEBUG] Colonnes finales disponibles :", df.columns.tolist())
    df = df[ordered]

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Données nettoyées sauvegardées dans : {output_path}")


# Exemple d'exécution
if __name__ == "__main__":
    clean_remplissage(
        "data/laverie1/remplissages.csv",
        "data_cleaned/laverie1/remplissages_cleaned.csv"
    )