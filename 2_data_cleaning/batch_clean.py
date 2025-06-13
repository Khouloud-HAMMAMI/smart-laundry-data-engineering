import os
from clean import clean_transactions_transaction

# Liste des laveries à traiter
laveries = ['laverie1', 'laverie2']

for laverie in laveries:
    input_path = f"1_data_raw/{laverie}/transactions_transaction.csv"
    output_path = f"3_data_cleaned/{laverie}/transactions_transaction_cleaned.csv"
    
    print(f"--- Nettoyage de {laverie} ---")
    clean_transactions_transaction(input_path, output_path)





from data_cleaned import clean_transactions_transaction
import os

RAW_DATA_PATH = "./1_data_ingestion/raw"
CLEANED_PATH = "./1_data_ingestion/cleaned"

laveries = ["laverie1", "laverie2"]

for laverie in laveries:
    input_file = os.path.join(RAW_DATA_PATH, laverie, "transactions_transaction.csv")
    output_file = os.path.join(CLEANED_PATH, laverie, "transactions_transaction_cleaned.csv")

    try:
        clean_transactions_transaction(input_file, output_file)
    except Exception as e:
        print(f"[!] Erreur pour {laverie} : {e}")
