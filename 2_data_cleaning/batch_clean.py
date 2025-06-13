import os
from clean import clean_transactions_transaction

# Liste des laveries à traiter
laveries = ['laverie1', 'laverie2']

for laverie in laveries:
    input_path = f"1_data_raw/{laverie}/transactions_transaction.csv"
    output_path = f"3_data_cleaned/{laverie}/transactions_transaction_cleaned.csv"
    
    print(f"--- Nettoyage de {laverie} ---")
    clean_transactions_transaction(input_path, output_path)
