import os
from clean import clean_transactions_transaction,clean_transactions_jour,clean_remplissage,clean_releves
# Liste des laveries à traiter
laveries = ['laverie1', 'laverie2']

for laverie in laveries:
    input_path = f"data/{laverie}/transactions_transaction.csv"
    output_path = f"data_cleaned/{laverie}/transactions_transaction_cleaned.csv"
    print(f"--- Nettoyage de {laverie} ---")
    clean_transactions_transaction(input_path, output_path)

    input_path = f"data/{laverie}/transactions_jour.csv"
    output_path = f"data_cleaned/{laverie}/transactions_jour_cleaned.csv"
    print(f"--- Nettoyage de {laverie} ---")
    clean_transactions_jour(input_path, output_path)

    input_path = f"data/{laverie}/remplissages.csv"
    output_path = f"data_cleaned/{laverie}/remplissages_cleaned.csv"
    print(f"--- Nettoyage de {laverie} ---")
    clean_remplissage(input_path, output_path)

    input_path = f"data/{laverie}/releves.csv"
    output_path = f"data_cleaned/{laverie}/releves_cleaned.csv"
    print(f"--- Nettoyage de {laverie} ---")
    clean_releves(input_path, output_path)

   





