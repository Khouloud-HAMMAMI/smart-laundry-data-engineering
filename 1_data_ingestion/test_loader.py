# test_loader.py (exemple temporaire pour tester le script)
from data_loader import load_all_data

base_path = "./data/laverie1"
data = load_all_data(base_path)

transactions_df = data['transactions_transaction']
