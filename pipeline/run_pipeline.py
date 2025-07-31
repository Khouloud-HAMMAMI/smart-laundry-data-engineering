
import os
from pipeline.merge_pipeline import merge_all_laverie1
from db_loader import load_csv_to_db

if __name__ == "__main__":
    # 1. Générer les données finales fusionnées
    df = merge_all_laverie1()
    output_path = "data_final/merged_laverie1.csv"
    df.to_csv(output_path, index=False)
    print("merged_laverie1.csv généré.")

    # 2. Charger dans Supabase
    load_csv_to_db(output_path, "laverie1_daily")

    # 3. Charger les autres jeux de données déjà prêts
    load_csv_to_db("data_final/affluence_laverie1.csv", "affluence_hourly")

    print("Pipeline terminé et base Supabase mise à jour.")
