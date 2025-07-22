from sqlalchemy import create_engine
import pandas as pd


DB_HOST = "db.rtqlbeculijkyokemkis.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "0123SmartLandry"

CONN_STR = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR)

def load_csv_to_db(csv_path, table_name, schema="smart_laundry"):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False)
    print(f"✅ Tabelle {schema}.{table_name} mise à jour avec {len(df)} lignes.")

if __name__ == "__main__":
    load_csv_to_db("data_final/merged_laverie1.csv", "laverie1_daily")
    load_csv_to_db("data_final/merged_laverie2.csv", "laverie2_daily")
    load_csv_to_db("data_final/affluence_laverie1.csv", "affluence_hourly")
