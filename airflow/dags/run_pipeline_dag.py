from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'smartlaundry',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'smartlaundry_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    run_pipeline = BashOperator(
        task_id='run_pipeline',
        bash_command='python /opt/airflow/dags/run_pipeline.py'
    )
