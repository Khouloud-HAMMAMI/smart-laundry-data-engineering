
import pandas as pd

def split_data(file_path, date_column="date", split_date="2024-01-01"):
    df = pd.read_csv(file_path, parse_dates=[date_column])
    train = df[df[date_column] < split_date]
    test = df[df[date_column] >= split_date]
    return train, test
