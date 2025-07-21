import pandas as pd
df_demo = pd.read_csv("data_external/tranche_age_bailleul.csv", sep=";", encoding="utf-8")
print(df_demo.head())
