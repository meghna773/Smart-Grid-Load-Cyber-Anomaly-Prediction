import pandas as pd

file_path = "data/bqeb_data_v1.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
for column in df.columns:
    print(column)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head().to_string())
