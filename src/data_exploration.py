import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetes.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(df["Outcome"].value_counts())

print("\n===== TARGET DISTRIBUTION (%) =====")
print(df["Outcome"].value_counts(normalize=True) * 100)
print("\n===== ZERO VALUE COUNT =====")

for column in df.columns:
    print(column, ":", (df[column] == 0).sum())