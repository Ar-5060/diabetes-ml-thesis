import pandas as pd
import numpy as np


# Load dataset
df = pd.read_csv("data/diabetes.csv")


zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# Replace invalid zero values with NaN
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)


print("Before Imputation:")
print(df.isnull().sum())


# Median Imputation
for column in zero_columns:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


print("\nAfter Imputation:")
print(df.isnull().sum())


print("\nDataset shape:")
print(df.shape)