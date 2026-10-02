import pandas as pd
import numpy as np


# Load dataset
df = pd.read_csv("data/diabetes.csv")


# Columns where zero is invalid
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# Replace zero with NaN
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)


# Median imputation
for column in zero_columns:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


# Separate features and target

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


print("Feature Data (X):")
print(X.head())


print("\nTarget Data (y):")
print(y.head())


print("\nX shape:")
print(X.shape)


print("\ny shape:")
print(y.shape)