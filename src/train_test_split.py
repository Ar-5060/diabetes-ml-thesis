import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/diabetes.csv")


# Replace invalid zeros
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)


# Median imputation
for column in zero_columns:
    df[column] = df[column].fillna(df[column].median())


# Separate X and y

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)