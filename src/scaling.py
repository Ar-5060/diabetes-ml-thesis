import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset

df = pd.read_csv("data/diabetes.csv")


zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)


for column in zero_columns:
    df[column] = df[column].fillna(df[column].median())


# X and y

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("Before Scaling:")
print(X_train.head())


print("\nAfter Scaling:")
print(X_train_scaled[:5])