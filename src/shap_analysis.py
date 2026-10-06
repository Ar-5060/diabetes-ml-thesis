import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("data/diabetes.csv")


# Replace invalid zeros with NaN
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


# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Tuned Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=4,
    random_state=42
)


# Train model
model.fit(
    X_train,
    y_train
)


# SHAP Explainer
explainer = shap.TreeExplainer(model)


# Calculate SHAP values
shap_values = explainer.shap_values(X_test)


# Check SHAP shape
print("SHAP type:", type(shap_values))
print("SHAP shape:", np.array(shap_values).shape)


# Select class 1 (Diabetes)
shap_values_class1 = shap_values[:, :, 1]


print("SHAP analysis completed")


# SHAP Summary Plot
shap.summary_plot(
    shap_values_class1,
    X_test
)


# SHAP Feature Importance Bar Plot
shap.summary_plot(
    shap_values_class1,
    X_test,
    plot_type="bar"
)