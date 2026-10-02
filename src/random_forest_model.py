import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


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


# X and y

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Training

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)


# Probability

y_prob = model.predict_proba(X_test)[:,1]


# Evaluation

print("Accuracy:",
      accuracy_score(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nAUC:",
      roc_auc_score(y_test, y_prob))


# Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")
print(importance)