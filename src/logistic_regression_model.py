import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


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


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# Model creation

model = LogisticRegression()


# Training

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)

# Probability prediction

y_prob = model.predict_proba(X_test)[:,1]


# AUC score

auc_score = roc_auc_score(y_test, y_prob)


print("\nAUC Score:", auc_score)


# ROC curve

fpr, tpr, threshold = roc_curve(
    y_test,
    y_prob
)


plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC={auc_score:.2f})"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# Accuracy

accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy)
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))
