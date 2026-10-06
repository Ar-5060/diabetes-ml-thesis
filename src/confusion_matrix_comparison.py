import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


for column in zero_columns:
    df[column] = df[column].fillna(df[column].median())


# Features and target

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Scaling for Logistic Regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Models

models = {

"Logistic Regression":
LogisticRegression(),

"Random Forest":
RandomForestClassifier(
    n_estimators=100,
    random_state=42
),

"Tuned Random Forest":
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=4,
    random_state=42
),

"XGBoost":
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
),

"Tuned XGBoost":
XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

}



# Create plots

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14,9)
)


axes = axes.flatten()



for i, (name, model) in enumerate(models.items()):


    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )


    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Diabetes",
            "Diabetes"
        ]
    )


    disp.plot(
        ax=axes[i]
    )


    axes[i].set_title(name)



# Remove empty subplot

axes[-1].axis("off")


plt.tight_layout()


plt.savefig(
    "results/confusion_matrix_comparison.png",
    dpi=300
)


plt.show()