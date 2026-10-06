import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc


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



# Scaling for Logistic Regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Define models

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



plt.figure(figsize=(8,6))


for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        y_prob = model.predict_proba(
            X_test_scaled
        )[:,1]


    else:

        model.fit(
            X_train,
            y_train
        )

        y_prob = model.predict_proba(
            X_test
        )[:,1]


    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison of ML Models"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    "results/ROC_comparison.png",
    dpi=300
)


plt.show()