import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
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



# Base XGBoost

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)



# Parameter grid

param_grid = {

    "n_estimators": [100,200,300],

    "max_depth": [2,3,5],

    "learning_rate": [0.01,0.05,0.1],

    "subsample": [0.8,1.0],

    "colsample_bytree": [0.8,1.0]

}



# Grid Search

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)



# Training

grid_search.fit(
    X_train,
    y_train
)



print("Best Parameters:")
print(grid_search.best_params_)


print("\nBest CV AUC:")
print(grid_search.best_score_)



# Best model

best_xgb = grid_search.best_estimator_



# Prediction

y_pred = best_xgb.predict(X_test)

y_prob = best_xgb.predict_proba(X_test)[:,1]



print("\nTest Accuracy:")
print(accuracy_score(y_test,y_pred))


print("\nTest AUC:")
print(roc_auc_score(y_test,y_prob))


print("\nClassification Report:")
print(classification_report(y_test,y_pred))