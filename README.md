# Explainable Machine Learning-Based Diabetes Prediction

## Overview

This repository contains the implementation of a machine learning framework for diabetes prediction using clinical features.

The study compares multiple classification algorithms and applies Explainable Artificial Intelligence (XAI) techniques using SHAP to interpret model predictions.

---

## Models Implemented

The following models were evaluated:

1. Logistic Regression
2. Random Forest
3. Optimized Random Forest
4. XGBoost
5. Optimized XGBoost


---

## Dataset

Dataset:
Pima Indians Diabetes Dataset

Features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age


Target:

Outcome
- 0: Non-diabetic
- 1: Diabetic


---

## Workflow

1. Data exploration
2. Data preprocessing
3. Missing value handling
4. Train-test splitting
5. Feature scaling
6. Model training
7. Hyperparameter optimization
8. Performance evaluation
9. SHAP-based explainability


---

## Results

Best performing model:

Optimized Random Forest

Accuracy:
76.62%


Important predictors:

1. Glucose
2. BMI
3. Age


---

## Explainable AI

SHAP was used to interpret model predictions and identify the contribution of individual clinical features.


---

## Installation

Clone repository:

git clone https://github.com/Ar-5060/diabetes-ml-thesis.git


Install dependencies:

pip install -r requirements.txt


Run:

python src/shap_analysis.py


---

## Requirements

Python >= 3.10

Libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib


---

## Author

Anisur Rahman
