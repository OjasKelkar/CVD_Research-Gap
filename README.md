Cardiovascular Disease Prediction Research

A Comparative Study of Machine Learning Models for Robust Cardiovascular Disease Prediction Under Missing Clinical Data

1. Research Objective

To investigate how missing clinical data and different imputation strategies affect machine learning performance and probability calibration in cardiovascular disease prediction.

2. Dataset

Dataset: UCI Cleveland Heart Disease Dataset

Source: UCI Machine Learning Repository

Instances: 303

Features: 13 clinical attributes

Task: Binary Classification

3. Experimental Design

Missingness Levels:

0%

10%

20%

30%

Imputation Methods:

Median Imputation

KNN Imputation

Iterative Imputation

4. Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

CatBoost

5. Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

PR-AUC

Brier Score

Calibration Curves

6. Validation Strategy

Stratified 5-Fold Cross-Validation

Two independent random seeds

Consistent evaluation across experimental configurations

Leakage-free preprocessing and imputation

7. Technologies Used

Python

NumPy

Pandas

Scikit-learn

XGBoost

CatBoost

Matplotlib

Seaborn

Jupyter Notebook

8. Research Workflow

Dataset acquisition and exploration

Data preprocessing

Missingness simulation

Missing value imputation

Machine learning model training

Performance evaluation

Probability calibration analysis

Comparative results and visualization

9. Research Contribution

This research aims to compare machine learning algorithms and imputation strategies to understand their robustness under different levels of missing clinical data, with particular emphasis on predictive performance and probability calibration.
