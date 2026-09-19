# ML and AI Course Work

This repository contains practical implementations, assignments, and mini-projects for the Machine Learning & AI course.

## Repository Structure

```
.
├── week1/
│   ├── Titanic-Dataset.csv                     # Raw Titanic dataset
│   ├── assignments_demo.py                     # Assignments 1, 2, & 3 (Exploration, Imputation, Encoding)
│   ├── titanic_data_cleaning.py                # Mini Project 1 (Titanic Data Cleaning Pipeline)
│   ├── titanic_cleaned.csv                     # Cleaned & encoded dataset
│   ├── age_distribution.png                    # Age distribution visualization
│   └── Week_1_Assignments_and_MiniProject.ipynb # Interactive Jupyter Notebook
└── week2/
    ├── Housing.csv                             # Kaggle House Prices dataset
    ├── Titanic-Dataset.csv                     # Titanic dataset for classification
    ├── assignments_demo.py                     # Assignments 1 & 2 (Linear & Logistic Regression + Tree Models)
    ├── house_price_prediction.py               # Mini Project 2 (House Price Prediction Pipeline)
    ├── housing_predictions.csv                 # Model test predictions output
    ├── predicted_vs_actual.png                 # Regression evaluation & residuals plots
    └── Week_2_Assignments_and_MiniProject.ipynb # Interactive Jupyter Notebook
```

## Week 1: ML Fundamentals + Data Preprocessing
- **Focus**: Basics of ML, Data Cleaning, Feature Engineering
- **Topics Covered**:
  - Importing & exploring datasets
  - Handling missing data (mean/median/mode imputation)
  - Encoding categorical variables (`LabelEncoder` & `OneHotEncoder`)
  - Mini Project 1: Titanic Survival Prediction - Data Cleaning

## Week 2: Supervised Learning (Regression & Classification)
- **Focus**: Model building, training, evaluation, and accuracy testing
- **Topics Covered**:
  - Linear Regression & Multiple Linear Regression
  - Logistic Regression for Binary Classification
  - Decision Trees & Random Forest Classifiers
  - Accuracy & Performance Metrics (MSE, RMSE, MAE, $R^2$, Confusion Matrix, ROC-AUC)
  - Mini Project 2: House Price Prediction Model (`Housing.csv`) with $R^2$ evaluation and Predicted vs Actual visualization
