"""
Week 2 Supervised Learning - Assignments 1 and 2
==================================================
Assignment 1: Build Linear Regression model on housing dataset (predict price).
Assignment 2: Train Logistic Regression on Titanic dataset for survival prediction.
Bonus Coverage: Comparison with Decision Tree and Random Forest Classifiers.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def print_separator(title: str):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def run_assignment_1():
    print_separator("ASSIGNMENT 1: Linear Regression on Housing Dataset (Predict Price)")
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dataset_path = os.path.join(current_dir, "Housing.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(current_dir, "week2", "Housing.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "Housing.csv"
    
    print(f"Loading Housing dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSample Data:")
    print(df.head(3))
    
    # 1. Feature Preprocessing
    df_processed = df.copy()
    
    # Binary categorical columns (yes/no -> 1/0)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
    # Multi-category column: furnishingstatus
    df_processed = pd.get_dummies(df_processed, columns=['furnishingstatus'], drop_first=True, dtype=int)
    
    # Define features (X) and target (y)
    X = df_processed.drop(columns=['price'])
    y = df_processed['price']
    
    print(f"\nFeatures ({X.shape[1]} total): {list(X.columns)}")
    print(f"Target variable: 'price'")
    
    # 2. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size:  {X_test.shape[0]} samples")
    
    # 3. Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # 4. Predict on Test Set
    y_pred = lr_model.predict(X_test)
    
    # 5. Evaluate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Squared Error (MSE)      : {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"Mean Absolute Error (MAE)     : {mae:,.2f}")
    print(f"R-squared (R²) Score          : {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    
    # Coefficients analysis
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    print(f"\nModel Intercept: {lr_model.intercept_:,.2f}")
    print("\nFeature Coefficients (Impact on Price):")
    print(coef_df.to_string(index=False))
    
    print("\nSample Predictions vs Actual:")
    sample_comparison = pd.DataFrame({
        'Actual Price': y_test.values[:5],
        'Predicted Price': np.round(y_pred[:5], 2),
        'Difference': np.round(y_pred[:5] - y_test.values[:5], 2)
    })
    print(sample_comparison.to_string(index=False))
    
    return lr_model

def run_assignment_2():
    print_separator("ASSIGNMENT 2: Logistic Regression on Titanic Dataset (Survival Prediction)")
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dataset_path = os.path.join(current_dir, "Titanic-Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(current_dir, "week2", "Titanic-Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "Titanic-Dataset.csv"
        
    print(f"Loading Titanic dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Data Cleaning & Feature Engineering
    df_clean = df.copy()
    
    # Impute missing values
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    # Select features
    # Drop columns not directly predictive in baseline model: PassengerId, Name, Ticket, Cabin
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df_clean[features].copy()
    y = df_clean['Survived']
    
    # Encode categorical features
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True, dtype=int)
    
    print(f"\nCleaned Features for Classification: {list(X.columns)}")
    print(f"Target Distribution (Survived): {dict(y.value_counts())}")
    
    # 2. Train-Test Split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Feature Scaling (important for Logistic Regression convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 5. Predict
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Logistic Regression Performance ---")
    print(f"Accuracy Score : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print("                Predicted Did Not Survive (0) | Predicted Survived (1)")
    print(f"Actual 0:                      {cm[0][0]:<14} | {cm[0][1]:<14}")
    print(f"Actual 1:                      {cm[1][0]:<14} | {cm[1][1]:<14}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive (0)', 'Survived (1)']))
    
    # -------------------------------------------------------------------------
    # BONUS: Decision Tree & Random Forest comparison (Week 2 Topics Covered)
    # -------------------------------------------------------------------------
    print_separator("TOPIC COMPARISON: Decision Tree & Random Forest Classifiers")
    
    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree (max_depth=4)', 'Random Forest (100 trees)'],
        'Accuracy': [f"{acc*100:.2f}%", f"{dt_acc*100:.2f}%", f"{rf_acc*100:.2f}%"],
        'ROC-AUC': [f"{roc_auc:.4f}", f"{roc_auc_score(y_test, dt_clf.predict_proba(X_test)[:, 1]):.4f}", f"{roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1]):.4f}"]
    })
    print(comparison_df.to_string(index=False))

def main():
    run_assignment_1()
    run_assignment_2()
    print_separator("Week 2 Assignments Completed Successfully!")

if __name__ == "__main__":
    main()
