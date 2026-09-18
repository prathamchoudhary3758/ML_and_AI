"""
Week 1 ML Fundamentals - Assignments 1, 2, and 3
=================================================
Assignment 1: Load a dataset using Pandas and summarize basic stats (.info(), .describe()).
Assignment 2: Handle missing data using mean/median imputation.
Assignment 3: Encode categorical variables using LabelEncoder & OneHotEncoder.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    # -------------------------------------------------------------
    # ASSIGNMENT 1: Load dataset & summarize basic statistics
    # -------------------------------------------------------------
    print_separator("ASSIGNMENT 1: Loading Dataset & Summarizing Basic Statistics")
    
    # 1. Load dataset using Pandas
    dataset_path = "Titanic-Dataset.csv"
    print(f"Loading dataset from: {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    
    # Preview first 5 rows
    print("\n[Preview - First 5 rows]:")
    print(df.head())
    
    # Dataset dimensions
    print(f"\n[Dataset Shape]: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Information about columns, datatypes, and non-null counts
    print("\n[Dataset Info (.info())]:")
    df.info()
    
    # Summary statistics for numerical variables
    print("\n[Numerical Statistics (.describe())]:")
    print(df.describe())
    
    # -------------------------------------------------------------
    # ASSIGNMENT 2: Handle missing data using mean/median imputation
    # -------------------------------------------------------------
    print_separator("ASSIGNMENT 2: Missing Data Handling with Mean/Median Imputation")
    
    # Focusing on 'Age' which has missing values
    print(f"Total missing values in 'Age' column: {df['Age'].isnull().sum()}")
    print(f"Original Age - Mean: {df['Age'].mean():.2f}, Median: {df['Age'].median():.2f}, Std: {df['Age'].std():.2f}")
    
    # Method A: Using Pandas fillna()
    df_pandas_mean = df.copy()
    df_pandas_median = df.copy()
    
    # Mean Imputation with Pandas
    df_pandas_mean['Age_Mean_Imputed'] = df_pandas_mean['Age'].fillna(df_pandas_mean['Age'].mean())
    # Median Imputation with Pandas
    df_pandas_median['Age_Median_Imputed'] = df_pandas_median['Age'].fillna(df_pandas_median['Age'].median())
    
    print("\n--- Pandas Imputation Results ---")
    print(f"After Mean Imputation   - Missing: {df_pandas_mean['Age_Mean_Imputed'].isnull().sum()}, Mean: {df_pandas_mean['Age_Mean_Imputed'].mean():.2f}, Std: {df_pandas_mean['Age_Mean_Imputed'].std():.2f}")
    print(f"After Median Imputation - Missing: {df_pandas_median['Age_Median_Imputed'].isnull().sum()}, Median: {df_pandas_median['Age_Median_Imputed'].median():.2f}, Std: {df_pandas_median['Age_Median_Imputed'].std():.2f}")

    # -------------------------------------------------------------
    # ASSIGNMENT 3: Encode categorical variables (LabelEncoder & OneHotEncoder)
    # -------------------------------------------------------------
    print_separator("ASSIGNMENT 3: Categorical Encoding (LabelEncoder & OneHotEncoder)")
    
    # 1. LabelEncoder (ideal for binary or ordinal categorical data like 'Sex')
    print("\n[1. LabelEncoder Demonstration on 'Sex' column]:")
    label_encoder = LabelEncoder()
    sex_encoded = label_encoder.fit_transform(df['Sex'])
    
    df_encoded = df[['Sex']].copy()
    df_encoded['Sex_LabelEncoded'] = sex_encoded
    print(df_encoded.drop_duplicates())
    print("Class mapping:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"  '{label}' -> {index}")

    # 2. OneHotEncoder from Scikit-Learn (ideal for nominal categories like 'Embarked')
    print("\n[2. OneHotEncoder Demonstration on 'Embarked' column]:")
    # Handling missing values in 'Embarked' first before encoding
    embarked_filled = df[['Embarked']].fillna('S')  # 'S' is mode
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    embarked_ohe = ohe.fit_transform(embarked_filled)
    feature_names = ohe.get_feature_names_out(['Embarked'])
    
    df_ohe = pd.DataFrame(embarked_ohe, columns=feature_names)
    print("One-Hot Encoded 'Embarked' features preview:")
    preview_ohe = pd.concat([embarked_filled, df_ohe], axis=1).drop_duplicates()
    print(preview_ohe)

    print_separator("ALL ASSIGNMENTS 1, 2, AND 3 COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
