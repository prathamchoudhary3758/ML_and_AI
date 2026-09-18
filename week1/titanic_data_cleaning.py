"""
Mini Project 1: Titanic Survival Prediction - Data Cleaning Project
===================================================================
Tasks:
1. Clean missing data (Age, Embarked, Cabin)
2. Encode Sex, Embarked columns
3. Visualize age distribution (Matplotlib/Seaborn)
4. Output cleaned dataset as new CSV (titanic_cleaned.csv)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def print_section(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

def main():
    # -------------------------------------------------------------
    # 1. Load the Dataset
    # -------------------------------------------------------------
    print_section("STEP 1: Load Titanic Dataset")
    input_file = "Titanic-Dataset.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find {input_file}. Please ensure it is in the working directory.")
    
    df = pd.read_csv(input_file)
    print(f"Loaded '{input_file}' successfully.")
    print(f"Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

    # -------------------------------------------------------------
    # 2. Inspect Missing Data
    # -------------------------------------------------------------
    print_section("STEP 2: Inspect Missing Data")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": missing_count, "Percentage (%)": missing_pct.round(2)})
    print("Missing values summary:")
    print(missing_df[missing_df["Missing Values"] > 0])

    # Keep a copy of raw Age for distribution comparison
    raw_age = df['Age'].copy()

    # -------------------------------------------------------------
    # 3. Clean Missing Data
    # -------------------------------------------------------------
    print_section("STEP 3: Clean Missing Data")
    
    # 3.1 Age Imputation: Use median age (robust against outliers)
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"- 'Age' column: Imputed {missing_count['Age']} missing values with median ({age_median:.1f} years).")
    
    # 3.2 Embarked Imputation: Use mode (most frequent port)
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"- 'Embarked' column: Imputed {missing_count['Embarked']} missing values with mode ('{embarked_mode}').")
    
    # 3.3 Cabin Handling:
    # Cabin has over 77% missing values. We drop Cabin as it contains too many missing entries for direct modeling,
    # and drop non-predictive identifiers ('Ticket', 'Name').
    print(f"- 'Cabin' column: Contains {missing_pct['Cabin']:.1f}% missing values. Dropping 'Cabin' column.")
    df.drop(columns=['Cabin'], inplace=True)
    
    # Verify no missing values remain in essential columns
    print(f"Remaining null values count in cleaned subset: {df[['Age', 'Embarked', 'Fare']].isnull().sum().sum()}")

    # -------------------------------------------------------------
    # 4. Encode Categorical Columns (Sex & Embarked)
    # -------------------------------------------------------------
    print_section("STEP 4: Encode Categorical Features")
    
    # 4.1 Encode 'Sex' using LabelEncoder (female: 0, male: 1)
    le = LabelEncoder()
    df['Sex_Code'] = le.fit_transform(df['Sex'])
    print("- 'Sex' encoded with LabelEncoder:")
    for cls, val in zip(le.classes_, le.transform(le.classes_)):
        print(f"   {cls} -> {val}")

    # 4.2 Encode 'Embarked' using OneHotEncoder (C, Q, S)
    ohe = OneHotEncoder(sparse_output=False, dtype=int)
    embarked_encoded = ohe.fit_transform(df[['Embarked']])
    embarked_col_names = [f"Embarked_{cat}" for cat in ohe.categories_[0]]
    
    embarked_df = pd.DataFrame(embarked_encoded, columns=embarked_col_names, index=df.index)
    print(f"- 'Embarked' encoded with OneHotEncoder into columns: {embarked_col_names}")
    
    # Merge one-hot encoded columns
    df_cleaned = pd.concat([df, embarked_df], axis=1)

    # -------------------------------------------------------------
    # 5. Visualize Age Distribution (Matplotlib & Seaborn)
    # -------------------------------------------------------------
    print_section("STEP 5: Visualize Age Distribution")
    
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Titanic Dataset - Age Distribution Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Age Distribution Before vs After Imputation
    ax1 = axes[0, 0]
    sns.histplot(raw_age.dropna(), color="#3498db", kde=True, stat="density", label=f"Original (n={raw_age.dropna().count()})", ax=ax1, alpha=0.4, bins=30)
    sns.histplot(df['Age'], color="#e74c3c", kde=True, stat="density", label=f"Median Imputed (n={len(df)})", ax=ax1, alpha=0.3, bins=30)
    ax1.set_title("Age Distribution: Raw vs. Median Imputed", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right")
    
    # Plot 2: Age Distribution by Survival Status
    ax2 = axes[0, 1]
    sns.kdeplot(data=df[df['Survived'] == 0], x='Age', label="Did not survive (0)", color="#e74c3c", fill=True, alpha=0.3, ax=ax2)
    sns.kdeplot(data=df[df['Survived'] == 1], x='Age', label="Survived (1)", color="#2ecc71", fill=True, alpha=0.3, ax=ax2)
    ax2.set_title("Age KDE Distribution by Survival Status", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper right")
    
    # Plot 3: Age Distribution by Passenger Class
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='Pclass', y='Age', hue='Pclass', palette="Set2", legend=False, ax=ax3)
    ax3.set_title("Age Distribution across Passenger Classes (Pclass)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    ax3.set_ylabel("Age (years)")
    
    # Plot 4: Age Distribution by Sex and Survival
    ax4 = axes[1, 1]
    sns.violinplot(data=df, x='Sex', y='Age', hue='Survived', split=True, palette={0: "#e74c3c", 1: "#2ecc71"}, ax=ax4)
    ax4.set_title("Age Distribution by Gender & Survival", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Gender")
    ax4.set_ylabel("Age (years)")
    ax4.legend(title="Survived", loc="upper right")
    
    plt.tight_layout()
    output_plot_path = "age_distribution.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization plot to: {output_plot_path}")

    # -------------------------------------------------------------
    # 6. Output Cleaned Dataset as new CSV
    # -------------------------------------------------------------
    print_section("STEP 6: Output Cleaned Dataset as new CSV")
    
    # Select clean, structured features for modeling
    output_columns = [
        'PassengerId', 'Survived', 'Pclass', 'Sex', 'Sex_Code', 'Age', 
        'SibSp', 'Parch', 'Fare', 'Embarked', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ]
    df_export = df_cleaned[output_columns].copy()
    
    output_csv = "titanic_cleaned.csv"
    df_export.to_csv(output_csv, index=False)
    print(f"Cleaned dataset successfully written to: {output_csv}")
    print(f"Final Cleaned Dataset Shape: {df_export.shape[0]} rows, {df_export.shape[1]} columns")
    print(f"Total null values in cleaned dataset: {df_export.isnull().sum().sum()}")
    print("\nPreview of Cleaned Dataset:")
    print(df_export.head())

    print_section("MINI PROJECT 1 COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
