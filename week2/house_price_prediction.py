"""
Mini Project 2: House Price Prediction Model
=============================================
Dataset: Kaggle - House Prices Dataset (Housing.csv)

Tasks:
1. Train Linear Regression model
2. Predict house prices
3. Evaluate R² score (along with MSE, RMSE, MAE)
4. Plot predicted vs actual values and residuals (save to 'predicted_vs_actual.png')
5. Export predictions to 'housing_predictions.csv'
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    # -------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------
    print_section("STEP 1: Load House Prices Dataset")
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    data_path = os.path.join(current_dir, "Housing.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, "week2", "Housing.csv")
    if not os.path.exists(data_path):
        data_path = "Housing.csv"
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset successfully from '{data_path}'.")
    print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nPreview (First 5 Rows):")
    print(df.head())

    # -------------------------------------------------------------
    # 2. Data Exploration & Missing Values Check
    # -------------------------------------------------------------
    print_section("STEP 2: Data Exploration & Validation")
    missing = df.isnull().sum()
    print("Missing Values per Column:")
    print(missing)
    if missing.sum() == 0:
        print(">> No missing values detected! Clean dataset.")

    # -------------------------------------------------------------
    # 3. Data Preprocessing & Feature Encoding
    # -------------------------------------------------------------
    print_section("STEP 3: Data Preprocessing & Encoding")
    df_clean = df.copy()

    # Binary columns: 'yes' -> 1, 'no' -> 0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})
    print(f"Binary encoded columns: {binary_cols}")

    # Multi-class categorical: 'furnishingstatus'
    df_clean = pd.get_dummies(df_clean, columns=['furnishingstatus'], drop_first=True, dtype=int)
    print(f"Dummies created for 'furnishingstatus'. Current shape: {df_clean.shape}")

    # Define Feature Matrix (X) and Target Vector (y)
    X = df_clean.drop(columns=['price'])
    y = df_clean['price']
    feature_names = list(X.columns)
    print(f"\nFinal Feature Count: {len(feature_names)}")
    print(f"Features: {feature_names}")

    # -------------------------------------------------------------
    # 4. Train-Test Split
    # -------------------------------------------------------------
    print_section("STEP 4: Train / Test Split (80 / 20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set : {X_test.shape[0]} samples")

    # -------------------------------------------------------------
    # 5. Model Training (Linear Regression)
    # -------------------------------------------------------------
    print_section("STEP 5: Train Linear Regression Model")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model training complete.")
    print(f"Intercept (Base Price): ${model.intercept_:,.2f}")

    # Benchmark with Ridge and Random Forest
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 6. Predict House Prices
    # -------------------------------------------------------------
    print_section("STEP 6: Predict House Prices on Test Set")
    y_pred = model.predict(X_test)
    y_pred_ridge = ridge.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Sample comparison
    comparison_sample = pd.DataFrame({
        'Actual Price ($)': y_test.values[:5],
        'Predicted Price ($)': np.round(y_pred[:5], 2),
        'Residual ($)': np.round(y_test.values[:5] - y_pred[:5], 2),
        'Error (%)': np.round(np.abs(y_test.values[:5] - y_pred[:5]) / y_test.values[:5] * 100, 2)
    })
    print("Sample Test Predictions:")
    print(comparison_sample.to_string(index=False))

    # -------------------------------------------------------------
    # 7. Evaluate R² Score and Regression Metrics
    # -------------------------------------------------------------
    print_section("STEP 7: Model Evaluation (R² Score & Accuracy Metrics)")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"1. R² (Coefficient of Determination) : {r2:.4f} ({r2 * 100:.2f}% of price variance explained)")
    print(f"2. Mean Squared Error (MSE)          : {mse:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE)    : ${rmse:,.2f}")
    print(f"4. Mean Absolute Error (MAE)         : ${mae:,.2f}")

    # Model comparison table
    model_eval_df = pd.DataFrame({
        'Model': ['Linear Regression (OLS)', 'Ridge Regression', 'Random Forest Regressor'],
        'R² Score': [f"{r2:.4f}", f"{r2_score(y_test, y_pred_ridge):.4f}", f"{r2_score(y_test, y_pred_rf):.4f}"],
        'RMSE ($)': [f"${rmse:,.2f}", f"${np.sqrt(mean_squared_error(y_test, y_pred_ridge)):,.2f}", f"${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}"],
        'MAE ($)': [f"${mae:,.2f}", f"${mean_absolute_error(y_test, y_pred_ridge):,.2f}", f"${mean_absolute_error(y_test, y_pred_rf):,.2f}"]
    })
    print("\n--- Model Benchmark Comparison ---")
    print(model_eval_df.to_string(index=False))

    # Feature Importance (Coefficients)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient ($)': model.coef_
    }).sort_values(by='Coefficient ($)', ascending=False)
    print("\nTop Positive Price Drivers (Coefficients):")
    print(coef_df.head(5).to_string(index=False))

    # -------------------------------------------------------------
    # 8. Plot Predicted vs Actual Values & Residuals
    # -------------------------------------------------------------
    print_section("STEP 8: Generate Visualizations (Predicted vs Actual)")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Predicted vs Actual Scatter Plot
    sns.scatterplot(
        x=y_test / 1e6,
        y=y_pred / 1e6,
        ax=axes[0],
        color='#1f77b4',
        alpha=0.8,
        s=60,
        edgecolor='black'
    )
    
    # 45-degree perfect prediction line
    min_val = min(y_test.min(), y_pred.min()) / 1e6
    max_val = max(y_test.max(), y_pred.max()) / 1e6
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', lw=2, label='Perfect Fit (y = x)')
    
    # Trendline
    z = np.polyfit(y_test / 1e6, y_pred / 1e6, 1)
    p = np.poly1d(z)
    axes[0].plot(np.sort(y_test / 1e6), p(np.sort(y_test / 1e6)), color='#2ca02c', linestyle='-', lw=1.5, label='Regression Trend')

    axes[0].set_title(f"Predicted vs Actual House Prices (R² = {r2:.4f})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Actual Price (in Millions $)", fontsize=11)
    axes[0].set_ylabel("Predicted Price (in Millions $)", fontsize=11)
    axes[0].legend(loc='upper left')

    # Subplot 2: Residuals Distribution
    residuals = y_test - y_pred
    sns.histplot(residuals / 1e6, kde=True, ax=axes[1], color='#9467bd', bins=20)
    axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_title(f"Residuals Distribution (RMSE = ${rmse/1e6:.2f}M)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Prediction Error / Residuals (in Millions $)", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(current_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved evaluation visualization to: {plot_path}")

    # -------------------------------------------------------------
    # 9. Export Predictions
    # -------------------------------------------------------------
    print_section("STEP 9: Export Predictions")
    output_df = X_test.copy()
    output_df['actual_price'] = y_test.values
    output_df['predicted_price'] = np.round(y_pred, 2)
    output_df['residual'] = np.round(residuals.values, 2)
    
    predictions_csv_path = os.path.join(current_dir, "housing_predictions.csv")
    output_df.to_csv(predictions_csv_path, index=False)
    print(f"Saved test predictions to: {predictions_csv_path}")

    print_section("Mini Project 2 Executed Successfully!")

if __name__ == "__main__":
    main()
