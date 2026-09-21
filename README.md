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
├── week2/
│   ├── Housing.csv                             # Kaggle House Prices dataset
│   ├── Titanic-Dataset.csv                     # Titanic dataset for classification
│   ├── assignments_demo.py                     # Assignments 1 & 2 (Linear & Logistic Regression + Tree Models)
│   ├── house_price_prediction.py               # Mini Project 2 (House Price Prediction Pipeline)
│   ├── housing_predictions.csv                 # Model test predictions output
│   ├── predicted_vs_actual.png                 # Regression evaluation & residuals plots
│   └── Week_2_Assignments_and_MiniProject.ipynb # Interactive Jupyter Notebook
└── week3/
    ├── iris.csv                                # UCI Iris flower dataset
    ├── assignments_demo.py                     # Assignments 1 & 2 (K-Means, PCA, & Model Serialization)
    ├── iris_clustering_project.py              # Mini Project 3 (Iris Flower Clustering Pipeline)
    ├── iris_clustered_results.csv              # Clustered results with PCA coords & ground-truth comparison
    ├── iris_clustering_analysis.png            # Multi-panel evaluation & cluster visualization plots
    ├── kmeans_iris_model.joblib                # Serialized trained K-Means model
    ├── scaler_iris.joblib                      # Serialized StandardScaler
    ├── pca_iris.joblib                         # Serialized PCA transformer
    └── Week_3_Assignments_and_MiniProject.ipynb # Interactive Jupyter Notebook
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

## Week 3: Unsupervised Learning + Model Evaluation
- **Focus**: Clustering, Dimensionality Reduction, Scaling
- **Topics Covered**:
  - K-Means Clustering (`k-means++`, inertia, centroid analysis)
  - Principal Component Analysis (PCA) for Dimensionality Reduction (4D to 2D/3D & explained variance ratio)
  - Elbow Method and Silhouette Analysis for choosing optimal $K$
  - Feature Scaling (`StandardScaler`) for distance-based algorithms
  - Model Persistence: Saving and loading models & scalers using `joblib` and `pickle`
  - Assignments:
    1. Perform K-Means on Iris dataset and visualize clusters across feature pairs.
    2. Apply PCA to reduce dataset dimensions and analyze principal component loadings.
  - Mini Project 3: Iris Flower Clustering Project (`iris.csv`) with K-Means ($k=3$), optimal Hungarian matching, Contingency Matrix, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and comprehensive multi-panel visualization (`iris_clustering_analysis.png`).
