"""
Week 3 Unsupervised Learning - Assignments 1 & 2 + Model Persistence
===================================================================
Assignment 1: Perform K-Means on Iris dataset and visualize clusters.
Assignment 2: Apply PCA (Principal Component Analysis) to reduce dataset dimensions.
Core Topic Demonstration: Model saving and loading using joblib and pickle.
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def print_separator(title: str):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def get_iris_data():
    """Helper to locate and load Iris dataset reliably."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris.csv") if "__file__" in globals() else "week3/iris.csv",
        "week3/iris.csv",
        "iris.csv",
        "iris.data.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "iris.data.csv") if "__file__" in globals() else "../iris.data.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            if "data" in os.path.basename(path):
                columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
                df = pd.read_csv(path, header=None, names=columns).dropna(how='all')
            else:
                df = pd.read_csv(path)
            return df
    raise FileNotFoundError("Could not find Iris dataset in search paths.")

def run_assignment_1():
    print_separator("ASSIGNMENT 1: K-Means Clustering on Iris Dataset & Elbow Method")

    df = get_iris_data()
    print(f"Loaded Iris Dataset successfully! Shape: {df.shape[0]} samples, {df.shape[1]} columns")
    print("\nSample Rows:")
    print(df.head(4))

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols].values
    y_true = df['species'].values

    # Step 1: Feature Scaling
    # K-Means computes Euclidean distance: unscaled features with larger variance dominate clustering.
    print("\n[Step 1] Feature Scaling with StandardScaler:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Scaled Features Mean (approx 0): {np.round(X_scaled.mean(axis=0), 4)}")
    print(f"  Scaled Features Std  (approx 1): {np.round(X_scaled.std(axis=0), 4)}")

    # Step 2: Elbow Method & Silhouette Scores to Find Optimal K
    print("\n[Step 2] Evaluating Elbow Method (Inertia / WCSS) & Silhouette Scores:")
    k_range = range(1, 11)
    inertias = []
    silhouette_scores = {}

    print(f"{'K (Clusters)':<15} | {'Inertia (WCSS)':<18} | {'Silhouette Score':<18}")
    print("-" * 57)
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        
        if k >= 2:
            sil = silhouette_score(X_scaled, km.labels_)
            silhouette_scores[k] = sil
            print(f"{k:<15} | {km.inertia_:<18.2f} | {sil:<18.4f}")
        else:
            print(f"{k:<15} | {km.inertia_:<18.2f} | {'N/A (k=1)':<18}")

    print("\nElbow Observation:")
    print("  Notice the sharp drop from k=1 to k=2 and k=3, where the curve forms an 'elbow'.")
    print("  k=3 aligns with both the dataset's biological reality (3 species) and strong cluster separation.")

    # Step 3: Train Final K-Means with K=3
    print("\n[Step 3] Training K-Means Model with k=3:")
    kmeans_3 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans_3.fit_predict(X_scaled)

    # Compute Clustering Evaluation Metrics (Unsupervised)
    sil_final = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    print(f"  Inertia (WCSS)             : {kmeans_3.inertia_:.2f}")
    print(f"  Silhouette Score           : {sil_final:.4f} (Higher is better, range [-1, 1])")
    print(f"  Calinski-Harabasz Index    : {ch_score:.2f} (Higher is better)")
    print(f"  Davies-Bouldin Index       : {db_score:.4f} (Lower is better)")

    # Display Cluster Centroids in Original Units (Inverse Transformed)
    centroids_scaled = kmeans_3.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroids_df.index = [f"Cluster {i}" for i in range(3)]
    print("\nLearned Cluster Centroids (in original feature measurements):")
    print(centroids_df.round(2))

    # Cluster distribution
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("\nSamples per Cluster:")
    for cluster_id, count in counts.items():
        print(f"  Cluster {cluster_id}: {count} flowers")

def run_assignment_2():
    print_separator("ASSIGNMENT 2: Principal Component Analysis (PCA) Dimensionality Reduction")

    df = get_iris_data()
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols].values

    # Step 1: Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA (Full 4 components to evaluate variance)
    pca_full = PCA(n_components=4, random_state=42)
    pca_full.fit(X_scaled)
    var_ratios = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(var_ratios)

    print("[Step 1] Full Dimensionality Decomposition (4 Principal Components):")
    for i, (var, cum) in enumerate(zip(var_ratios, cumulative_var), start=1):
        print(f"  PC{i}: Explained Variance = {var*100:6.2f}% | Cumulative Variance = {cum*100:6.2f}%")

    # Step 3: Reduce to 2 Principal Components
    print("\n[Step 2] Dimensionality Reduction: 4D -> 2D (PC1 & PC2):")
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)

    total_2d_var = np.sum(pca_2d.explained_variance_ratio_) * 100
    print(f"  PC1 + PC2 retain {total_2d_var:.2f}% of total data variance!")
    print(f"  Reduced feature matrix shape: {X_pca.shape}")

    # PCA Component Loadings / Eigenvectors
    loadings = pd.DataFrame(
        pca_2d.components_,
        columns=feature_cols,
        index=['PC1', 'PC2']
    )
    print("\nPrincipal Component Loadings (Feature Contributions):")
    print(loadings.round(4))
    print("  Notice: PC1 is heavily dominated by Petal Length and Petal Width.")

    # Step 4: Run K-Means directly on PCA-reduced features
    print("\n[Step 3] K-Means Clustering on 2D PCA Space:")
    km_pca = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    pca_clusters = km_pca.fit_predict(X_pca)
    sil_pca = silhouette_score(X_pca, pca_clusters)
    print(f"  Silhouette Score on 2D PCA Space: {sil_pca:.4f}")
    print(f"  Inertia on 2D PCA Space: {km_pca.inertia_:.2f}")

def run_model_persistence_demo():
    print_separator("CORE TOPIC: Model Saving & Loading (joblib vs pickle)")

    df = get_iris_data()
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    original_preds = km.predict(X_scaled[:5])

    sample_query = np.array([
        [5.0, 3.4, 1.5, 0.2],  # Expected Setosa-like
        [6.2, 2.9, 4.3, 1.3],  # Expected Versicolor-like
        [7.7, 3.0, 6.1, 2.3]   # Expected Virginica-like
    ])
    sample_scaled = scaler.transform(sample_query)
    sample_preds = km.predict(sample_scaled)

    # 1. Joblib Saving and Loading
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "week3"
    joblib_path = os.path.join(current_dir, "kmeans_demo.joblib")
    scaler_joblib_path = os.path.join(current_dir, "scaler_demo.joblib")

    print("[1] Saving model and scaler with joblib...")
    joblib.dump(km, joblib_path)
    joblib.dump(scaler, scaler_joblib_path)
    print(f"  Saved: '{joblib_path}' ({os.path.getsize(joblib_path)} bytes)")

    print("    Reloading model with joblib.load()...")
    loaded_km_joblib = joblib.load(joblib_path)
    loaded_scaler_joblib = joblib.load(scaler_joblib_path)
    joblib_preds = loaded_km_joblib.predict(loaded_scaler_joblib.transform(sample_query))
    print(f"    Predictions match original exactly: {np.array_equal(sample_preds, joblib_preds)}")

    # 2. Pickle Saving and Loading
    pickle_path = os.path.join(current_dir, "kmeans_demo.pkl")
    print("\n[2] Saving model with pickle...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(km, f)
    print(f"  Saved: '{pickle_path}' ({os.path.getsize(pickle_path)} bytes)")

    print("    Reloading model with pickle.load()...")
    with open(pickle_path, 'rb') as f:
        loaded_km_pickle = pickle.load(f)
    pickle_preds = loaded_km_pickle.predict(sample_scaled)
    print(f"    Predictions match original exactly: {np.array_equal(sample_preds, pickle_preds)}")

    print("\nSample Test Predictions:")
    for flower, pred in zip(sample_query, sample_preds):
        print(f"  Features {flower} -> Assigned Cluster {pred}")

    # Cleanup demo files
    for p in [joblib_path, scaler_joblib_path, pickle_path]:
        if os.path.exists(p):
            os.remove(p)
    print("\nCleaned up temporary demonstration model artifacts.")

def main():
    print("==========================================================================")
    print("   MACHINE LEARNING COURSEWORK: WEEK 3 ASSIGNMENTS DEMONSTRATION")
    print("==========================================================================")
    run_assignment_1()
    run_assignment_2()
    run_model_persistence_demo()
    print("\n" + "=" * 75)
    print(" ALL WEEK 3 ASSIGNMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
