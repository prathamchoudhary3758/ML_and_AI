"""
Mini Project 3: Iris Flower Clustering Project
==============================================
Dataset: Iris Dataset (UCI / Kaggle)

Tasks:
1. Apply K-Means clustering (k=3) on standardized Iris measurements.
2. Determine optimal clusters using the Elbow Method and Silhouette Analysis.
3. Dimensionality reduction and visualization via Principal Component Analysis (PCA).
4. Visualize clusters with scatter plots and centroid markers.
5. Compare predicted clusters vs true ground-truth labels:
   - Contingency Matrix / Confusion Matrix
   - Adjusted Rand Index (ARI) & Normalized Mutual Information (NMI)
   - Optimal cluster-to-class alignment (Hungarian algorithm / Hungarian matching)
   - Accuracy, Precision, Recall, and F1-Score
6. Save production-ready trained model & scaler with joblib.
7. Export full prediction results to 'iris_clustered_results.csv' and charts to 'iris_clustering_analysis.png'.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)

def print_section(title: str):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    # -------------------------------------------------------------
    # STEP 1: Load Dataset & Preliminary EDA
    # -------------------------------------------------------------
    print_section("STEP 1: Load Iris Dataset & Exploratory Data Analysis")
    data_paths = [
        os.path.join(current_dir, "iris.csv"),
        os.path.join(current_dir, "week3", "iris.csv"),
        "week3/iris.csv",
        "iris.csv",
        "iris.data.csv"
    ]
    data_path = None
    for p in data_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        raise FileNotFoundError("Could not find Iris dataset!")

    print(f"Loading dataset from: {data_path}")
    if "data" in os.path.basename(data_path):
        cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df = pd.read_csv(data_path, header=None, names=cols).dropna(how='all')
    else:
        df = pd.read_csv(data_path)

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSummary Statistics:")
    print(df.describe().round(2))

    print("\nClass Distribution:")
    print(df['species'].value_counts())

    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values detected: {missing.sum()}")

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols].values
    y_true = df['species'].values

    # -------------------------------------------------------------
    # STEP 2: Feature Standardization
    # -------------------------------------------------------------
    print_section("STEP 2: Feature Scaling (StandardScaler)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features standardized with mean=0 and variance=1.")
    print(f"Features: {feature_cols}")

    # -------------------------------------------------------------
    # STEP 3: Elbow Method & Silhouette Analysis for Optimal K
    # -------------------------------------------------------------
    print_section("STEP 3: Optimal K Selection (Elbow Method & Silhouette Analysis)")
    k_values = list(range(1, 11))
    inertias = []
    sil_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            score = silhouette_score(X_scaled, km.labels_)
            sil_scores.append(score)
        else:
            sil_scores.append(np.nan)

    print(f"{'K':<5} | {'Inertia (WCSS)':<18} | {'Silhouette Score':<18}")
    print("-" * 47)
    for k, inert, sil in zip(k_values, inertias, sil_scores):
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        print(f"{k:<5} | {inert:<18.2f} | {sil_str:<18}")

    # -------------------------------------------------------------
    # STEP 4: Fit K-Means (k=3)
    # -------------------------------------------------------------
    print_section("STEP 4: Train K-Means Clustering Model (k=3)")
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=20, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Calculate Unsupervised Clustering Metrics
    sil = silhouette_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)
    db = davies_bouldin_score(X_scaled, clusters)

    print(f"K-Means Configuration : k={k_optimal}, init='k-means++', n_init=20")
    print(f"Model Inertia (WCSS)  : {kmeans.inertia_:.2f}")
    print(f"Silhouette Score      : {sil:.4f}")
    print(f"Calinski-Harabasz     : {ch:.2f}")
    print(f"Davies-Bouldin Index  : {db:.4f}")

    # Centroids in original feature scale
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroids_df.index = [f"Cluster {i}" for i in range(k_optimal)]
    print("\nCluster Centroids (Original Metric Units):")
    print(centroids_df.round(2))

    # -------------------------------------------------------------
    # STEP 5: PCA Dimensionality Reduction
    # -------------------------------------------------------------
    print_section("STEP 5: Dimensionality Reduction via PCA (4D -> 2D)")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_centroids = pca.transform(kmeans.cluster_centers_)

    var_pc1, var_pc2 = pca.explained_variance_ratio_
    print(f"Principal Component 1 (PC1) Variance: {var_pc1*100:.2f}%")
    print(f"Principal Component 2 (PC2) Variance: {var_pc2*100:.2f}%")
    print(f"Total Retained Variance (PC1 + PC2) : {(var_pc1 + var_pc2)*100:.2f}%")

    # -------------------------------------------------------------
    # STEP 6: Cluster Comparison vs True Labels
    # -------------------------------------------------------------
    print_section("STEP 6: Cluster Evaluation vs Ground Truth Species")
    
    # 1. Contingency Matrix (Crosstab)
    contingency = pd.crosstab(
        pd.Series(y_true, name='True Species'),
        pd.Series(clusters, name='Assigned Cluster')
    )
    print("Contingency Matrix (True Species vs Predicted Cluster):")
    print(contingency)

    # 2. Optimal Bipartite Mapping (Hungarian Algorithm)
    # Match each unsupervised cluster to its most corresponding ground truth label
    classes = np.unique(y_true)
    cost_matrix = np.zeros((k_optimal, len(classes)))
    for c_idx, cluster_id in enumerate(range(k_optimal)):
        for l_idx, class_name in enumerate(classes):
            cost_matrix[c_idx, l_idx] = -np.sum((clusters == cluster_id) & (y_true == class_name))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_label = {row: classes[col] for row, col in zip(row_ind, col_ind)}
    mapped_predictions = np.array([cluster_to_label[c] for c in clusters])

    print("\nOptimal Cluster-to-Species Mapping (via Hungarian Algorithm):")
    for c_id, species_name in cluster_to_label.items():
        print(f"  Cluster {c_id} -> {species_name}")

    # 3. Supervised Comparison Metrics
    ari = adjusted_rand_score(y_true, clusters)
    nmi = normalized_mutual_info_score(y_true, clusters)
    homo = homogeneity_score(y_true, clusters)
    comp = completeness_score(y_true, clusters)
    v_meas = v_measure_score(y_true, clusters)
    acc = accuracy_score(y_true, mapped_predictions)

    print("\nExternal Validation Metrics (Clusters vs True Labels):")
    print(f"  Adjusted Rand Index (ARI)       : {ari:.4f} (Measures agreement, 1.0 is perfect)")
    print(f"  Normalized Mutual Info (NMI)   : {nmi:.4f} (Information shared between partitions)")
    print(f"  Homogeneity Score               : {homo:.4f} (Each cluster contains only 1 class)")
    print(f"  Completeness Score              : {comp:.4f} (All members of a class in same cluster)")
    print(f"  V-Measure Score                 : {v_meas:.4f} (Harmonic mean of homo & comp)")
    print(f"  Mapped Clustering Accuracy      : {acc*100:.2f}%")

    print("\nClassification Report (after cluster-to-label alignment):")
    print(classification_report(y_true, mapped_predictions))

    cm = confusion_matrix(y_true, mapped_predictions, labels=classes)

    # -------------------------------------------------------------
    # STEP 7: Generate Publication-Quality Visualizations
    # -------------------------------------------------------------
    print_section("STEP 7: Generate Visualizations & Plots")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Mini Project 3: Iris Flower Clustering & PCA Evaluation", fontsize=18, fontweight='bold', y=0.98)

    palette_clusters = {0: '#2b5c8f', 1: '#d95f02', 2: '#2ca02c'}
    species_list = sorted(list(classes))
    palette_species = {
        'Iris-setosa': '#1f77b4',
        'Iris-versicolor': '#ff7f0e',
        'Iris-virginica': '#2ca02c'
    }

    # Subplot 1: Elbow Method (Inertia vs K)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(k_values, inertias, marker='o', linewidth=2.2, color='#1f77b4', markersize=7)
    ax1.axvline(x=3, color='#d95f02', linestyle='--', linewidth=1.8, label='Optimal K=3 (Elbow)')
    ax1.set_title("1. Elbow Method (Inertia vs. K)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax1.set_xticks(k_values)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Silhouette Scores vs K
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(k_values[1:], sil_scores[1:], marker='s', linewidth=2.2, color='#2ca02c', markersize=7)
    ax2.axvline(x=3, color='#d95f02', linestyle='--', linewidth=1.8, label=f'k=3 (Score={sil:.3f})')
    ax2.set_title("2. Silhouette Analysis across K", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_xticks(k_values[1:])
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Original Feature Space (Petal Length vs Width) - Ground Truth
    ax3 = plt.subplot(2, 3, 3)
    sns.scatterplot(
        x=df['petal_length'],
        y=df['petal_width'],
        hue=df['species'],
        palette=palette_species,
        s=70,
        alpha=0.85,
        edgecolor='k',
        ax=ax3
    )
    ax3.set_title("3. Ground Truth: Petal Dimensions", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Petal Length (cm)", fontsize=11)
    ax3.set_ylabel("Petal Width (cm)", fontsize=11)
    ax3.legend(title="Species", frameon=True)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Original Feature Space - K-Means Clusters & Centroids
    ax4 = plt.subplot(2, 3, 4)
    sns.scatterplot(
        x=df['petal_length'],
        y=df['petal_width'],
        hue=[f"Cluster {c} ({cluster_to_label[c].replace('Iris-', '')})" for c in clusters],
        palette=['#2b5c8f', '#d95f02', '#2ca02c'],
        s=70,
        alpha=0.85,
        edgecolor='k',
        ax=ax4
    )
    # Plot Petal Length / Petal Width Centroids
    ax4.scatter(
        centroids_df['petal_length'],
        centroids_df['petal_width'],
        c='red',
        s=220,
        marker='X',
        edgecolor='black',
        linewidth=1.5,
        label='Centroids'
    )
    ax4.set_title("4. K-Means Clusters & Centroids (Petal Space)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Petal Length (cm)", fontsize=11)
    ax4.set_ylabel("Petal Width (cm)", fontsize=11)
    ax4.legend(frameon=True)
    ax4.grid(True, alpha=0.3)

    # Subplot 5: PCA 2D Space - Clusters & Projected Centroids
    ax5 = plt.subplot(2, 3, 5)
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=[f"Cluster {c}" for c in clusters],
        palette=['#2b5c8f', '#d95f02', '#2ca02c'],
        s=70,
        alpha=0.85,
        edgecolor='k',
        ax=ax5
    )
    ax5.scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        c='red',
        s=220,
        marker='X',
        edgecolor='black',
        linewidth=1.5,
        label='PCA Centroids'
    )
    ax5.set_title(f"5. PCA 2D Projection (Clusters) - {var_pc1*100 + var_pc2*100:.1f}% Var", fontsize=13, fontweight='bold')
    ax5.set_xlabel(f"Principal Component 1 ({var_pc1*100:.1f}%)", fontsize=11)
    ax5.set_ylabel(f"Principal Component 2 ({var_pc2*100:.1f}%)", fontsize=11)
    ax5.legend(frameon=True)
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Confusion / Contingency Matrix Heatmap
    ax6 = plt.subplot(2, 3, 6)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[s.replace('Iris-', '') for s in classes],
        yticklabels=[s.replace('Iris-', '') for s in classes],
        cbar=False,
        ax=ax6,
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    ax6.set_title(f"6. Confusion Matrix (Accuracy: {acc*100:.1f}%)", fontsize=13, fontweight='bold')
    ax6.set_xlabel("Mapped Predicted Species", fontsize=11)
    ax6.set_ylabel("True Ground Truth Species", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = "iris_clustering_analysis.png"
    plot_path = os.path.join(current_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Successfully generated and saved figure to '{plot_path}'")

    # -------------------------------------------------------------
    # STEP 8: Export Enriched Predictions to CSV
    # -------------------------------------------------------------
    print_section("STEP 8: Export Enriched Clustered Results to CSV")
    df_results = df.copy()
    df_results['cluster_id'] = clusters
    df_results['mapped_species'] = mapped_predictions
    df_results['is_correct'] = (df_results['species'] == df_results['mapped_species']).astype(int)
    df_results['pca_component_1'] = np.round(X_pca[:, 0], 4)
    df_results['pca_component_2'] = np.round(X_pca[:, 1], 4)

    csv_path = os.path.join(current_dir, "iris_clustered_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Exported {len(df_results)} rows to '{csv_path}'")
    print("Preview of Exported Table:")
    print(df_results[['sepal_length', 'petal_length', 'species', 'cluster_id', 'mapped_species', 'is_correct']].head(6))

    # -------------------------------------------------------------
    # STEP 9: Save Trained Models with joblib
    # -------------------------------------------------------------
    print_section("STEP 9: Persist Model Artifacts with joblib")
    model_path = os.path.join(current_dir, "kmeans_iris_model.joblib")
    scaler_path = os.path.join(current_dir, "scaler_iris.joblib")
    pca_path = os.path.join(current_dir, "pca_iris.joblib")

    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print(f"Saved K-Means Model  : '{model_path}' ({os.path.getsize(model_path)} bytes)")
    print(f"Saved Feature Scaler : '{scaler_path}' ({os.path.getsize(scaler_path)} bytes)")
    print(f"Saved PCA Transformer: '{pca_path}' ({os.path.getsize(pca_path)} bytes)")

    print("\n" + "=" * 75)
    print(" MINI PROJECT 3 (IRIS CLUSTERING) COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
