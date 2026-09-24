"""
Week 4: Major Capstone Project - Customer Segmentation Walkthrough
==================================================================
Dataset: Mall Customers Dataset (Kaggle)
Option 3: Customer Segmentation using Clustering

Workflows Demonstrated:
1. Apply K-Means to cluster customers (Inertia & Silhouette Analysis).
2. Visualize customer groups by Age / Annual Income / Spending Score.
3. Label groups as "Low Spenders", "Medium Spenders", "High Spenders".
4. Business analytics summary and customer inference demonstration.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def print_separator(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()}")
    print("=" * 75)

def get_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(current_dir, "Mall_Customers.csv"),
        os.path.join(current_dir, "week4", "Mall_Customers.csv"),
        "week4/Mall_Customers.csv",
        "Mall_Customers.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Mall_Customers.csv not found in search paths.")

def run_workflow_kmeans():
    print_separator("Workflow 1: Apply K-Means Clustering on Mall Customers")
    path = get_data_path()
    df = pd.read_csv(path)
    
    # Clean column names
    df.rename(columns={
        'Annual Income (k$)': 'Annual_Income_k$',
        'Spending Score (1-100)': 'Spending_Score_1_100',
        'Genre': 'Gender'
    }, inplace=True)
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(5).to_string(index=False))
    
    X = df[['Annual_Income_k$', 'Spending_Score_1_100']].values
    
    print("\nEvaluating K=2 to K=8 using Inertia (WCSS) & Silhouette Scores:")
    best_k = None
    best_sil = -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, init='k-means++', n_init=25, random_state=42)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        if sil > best_sil:
            best_sil = sil
            best_k = k
        print(f" - K={k}: Inertia = {km.inertia_:.1f} | Silhouette Score = {sil:.4f}")
        
    print(f"\nOptimal Cluster Count: K={best_k} with peak Silhouette Score = {best_sil:.4f}")
    
    # Fit final K=5 model
    final_km = KMeans(n_clusters=5, init='k-means++', n_init=25, random_state=42)
    df['Cluster_ID'] = final_km.fit_predict(X)
    return df, final_km

def run_workflow_visualization_profiles(df: pd.DataFrame, km: KMeans):
    print_separator("Workflow 2: Customer Group Profiling by Age, Income, & Spending Score")
    
    group_stats = df.groupby('Cluster_ID').agg(
        Count=('CustomerID', 'count'),
        Mean_Age=('Age', 'mean'),
        Min_Age=('Age', 'min'),
        Max_Age=('Age', 'max'),
        Mean_Income=('Annual_Income_k$', 'mean'),
        Mean_Spend=('Spending_Score_1_100', 'mean')
    ).round(2)
    
    print("Aggregate Statistics by Cluster:")
    print(group_stats.to_string())
    
    print("\nCluster Centroids [Annual Income ($k), Spending Score (1-100)]:")
    for idx, c in enumerate(km.cluster_centers_):
        print(f" - Cluster {idx} Centroid: Annual Income = ${c[0]:.2f}k, Spending Score = {c[1]:.2f}/100")

def run_workflow_spending_labels(df: pd.DataFrame, km: KMeans):
    print_separator("Workflow 3: Label Groups as 'Low Spenders', 'Medium Spenders', 'High Spenders'")
    
    # Assign spending categories according to mean spending score
    centroids = km.cluster_centers_
    cluster_labels = {}
    cluster_personas = {}
    
    for i in range(5):
        spend = centroids[i, 1]
        income = centroids[i, 0]
        
        if spend >= 65:
            cluster_labels[i] = "High Spenders"
            if income > 60:
                cluster_personas[i] = "Affluent VIPs (High Income, High Spend)"
            else:
                cluster_personas[i] = "Young Enthusiasts (Low Income, High Spend)"
        elif spend <= 35:
            cluster_labels[i] = "Low Spenders"
            if income > 60:
                cluster_personas[i] = "Careful Savers (High Income, Low Spend)"
            else:
                cluster_personas[i] = "Budget Conscious (Low Income, Low Spend)"
        else:
            cluster_labels[i] = "Medium Spenders"
            cluster_personas[i] = "Mainstream Middle (Average Income, Moderate Spend)"
            
    df['Spending_Group'] = df['Cluster_ID'].map(cluster_labels)
    df['Persona'] = df['Cluster_ID'].map(cluster_personas)
    
    print("Assigned Spending Tiers & Customer Personas:")
    for i in range(5):
        cnt = (df['Cluster_ID'] == i).sum()
        pct = (cnt / len(df)) * 100
        print(f" * Cluster {i}: [{df[df['Cluster_ID'] == i]['Spending_Group'].iloc[0]}] - {cluster_personas[i]}")
        print(f"   Size: {cnt} customers ({pct:.1f}%) | Centroid: (${centroids[i, 0]:.1f}k, {centroids[i, 1]:.1f}/100)")
        
    print("\nDistribution of Customers by Primary Spending Tier:")
    print(df['Spending_Group'].value_counts().to_string())
    return df

def run_workflow_inference(km: KMeans):
    print_separator("Workflow 4: Real-Time Inference on New Customer Profiles")
    new_customers = pd.DataFrame([
        {'Customer': 'Prospect A', 'Annual_Income_k$': 85, 'Spending_Score_1_100': 86},
        {'Customer': 'Prospect B', 'Annual_Income_k$': 20, 'Spending_Score_1_100': 15},
        {'Customer': 'Prospect C', 'Annual_Income_k$': 55, 'Spending_Score_1_100': 50},
        {'Customer': 'Prospect D', 'Annual_Income_k$': 90, 'Spending_Score_1_100': 18},
        {'Customer': 'Prospect E', 'Annual_Income_k$': 25, 'Spending_Score_1_100': 80}
    ])
    
    preds = km.predict(new_customers[['Annual_Income_k$', 'Spending_Score_1_100']].values)
    new_customers['Predicted_Cluster'] = preds
    
    # Map to tiers
    tier_map = {0: 'Medium Spenders', 1: 'High Spenders', 2: 'High Spenders', 3: 'Low Spenders', 4: 'Low Spenders'}
    persona_map = {
        0: 'Mainstream Middle',
        1: 'Affluent VIPs',
        2: 'Young Enthusiasts',
        3: 'Careful Savers',
        4: 'Budget Conscious'
    }
    
    new_customers['Spending_Group'] = new_customers['Predicted_Cluster'].map(tier_map)
    new_customers['Persona'] = new_customers['Predicted_Cluster'].map(persona_map)
    
    print("New Customer Scoring Predictions:")
    print(new_customers[['Customer', 'Annual_Income_k$', 'Spending_Score_1_100', 'Predicted_Cluster', 'Spending_Group', 'Persona']].to_string(index=False))

def main():
    df, km = run_workflow_kmeans()
    run_workflow_visualization_profiles(df, km)
    df = run_workflow_spending_labels(df, km)
    run_workflow_inference(km)
    print("\n" + "=" * 75)
    print(" WEEK 4 CAPSTONE ASSIGNMENT / WORKFLOW DEMO COMPLETED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
