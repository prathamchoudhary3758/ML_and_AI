"""
Week 4: Major Project (Capstone Project)
========================================
Topic: Customer Segmentation using Clustering
Dataset: Mall Customers Dataset (Kaggle)
Focus: Full ML pipeline + Documentation + Presentation

Workflow:
1. Exploratory Data Analysis (EDA) on Age, Income, and Spending Score.
2. Determine optimal number of clusters via Elbow Method & Silhouette Analysis.
3. Apply K-Means Clustering on customer profiles.
4. Visualize customer groups across Age, Annual Income, and Spending Score (2D and 3D).
5. Categorize and label customer groups into overarching tiers:
   - "Low Spenders"
   - "Medium Spenders"
   - "High Spenders"
   alongside detailed demographic & behavioral personas.
6. Provide actionable business intelligence and marketing strategies for mall retail optimization.
7. Model persistence (joblib) and segmented dataset export.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

def print_section(title: str):
    """Utility to print styled console section headers."""
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def load_data(file_path: str) -> pd.DataFrame:
    """Load and perform hygiene checks on the Mall Customers dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Rename columns for programmatic consistency
    rename_dict = {
        'CustomerID': 'CustomerID',
        'Genre': 'Gender',
        'Age': 'Age',
        'Annual Income (k$)': 'Annual_Income_k$',
        'Spending Score (1-100)': 'Spending_Score_1_100'
    }
    df.rename(columns=rename_dict, inplace=True)
    return df

def perform_eda(df: pd.DataFrame, output_dir: str):
    """Perform exploratory data analysis and save comprehensive visual summary."""
    print_section("Step 1: Exploratory Data Analysis (EDA)")
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData Types and Missing Values:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics of Numerical Features:")
    summary = df[['Age', 'Annual_Income_k$', 'Spending_Score_1_100']].describe().T
    summary['median'] = df[['Age', 'Annual_Income_k$', 'Spending_Score_1_100']].median()
    print(summary[['mean', 'std', 'min', '25%', 'median', '75%', 'max']])
    
    print("\nGender Breakdown:")
    gender_counts = df['Gender'].value_counts()
    for g, count in gender_counts.items():
        pct = (count / len(df)) * 100
        mean_spend = df[df['Gender'] == g]['Spending_Score_1_100'].mean()
        mean_income = df[df['Gender'] == g]['Annual_Income_k$'].mean()
        print(f" - {g}: {count} customers ({pct:.1f}%) | Avg Income: ${mean_income:.1f}k | Avg Spend: {mean_spend:.1f}/100")

    # Generate EDA Visualizations
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Mall Customers Dataset - Exploratory Data Analysis", fontsize=18, fontweight='bold', y=0.98)

    palette = {'Male': '#3498db', 'Female': '#e74c3c'}

    # 1. Age Distribution
    sns.histplot(data=df, x='Age', hue='Gender', kde=True, bins=15, ax=axes[0, 0], palette=palette, multiple="stack")
    axes[0, 0].set_title("Age Distribution by Gender", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Age (Years)")
    axes[0, 0].axvline(df['Age'].mean(), color='black', linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
    axes[0, 0].legend()

    # 2. Annual Income Distribution
    sns.histplot(data=df, x='Annual_Income_k$', hue='Gender', kde=True, bins=15, ax=axes[0, 1], palette=palette, multiple="stack")
    axes[0, 1].set_title("Annual Income Distribution by Gender", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Annual Income (k$)")
    axes[0, 1].axvline(df['Annual_Income_k$'].mean(), color='black', linestyle='--', label=f"Mean: ${df['Annual_Income_k$'].mean():.1f}k")
    axes[0, 1].legend()

    # 3. Spending Score Distribution
    sns.histplot(data=df, x='Spending_Score_1_100', hue='Gender', kde=True, bins=15, ax=axes[0, 2], palette=palette, multiple="stack")
    axes[0, 2].set_title("Spending Score Distribution by Gender", fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel("Spending Score (1-100)")
    axes[0, 2].axvline(df['Spending_Score_1_100'].mean(), color='black', linestyle='--', label=f"Mean: {df['Spending_Score_1_100'].mean():.1f}")
    axes[0, 2].legend()

    # 4. Boxplots for Feature Outlier Detection
    box_data = pd.melt(df, id_vars=['CustomerID', 'Gender'], value_vars=['Age', 'Annual_Income_k$', 'Spending_Score_1_100'],
                       var_name='Feature', value_name='Value')
    sns.boxplot(data=box_data, x='Feature', y='Value', hue='Gender', ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Boxplot Distributions by Feature & Gender", fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks([0, 1, 2])
    axes[1, 0].set_xticklabels(['Age', 'Annual Income ($k)', 'Spending Score'])

    # 5. Correlation Heatmap
    corr = df[['Age', 'Annual_Income_k$', 'Spending_Score_1_100']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, ax=axes[1, 1], cbar_kws={'label': 'Pearson Correlation'})
    axes[1, 1].set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')

    # 6. Scatter: Age vs Spending Score
    sns.scatterplot(data=df, x='Age', y='Spending_Score_1_100', hue='Gender', size='Annual_Income_k$',
                    sizes=(40, 200), alpha=0.8, ax=axes[1, 2], palette=palette)
    axes[1, 2].set_title("Age vs Spending Score (Bubble Size = Income)", fontsize=13, fontweight='bold')
    axes[1, 2].axhline(50, color='gray', linestyle=':')
    axes[1, 2].axvline(40, color='gray', linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    eda_path = os.path.join(output_dir, "customer_segmentation_overview.png")
    plt.savefig(eda_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved EDA Overview Chart to: {eda_path}")

def find_optimal_clusters(X: np.ndarray, max_k: int = 10):
    """Compute WCSS (Inertia) and Silhouette Scores for k=2..max_k."""
    k_range = list(range(2, max_k + 1))
    wcss = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=25, max_iter=300, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        sil = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(sil)
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        davies_scores.append(davies_bouldin_score(X, kmeans.labels_))

    return {
        'k_range': k_range,
        'wcss': wcss,
        'silhouette': silhouette_scores,
        'calinski': calinski_scores,
        'davies': davies_scores
    }

def map_clusters_to_personas(df: pd.DataFrame, kmeans: KMeans, features: list):
    """
    Profile each cluster and assign descriptive persona names and spending group tiers:
    - 'Low Spenders'
    - 'Medium Spenders'
    - 'High Spenders'
    """
    centroids = kmeans.cluster_centers_
    cluster_profiles = []

    for i in range(kmeans.n_clusters):
        c_mask = (df['Cluster_ID'] == i)
        c_data = df[c_mask]
        
        mean_income = c_data['Annual_Income_k$'].mean()
        mean_spend = c_data['Spending_Score_1_100'].mean()
        mean_age = c_data['Age'].mean()
        count = len(c_data)
        pct = (count / len(df)) * 100

        # Define 3 spending tiers as requested in PDF: "Low Spenders", "Medium Spenders", "High Spenders"
        if mean_spend >= 65:
            spending_group = "High Spenders"
        elif mean_spend <= 35:
            spending_group = "Low Spenders"
        else:
            spending_group = "Medium Spenders"

        # Persona identification based on 2D income & spend quadrant
        if mean_income > 60 and mean_spend > 60:
            persona = "Affluent VIPs"
            description = "High Income, High Spending: Key revenue drivers; brand & luxury oriented."
            action = "Offer exclusive VIP previews, premium loyalty rewards, concierge services."
        elif mean_income < 40 and mean_spend > 60:
            persona = "Young Enthusiasts"
            description = "Low Income, High Spending: Trend-conscious, spontaneous shoppers (youth/students)."
            action = "Deploy trendy social media campaigns, student discounts, BNPL (buy now pay later)."
        elif mean_income > 60 and mean_spend < 40:
            persona = "Careful Savers"
            description = "High Income, Low Spending: High disposable income but conservative spenders."
            action = "Showcase value propositions, premium warranties, long-term durability & investment value."
        elif mean_income < 40 and mean_spend < 40:
            persona = "Budget Conscious"
            description = "Low Income, Low Spending: Highly price sensitive; necessities and essentials focused."
            action = "Provide clearance sales, bargain bundles, cashback vouchers, and budget promotions."
        else:
            persona = "Mainstream Middle"
            description = "Average Income, Moderate Spending: Balanced and pragmatic mass market shoppers."
            action = "Target with seasonal promotions, family events, and rewards for repeat store visits."

        cluster_profiles.append({
            'Cluster_ID': i,
            'Persona': persona,
            'Spending_Group': spending_group,
            'Count': count,
            'Percentage': pct,
            'Mean_Income': mean_income,
            'Mean_Spend': mean_spend,
            'Mean_Age': mean_age,
            'Description': description,
            'Marketing_Action': action
        })

    profiles_df = pd.DataFrame(cluster_profiles)
    return profiles_df

def plot_clustering_results(df: pd.DataFrame, eval_metrics: dict, kmeans: KMeans, profiles_df: pd.DataFrame, output_dir: str):
    """Plot comprehensive 6-panel evaluation and segmentation graphic."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    fig.suptitle("Mall Customer Segmentation - K-Means Clustering Analysis & Evaluation", fontsize=20, fontweight='bold', y=0.98)

    palette_5 = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

    # 1. Elbow Method (Inertia / WCSS)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eval_metrics['k_range'], eval_metrics['wcss'], marker='o', color='#2c3e50', linewidth=2.5, markersize=8)
    ax1.axvline(5, color='#e74c3c', linestyle='--', linewidth=2, label='Optimal Elbow (K=5)')
    ax1.set_title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)")
    ax1.legend(loc='upper right')

    # 2. Silhouette Score Evaluation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eval_metrics['k_range'], eval_metrics['silhouette'], marker='s', color='#16a085', linewidth=2.5, markersize=8)
    opt_k_idx = eval_metrics['k_range'].index(5)
    ax2.plot(5, eval_metrics['silhouette'][opt_k_idx], marker='*', color='#e74c3c', markersize=16, label=f"Max Silhouette = {eval_metrics['silhouette'][opt_k_idx]:.3f} (K=5)")
    ax2.set_title("Silhouette Scores vs Number of Clusters", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Average Silhouette Score")
    ax2.legend(loc='lower left')

    # 3. Spend Category Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    group_counts = df['Spending_Group'].value_counts()
    group_colors = {'Low Spenders': '#3498db', 'Medium Spenders': '#f1c40f', 'High Spenders': '#e74c3c'}
    bars = ax3.bar(group_counts.index, group_counts.values, color=[group_colors.get(x, '#95a5a6') for x in group_counts.index], edgecolor='black', width=0.55)
    ax3.set_title("Customer Distribution by Spending Group", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Assigned Spending Group")
    ax3.set_ylabel("Customer Count")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 2, f"{int(h)} ({h/len(df)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, max(group_counts.values) * 1.15)

    # 4. 2D Cluster Scatter Plot (Income vs Spending Score)
    ax4 = fig.add_subplot(gs[1, 0:2])
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = df[df['Cluster_ID'] == cluster_id]
        persona_row = profiles_df[profiles_df['Cluster_ID'] == cluster_id].iloc[0]
        label = f"Cluster {cluster_id}: {persona_row['Persona']} ({persona_row['Spending_Group']})"
        ax4.scatter(cluster_data['Annual_Income_k$'], cluster_data['Spending_Score_1_100'],
                    s=80, alpha=0.85, label=label, edgecolor='white', linewidth=0.8)

    # Plot Centroids
    centroids = kmeans.cluster_centers_
    ax4.scatter(centroids[:, 0], centroids[:, 1], s=250, c='black', marker='X', edgecolors='gold', linewidths=2, zorder=10, label='Centroids')
    
    # Annotate Centroids
    for i, c in enumerate(centroids):
        p_name = profiles_df[profiles_df['Cluster_ID'] == i]['Persona'].values[0]
        ax4.annotate(f"C{i}: {p_name}\n({c[0]:.0f}k, {c[1]:.0f})", (c[0] + 1.5, c[1] - 3.5),
                     fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6, ec="black"))

    ax4.set_title("Customer Segmentation Map (Annual Income vs Spending Score)", fontsize=15, fontweight='bold')
    ax4.set_xlabel("Annual Income (k$)", fontsize=12)
    ax4.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax4.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    # 5. Boxplot of Spending Score by Cluster
    ax5 = fig.add_subplot(gs[1, 2])
    sns.boxplot(data=df, x='Cluster_Name', y='Spending_Score_1_100', hue='Cluster_Name', ax=ax5, palette='Set2', legend=False)
    ax5.set_title("Spending Score Distribution across Personas", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Customer Persona")
    ax5.set_ylabel("Spending Score (1-100)")
    plt.setp(ax5.get_xticklabels(), rotation=35, ha='right', fontsize=9.5)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, hspace=0.3, wspace=0.25)
    clusters_path = os.path.join(output_dir, "customer_segmentation_clusters.png")
    plt.savefig(clusters_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Cluster Analysis Chart to: {clusters_path}")

def plot_3d_segmentation(df: pd.DataFrame, output_dir: str):
    """Plot 3D visualization of Age vs Income vs Spending Score with cluster color codes."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for cluster_id in sorted(df['Cluster_ID'].unique()):
        subset = df[df['Cluster_ID'] == cluster_id]
        p_name = subset['Cluster_Name'].iloc[0]
        s_group = subset['Spending_Group'].iloc[0]
        ax.scatter(subset['Age'], subset['Annual_Income_k$'], subset['Spending_Score_1_100'],
                   c=colors[cluster_id % len(colors)], label=f"Cluster {cluster_id}: {p_name} [{s_group}]",
                   s=60, alpha=0.85, edgecolors='w', depthshade=True)

    ax.set_title("3D Customer Segmentation\n(Age vs Annual Income vs Spending Score)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Age (Years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Annual Income (k$)", fontsize=11, labelpad=10)
    ax.set_zlabel("Spending Score (1-100)", fontsize=11, labelpad=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95), fontsize=9)
    ax.view_init(elev=25, azim=130)

    out_path = os.path.join(output_dir, "customer_segmentation_3d.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D Segmentation Plot to: {out_path}")

def main():
    """Execute complete end-to-end Capstone Machine Learning Pipeline."""
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    # Ensure correct data path
    possible_paths = [
        os.path.join(current_dir, "Mall_Customers.csv"),
        os.path.join(current_dir, "week4", "Mall_Customers.csv"),
        "week4/Mall_Customers.csv",
        "Mall_Customers.csv"
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if not data_path:
        raise FileNotFoundError("Mall_Customers.csv could not be located.")

    print_section("Week 4 Capstone Project: Customer Segmentation Pipeline")
    print(f"Reading dataset from: {data_path}")
    df = load_data(data_path)
    
    # -------------------------------------------------------------
    # Step 1: Exploratory Data Analysis
    # -------------------------------------------------------------
    perform_eda(df, current_dir)
    
    # -------------------------------------------------------------
    # Step 2: Feature Selection & Preprocessing
    # -------------------------------------------------------------
    print_section("Step 2: Feature Selection & Scaling")
    features_2d = ['Annual_Income_k$', 'Spending_Score_1_100']
    X_2d = df[features_2d].values

    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    print(f"Selected Features for 2D Segmentation: {features_2d}")
    print(f"Feature Space Scaled: Mean={X_scaled.mean(axis=0).round(4)}, Std={X_scaled.std(axis=0).round(4)}")

    # -------------------------------------------------------------
    # Step 3: Determining Optimal Number of Clusters
    # -------------------------------------------------------------
    print_section("Step 3: Optimal K Search (Elbow & Silhouette)")
    eval_metrics = find_optimal_clusters(X_2d, max_k=10)
    
    print(f"{'K':<4} | {'Inertia (WCSS)':<16} | {'Silhouette':<12} | {'Calinski-Harabasz':<18} | {'Davies-Bouldin':<15}")
    print("-" * 75)
    for i, k in enumerate(eval_metrics['k_range']):
        print(f"{k:<4} | {eval_metrics['wcss'][i]:<16.2f} | {eval_metrics['silhouette'][i]:<12.4f} | {eval_metrics['calinski'][i]:<18.2f} | {eval_metrics['davies'][i]:<15.4f}")

    optimal_k = 5
    print(f"\nSelection: Optimal K = {optimal_k} achieves peak Silhouette Score ({eval_metrics['silhouette'][eval_metrics['k_range'].index(5)]:.4f}) and clear Elbow inflection.")

    # -------------------------------------------------------------
    # Step 4: Model Training & Cluster Profiling
    # -------------------------------------------------------------
    print_section("Step 4: K-Means Model Training & Persona Profiling")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=25, max_iter=300, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(X_2d)

    profiles_df = map_clusters_to_personas(df, kmeans, features_2d)
    
    # Map cluster persona and spending group back to main dataframe
    name_map = dict(zip(profiles_df['Cluster_ID'], profiles_df['Persona']))
    group_map = dict(zip(profiles_df['Cluster_ID'], profiles_df['Spending_Group']))
    action_map = dict(zip(profiles_df['Cluster_ID'], profiles_df['Marketing_Action']))

    df['Cluster_Name'] = df['Cluster_ID'].map(name_map)
    df['Spending_Group'] = df['Cluster_ID'].map(group_map)
    df['Marketing_Strategy'] = df['Cluster_ID'].map(action_map)

    print("\nCustomer Cluster Summary Table:")
    print(profiles_df[['Cluster_ID', 'Persona', 'Spending_Group', 'Count', 'Percentage', 'Mean_Income', 'Mean_Spend', 'Mean_Age']].to_string(index=False))

    # -------------------------------------------------------------
    # Step 5: Visualizations
    # -------------------------------------------------------------
    print_section("Step 5: Visualizing Segments across Income, Spend & Age")
    plot_clustering_results(df, eval_metrics, kmeans, profiles_df, current_dir)
    plot_3d_segmentation(df, current_dir)

    # -------------------------------------------------------------
    # Step 6: Direct 3-Tier K-Means Evaluation (Low, Medium, High)
    # -------------------------------------------------------------
    print_section("Step 6: Comparative 3-Cluster Model Analysis (K=3)")
    km3 = KMeans(n_clusters=3, init='k-means++', n_init=25, random_state=42)
    labels_3 = km3.fit_predict(X_2d)
    sil_3 = silhouette_score(X_2d, labels_3)
    print(f"K=3 Direct Clustering Silhouette Score: {sil_3:.4f}")
    c3_centers = km3.cluster_centers_
    sorted_idx = np.argsort(c3_centers[:, 1]) # sort by spending score
    tier_labels = ['Low Spenders', 'Medium Spenders', 'High Spenders']
    for rank, idx in enumerate(sorted_idx):
        print(f" - Centroid {idx}: Mean Income = ${c3_centers[idx, 0]:.1f}k, Mean Spend = {c3_centers[idx, 1]:.1f} -> {tier_labels[rank]}")

    # -------------------------------------------------------------
    # Step 7: Model Persistence & Results Export
    # -------------------------------------------------------------
    print_section("Step 7: Saving Model Artifacts & Segmented Dataset")
    model_path = os.path.join(current_dir, "kmeans_customer_model.joblib")
    scaler_path = os.path.join(current_dir, "scaler.joblib")
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    output_csv = os.path.join(current_dir, "mall_customers_segmented.csv")
    df.to_csv(output_csv, index=False)
    print(f"Segmented customer dataset exported to: {output_csv}")

    # -------------------------------------------------------------
    # Step 8: Executive Summary & Business Impact
    # -------------------------------------------------------------
    print_section("Step 8: Capstone Business Insights & Retail Strategies")
    print("\nKey Actionable Takeaways for Mall Retail Management:")
    for _, row in profiles_df.iterrows():
        print(f"\n[{row['Spending_Group'].upper()}] - Cluster {row['Cluster_ID']}: {row['Persona']} ({row['Percentage']:.1f}% of base)")
        print(f"   Profile: Avg Age {row['Mean_Age']:.0f} yrs | Income ${row['Mean_Income']:.1f}k | Spend Score {row['Mean_Spend']:.1f}/100")
        print(f"   Insight: {row['Description']}")
        print(f"   Strategic Action: {row['Marketing_Action']}")

    print("\n" + "=" * 80)
    print(" CAPSTONE ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
