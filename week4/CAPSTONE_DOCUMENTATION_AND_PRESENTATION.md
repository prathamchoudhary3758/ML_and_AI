# Capstone Project: Customer Segmentation using K-Means Clustering
**Course**: Machine Learning & AI  
**Deliverable**: Week 4 – Major Project (Capstone Project)  
**Dataset**: Kaggle – Mall Customers Dataset  
**Focus**: Full ML Pipeline + Comprehensive Documentation + Executive Presentation  

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Business Problem & Objectives](#2-business-problem--objectives)
3. [Dataset Architecture & Hygiene](#3-dataset-architecture--hygiene)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Mathematical & Algorithmic Foundation](#5-mathematical--algorithmic-foundation)
6. [Optimal Cluster Selection (Elbow & Silhouette)](#6-optimal-cluster-selection-elbow--silhouette)
7. [Customer Segmentation & Persona Profiling](#7-customer-segmentation--persona-profiling)
8. [Overarching Spending Tiers ("Low", "Medium", "High Spenders")](#8-overarching-spending-tiers)
9. [Visual Analytics (2D and 3D Visualizations)](#9-visual-analytics)
10. [Business Analytics & Retail Optimization Strategy](#10-business-analytics--retail-optimization-strategy)
11. [Executive Presentation Slide Deck](#11-executive-presentation-slide-deck)
12. [Model Serialization & Production Deployment](#12-model-serialization--production-deployment)

---

## 1. Executive Summary
In retail real estate and commercial shopping malls, blanket marketing campaigns suffer from high acquisition costs and low conversion rates. Customer Segmentation solves this by applying unsupervised machine learning algorithms to identify natural consumer groupings.

By clustering mall shoppers across **Age**, **Annual Income (k$)**, and **Spending Score (1–100)**, we reveal 5 mathematically distinct clusters that roll up into three strategic spending tiers:
- **High Spenders (30.5%)**: High-value shoppers comprising *Affluent VIPs* and *Young Trendsetters*.
- **Medium Spenders (40.5%)**: The stable revenue backbone of *Mainstream Families*.
- **Low Spenders (29.0%)**: Value-conscious shoppers comprising *Careful Savers* and *Budget-Focused Shoppers*.

---

## 2. Business Problem & Objectives
Traditional marketing treats customers as a monolithic bloc. The primary objectives of this capstone project are:
1. **Unsupervised Grouping**: Group 200 mall customers into distinct segments using K-Means clustering.
2. **Optimal Cluster Evaluation**: Validate cluster stability and separability using the Elbow Method and Silhouette Analysis.
3. **Multi-Dimensional Profiling**: Analyze interactions across Age, Income, and Spending Score.
4. **Strategic Labeling**: Classify groups into **"Low Spenders"**, **"Medium Spenders"**, and **"High Spenders"** with detailed demographic personas.
5. **Actionable ROI Strategies**: Deliver personalized promotional, leasing, and merchandising strategies to maximize mall revenue and customer lifetime value (CLV).

---

## 3. Dataset Architecture & Hygiene
The project utilizes the benchmark **Mall Customers Dataset** from Kaggle:
- **Total Records**: 200 customers
- **Total Attributes**: 5 features (1 ID, 1 categorical, 3 continuous)
- **Data Integrity**: 0 missing values, 0 duplicated entries.

| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Integer | 1 – 200 | Unique numeric identifier for each mall visitor |
| `Gender` (`Genre`) | Categorical | Male (44%), Female (56%) | Customer biological gender |
| `Age` | Continuous | 18 – 70 years | Customer chronological age (Mean: 38.85 yrs) |
| `Annual Income (k$)` | Continuous | $15k – $137k | Self-reported or transaction-inferred annual income |
| `Spending Score (1-100)` | Continuous | 1 – 99 | Score assigned based on customer purchase history and shopping behavior |

---

## 4. Exploratory Data Analysis (EDA)

### Key Descriptive Statistics
| Metric | Age (Years) | Annual Income ($k) | Spending Score (1-100) |
| :--- | :--- | :--- | :--- |
| **Mean** | 38.85 | $60.56k | 50.20 |
| **Std Dev** | 13.97 | $26.26k | 25.82 |
| **Median** | 36.00 | $61.50k | 50.00 |
| **IQR (25% - 75%)** | 28.75 – 49.00 | $41.50k – $78.00k | 34.75 – 73.00 |
| **Min / Max** | 18 / 70 | $15k / $137k | 1 / 99 |

### Demographic Insights
1. **Gender Distribution**: Females represent **56%** of visitors with an average spending score of **51.5**, compared to Males (**44%**) with an average spending score of **48.5** and slightly higher average income ($62.2k vs $59.2k).
2. **Age Correlation**: A moderate negative correlation ($r \approx -0.327$) exists between Age and Spending Score, indicating younger demographics exhibit higher willingness to spend on lifestyle and impulse purchases.
3. **Income vs. Spending**: Annual income shows near-zero correlation ($r \approx 0.010$) with spending score across the whole dataset, underscoring that **high income does not automatically imply high spending**.

Visual summary saved to: [`customer_segmentation_overview.png`](customer_segmentation_overview.png)

---

## 5. Mathematical & Algorithmic Foundation

### K-Means Objective Function
K-Means partitions $N$ observations into $K$ mutually exclusive clusters $S = \{S_1, S_2, \dots, S_K\}$ by minimizing the **Within-Cluster Sum of Squares (WCSS / Inertia)**:
$$\arg\min_{S} \sum_{i=1}^{K} \sum_{x \in S_i} ||x - \mu_i||^2$$
where $\mu_i$ denotes the geometric centroid of cluster $S_i$.

### Initialization: `k-means++`
To avoid poor local minima, `k-means++` initializes centroids with probability proportional to the squared Euclidean distance to the nearest existing center:
$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$

### Validation Metrics
1. **Silhouette Coefficient ($s$)**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   $a(i)$: average intra-cluster distance; $b(i)$: average nearest-cluster distance.
2. **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion.
3. **Davies-Bouldin Index**: Average similarity between each cluster and its most similar counterpart.

---

## 6. Optimal Cluster Selection (Elbow & Silhouette)

We evaluated candidate cluster counts $K \in [2, 10]$ across multiple validation metrics:

| Number of Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 181,363.60 | 0.2969 | 96.75 | 1.2568 |
| 3 | 106,348.37 | 0.4676 | 151.56 | 0.7153 |
| 4 | 73,679.79 | 0.4932 | 174.06 | 0.7104 |
| **5 (Optimal)** | **44,448.46** | **0.5539** | **247.36** | **0.5726** |
| 6 | 37,233.81 | 0.5398 | 242.54 | 0.6522 |
| 7 | 30,241.34 | 0.5288 | 255.00 | 0.7108 |
| 8 | 24,995.97 | 0.4593 | 268.83 | 0.7541 |

### Selection Verdict:
- **Elbow Point**: The sharpest decline in inertia flattens prominently at **$K=5$**.
- **Peak Silhouette**: The silhouette score peaks at **0.5539** for $K=5$, demonstrating maximum separation and intra-cluster cohesion.
- **Davies-Bouldin**: Minimum index (**0.5726**) confirms minimal cluster overlap.

---

## 7. Customer Segmentation & Persona Profiling

The optimal 5-cluster model maps directly into actionable consumer personas:

| Cluster ID | Persona Name | Spending Tier | Market Share | Mean Age | Mean Income | Mean Spend Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cluster 1** | **Affluent VIPs** | **High Spenders** | 19.5% (39) | 32.7 yrs | $86.5k | **82.1 / 100** |
| **Cluster 2** | **Young Enthusiasts** | **High Spenders** | 11.0% (22) | 25.3 yrs | $25.7k | **79.4 / 100** |
| **Cluster 0** | **Mainstream Middle** | **Medium Spenders** | 40.5% (81) | 42.7 yrs | $55.3k | **49.5 / 100** |
| **Cluster 3** | **Careful Savers** | **Low Spenders** | 17.5% (35) | 41.1 yrs | $88.2k | **17.1 / 100** |
| **Cluster 4** | **Budget Conscious** | **Low Spenders** | 11.5% (23) | 45.2 yrs | $26.3k | **20.9 / 100** |

---

## 8. Overarching Spending Tiers

As required by the capstone specification, clusters are grouped into three primary spending tiers:

```
                              [All Mall Customers (200)]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  HIGH SPENDERS                   MEDIUM SPENDERS                   LOW SPENDERS
  (61 customers, 30.5%)           (81 customers, 40.5%)            (58 customers, 29.0%)
  Avg Spend: 81.1/100             Avg Spend: 49.5/100              Avg Spend: 18.6/100
  Avg Income: $64.6k              Avg Income: $55.3k               Avg Income: $63.6k
  Avg Age: 30.0 yrs               Avg Age: 42.7 yrs                Avg Age: 42.7 yrs
  ┌─────────────┬─────────────┐          │                  ┌─────────────┬─────────────┐
  ▼             ▼                        ▼                  ▼             ▼
Affluent      Young                  Mainstream          Careful        Budget
 VIPs      Enthusiasts                 Middle             Savers       Conscious
(19.5%)      (11.0%)                  (40.5%)            (17.5%)        (11.5%)
```

---

## 9. Visual Analytics

The pipeline generates three high-resolution visual deliverables:

1. **`customer_segmentation_overview.png`**:
   - Stacked histograms and KDE curves for Age, Income, and Spending Score.
   - Comparative boxplots showing feature dispersion across genders.
   - Pearson correlation heatmap.
   - Age vs. Spending Score scatter plot with bubble size scaled by income.

2. **`customer_segmentation_clusters.png`**:
   - Panel 1: WCSS Elbow curve highlighting $K=5$.
   - Panel 2: Silhouette analysis curve with peak marker.
   - Panel 3: Spending Tier distribution bar chart.
   - Panel 4: 2D Cluster Map (Annual Income vs Spending Score) showing colored clusters, labeled centroids, and decision boundaries.
   - Panel 5: Spending Score distribution boxplots across the 5 personas.

3. **`customer_segmentation_3d.png`**:
   - 3D spatial plot displaying Age on X-axis, Annual Income on Y-axis, and Spending Score on Z-axis with distinctive cluster markers.

---

## 10. Business Analytics & Retail Optimization Strategy

### 🌟 Cluster 1: Affluent VIPs (High Spenders)
- **Profile**: Younger professionals (Mean Age 33), high disposable income ($86.5k), peak spending (82.1).
- **Retail Strategy**: 
  - Host invite-only luxury fashion events and private product showcases.
  - Dedicate premium parking and personal concierge shopping assistance.
  - Implement white-glove loyalty tiers (e.g., Platinum/Black Card) with zero-friction checkout.

### ⚡ Cluster 2: Young Enthusiasts (High Spenders)
- **Profile**: Gen-Z and college students (Mean Age 25), low income ($25.7k), very high spending (79.4).
- **Retail Strategy**:
  - Introduce trend-driven pop-ups, fast-fashion boutiques, and experiential entertainment (VR gaming, trendy cafes).
  - Launch social media campaigns on TikTok and Instagram with influencer meet-and-greets.
  - Partner with Buy-Now-Pay-Later (BNPL) providers (e.g., Klarna, Afterpay) to facilitate higher transaction frequency.

### ⚖️ Cluster 0: Mainstream Middle (Medium Spenders)
- **Profile**: Core families (Mean Age 43), moderate income ($55.3k), balanced spending (49.5). Largest demographic (40.5%).
- **Retail Strategy**:
  - Focus on family entertainment (cinemas, family restaurants, kids' play areas).
  - Seasonal marketing: Back-to-school bundles, holiday festivals, weekend promotions.
  - Cross-store reward programs redeemable across supermarket and apparel anchors.

### 🛡️ Cluster 3: Careful Savers (Low Spenders)
- **Profile**: Mature consumers (Mean Age 41), high income ($88.2k), very low spending (17.1).
- **Retail Strategy**:
  - Emphasize durability, brand heritage, and long-term investment value rather than transient fashion.
  - Promote high-ticket quality goods (smart home tech, premium appliances, luxury watches).
  - Provide extended warranties, comprehensive service plans, and educational product demos.

### 🏷️ Cluster 4: Budget Conscious (Low Spenders)
- **Profile**: Older shoppers (Mean Age 45), low income ($26.3k), low spending (20.9).
- **Retail Strategy**:
  - Position discount retailers, outlet stores, and essential grocery chains.
  - Distribute coupon booklets, clearance notifications, and bulk-buy discount vouchers.
  - Run weekday off-peak discount hours to drive foot traffic during slow periods.

---

## 11. Executive Presentation Slide Deck

### Slide 1: Title & Purpose
- **Title**: *Data-Driven Customer Segmentation for Retail Mall Strategy*
- **Presenter**: Machine Learning & AI Team
- **Objective**: Maximize tenant sales, foot-traffic monetization, and customer lifetime value using Unsupervised Machine Learning.

### Slide 2: The Challenge
- Malls face rising competition from e-commerce and changing consumer spending habits.
- Mass-broadcast marketing yields dwindling engagement and wasted budget.
- Solution: Segment consumers according to empirical purchasing behaviors.

### Slide 3: Methodology & Pipeline
- Rigorous exploratory data analysis on 200 mall shoppers.
- Scaling and feature normalization.
- Quantitative hyperparameter selection via Elbow and Silhouette methods.
- K-Means ($K=5$) clustering with centroid convergence.

### Slide 4: Optimal $K$ Discovery
- Evaluation over $K \in [2, 10]$.
- Elbow inflection at $K=5$.
- Silhouette coefficient peak of **0.5539** ($K=5$) vs 0.4676 ($K=3$) and 0.4932 ($K=4$).

### Slide 5: The 5 Customer Personas
- **Affluent VIPs** (19.5%): Young, affluent, luxury seekers.
- **Young Enthusiasts** (11.0%): Trendy, impulsive, youth trendsetters.
- **Mainstream Middle** (40.5%): Balanced, family-oriented base.
- **Careful Savers** (17.5%): Wealthy but conservative spenders.
- **Budget Conscious** (11.5%): Price-sensitive essentials shoppers.

### Slide 6: Three Strategic Spending Tiers
- **High Spenders**: 30.5% of visitors | Strategic focus: VIP exclusivity & fast fashion.
- **Medium Spenders**: 40.5% of visitors | Strategic focus: Family loyalty & community events.
- **Low Spenders**: 29.0% of visitors | Strategic focus: Value messaging & discount clearance.

### Slide 7: Actionable ROI Implementation
- Tenant Mix Optimization: Balance luxury flagships, trendy boutiques, and discount anchors.
- Marketing Budget Allocation: Direct 50% of ad spend to High Spenders, 35% to Mainstream Middle, 15% to Low Spenders.
- Real-Time Integration: Instant point-of-sale customer scoring API.

---

## 12. Model Serialization & Production Deployment

The trained clustering model and preprocessing objects are serialized for real-time inference:

- **Model File**: [`kmeans_customer_model.joblib`](kmeans_customer_model.joblib)
- **Scaler File**: [`scaler.joblib`](scaler.joblib)
- **Enriched Dataset**: [`mall_customers_segmented.csv`](mall_customers_segmented.csv)

### Python Inference Code Snippet:
```python
import joblib

# Load serialized model
model = joblib.load('week4/kmeans_customer_model.joblib')

# Score new customer: Income $85k, Spending Score 82
prediction = model.predict([[85, 82]])[0]
tier_map = {0: 'Medium Spenders', 1: 'High Spenders', 2: 'High Spenders', 3: 'Low Spenders', 4: 'Low Spenders'}
print(f"Assigned Cluster: {prediction} | Spending Tier: {tier_map[prediction]}")
# Output: Assigned Cluster: 1 | Spending Tier: High Spenders
```

---
*Capstone Project Completed as part of Machine Learning & AI Course Work.*
