# Week 7 Tutorial: Unsupervised Learning — Customer Segmentation
## "Your First Clustering Pipeline"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a complete customer segmentation pipeline — standardizing features, reducing dimensions with PCA, finding the optimal number of clusters, training K-Means and DBSCAN, detecting anomalies, and profiling each segment into actionable business insights.

---

### 1. What We Are Building

Last week, we classified fraudulent transactions using labeled data. Today, we have **no labels**. We are given a dataset of customer behavior and must discover natural groupings — without being told what those groups should be.

**By the end of this session, you will have:**
- Built a clean preprocessing pipeline with standardization and PCA.
- Used the elbow method and silhouette analysis to pick the optimal $K$.
- Trained K-Means and visualized the clusters.
- Applied DBSCAN and compared results to K-Means.
- Detected anomalous customers using Isolation Forest.
- Profiled each cluster to understand *who* lives inside each segment.

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Verify key versions
print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")
print("\nReady to discover hidden customer segments!")
```

---

### 3. Creating Our Dataset

We'll generate a synthetic dataset of **2,000 e-commerce customers** with realistic behavioral features.

```python
np.random.seed(42)
n = 2000

# Define 4 latent customer segments — the "ground truth" we want to discover
# Segment 0: "High-Value Loyals"  — high spend, frequent, long tenure
# Segment 1: "Bargain Hunters"    — low spend, high frequency, short tenure
# Segment 2: "Dormant At-Risk"    — medium spend, very low frequency
# Segment 3: "New High-Spenders"  — high spend, low frequency, new

segment_sizes = [500, 600, 550, 350]

def make_segment(size, spend_mean, freq_mean, tenure_mean, recency_mean, categories_mean):
    return {
        'avg_order_value':   np.random.normal(spend_mean,   spend_mean * 0.2,   size).clip(5, 500),
        'purchase_frequency': np.random.normal(freq_mean,   freq_mean * 0.3,    size).clip(0.5, 50),
        'account_tenure_months': np.random.normal(tenure_mean, tenure_mean * 0.25, size).clip(1, 60),
        'days_since_last_purchase': np.random.normal(recency_mean, recency_mean * 0.3, size).clip(1, 365),
        'num_categories_purchased': np.random.normal(categories_mean, 1.5, size).clip(1, 15).astype(int),
    }

segments_data = [
    make_segment(segment_sizes[0], spend_mean=180, freq_mean=12, tenure_mean=36, recency_mean=15,  categories_mean=8),
    make_segment(segment_sizes[1], spend_mean=45,  freq_mean=20, tenure_mean=12, recency_mean=10,  categories_mean=4),
    make_segment(segment_sizes[2], spend_mean=95,  freq_mean=2,  tenure_mean=42, recency_mean=150, categories_mean=3),
    make_segment(segment_sizes[3], spend_mean=220, freq_mean=3,  tenure_mean=5,  recency_mean=20,  categories_mean=6),
]

# Concatenate all segments
data = {}
for key in segments_data[0].keys():
    data[key] = np.concatenate([seg[key] for seg in segments_data])

df = pd.DataFrame(data)
df['segment_gt'] = np.concatenate([
    np.full(sz, i) for i, sz in enumerate(segment_sizes)
])  # Ground-truth labels — we will NOT use these in training

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features (no labels used in training!)
feature_cols = ['avg_order_value', 'purchase_frequency', 'account_tenure_months',
                'days_since_last_purchase', 'num_categories_purchased']
X = df[feature_cols].values

print(f"Dataset shape: {df.shape}")
print(f"\nFeature summary:")
df[feature_cols].describe().round(2)
```

---

### 4. Step 1 — Exploratory Data Analysis

#### 4.1 Feature Distributions

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#4ade80', '#ef4444']

for ax, col, color in zip(axes.flatten(), feature_cols, colors):
    ax.hist(df[col], bins=40, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide the 6th subplot (we only have 5 features)
axes[1, 2].set_visible(False)

plt.suptitle('Customer Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('customer_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
```

#### 4.2 Correlation Heatmap

```python
import matplotlib.colors as mcolors

corr = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
labels = [c.replace('_', '\n') for c in feature_cols]
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)

# Add values
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)

ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: Which features are most correlated? Does this make business sense?

---

### 5. Step 2 — Preprocessing: Standardization & PCA

#### 5.1 Why Standardize?

K-Means uses Euclidean distance. Without standardization, features with large scales (e.g., `days_since_last_purchase`, range: 1–365) dominate over features with small scales (`num_categories_purchased`, range: 1–15).

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before scaling:")
print(f"  avg_order_value:         mean={X[:, 0].mean():.1f}, std={X[:, 0].std():.1f}")
print(f"  days_since_last_purchase: mean={X[:, 3].mean():.1f}, std={X[:, 3].std():.1f}")
print("\nAfter scaling:")
print(f"  avg_order_value:         mean={X_scaled[:, 0].mean():.1f}, std={X_scaled[:, 0].std():.1f}")
print(f"  days_since_last_purchase: mean={X_scaled[:, 3].mean():.1f}, std={X_scaled[:, 3].std():.1f}")
```

#### 5.2 PCA — Reduce to 2D for Visualization

```python
# Full PCA to see variance explained
pca_full = PCA()
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Individual variance explained
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, color='#3b82f6', alpha=0.8)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Variance Explained per Component')

# Plot 2: Cumulative variance explained
axes[1].plot(range(1, len(cumvar)+1), cumvar, 'bo-', markersize=8)
axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].set_ylim(0, 1.05)

plt.suptitle('PCA Variance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=100, bbox_inches='tight')
plt.show()

n_for_95 = np.argmax(cumvar >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_for_95}")
print(f"Individual component variance: {pca_full.explained_variance_ratio_.round(3)}")
```

```python
# Reduce to 2D for clustering and visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

print(f"\nOriginal dimensions: {X_scaled.shape[1]}")
print(f"Reduced dimensions:  {X_pca.shape[1]}")
print(f"Total variance retained: {pca_2d.explained_variance_ratio_.sum():.2%}")
```

---

### 6. Step 3 — Finding the Optimal K

#### 6.1 Elbow Method

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(X_pca)
    inertias.append(km.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(K_range, inertias, 'bo-', markersize=9, linewidth=2)
for k, inertia in zip(K_range, inertias):
    plt.annotate(f'{inertia:.0f}', (k, inertia), textcoords='offset points', xytext=(8, 5), fontsize=8)
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=100, bbox_inches='tight')
plt.show()
```

#### 6.2 Silhouette Analysis

```python
silhouette_scores = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)
    print(f"K={k:2d}: Silhouette Score = {score:.4f}")

best_k = np.argmax(silhouette_scores) + 2   # +2 because we start at K=2

plt.figure(figsize=(9, 5))
plt.plot(range(2, 11), silhouette_scores, 'go-', markersize=9, linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('Number of Clusters K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. K')
plt.legend()
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\n✓ Optimal K by Silhouette: {best_k}")
```

---

### 7. Step 4 — Train K-Means

#### 7.1 Fit the Final K-Means Model

```python
# Use best_k found above
km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = km_final.fit_predict(X_pca)

print(f"K-Means fitted with K={best_k}")
print(f"Inertia: {km_final.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_pca, cluster_labels):.4f}")
print(f"\nCluster sizes:")
for k in range(best_k):
    count = (cluster_labels == k).sum()
    print(f"  Cluster {k}: {count} customers ({count/n:.1%})")
```

#### 7.2 Visualize K-Means Clusters (2D PCA Projection)

```python
colors = ['#3b82f6', '#ef4444', '#4ade80', '#f59e0b', '#8b5cf6', '#06b6d4']

fig, ax = plt.subplots(figsize=(11, 7))

for k in range(best_k):
    mask = cluster_labels == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[k], label=f'Cluster {k}', alpha=0.6, s=40, edgecolors='white', linewidth=0.3)

# Plot centroids (transformed to PCA space)
centroids_pca = km_final.cluster_centers_
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='black', marker='X', s=200, zorder=5, label='Centroids')

for k, (cx, cy) in enumerate(centroids_pca):
    ax.annotate(f'C{k}', (cx, cy), textcoords='offset points', xytext=(8, 8),
                fontsize=12, fontweight='bold')

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax.set_title(f'K-Means Customer Clusters (K={best_k}) — PCA 2D Projection', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=100, bbox_inches='tight')
plt.show()
```

#### 7.3 Deep Silhouette Plot

```python
fig, ax = plt.subplots(figsize=(10, 7))
sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)

y_lower = 10
for k in range(best_k):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == k]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(k) / best_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))
    y_lower = y_upper + 10

avg_score = silhouette_score(X_pca, cluster_labels)
ax.axvline(x=avg_score, color="red", linestyle="--", label=f'Mean: {avg_score:.3f}')
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot for K={best_k}")
ax.legend()
plt.tight_layout()
plt.savefig('silhouette_plot.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 8. Step 5 — DBSCAN Clustering

```python
# DBSCAN on PCA-reduced data
# eps and min_samples require experimentation
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels_db = dbscan.fit_predict(X_pca)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise_db = (labels_db == -1).sum()

print(f"DBSCAN Results:")
print(f"  Estimated clusters: {n_clusters_db}")
print(f"  Noise points:       {n_noise_db} ({n_noise_db/n:.1%})")

if n_clusters_db >= 2:
    db_mask = labels_db != -1
    db_score = silhouette_score(X_pca[db_mask], labels_db[db_mask])
    print(f"  Silhouette (excl. noise): {db_score:.4f}")

# Visualize DBSCAN results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means
for k in range(best_k):
    mask = cluster_labels == k
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[k % len(colors)],
                    label=f'Cluster {k}', alpha=0.6, s=30)
axes[0].set_title(f'K-Means (K={best_k})', fontsize=12)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(fontsize=9)

# DBSCAN
db_labels_unique = sorted(set(labels_db))
for label in db_labels_unique:
    mask = labels_db == label
    if label == -1:
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x',
                        s=30, label='Noise', alpha=0.5)
    else:
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[label % len(colors)],
                        label=f'Cluster {label}', alpha=0.6, s=30)

axes[1].set_title(f'DBSCAN (eps=0.5, min_samples=10)\n{n_clusters_db} clusters, {n_noise_db} noise', fontsize=12)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(fontsize=9)

plt.suptitle('K-Means vs. DBSCAN Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: How do the cluster shapes differ between K-Means and DBSCAN? Which noise points did DBSCAN find? Would you investigate them further?

---

### 9. Step 6 — Anomaly Detection with Isolation Forest

```python
# Detect anomalous customers on the original scaled space
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.03,   # we expect ~3% anomalies
    random_state=42
)
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores  = iso_forest.score_samples(X_scaled)  # lower = more anomalous

n_anomalies = (anomaly_labels == -1).sum()
print(f"Isolation Forest — Anomaly Detection")
print(f"  Total customers:        {n}")
print(f"  Anomalies detected:     {n_anomalies} ({n_anomalies/n:.1%})")

# Visualize anomalies on PCA plot
fig, ax = plt.subplots(figsize=(11, 7))

normal_mask  = anomaly_labels == 1
anomaly_mask = anomaly_labels == -1

ax.scatter(X_pca[normal_mask, 0],  X_pca[normal_mask, 1],
           c='#3b82f6', alpha=0.5, s=30, label='Normal', edgecolors='none')
ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
           c='#ef4444', marker='*', s=150, label=f'Anomaly ({n_anomalies})',
           edgecolors='black', linewidth=0.5, zorder=5)

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('Isolation Forest — Anomaly Detection on Customer Data')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=100, bbox_inches='tight')
plt.show()

# Profile the anomalies
df_anomalies = df[feature_cols][anomaly_mask]
df_normals   = df[feature_cols][normal_mask]

print("\nAnomaly profile vs. Normal profile:")
comparison = pd.DataFrame({
    'Normal Mean':  df_normals.mean(),
    'Anomaly Mean': df_anomalies.mean(),
}).round(2)
print(comparison)
```

---

### 10. Step 7 — Cluster Profiling & Business Insights

This is the most important step: turning cluster numbers into **business knowledge**.

```python
# Attach K-Means cluster labels back to the dataframe
df['cluster'] = cluster_labels

# Compute mean features per cluster
cluster_profiles = df.groupby('cluster')[feature_cols].mean().round(2)
print("=" * 70)
print("CLUSTER PROFILES — Mean Feature Values per Cluster")
print("=" * 70)
print(cluster_profiles.to_string())
```

```python
# Radar / bar chart comparison of clusters
fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 5), sharey=False)

cluster_labels_list = range(best_k)

# Normalize profile values between 0 and 1 for comparison
profile_normalized = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

for k, ax in zip(cluster_labels_list, axes.flatten()):
    values = profile_normalized.loc[k]
    bars = ax.barh(feature_cols, values, color=colors[k % len(colors)], alpha=0.8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Normalized Value')
    ax.set_title(f'Cluster {k}\n(n={( df["cluster"]==k).sum()})', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, (feat, val) in zip(bars, zip(feature_cols, cluster_profiles.loc[k])):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)

plt.suptitle('Customer Cluster Profiles (Normalized)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=100, bbox_inches='tight')
plt.show()
```

```python
# Assign business names to clusters (adjust based on your actual results)
# This is the step that turns data science into business intelligence
print("\n" + "=" * 60)
print("CLUSTER BUSINESS INTERPRETATION")
print("=" * 60)

for k in range(best_k):
    profile = cluster_profiles.loc[k]
    size = (df['cluster'] == k).sum()
    print(f"\n--- Cluster {k} ({size} customers, {size/n:.1%}) ---")
    print(f"  Avg Order Value:          {profile['avg_order_value']:.1f}")
    print(f"  Purchase Frequency/month: {profile['purchase_frequency']:.1f}")
    print(f"  Account Tenure (months):  {profile['account_tenure_months']:.1f}")
    print(f"  Days Since Last Purchase: {profile['days_since_last_purchase']:.1f}")
    print(f"  Categories Purchased:     {profile['num_categories_purchased']:.1f}")

    # Simple rule-based interpretation
    if profile['avg_order_value'] > 150 and profile['purchase_frequency'] > 8:
        tag = "🏆 High-Value Loyals — Reward & Upsell"
    elif profile['days_since_last_purchase'] > 120:
        tag = "⚠️  Dormant At-Risk — Re-engagement Campaign"
    elif profile['avg_order_value'] < 70 and profile['purchase_frequency'] > 10:
        tag = "💰 Bargain Hunters — Promotions & Cross-sell"
    elif profile['avg_order_value'] > 150 and profile['account_tenure_months'] < 15:
        tag = "🚀 New High-Spenders — Nurture & Retain"
    else:
        tag = "📊 Mixed Segment — Investigate Further"

    print(f"  → Business Tag: {tag}")
```

---

### 11. Step 8 — Full Pipeline Summary

```python
# Consolidated pipeline for reproducible segmentation
print("=" * 60)
print("FULL UNSUPERVISED PIPELINE SUMMARY")
print("=" * 60)

summary = {
    "Dataset Size": n,
    "Features Used": len(feature_cols),
    "Scaling": "StandardScaler",
    "Dimensionality Reduction": f"PCA 2D ({pca_2d.explained_variance_ratio_.sum():.1%} variance retained)",
    "Optimal K (Silhouette)": best_k,
    "K-Means Inertia": f"{km_final.inertia_:.2f}",
    "Silhouette Score": f"{silhouette_score(X_pca, cluster_labels):.4f}",
    "DBSCAN Clusters Found": n_clusters_db,
    "DBSCAN Noise Points": n_noise_db,
    "Anomalies (Isolation Forest)": n_anomalies,
}

for key, val in summary.items():
    print(f"  {key:<40} {val}")
```

---

### 12. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Analysis</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Skip the PCA step and run K-Means directly on `X_scaled`. Does the silhouette score improve or worsen? Do the cluster shapes change in the 2D plot?</li>
        <li><strong>Experiment B:</strong> Tune DBSCAN's `eps` parameter: try 0.3, 0.5, 0.8, 1.2. How does the number of clusters and noise points change? Plot results side-by-side.</li>
        <li><strong>Experiment C:</strong> Add a 6th feature: `total_lifetime_value = avg_order_value * purchase_frequency`. Does this new feature change the cluster composition?</li>
        <li><strong>Experiment D:</strong> Download the real UCI Online Retail dataset from Kaggle (contains real purchase histories). Apply this same pipeline and discover real customer segments. Do your segments match RFM (Recency, Frequency, Monetary) analysis?</li>
    </ul>
</div>

---

### 13. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that:
    1. Loads and preprocesses a dataset of your choice (e.g., UCI Online Retail, Mall Customer Segmentation from Kaggle, or our synthetic data).
    2. Standardizes features and applies PCA, reporting variance retained.
    3. Uses **both** the elbow method and silhouette score to justify the choice of $K$.
    4. Trains K-Means and DBSCAN, comparing their results visually and quantitatively.
    5. Applies Isolation Forest to detect anomalies and profiles the anomalous points.
    6. Produces a **cluster profile table** with a written business interpretation for each cluster (3-5 sentences per cluster).
*   **Report**: Write a paragraph (4-6 sentences) explaining why you cannot use accuracy to evaluate clustering, and what metrics you use instead. Include the silhouette score formula.
*   **Bonus**: Implement K-Means from scratch using only NumPy — no Scikit-learn `KMeans`. Compare your results to Scikit-learn's implementation on the same data. How many iterations until convergence?

**See you at the Lecture! Without labels, data doesn't shout — it whispers.**
