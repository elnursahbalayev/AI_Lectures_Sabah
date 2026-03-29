# Week 7 Lecture: Unsupervised Learning
## "Finding Structure in the Dark — Learning Without Labels"

*Module 2: Applied Machine Learning | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **In the real world, labeled data is expensive. Unsupervised learning finds hidden structure in raw data without any labels — discovering natural groupings, compressing information, and detecting anomalies that don't belong.**

In Weeks 4-6, every algorithm had a target label $y$. Today, we remove that crutch. We only have $X$ — and we must discover what the data is telling us.

**We will cover:**
1. The Landscape of Unsupervised Learning
2. K-Means Clustering — The Classic Algorithm
3. Choosing K — The Elbow Method & Silhouette Score
4. The Limitations of K-Means
5. DBSCAN — Density-Based Clustering
6. Dimensionality Reduction — The Curse and the Cure
7. Principal Component Analysis (PCA)
8. PCA Mathematics — Eigenvectors & Variance Explained
9. Anomaly Detection Concepts
10. Practical Implementation

---

### 1. The Landscape of Unsupervised Learning

#### 1.1 Supervised vs. Unsupervised

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Data** | $(X, y)$ — features + labels | $X$ — features only |
| **Goal** | Predict $y$ for new $X$ | Discover structure in $X$ |
| **Evaluation** | Accuracy, F1, MSE (vs. ground truth) | Silhouette score, inertia, visual inspection |
| **Examples** | Regression, Classification | Clustering, Dimensionality Reduction |

#### 1.2 Why Unsupervised Learning Matters

| Use Case | Description |
|---|---|
| **Customer Segmentation** | Group users by purchasing behavior — without knowing what the groups should be |
| **Anomaly Detection** | Identify fraudulent transactions, server failures, or defective products |
| **Data Compression** | Reduce 1000 features to 10, preserving most information |
| **Exploratory Analysis** | Visualize high-dimensional data in 2D/3D to find patterns |
| **Preprocessing** | Use PCA to reduce noise before training a supervised model |

#### 1.3 The Three Pillars

```
Unsupervised Learning
├── Clustering          — "What groups exist?"
│   ├── K-Means
│   ├── DBSCAN
│   └── Hierarchical Clustering
├── Dimensionality Reduction  — "What's the essence of this data?"
│   ├── PCA (Principal Component Analysis)
│   ├── t-SNE (for visualization)
│   └── Autoencoders (Week 13)
└── Anomaly Detection   — "What doesn't belong?"
    ├── Isolation Forest
    └── One-Class SVM
```

---

### 2. K-Means Clustering

#### 2.1 The Intuition

Imagine you drop $K$ pins randomly on a map and assign each city to its nearest pin. Then you move each pin to the center of its assigned cities. Repeat until the pins stop moving. That's K-Means.

#### 2.2 The Algorithm

**Input**: Data $X = \{x_1, \ldots, x_n\}$, number of clusters $K$.

1. **Initialize**: Randomly place $K$ cluster centroids $\mu_1, \ldots, \mu_K$.
2. **Assign**: Each point $x_i$ is assigned to the nearest centroid:
   $$c_i = \arg\min_{k} \|x_i - \mu_k\|^2$$
3. **Update**: Move each centroid to the mean of its assigned points:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
4. **Repeat** steps 2-3 until centroids stop moving (convergence).

#### 2.3 The Objective Function

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

> **K-Means is an optimization algorithm.** It performs coordinate descent on WCSS, alternating between fixing centroids and updating assignments.

#### 2.4 Convergence Guarantee

K-Means always converges but may converge to a **local minimum**, not the global minimum. The initialization matters:

| Initialization Method | Description |
|---|---|
| **Random** | Centroids placed at random points — may give poor results |
| **K-Means++** | Smart initialization — first centroid random, each subsequent centroid chosen with probability proportional to distance from existing centroids |

**K-Means++ is the default in Scikit-learn** and dramatically improves results in practice.

#### 2.5 K-Means in Python

```python
from sklearn.cluster import KMeans
import numpy as np

# Fit K-Means with K=3 clusters
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',    # smart initialization
    n_init=10,           # run 10 times, keep best result
    max_iter=300,        # maximum iterations per run
    random_state=42
)
kmeans.fit(X)

# Results
labels = kmeans.labels_          # cluster assignment for each point
centroids = kmeans.cluster_centers_  # location of each centroid
inertia = kmeans.inertia_        # WCSS value (lower = tighter clusters)

print(f"Cluster labels: {np.unique(labels)}")
print(f"Inertia (WCSS): {inertia:.2f}")
```

#### 2.6 Assumptions and Limitations

K-Means makes strong assumptions about cluster shape:

| Assumption | What It Means | When It Fails |
|---|---|---|
| **Spherical clusters** | Each cluster is roughly circular | Elongated, crescent-shaped, or irregular clusters |
| **Similar size** | All clusters have roughly equal number of points | One cluster has 10 points, another has 10,000 |
| **Similar density** | Points are equally dense across clusters | Sparse outer clusters mixed with dense core clusters |
| **K is known** | You must specify $K$ in advance | You don't know how many groups exist |

---

### 3. Choosing K — The Elbow Method & Silhouette Score

#### 3.1 The Elbow Method

Run K-Means for $K = 1, 2, \ldots, 10$ and plot the inertia (WCSS) vs. $K$. Look for the "elbow" — the point where adding more clusters stops significantly reducing inertia.

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# Plot
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.tight_layout()
plt.show()
```

> **The elbow point** is where the inertia "bends" — adding more clusters after this point gives diminishing returns.

#### 3.2 Silhouette Score — A Better Metric

The **silhouette score** measures how well each point fits its cluster vs. the nearest alternative cluster:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance from point $i$ to all other points **in its cluster** (intra-cluster distance)
- $b(i)$ = average distance from point $i$ to all points **in the nearest other cluster** (inter-cluster distance)

| Score Range | Interpretation |
|---|---|
| $s \approx +1$ | Point is well inside its cluster, far from others — **excellent** |
| $s \approx 0$ | Point is on the boundary between two clusters |
| $s \approx -1$ | Point is likely assigned to the wrong cluster |

The **mean silhouette score** across all points: higher is better. Choose $K$ that maximizes it.

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):  # silhouette requires at least 2 clusters
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")
```

---

### 4. DBSCAN — Density-Based Spatial Clustering

#### 4.1 The Core Idea

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as **dense regions separated by sparse regions**. Unlike K-Means:

- You don't specify $K$ — the number of clusters is discovered automatically.
- It can find **arbitrary-shaped clusters** (crescents, spirals, rings).
- It explicitly identifies **noise points** (outliers) that belong to no cluster.

#### 4.2 Key Concepts

| Concept | Definition |
|---|---|
| **eps (ε)** | The radius of the neighborhood around each point |
| **min_samples** | Minimum number of points within eps to be a core point |
| **Core Point** | A point with at least `min_samples` neighbors within eps |
| **Border Point** | A point within eps of a core point but not itself a core point |
| **Noise Point** | A point that is neither a core point nor a border point — **outlier**, labeled -1 |

#### 4.3 The DBSCAN Algorithm

1. For each point $x_i$, find all points within radius $\varepsilon$: the **ε-neighborhood** $N_\varepsilon(x_i)$.
2. If $|N_\varepsilon(x_i)| \geq \text{min\_samples}$, mark $x_i$ as a **core point** and start a new cluster.
3. Expand the cluster by adding all reachable points (transitively through core points).
4. Points not reachable from any core point are labeled **noise** (label = -1).

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,          # neighborhood radius
    min_samples=5     # minimum points to be core
)
labels_db = dbscan.fit_predict(X)

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")
```

#### 4.4 K-Means vs. DBSCAN

| Feature | K-Means | DBSCAN |
|---|---|---|
| **Number of clusters** | Must specify $K$ | Automatically discovered |
| **Cluster shape** | Spherical only | Arbitrary shapes |
| **Outliers** | All points assigned to a cluster | Noise points explicitly labeled |
| **Scalability** | Fast — $O(nKI)$ | Slower — $O(n \log n)$ with spatial index |
| **Hyperparameters** | $K$ | $\varepsilon$, min_samples |
| **Best for** | Well-separated, spherical clusters | Irregular shapes, noisy data |

> **Rule of thumb**: Start with K-Means. If clusters are irregular or you don't know $K$, try DBSCAN.

---

### 5. Dimensionality Reduction — The Curse and the Cure

#### 5.1 The Curse of Dimensionality

As the number of features $p$ grows:

1. **Data becomes sparse**: In a 100-dimensional space, every point is far from every other point. "Nearest neighbor" loses meaning.
2. **Distance concentration**: All pairwise distances converge to the same value — the maximum distance provides no contrast.
3. **Exponential sample requirements**: To maintain the same statistical coverage, you need exponentially more data as $p$ increases.

$$\text{Volume of a hypersphere in } d \text{ dimensions} \propto r^d \to 0 \text{ as } d \to \infty$$

#### 5.2 Why Reduce Dimensions?

| Reason | Benefit |
|---|---|
| **Visualization** | Reduce to 2D/3D to plot and inspect data |
| **Remove noise** | Low-variance components often capture noise |
| **Speed up training** | Fewer features → faster models |
| **Avoid the curse** | Dense data in lower dimensions works better |
| **Storage** | Compress data with little information loss |

---

### 6. Principal Component Analysis (PCA)

#### 6.1 The Core Idea

PCA finds a new coordinate system where:
- The **first axis (PC1)** points in the direction of **maximum variance** in the data.
- The **second axis (PC2)** is **perpendicular to PC1** and captures the next most variance.
- Each subsequent PC is perpendicular to all previous ones.

> **PCA rotates your data to find the most informative directions — then you can drop the least informative directions.**

#### 6.2 PCA Mathematics

Given a centered data matrix $X \in \mathbb{R}^{n \times p}$ (zero mean):

1. **Compute the covariance matrix**:
   $$\Sigma = \frac{1}{n-1} X^T X \in \mathbb{R}^{p \times p}$$

2. **Eigendecomposition**:
   $$\Sigma = V \Lambda V^T$$
   Where $V$ contains the **eigenvectors** (principal component directions) and $\Lambda$ is a diagonal matrix of **eigenvalues** $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$.

3. **Select top $d$ components**:
   $$W = V_{[:, 1:d]} \in \mathbb{R}^{p \times d}$$

4. **Project the data**:
   $$Z = X W \in \mathbb{R}^{n \times d}$$

#### 6.3 Variance Explained

Each eigenvalue $\lambda_k$ represents the **variance captured** by the $k$-th principal component:

$$\text{Variance Explained by PC}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}$$

**Cumulative variance explained** tells you how much information is retained with $d$ components:

$$\text{Cumulative Variance} = \frac{\sum_{k=1}^{d} \lambda_k}{\sum_{j=1}^{p} \lambda_j}$$

> **Common rule**: Choose $d$ such that cumulative variance explained ≥ 95%.

#### 6.4 PCA in Python

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Standardize (critical! PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)        # reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape:  {X_scaled.shape}")
print(f"Reduced shape:   {X_pca.shape}")
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total variance retained:   {pca.explained_variance_ratio_.sum():.2%}")
```

#### 6.5 Choosing the Number of Components

```python
# Fit PCA with all components to inspect variance
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative variance explained
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(range(1, len(cumvar)+1), cumvar, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA — Variance Explained')
plt.legend()
plt.show()

# How many components for 95%?
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

#### 6.6 PCA for Visualization: Applying to Labeled Data

PCA is extremely useful for **visualizing high-dimensional data** even when you aren't doing clustering:

```python
# Reduce to 2D and visualize with cluster labels
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA 2D Projection of Clusters')
plt.show()
```

---

### 7. PCA vs. t-SNE (A Brief Note)

| Method | PCA | t-SNE |
|---|---|---|
| **Type** | Linear transformation | Non-linear |
| **Speed** | Fast | Slow (scales poorly to large $n$) |
| **Preserves** | Global variance structure | Local neighborhood structure |
| **Use case** | Preprocessing, dimensionality reduction for ML | Visualization only |
| **Reproducible** | Yes | Stochastic (set random_state) |

> **Rule**: Use PCA for preprocessing (before training a model). Use t-SNE only for visualization — never use t-SNE output as features for downstream ML.

---

### 8. Anomaly Detection

#### 8.1 The Problem

**Anomaly detection** (also called outlier detection) identifies observations that are **significantly different from the majority**. Unlike fraud classification (supervised), we often don't have labels for what "anomalous" means.

| Type | Description | Example |
|---|---|---|
| **Point anomaly** | A single data point is anomalous | A temperature reading of 500°C |
| **Contextual anomaly** | Anomalous in context but not globally | $30°C$ in January (but normal in July) |
| **Collective anomaly** | A group of points is anomalous together | Network packets individually normal but collectively forming a DDoS |

#### 8.2 Isolation Forest

The **Isolation Forest** algorithm isolates anomalies rather than modeling normal behavior:

- **Insight**: Anomalies are rare and different → they are **easier to isolate** (require fewer splits).
- Build many random trees. Count how many splits are needed to isolate each point.
- Points that are isolated in **fewer splits** → anomalies (anomaly score close to 1).

$$\text{Anomaly Score}(x) \approx 1 \implies \text{likely anomaly}$$
$$\text{Anomaly Score}(x) \approx 0 \implies \text{likely normal}$$

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expected fraction of anomalies (5%)
    random_state=42
)
anomaly_labels = iso_forest.fit_predict(X)
# Returns: +1 for normal, -1 for anomaly

n_anomalies = (anomaly_labels == -1).sum()
print(f"Detected anomalies: {n_anomalies} ({n_anomalies/len(X):.1%})")
```

#### 8.3 DBSCAN for Anomaly Detection

DBSCAN naturally identifies outliers as its noise points (label = -1). This makes it a powerful, parameter-free anomaly detector when cluster structure is known.

```python
# Treat DBSCAN noise points as anomalies
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

anomalies = X[labels == -1]
normals = X[labels != -1]
print(f"Anomalies detected by DBSCAN: {len(anomalies)}")
```

---

### 9. Practical Workflow: Customer Segmentation

Here is the end-to-end pipeline we will implement in the tutorial, connecting all concepts:

```python
# Complete Unsupervised Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. Reduce dimensions with PCA (optional preprocessing)
pca = PCA(n_components=0.95)   # keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} features")

# 3. Find optimal K
inertias, silhouettes = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

# 4. Fit final model
best_k = np.argmax(silhouettes) + 2   # +2 because we started at K=2
km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = km_final.fit_predict(X_pca)

# 5. Interpret clusters
X_df = pd.DataFrame(X_raw, columns=feature_names)
X_df['Cluster'] = cluster_labels
print(X_df.groupby('Cluster').mean().round(2))
```

---

### 10. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| K-Means Objective (WCSS) | $\sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$ |
| Centroid Update | $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ |
| Silhouette Score | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ |
| PCA Covariance Matrix | $\Sigma = \frac{1}{n-1} X^T X$ |
| Variance Explained by PC$_k$ | $\lambda_k / \sum_j \lambda_j$ |

---

### 11. Summary

1. **Unsupervised learning** finds structure in data without labels — clustering, dimensionality reduction, and anomaly detection.
2. **K-Means** partitions data into $K$ spherical clusters by minimizing within-cluster sum of squares. Use the elbow method and silhouette score to choose $K$.
3. **DBSCAN** finds arbitrary-shaped clusters and explicitly labels outliers as noise — no need to specify $K$.
4. **The Curse of Dimensionality** makes high-dimensional data sparse and distances meaningless — dimensionality reduction is essential.
5. **PCA** rotates data to find directions of maximum variance, allowing you to retain 95%+ of information with far fewer features.
6. **Anomaly Detection** identifies points that don't belong — Isolation Forest isolates anomalies through random splitting, DBSCAN labels them as noise.
7. **Always standardize** your data before clustering or PCA — algorithms based on distance are sensitive to feature scale.
8. **Profiling the clusters** (computing mean feature values per cluster) is how you turn cluster labels into business insights.

---

### 12. Bridge to the Tutorial

In the tutorial session, you will:
- Generate a synthetic customer behavioral dataset.
- Standardize features and apply PCA for dimension reduction.
- Use the elbow method and silhouette analysis to find the optimal $K$.
- Train K-Means and visualize clusters in 2D (PCA projection).
- Apply DBSCAN and compare its clusters to K-Means.
- Detect anomalies using Isolation Forest.
- Profile each customer segment (what does Cluster 0 vs. Cluster 1 look like?).

**Come to the tutorial. Without labels, you must listen to what the data whispers.**
