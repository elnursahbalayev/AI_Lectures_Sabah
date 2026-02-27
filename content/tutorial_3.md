# Week 3 Tutorial: Data Cleaning Lab
## "Turning Chaos into Features"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have taken a messy, "dirty" dataset and transformed it into clean, model-ready input using Pandas and scikit-learn.

---

### 1. What We Are Building

Last week, we built an AI that plays Tic-Tac-Toe perfectly. Today, we tackle the less glamorous but equally critical task: **data engineering**. We will clean a messy dataset of system server logs — the kind of data you'll encounter in real industry projects.

**By the end of this session, you will have:**
- Explored a dirty dataset and identified all its problems.
- Handled missing values using multiple strategies.
- Encoded categorical features (One-Hot and Label Encoding).
- Scaled numerical features to prepare them for ML algorithms.
- Built a reusable scikit-learn preprocessing pipeline.

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Verify versions
print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")

print("\nReady to clean some data!")
```

---

### 3. Creating Our Dirty Dataset

In practice, you'd load a CSV. For this tutorial, we'll create a realistic dirty dataset programmatically — a server monitoring log with the kinds of problems you'll face in production.

```python
np.random.seed(42)
n = 200

# Generate base data
data = {
    'server_id': [f'SRV-{i:03d}' for i in range(1, n+1)],
    'cpu_usage': np.random.uniform(10, 95, n),
    'memory_gb': np.random.uniform(2, 64, n),
    'disk_io': np.random.uniform(50, 500, n),
    'network_mbps': np.random.uniform(10, 1000, n),
    'os_type': np.random.choice(['Linux', 'Windows', 'linux', 'LINUX', 'windows', 'Win'], n),
    'region': np.random.choice(['US-East', 'US-West', 'EU-Central', 'Asia-Pacific', None], n),
    'priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n),
    'uptime_hours': np.random.uniform(0, 8760, n),
    'failed': np.random.choice([0, 1], n, p=[0.7, 0.3])  # Target variable
}

df = pd.DataFrame(data)

# Introduce missing values
for col in ['cpu_usage', 'memory_gb', 'disk_io']:
    mask = np.random.random(n) < 0.1  # 10% missing
    df.loc[mask, col] = np.nan

# Introduce outliers
df.loc[5, 'cpu_usage'] = 350.0   # Impossible: CPU usage > 100%
df.loc[42, 'memory_gb'] = -16.0  # Impossible: negative memory
df.loc[99, 'network_mbps'] = 50000  # Extreme outlier

# Introduce duplicates
df = pd.concat([df, df.iloc[10:15]], ignore_index=True)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()
```

---

### 4. Step 1 — Exploration (Know Your Data)

Before cleaning, you must **understand** the mess. Never clean blindly.

#### 4.1 Basic Info

```python
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nBasic statistics:")
df.describe()
```

#### 4.2 Missing Values Report

```python
def missing_report(df):
    """Generate a missing values report."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    report = pd.DataFrame({
        'Missing Count': missing,
        'Percent (%)': percent.round(1)
    })
    report = report[report['Missing Count'] > 0].sort_values('Percent (%)', ascending=False)
    return report

print("\nMissing Values Report:")
print(missing_report(df))
```

#### 4.3 Identify Problems

```python
print("\n--- Problem 1: Inconsistent categories ---")
print(f"Unique os_type values: {df['os_type'].unique()}")
# Notice: 'Linux', 'linux', 'LINUX' are the same thing!

print("\n--- Problem 2: Outliers ---")
print(f"CPU usage range: [{df['cpu_usage'].min():.1f}, {df['cpu_usage'].max():.1f}]")
print(f"Memory range: [{df['memory_gb'].min():.1f}, {df['memory_gb'].max():.1f}]")
print(f"Network range: [{df['network_mbps'].min():.1f}, {df['network_mbps'].max():.1f}]")

print("\n--- Problem 3: Duplicates ---")
print(f"Duplicate rows: {df.duplicated().sum()}")
```

**Discussion Point**: What problems did you find? List them before moving on.

---

### 5. Step 2 — Cleaning

Now we fix the problems one by one.

#### 5.1 Remove Duplicates

```python
print(f"Before: {len(df)} rows")
df = df.drop_duplicates()
print(f"After:  {len(df)} rows")
print(f"Removed {205 - len(df)} duplicate rows")
```

#### 5.2 Standardize Categorical Values

```python
# Standardize os_type: all variations -> consistent format
os_mapping = {
    'Linux': 'Linux', 'linux': 'Linux', 'LINUX': 'Linux',
    'Windows': 'Windows', 'windows': 'Windows', 'Win': 'Windows'
}
df['os_type'] = df['os_type'].map(os_mapping)

print(f"OS types after cleaning: {df['os_type'].unique()}")
# Now we only have: ['Linux', 'Windows']
```

#### 5.3 Handle Outliers

```python
# CPU usage must be between 0 and 100
print(f"\nCPU outliers (>100 or <0): {((df['cpu_usage'] > 100) | (df['cpu_usage'] < 0)).sum()}")
df.loc[df['cpu_usage'] > 100, 'cpu_usage'] = np.nan  # Replace with NaN for imputation
df.loc[df['cpu_usage'] < 0, 'cpu_usage'] = np.nan

# Memory must be positive
print(f"Memory outliers (<0): {(df['memory_gb'] < 0).sum()}")
df.loc[df['memory_gb'] < 0, 'memory_gb'] = np.nan

# Cap network at a reasonable maximum (10 Gbps = 10000 Mbps)
print(f"Network outliers (>10000): {(df['network_mbps'] > 10000).sum()}")
df.loc[df['network_mbps'] > 10000, 'network_mbps'] = np.nan

print("\nOutliers replaced with NaN for imputation.")
```

---

### 6. Step 3 — Handle Missing Values

```python
# Check missing values after outlier handling
print("Missing values after outlier cleanup:")
print(missing_report(df))
```

#### 6.1 Impute Numerical Columns

```python
# Strategy: use median for numerical columns (robust to remaining outliers)
numerical_cols = ['cpu_usage', 'memory_gb', 'disk_io', 'network_mbps', 'uptime_hours']

for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        count = df[col].isnull().sum()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {count} missing values in '{col}' with median = {median_val:.2f}")
```

#### 6.2 Impute Categorical Columns

```python
# Strategy: use mode (most frequent) for categorical columns
categorical_cols = ['region']

for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        count = df[col].isnull().sum()
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled {count} missing values in '{col}' with mode = '{mode_val}'")

# Verify: no more missing values
print(f"\nRemaining missing values: {df.isnull().sum().sum()}")
```

---

### 7. Step 4 — Feature Encoding

#### 7.1 One-Hot Encoding for Nominal Data

```python
# os_type and region have NO natural order -> One-Hot Encoding
print("Before encoding:")
print(f"  os_type unique: {df['os_type'].unique()}")
print(f"  region unique: {df['region'].unique()}")

df_encoded = pd.get_dummies(df, columns=['os_type', 'region'], drop_first=True)

print(f"\nAfter encoding — new columns:")
new_cols = [c for c in df_encoded.columns if 'os_type' in c or 'region' in c]
print(f"  {new_cols}")
print(f"\nDataset shape: {df_encoded.shape}")
```

#### 7.2 Label Encoding for Ordinal Data

```python
# priority has a natural order: Low < Medium < High < Critical
priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
df_encoded['priority'] = df_encoded['priority'].map(priority_map)

print(f"Priority encoding: {priority_map}")
print(f"Priority values: {df_encoded['priority'].unique()}")
```

---

### 8. Step 5 — Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Identify numerical columns to scale
scale_cols = ['cpu_usage', 'memory_gb', 'disk_io', 'network_mbps', 'uptime_hours']

# Demonstrate both scalers
print("=== Before Scaling ===")
print(df_encoded[scale_cols].describe().round(2))

# StandardScaler (Z-score)
scaler_std = StandardScaler()
df_standard = df_encoded.copy()
df_standard[scale_cols] = scaler_std.fit_transform(df_standard[scale_cols])

print("\n=== After StandardScaler ===")
print(df_standard[scale_cols].describe().round(2))
# Notice: mean ≈ 0, std ≈ 1

# MinMaxScaler
scaler_mm = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[scale_cols] = scaler_mm.fit_transform(df_minmax[scale_cols])

print("\n=== After MinMaxScaler ===")
print(df_minmax[scale_cols].describe().round(2))
# Notice: min = 0, max = 1
```

---

### 9. Step 6 — Visualization

Let's visualize the impact of our preprocessing.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Before scaling
axes[0].hist(df_encoded['cpu_usage'], bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
axes[0].set_title('CPU Usage (Original)')
axes[0].set_xlabel('Value')

# After StandardScaler
axes[1].hist(df_standard['cpu_usage'], bins=20, color='#8b5cf6', alpha=0.7, edgecolor='black')
axes[1].set_title('CPU Usage (StandardScaler)')
axes[1].set_xlabel('Z-Score')

# After MinMaxScaler
axes[2].hist(df_minmax['cpu_usage'], bins=20, color='#10b981', alpha=0.7, edgecolor='black')
axes[2].set_title('CPU Usage (MinMaxScaler)')
axes[2].set_xlabel('Scaled Value')

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: scaling_comparison.png")
```

---

### 10. The Complete Pipeline with Scikit-learn

In practice, you should wrap everything into a **Pipeline** to avoid data leakage and ensure reproducibility.

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Prepare features and target
X = df.drop(columns=['server_id', 'failed'])
y = df['failed']

# Define column groups
numeric_features = ['cpu_usage', 'memory_gb', 'disk_io', 'network_mbps', 'uptime_hours']
nominal_features = ['os_type', 'region']
ordinal_features = ['priority']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Critical']]))
])

# Combine all transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('nom', nominal_transformer, nominal_features),
    ('ord', ordinal_transformer, ordinal_features)
])

# Split FIRST, then preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit on train, transform both
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

print(f"Training set: {X_train_clean.shape}")
print(f"Test set:     {X_test_clean.shape}")
print(f"\nData is now clean and model-ready!")
```

**Key Points:**
- `fit_transform()` on training data — learns the parameters (mean, std, categories).
- `transform()` on test data — applies the same parameters. Never `fit()` on test data!
- This prevents **data leakage**: the model never sees test data statistics during training.

---

### 11. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Pipeline</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Replace <code>SimpleImputer(strategy='median')</code> with <code>strategy='mean'</code>. Compare the resulting distributions. When would one be better than the other?</li>
        <li><strong>Experiment B:</strong> Add a new feature: <code>cpu_per_memory = cpu_usage / memory_gb</code>. This is called <strong>Feature Engineering</strong> — creating new informative features from existing ones.</li>
        <li><strong>Experiment C:</strong> Load a real dataset from Kaggle (e.g., the Titanic dataset) and apply the same pipeline. What additional cleaning steps are needed?</li>
    </ul>
</div>

---

### 12. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that takes a "dirty" CSV file (provided on the course page), cleans it, and outputs the preprocessed NumPy arrays ready for training. Your notebook must include:
    1. An exploration section with missing value report and data type summary.
    2. Handling of at least 3 types of data problems (missing values, outliers, inconsistent categories).
    3. Proper encoding (One-Hot for nominal, Label/Ordinal for ordinal).
    4. Feature scaling with justification of your scaler choice.
    5. A scikit-learn Pipeline that wraps all preprocessing steps.
*   **Report**: Write a brief paragraph (3-5 sentences) explaining why you should never fit the scaler on the full dataset before splitting.
*   **Bonus**: Implement a custom function that takes any CSV and automatically detects column types, missing patterns, and suggests appropriate preprocessing strategies.

**See you at the Lecture!**
