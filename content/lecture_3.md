# Week 3 Lecture: Data Engineering for AI
## "Garbage In, Garbage Out"

*Module 1: Algorithmic Foundations | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **The most sophisticated algorithm in the world is useless if it's trained on bad data. Data Engineering is the unglamorous backbone of every successful AI system.**

In Weeks 1-2, we built optimization and search algorithms. We assumed the data was clean, numeric, and ready. In the real world, it never is. Today, we learn the bridge between raw, messy data and the clean numerical inputs our algorithms demand.

**We will cover:**
1. Why Data Engineering Matters — The 80/20 Rule of ML
2. Probability Review — Bayes' Theorem
3. Types of Data and Their Challenges
4. Handling Missing Data
5. Feature Encoding — From Categories to Numbers
6. Feature Scaling — Normalization & Standardization
7. The Complete Preprocessing Pipeline
8. Key Equations — Quick Reference

---

### 1. Why Data Engineering Matters

#### 1.1 The 80/20 Rule

In industry, data scientists spend approximately **80% of their time** on data preparation and only **20% on actual modeling**. This isn't a failure of process — it reflects a fundamental truth:

> A simple model on clean data will almost always outperform a complex model on dirty data.

#### 1.2 What Can Go Wrong?

Real-world datasets are messy in predictable ways:

| Problem | Example | Consequence |
|---|---|---|
| **Missing values** | Sensor went offline, user skipped a field | Model crashes or learns noise |
| **Inconsistent formats** | "Male", "male", "M", "1" for the same concept | Model treats them as different categories |
| **Outliers** | A salary of $1,000,000 in a dataset of students | Skews the mean, distorts learning |
| **Different scales** | Age (0-100) vs. Income (0-1,000,000) | Gradient Descent oscillates wildly |
| **Categorical data** | "Red", "Blue", "Green" | Algorithms need numbers, not strings |
| **Duplicates** | Same record entered twice | Model overweights that example |

#### 1.3 The Data Pipeline

Every ML project follows this pipeline:

```
Raw Data → Clean → Transform → Encode → Scale → Model-Ready Data
              ↑        ↑          ↑        ↑
           Missing   Feature    One-Hot   MinMax /
           values    engineering Encoding  StandardScaler
```

Today, we learn each stage.

---

### 2. Probability Review — Bayes' Theorem

Before we dive into data preprocessing, we need to refresh a critical mathematical tool that underlies many ML algorithms: **Bayes' Theorem**.

#### 2.1 Conditional Probability

The probability of event $A$ given that event $B$ has occurred:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Example**: What is the probability that an email is spam ($A$), given that it contains the word "free" ($B$)?

#### 2.2 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

In ML terminology:

$$P(\text{hypothesis} | \text{data}) = \frac{P(\text{data} | \text{hypothesis}) \cdot P(\text{hypothesis})}{P(\text{data})}$$

| Term | Name | Meaning |
|---|---|---|
| $P(A|B)$ | **Posterior** | What we want to know (updated belief) |
| $P(B|A)$ | **Likelihood** | How probable is the data under this hypothesis? |
| $P(A)$ | **Prior** | Our initial belief before seeing data |
| $P(B)$ | **Evidence** | Total probability of the observed data |

#### 2.3 Worked Example: Medical Diagnosis

A disease affects 1% of the population. A test for the disease is 95% accurate (true positive) and has a 5% false positive rate. If a patient tests positive, what is the probability they actually have the disease?

**Given:**
- $P(\text{Disease}) = 0.01$ (Prior)
- $P(\text{Positive} | \text{Disease}) = 0.95$ (Likelihood / Sensitivity)
- $P(\text{Positive} | \text{No Disease}) = 0.05$ (False Positive Rate)

**Step 1**: Calculate $P(\text{Positive})$ using the Law of Total Probability:

$$P(\text{Positive}) = P(\text{Pos}|\text{Dis}) \cdot P(\text{Dis}) + P(\text{Pos}|\text{No Dis}) \cdot P(\text{No Dis})$$
$$= 0.95 \times 0.01 + 0.05 \times 0.99 = 0.0095 + 0.0495 = 0.059$$

**Step 2**: Apply Bayes' Theorem:

$$P(\text{Disease} | \text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$

**Result**: Only **16.1%** chance of actually having the disease despite a positive test! This counterintuitive result is the **Base Rate Fallacy** — when the disease is rare, even a good test produces many false positives.

#### 2.4 Why Bayes Matters for ML

- **Naive Bayes Classifier** (Week 5): Uses Bayes' Theorem directly for classification.
- **Prior knowledge**: Bayesian thinking lets us incorporate domain knowledge into models.
- **Probabilistic interpretation**: Many ML models output probabilities, not just labels.

---

### 3. Types of Data

Before preprocessing, we must understand what kind of data we're working with. Different types require different treatments.

#### 3.1 Numerical Data

Data that is inherently numeric and has mathematical meaning.

| Subtype | Definition | Example | Operations |
|---|---|---|---|
| **Continuous** | Any value in a range | Temperature: 36.7°C | Mean, std, all math |
| **Discrete** | Countable integers | Number of rooms: 3 | Count, mode |

#### 3.2 Categorical Data

Data that represents groups or labels, not quantities.

| Subtype | Definition | Example | Encoding |
|---|---|---|---|
| **Nominal** | No natural order | Color: Red, Blue, Green | One-Hot Encoding |
| **Ordinal** | Has natural order | Education: High School < Bachelor's < Master's | Label Encoding |

#### 3.3 The Cardinal Rule

> **Never treat categorical data as numerical.** If you encode "Red=1, Blue=2, Green=3", the model will learn that Green > Blue > Red, which is meaningless.

---

### 4. Handling Missing Data

Missing data is inevitable. How you handle it can make or break your model.

#### 4.1 Types of Missingness

| Type | Meaning | Example |
|---|---|---|
| **MCAR** (Missing Completely At Random) | No pattern to the missingness | Sensor randomly malfunctions |
| **MAR** (Missing At Random) | Missingness depends on other observed data | Older patients skip online surveys |
| **MNAR** (Missing Not At Random) | Missingness depends on the missing value itself | High-income people don't report salary |

#### 4.2 Strategies

**Strategy 1: Drop rows with missing values**
```python
df.dropna()  # Drop any row with at least one NaN
```
- **Pro**: Simple, no introduced bias.
- **Con**: Lose data. If 20% of rows have missing values, you lose 20% of your dataset.

**Strategy 2: Drop columns with too many missing values**
```python
# Drop columns where more than 50% of values are missing
threshold = len(df) * 0.5
df.dropna(axis=1, thresh=threshold)
```

**Strategy 3: Imputation — Fill in the blanks**

| Method | Code | When to Use |
|---|---|---|
| **Mean** | `df['col'].fillna(df['col'].mean())` | Normally distributed numerical data |
| **Median** | `df['col'].fillna(df['col'].median())` | Skewed numerical data |
| **Mode** | `df['col'].fillna(df['col'].mode()[0])` | Categorical data |
| **Forward Fill** | `df['col'].fillna(method='ffill')` | Time-series data |
| **Constant** | `df['col'].fillna(0)` | When missingness has semantic meaning |

**Strategy 4: Indicator variable**
```python
df['col_was_missing'] = df['col'].isna().astype(int)
df['col'].fillna(df['col'].median(), inplace=True)
```
This preserves the information that the value was missing — sometimes missingness itself is a feature!

---

### 5. Feature Encoding

Machine Learning algorithms speak numbers. Categorical features must be translated.

#### 5.1 Label Encoding (Ordinal Data)

For data with a natural order, assign integers that preserve the ranking.

```python
from sklearn.preprocessing import LabelEncoder

education = ['High School', 'Bachelor', 'Master', 'PhD']
# Encoding: High School=0, Bachelor=1, Master=2, PhD=3
```

The ordering 0 < 1 < 2 < 3 correctly reflects High School < Bachelor's < Master's < PhD.

#### 5.2 One-Hot Encoding (Nominal Data)

For data with **no natural order**, create a binary column for each category.

```python
import pandas as pd

df = pd.DataFrame({'color': ['Red', 'Blue', 'Green', 'Red', 'Blue']})
df_encoded = pd.get_dummies(df, columns=['color'])
```

**Result:**

| color_Blue | color_Green | color_Red |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |

Each row has exactly one `1` — no false ordering imposed.

#### 5.3 The Dummy Variable Trap

If you have $k$ categories, you only need $k-1$ binary columns. The last column is redundant (it's determined by the others). Many frameworks handle this automatically:

```python
pd.get_dummies(df, columns=['color'], drop_first=True)
```

---

### 6. Feature Scaling

#### 6.1 Why Scaling Matters

Consider two features: **Age** (range: 18-65) and **Salary** (range: 20,000-200,000). If we feed these directly into Gradient Descent:

- A 1-unit change in Age is a meaningful change.
- A 1-unit change in Salary is negligible.

The result: Gradient Descent will oscillate wildly along the Salary axis and crawl along the Age axis. **Scaling puts all features on equal footing.**

#### 6.2 Min-Max Normalization (Scaling to [0, 1])

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

- **Pro**: Bounded output [0, 1]. Good for algorithms sensitive to magnitude (Neural Networks, KNN).
- **Con**: Sensitive to outliers. One extreme value compresses all others.

#### 6.3 Standardization (Z-Score Normalization)

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

- **Pro**: Not bounded — handles outliers better. Centers data at 0 with unit variance.
- **Con**: Doesn't guarantee a specific range. Output can be any real number.

#### 6.4 When to Use Which?

| Scaler | Use When | Algorithms |
|---|---|---|
| **MinMaxScaler** | You need bounded values; data has no extreme outliers | Neural Networks, KNN, Image pixels |
| **StandardScaler** | Data has outliers; algorithm assumes Gaussian distribution | Linear Regression, SVM, PCA |
| **No scaling** | Tree-based algorithms (they split on thresholds, not magnitudes) | Decision Trees, Random Forest, XGBoost |

---

### 7. The Complete Preprocessing Pipeline

Let's put it all together. Here's the standard preprocessing pipeline in scikit-learn:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 4. Define preprocessing for each type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 5. Combine into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 6. Split data BEFORE fitting the preprocessor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fit on training data, transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)  # No .fit() on test data!
```

**Critical Rule**: Always fit the preprocessor on the **training data only**, then transform both train and test. Fitting on the full dataset causes **data leakage** — the model indirectly sees test data statistics during training.

---

### 8. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Conditional Probability | $P(A|B) = \frac{P(A \cap B)}{P(B)}$ |
| Bayes' Theorem | $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$ |
| Law of Total Probability | $P(B) = \sum_i P(B|A_i) \cdot P(A_i)$ |
| Min-Max Normalization | $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ |
| Standardization (Z-Score) | $x_{\text{std}} = \frac{x - \mu}{\sigma}$ |
| Mean Imputation | $x_{\text{missing}} \leftarrow \bar{x}$ |

---

### 9. Summary

1. **Data quality determines model quality** — the 80/20 rule of ML means most of your time should be spent on data, not algorithms.
2. **Bayes' Theorem** is the mathematical foundation for updating beliefs with evidence. It powers classifiers (Naive Bayes) and gives us a probabilistic framework for reasoning under uncertainty.
3. **Missing data** must be handled strategically — dropping, imputing, or flagging, depending on the type and amount of missingness.
4. **Categorical encoding** (One-Hot for nominal, Label for ordinal) translates human-readable categories into algorithm-readable numbers without imposing false structure.
5. **Feature scaling** (MinMax or StandardScaler) ensures all features contribute equally to learning, preventing dominant features from hijacking Gradient Descent.
6. **The preprocessing pipeline** in scikit-learn automates the entire process and prevents data leakage.

---

### 10. Bridge to the Tutorial

In the tutorial session, you will:
- Load a real "dirty" dataset with missing values, inconsistent formats, and mixed data types.
- Use Pandas to explore, clean, and transform the data step by step.
- Implement One-Hot Encoding and feature scaling.
- Build a complete scikit-learn preprocessing pipeline.
- Verify your cleaned data is ready for model training.

**Come to the tutorial. Clean data isn't just better — it's the difference between a model that works and one that doesn't.**
