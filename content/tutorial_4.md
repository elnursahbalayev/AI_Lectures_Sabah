# Week 4 Tutorial: Predicting Metrics — Housing Prices
## "Your First ML Model"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a Linear Regression model from scratch AND with Scikit-learn, and evaluated it using R², RMSE, and MAE.

---

### 1. What We Are Building

Last week, we cleaned messy data. Today, we use clean data to **make predictions**. We will build a model that predicts **housing prices** based on features like square footage, number of bedrooms, and distance to the city center.

**By the end of this session, you will have:**
- Implemented Linear Regression from scratch using NumPy.
- Compared your implementation to Scikit-learn's `LinearRegression`.
- Visualized the regression line and residuals.
- Experimented with Polynomial Regression and observed overfitting.
- Built a complete Pipeline (preprocess + train + evaluate).

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Verify versions
print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")

print("\nReady to build our first ML model!")
```

---

### 3. Creating Our Dataset

We'll generate a realistic housing dataset with known relationships so we can verify our model learns them correctly.

```python
np.random.seed(42)
n = 300  # Number of houses

# Generate features
data = {
    'sqft': np.random.uniform(500, 4000, n),           # Square footage
    'bedrooms': np.random.randint(1, 6, n),             # Number of bedrooms
    'age': np.random.uniform(0, 50, n),                 # Age of house in years
    'distance_km': np.random.uniform(1, 30, n),         # Distance to city center
    'garage': np.random.choice([0, 1], n, p=[0.3, 0.7]) # Has garage? (0/1)
}

df = pd.DataFrame(data)

# Generate price with known formula + noise
# TRUE RELATIONSHIP (what the model should learn):
#   price = 150 * sqft + 25000 * bedrooms - 2000 * age - 3000 * distance + 40000 * garage + 50000
df['price'] = (
    150 * df['sqft'] +
    25000 * df['bedrooms'] -
    2000 * df['age'] -
    3000 * df['distance_km'] +
    40000 * df['garage'] +
    50000 +
    np.random.normal(0, 30000, n)  # Random noise
)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()
```

---

### 4. Step 1 — Exploration

#### 4.1 Understand the Data

```python
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nBasic statistics:")
df.describe().round(1)
```

#### 4.2 Visualize Relationships

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plots: each feature vs price
features = ['sqft', 'bedrooms', 'age', 'distance_km']
colors = ['#3b82f6', '#8b5cf6', '#ef4444', '#f59e0b']

for ax, feat, color in zip(axes.flatten(), features, colors):
    ax.scatter(df[feat], df['price'], alpha=0.4, s=15, color=color)
    ax.set_xlabel(feat)
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Price vs {feat}')

plt.tight_layout()
plt.savefig('feature_relationships.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: Which features show the strongest linear relationship with price? Which look noisy?

---

### 5. Step 2 — Simple Linear Regression from Scratch

Let's first implement regression with **one feature** (sqft) to understand the mechanics.

#### 5.1 Prepare Data

```python
# Use only sqft for simple regression
X_simple = df['sqft'].values
y = df['price'].values

# Manual train-test split
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train_s)} samples")
print(f"Test:     {len(X_test_s)} samples")
```

#### 5.2 Gradient Descent Implementation

```python
def linear_regression_gd(X, y, lr=0.0000001, epochs=1000):
    """Train linear regression using Gradient Descent."""
    w = 0.0  # Initialize weight to zero
    b = 0.0  # Initialize bias to zero
    n = len(X)
    history = []  # Track MSE over time
    
    for epoch in range(epochs):
        # Forward pass: predictions
        y_pred = w * X + b
        
        # Compute MSE
        mse = np.mean((y - y_pred) ** 2)
        history.append(mse)
        
        # Compute gradients
        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
        # Log progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | MSE: {mse:,.0f} | w: {w:.4f} | b: {b:.2f}")
    
    return w, b, history

# Train
w, b, history = linear_regression_gd(X_train_s, y_train, lr=0.0000001, epochs=1000)
print(f"\nFinal: w = {w:.4f}, b = {b:.2f}")
print(f"Interpretation: each sqft adds ~${w:.0f} to price")
```

#### 5.3 Visualize Training Progress

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history, color='#3b82f6', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_title('Training Loss Curve')
axes[0].set_yscale('log')

# Regression line on test data
axes[1].scatter(X_test_s, y_test, alpha=0.5, s=20, color='#6b7280', label='Test data')
x_line = np.linspace(X_test_s.min(), X_test_s.max(), 100)
y_line = w * x_line + b
axes[1].plot(x_line, y_line, color='#ef4444', linewidth=2, label='Our model')
axes[1].set_xlabel('Square Footage')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Regression Line (from scratch)')
axes[1].legend()

plt.tight_layout()
plt.savefig('regression_from_scratch.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 6. Step 3 — Compare with Scikit-learn

#### 6.1 Fit Scikit-learn Model

```python
# Scikit-learn version (one line!)
sk_model = LinearRegression()
sk_model.fit(X_train_s.reshape(-1, 1), y_train)

print(f"Scikit-learn w = {sk_model.coef_[0]:.4f}")
print(f"Scikit-learn b = {sk_model.intercept_:.2f}")
print(f"\nOur model    w = {w:.4f}")
print(f"Our model    b = {b:.2f}")
```

#### 6.2 Evaluate Both Models

```python
# Our model predictions
y_pred_ours = w * X_test_s + b

# Scikit-learn predictions
y_pred_sk = sk_model.predict(X_test_s.reshape(-1, 1))

# Compare metrics
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"\n{'Metric':<10} {'Our Model':>15} {'Scikit-learn':>15}")
print("-" * 42)
print(f"{'R²':<10} {r2_score(y_test, y_pred_ours):>15.4f} {r2_score(y_test, y_pred_sk):>15.4f}")
print(f"{'RMSE':<10} ${np.sqrt(mean_squared_error(y_test, y_pred_ours)):>14,.0f} ${np.sqrt(mean_squared_error(y_test, y_pred_sk)):>14,.0f}")
print(f"{'MAE':<10} ${mean_absolute_error(y_test, y_pred_ours):>14,.0f} ${mean_absolute_error(y_test, y_pred_sk):>14,.0f}")
```

**Discussion**: Is scikit-learn better? Why?

---

### 7. Step 4 — Multiple Linear Regression

Now let's use **all features** to make better predictions.

```python
# Prepare all features
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features + Fit model (Pipeline)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Metrics
print("=" * 50)
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"\nR² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:      ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"MAE:       ${mean_absolute_error(y_test, y_pred):,.0f}")

# Feature importance (weights after scaling)
print(f"\n--- Feature Importance (scaled weights) ---")
weights = pipeline.named_steps['model'].coef_
for feat, w in sorted(zip(X.columns, weights), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if w > 0 else "↓"
    print(f"  {feat:<15} {direction} {w:+10,.0f}")
```

---

### 8. Step 5 — Polynomial Regression

Let's see what happens when we add polynomial features.

```python
# Use just sqft for visualization clarity
X_sqft = df[['sqft']]
y = df['price']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_sqft, y, test_size=0.2, random_state=42
)

# Try different polynomial degrees
degrees = [1, 2, 3, 10]
results = {}

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, deg in zip(axes, degrees):
    # Build pipeline: polynomial features → scale → linear regression
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    poly_pipeline.fit(X_train_p, y_train_p)
    
    # Metrics
    train_r2 = r2_score(y_train_p, poly_pipeline.predict(X_train_p))
    test_r2 = r2_score(y_test_p, poly_pipeline.predict(X_test_p))
    results[deg] = {'train_r2': train_r2, 'test_r2': test_r2}
    
    # Plot
    ax.scatter(X_test_p, y_test_p, alpha=0.3, s=15, color='#6b7280')
    x_range = np.linspace(X_sqft.min(), X_sqft.max(), 200)
    y_range = poly_pipeline.predict(x_range)
    ax.plot(x_range, y_range, color='#ef4444', linewidth=2)
    ax.set_title(f'Degree {deg}\nTrain R²={train_r2:.3f} | Test R²={test_r2:.3f}')
    ax.set_xlabel('sqft')

plt.tight_layout()
plt.savefig('polynomial_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# Summary table
print("\n--- Polynomial Degree Comparison ---")
print(f"{'Degree':<8} {'Train R²':>10} {'Test R²':>10} {'Status':>12}")
print("-" * 42)
for deg, scores in results.items():
    if scores['test_r2'] < scores['train_r2'] - 0.1:
        status = "Overfitting!"
    elif scores['test_r2'] < 0.5:
        status = "Underfitting"
    else:
        status = "Good"
    print(f"{deg:<8} {scores['train_r2']:>10.4f} {scores['test_r2']:>10.4f} {status:>12}")
```

---

### 9. Step 6 — Residual Analysis

A good model should have **random** residuals — no patterns.

```python
# Using our best model (multiple regression pipeline)
X_all = df.drop('price', axis=1)
y_all = df['price']
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

pipeline.fit(X_train_a, y_train_a)
y_pred_a = pipeline.predict(X_test_a)
residuals = y_test_a - y_pred_a

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot
axes[0].scatter(y_pred_a, residuals, alpha=0.5, s=20, color='#3b82f6')
axes[0].axhline(y=0, color='#ef4444', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Price ($)')
axes[0].set_ylabel('Residual ($)')
axes[0].set_title('Residual Plot')

# Distribution of residuals
axes[1].hist(residuals, bins=25, color='#8b5cf6', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='#ef4444', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"Mean residual: ${np.mean(residuals):,.0f} (should be ≈ 0)")
print(f"Std residual:  ${np.std(residuals):,.0f}")
```

---

### 10. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Analysis</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Remove one feature at a time and observe how R² changes. Which feature matters most?</li>
        <li><strong>Experiment B:</strong> Increase the noise level (change `30000` to `100000` in the data generation). How does this affect model performance?</li>
        <li><strong>Experiment C:</strong> Try using `MinMaxScaler` instead of `StandardScaler`. Does it change the R² score? Why or why not?</li>
        <li><strong>Experiment D:</strong> Find a real housing dataset on Kaggle (e.g., "California Housing") and apply the same pipeline.</li>
    </ul>
</div>

---

### 11. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that:
    1. Loads a dataset (provided or from Kaggle).
    2. Explores the data with scatter plots and correlation analysis.
    3. Implements simple Linear Regression from scratch using Gradient Descent.
    4. Trains a Scikit-learn `LinearRegression` and compares results.
    5. Evaluates using R², RMSE, and MAE on a held-out test set.
    6. Experiments with at least two polynomial degrees and documents the overfitting behavior.
*   **Report**: Write a brief paragraph (3-5 sentences) explaining the Bias-Variance Tradeoff using your polynomial regression results.
*   **Bonus**: Implement regularization (Ridge or Lasso regression) and compare it against plain LinearRegression. Which performs better and why?

**See you at the Lecture!**
