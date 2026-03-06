# Week 4 Lecture: Regression Analysis
## "Drawing the Line — Literally"

*Module 2: Applied Machine Learning | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **Regression is the first "real" Machine Learning algorithm you will learn — it takes everything from Weeks 1-3 (optimization, data engineering) and assembles them into a model that makes predictions.**

In Weeks 1-2, we built optimizers and game-playing agents. In Week 3, we learned to clean data. Today, we finally build something that **learns from data** to **predict outcomes**.

**We will cover:**
1. What is Regression? — From Rules to Learning
2. Simple Linear Regression — One Feature, One Line
3. The Loss Function — Mean Squared Error (MSE)
4. Gradient Descent for Regression — Finding Optimal Parameters
5. Multiple Linear Regression — Many Features
6. Polynomial Regression — Beyond Straight Lines
7. Evaluating Regression Models — R², MAE, RMSE
8. Practical Implementation with Scikit-learn
9. Key Equations — Quick Reference

---

### 1. What is Regression?

#### 1.1 Two Types of Predictions

In Machine Learning, we make two broad types of predictions:

| Type | Output | Example |
|---|---|---|
| **Regression** | A continuous number | Predict house price: $342,500 |
| **Classification** | A discrete label | Predict spam or not: "Spam" |

Today we focus on **Regression** — predicting a continuous numerical value.

#### 1.2 The Fundamental Idea

Traditional programming:

```
Rules + Data → Answer
```

Machine Learning:

```
Data + Answers → Rules (Model)
```

We give the machine **examples** (input-output pairs), and it learns the **relationship** between inputs and outputs. This learned relationship is the **model**.

#### 1.3 Where is Regression Used?

- **Housing prices** — predict price from square footage, bedrooms, location.
- **CPU temperature** — predict temperature from load, fan speed, ambient temp.
- **Stock prices** — predict next day's close from historical data (risky!).
- **Medical dosage** — predict optimal drug dosage from patient weight, age.
- **Energy consumption** — predict power usage from weather, time of day.

---

### 2. Simple Linear Regression

#### 2.1 The Model

The simplest regression model fits a straight line through data:

$$\hat{y} = wx + b$$

Where:
- $\hat{y}$ (y-hat) = the **predicted** value
- $x$ = the **input feature**
- $w$ = the **weight** (slope of the line)
- $b$ = the **bias** (y-intercept)

The model's job: find the optimal values of $w$ and $b$ that make the predictions $\hat{y}$ as close as possible to the real values $y$.

#### 2.2 Weights and Biases — The Core Parameters

| Parameter | Role | Analogy |
|---|---|---|
| **Weight** ($w$) | Controls the slope — how much $x$ influences $\hat{y}$ | The "knob" that adjusts sensitivity |
| **Bias** ($b$) | Shifts the line up/down — the baseline prediction | The starting point before any input |

> **Why "weight" and "bias"?** The weight determines how much *weight* or importance each feature has. The bias represents the model's *bias* or default assumption when all inputs are zero.

#### 2.3 Geometric Intuition

Given a scatter plot of data points $(x_i, y_i)$:
- The model draws a line through the cloud of points.
- A **good** line passes close to most points.
- A **bad** line misses badly — high error.

The question becomes: **How do we define "close"?** That's where the Loss Function comes in.

---

### 3. The Loss Function — Mean Squared Error (MSE)

#### 3.1 What is a Loss Function?

A loss function measures **how wrong** the model is. It takes the model's predictions and the true values, and outputs a single number: the **error**.

$$\text{Loss} = f(\text{predictions}, \text{reality})$$

A lower loss means better predictions. Training = minimizing the loss.

#### 3.2 Mean Squared Error (MSE)

The most common loss function for regression:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $n$ = number of data points
- $y_i$ = the **actual** value for data point $i$
- $\hat{y}_i$ = the **predicted** value for data point $i$

#### 3.3 Why Squared?

| Reason | Explanation |
|---|---|
| **Positive errors** | Squaring ensures negative and positive errors don't cancel out |
| **Penalizes big mistakes** | An error of 10 costs 100 (10²), not just 10 — big mistakes are punished harder |
| **Differentiable** | The squared function is smooth, so we can compute gradients for optimization |
| **Convex** | MSE creates a bowl-shaped surface — Gradient Descent is guaranteed to find the minimum |

#### 3.4 Worked Example

```python
# Suppose our model predicts:
predictions = [3.0, 5.5, 7.2]
actuals     = [3.5, 5.0, 7.0]

# Error for each:
# (3.5 - 3.0)² = 0.25
# (5.0 - 5.5)² = 0.25
# (7.0 - 7.2)² = 0.04

# MSE = (0.25 + 0.25 + 0.04) / 3 = 0.18
```

An MSE of 0.18 means our model's average squared error is 0.18 — quite good.

---

### 4. Gradient Descent for Regression

#### 4.1 Connecting to Week 1

Remember Gradient Descent from Week 1? We used it to find the minimum of a function. Now, the function we're minimizing is the **Loss Function (MSE)**.

The parameters we're optimizing are $w$ and $b$.

#### 4.2 The Gradients

For MSE with a simple linear model $\hat{y} = wx + b$:

$$\frac{\partial \text{MSE}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$

$$\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

#### 4.3 The Update Rule

At each step, move $w$ and $b$ in the direction that reduces the loss:

$$w \leftarrow w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$$

Where $\alpha$ is the **learning rate** (from Week 1).

#### 4.4 The Training Loop

```python
# Gradient Descent from scratch
import numpy as np

def train_linear_regression(X, y, lr=0.01, epochs=1000):
    w = 0.0  # Initialize weight
    b = 0.0  # Initialize bias
    n = len(X)
    
    for epoch in range(epochs):
        # Forward pass: make predictions
        y_pred = w * X + b
        
        # Compute gradients
        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
        # Print progress every 200 epochs
        if epoch % 200 == 0:
            mse = np.mean((y - y_pred) ** 2)
            print(f"Epoch {epoch}: MSE = {mse:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b
```

#### 4.5 The Normal Equation — An Alternative

For linear regression, there's actually a **closed-form solution** — no iteration needed:

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

| Method | Pros | Cons |
|---|---|---|
| **Gradient Descent** | Works for any model; scales to huge datasets | Requires tuning learning rate; iterative |
| **Normal Equation** | Exact answer in one step; no hyperparameters | Slow for large datasets ($O(n^3)$ matrix inversion); only works for linear models |

> In practice, scikit-learn uses optimized variants of both.

---

### 5. Multiple Linear Regression

#### 5.1 From One Feature to Many

Real-world predictions depend on **multiple features**, not just one:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \ldots + w_p x_p + b$$

In vector notation:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

Where $\mathbf{w} = [w_1, w_2, \ldots, w_p]$ is the weight vector and $\mathbf{x} = [x_1, x_2, \ldots, x_p]$ is the feature vector.

#### 5.2 Example: House Price Prediction

$$\hat{\text{price}} = w_1 \cdot \text{sqft} + w_2 \cdot \text{bedrooms} + w_3 \cdot \text{age} + b$$

Each weight tells us **how much** that feature contributes:
- $w_1 = 150$ means each extra square foot adds $150 to the predicted price.
- $w_3 = -2000$ means each year of age reduces the predicted price by $2,000.

#### 5.3 Feature Importance

One advantage of linear regression: the **weights are interpretable**. Larger absolute weights mean more important features (after scaling!).

> **Critical**: You must **scale your features** (Week 3) before interpreting weights! If Square Footage is in range [500, 5000] and Bedrooms is in [1, 6], the raw weights are not comparable.

---

### 6. Polynomial Regression — Beyond Straight Lines

#### 6.1 When a Line Isn't Enough

Not all relationships are linear. If data follows a curve, we need polynomial features:

$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + b$$

This is still "linear" regression — it's linear in the **parameters** ($w_1, w_2, w_3$), even though it's polynomial in the **features**.

#### 6.2 The Trick: Feature Engineering

Instead of changing the algorithm, we change the **input**:

```python
# Transform: x → [x, x², x³]
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)  # Adds x², x³ as new columns
```

Now we apply standard linear regression to the expanded features.

#### 6.3 The Danger: Overfitting

| Degree | Behavior |
|---|---|
| **Too low** (degree=1) | **Underfitting** — too simple, misses the pattern |
| **Just right** (degree=2-3) | Captures the trend, generalizes well |
| **Too high** (degree=15) | **Overfitting** — memorizes the training data, fails on new data |

> This is the **Bias-Variance Tradeoff** — you'll see it again and again throughout this course.

---

### 7. Evaluating Regression Models

#### 7.1 Metrics Overview

MSE is not the only metric. Here are the key ones:

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error. Penalizes outliers. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as $y$. More interpretable. |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average absolute error. Robust to outliers. |
| **R² Score** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | How much variance explained. 1.0 = perfect. |

#### 7.2 R² Score — The "Report Card"

R² (R-Squared, or Coefficient of Determination) tells you how much better your model is than simply predicting the mean:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

| R² Value | Meaning |
|---|---|
| **1.0** | Perfect predictions |
| **0.8–0.99** | Strong model |
| **0.5–0.8** | Moderate model |
| **< 0.5** | Weak model — consider more features or a different algorithm |
| **< 0** | Worse than predicting the mean — your model is broken |

#### 7.3 Train vs. Test Performance

Always evaluate on **held-out test data** (Week 3's train/test split):

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train on training data
model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

### 8. Practical Implementation with Scikit-learn

#### 8.1 Simple Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.uniform(500, 3500, 100).reshape(-1, 1)  # Square footage
y = 50 + 150 * X.flatten() + np.random.normal(0, 15000, 100)  # Price

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model = LinearRegression()
model.fit(X_train, y_train)

# Results
print(f"Weight (slope): {model.coef_[0]:.2f}")
print(f"Bias (intercept): {model.intercept_:.2f}")
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
```

#### 8.2 Multiple Linear Regression with Pipeline

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a realistic dataset
np.random.seed(42)
n = 200
data = {
    'sqft': np.random.uniform(500, 3500, n),
    'bedrooms': np.random.randint(1, 6, n),
    'age': np.random.uniform(0, 50, n),
    'distance_to_center': np.random.uniform(1, 30, n),
}
df = pd.DataFrame(data)
df['price'] = (
    150 * df['sqft']
    + 20000 * df['bedrooms']
    - 2000 * df['age']
    - 5000 * df['distance_to_center']
    + 50000
    + np.random.normal(0, 25000, n)
)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Scale → Fit
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

# Feature importance (scaled weights)
weights = pipeline.named_steps['model'].coef_
features = X.columns
for f, w in sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f}: {w:+.2f}")
```

---

### 9. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Simple Linear Model | $\hat{y} = wx + b$ |
| Multiple Linear Model | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ |
| Mean Squared Error (MSE) | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ |
| Gradient (w.r.t. $w$) | $-\frac{2}{n}\sum x_i(y_i - \hat{y}_i)$ |
| Gradient (w.r.t. $b$) | $-\frac{2}{n}\sum(y_i - \hat{y}_i)$ |
| Weight Update | $w \leftarrow w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$ |
| Normal Equation | $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| R² Score | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |

---

### 10. Summary

1. **Linear Regression** is the foundation of all supervised learning — a model that learns the relationship $\hat{y} = wx + b$ from data.
2. **Weights and Biases** are the learnable parameters — $w$ controls how much each feature matters, $b$ sets the baseline.
3. **MSE** measures error by squaring differences — it's differentiable, convex, and penalizes big mistakes.
4. **Gradient Descent** iteratively adjusts $w$ and $b$ to minimize MSE — `learning_rate` controls step size.
5. **Multiple regression** extends to many features: $\hat{y} = \mathbf{w}^T\mathbf{x} + b$. Weights become interpretable after scaling.
6. **R² Score** is the gold standard metric — 1.0 = perfect, 0 = no better than guessing the mean.
7. **Always evaluate on test data** — training metrics are misleading due to potential overfitting.

---

### 11. Bridge to the Tutorial

In the tutorial session, you will:
- Generate a realistic housing price dataset.
- Implement Linear Regression from scratch (gradient descent loop).
- Compare your implementation against Scikit-learn's `LinearRegression`.
- Experiment with Polynomial Regression and observe overfitting.
- Build a complete Pipeline that preprocesses and predicts in one step.
- Evaluate models using R², RMSE, and MAE on held-out test data.

**Come to the tutorial. A model that never sees data never learns.**
