# Week 8 Lecture: Model Evaluation & Deployment
## "How Do You Know Your Model Is Actually Good?"

*Module 2: Applied Machine Learning | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **A model that performs perfectly on training data and fails in production is worse than useless — it's dangerous. Model evaluation is the discipline of measuring generalization, not memorization. Deployment is the art of making your model useful to the world.**

In Weeks 4–7, we built models. Today, we ask the harder question: *How do we know any of them actually work?*

**We will cover:**
1. The Generalization Problem — Overfitting & Underfitting
2. The Bias-Variance Tradeoff
3. Proper Evaluation: The Train/Validation/Test Split
4. Cross-Validation — K-Fold and Stratified K-Fold
5. Classification Metrics — Precision, Recall, F1-Score, AUC-ROC
6. Regression Metrics — MSE, RMSE, MAE, R²
7. The Confusion Matrix — Reading It Like a Professional
8. The ROC Curve & AUC
9. Hyperparameter Tuning — Grid Search & Randomized Search
10. Deployment — Wrapping Your Model as a Service

---

### 1. The Generalization Problem

#### 1.1 What We Actually Want

When we train a machine learning model, the goal is **never** to perform well on the training data. The goal is to perform well on **unseen data** — data the model has never seen before.

> **Generalization** is the ability of a model to perform well on new, unseen data from the same distribution it was trained on.

The three failure modes:

| Failure | Symptom | Cause |
|---|---|---|
| **Overfitting** | High train accuracy, low test accuracy | Model memorized the training data, including its noise |
| **Underfitting** | Low train accuracy, low test accuracy | Model is too simple to capture the pattern |
| **Data Leakage** | Unrealistically high test accuracy | Test data accidentally contaminated the training process |

#### 1.2 Visualizing Overfitting vs. Underfitting

Consider fitting a polynomial to data:

| Model | Behavior |
|---|---|
| Degree 1 (line) | Underfits — can't capture the curve |
| Degree 5 (appropriate) | Good fit — captures the trend |
| Degree 20 (too complex) | Overfits — wiggles through every training point, fails on new data |

The goal: find the **sweet spot** of model complexity.

---

### 2. The Bias-Variance Tradeoff

Every model's error decomposes into three components:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Component | Definition | Symptom |
|---|---|---|
| **Bias** | Systematic error from wrong model assumptions | Model consistently wrong in the same direction (underfitting) |
| **Variance** | Sensitivity to fluctuations in training data | Model gives wildly different results on different subsets (overfitting) |
| **Noise** | Irreducible error in the data | Cannot be reduced by any model |

> **The Core Tradeoff**: As you increase model complexity, bias decreases but variance increases. Optimal complexity is the point of minimum *total* expected error.

| Model Complexity | Bias | Variance | Total Error |
|---|---|---|---|
| Too simple (linear) | High | Low | High (underfitting) |
| Just right | Low | Low | **Minimum** |
| Too complex (degree-20 poly) | Low | High | High (overfitting) |

**Practical solutions:**

| Problem | Solution |
|---|---|
| Overfitting (high variance) | More data, regularization (L1/L2), simpler model, dropout (deep learning), cross-validation |
| Underfitting (high bias) | More complex model, more features, less regularization, more training time |

---

### 3. Proper Evaluation: The Split Strategy

#### 3.1 Train/Test Split

The fundamental rule: **never evaluate a model on data it has seen during training.**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,
    stratify=y           # preserve class proportions — critical for imbalanced data
)

print(f"Train size: {X_train.shape[0]} ({X_train.shape[0]/len(X):.0%})")
print(f"Test size:  {X_test.shape[0]} ({X_test.shape[0]/len(X):.0%})")
```

> **The test set is sacred.** You look at it exactly **once** — at the very end, after all decisions about model architecture and hyperparameters are finalized. If you evaluate on the test set and then adjust your model, you have leaked information.

#### 3.2 Train/Validation/Test Split

For hyperparameter tuning, you need a third set:

```
All Data (100%)
├── Train (60%) — model learns weights from this
├── Validation (20%) — used to tune hyperparameters
└── Test (20%) — final, one-time evaluation
```

```python
# Two-stage split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# 0.25 × 0.8 = 0.20 → final split is 60/20/20

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
```

> **A critical warning**: Fit your preprocessing (e.g., `StandardScaler`, `PCA`) using **only training data**, then apply the same transform to validation and test data. Fitting on all data is data leakage.

---

### 4. Cross-Validation

#### 4.1 The Problem with a Single Split

A single 80/20 split has a problem: the results depend on **which 20%** ended up in the test set. With small datasets, this variance can be large — you might get lucky or unlucky.

**K-Fold Cross-Validation** solves this:

1. Split data into $K$ equal-sized **folds**.
2. Train on $K-1$ folds, evaluate on the held-out fold.
3. Repeat $K$ times — each fold serves as the test set once.
4. Report the **mean and standard deviation** of scores across all $K$ folds.

$$\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{score}_i$$

#### 4.2 K-Fold in Python

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted')

print(f"CV Scores per fold: {cv_scores.round(4)}")
print(f"Mean CV Score:      {cv_scores.mean():.4f}")
print(f"Std Dev:            {cv_scores.std():.4f}")
print(f"95% CI:             ({cv_scores.mean() - 2*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 2*cv_scores.std():.4f})")
```

> A **high standard deviation** across folds means the model is sensitive to the training data split — a warning sign of high variance (overfitting).

#### 4.3 Stratified K-Fold — For Classification

Regular K-Fold splits randomly, so folds may have imbalanced class distributions. **Stratified K-Fold** preserves the original class ratio in each fold:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

# Use StratifiedKFold for classification problems
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_validate returns multiple metrics
results = cross_validate(
    model, X_train, y_train,
    cv=skfold,
    scoring=['accuracy', 'f1_weighted', 'roc_auc'],
    return_train_score=True
)

for metric in ['accuracy', 'f1_weighted', 'roc_auc']:
    train_scores = results[f'train_{metric}']
    test_scores  = results[f'test_{metric}']
    print(f"{metric:20s}: Train={train_scores.mean():.4f}  Validation={test_scores.mean():.4f}")
```

> **Always use Stratified K-Fold for classification**, especially with imbalanced classes. This is the default when you pass `cv=5` to `cross_val_score` with a classifier.

#### 4.4 Common K Values

| K | Trade-off |
|---|---|
| K=5 | Fast, less variance in the estimate — good default |
| K=10 | More reliable estimate, more compute — common in research |
| K=n (Leave-One-Out) | Maximum data use, very slow, high variance in estimate — avoid for large datasets |

---

### 5. Classification Metrics

#### 5.1 Why Accuracy Is Insufficient

Consider a fraud detection system where 99% of transactions are legitimate:

- A model that **always predicts "not fraud"** achieves **99% accuracy**.
- But it catches **zero fraud cases** — completely useless.

Accuracy is misleading for **imbalanced classes**. We need richer metrics.

#### 5.2 The Confusion Matrix

For binary classification, all predictions fall into one of four categories:

|  | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) — *missed it* |
| **Actual Negative** | False Positive (FP) — *false alarm* | True Negative (TN) |

> **Critical intuition**: TP and TN are correct. FP is a "false alarm" (said yes, was no). FN is a "missed hit" (said no, was yes).

#### 5.3 Precision, Recall, and F1-Score

**Precision** — "Of all the things I called positive, how many actually are?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)** — "Of all the actual positives, how many did I catch?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** — The harmonic mean of Precision and Recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

> **Why harmonic mean?** The arithmetic average of (1.0, 0.0) = 0.5 — misleadingly high. The harmonic mean = 0.0, correctly showing the model is useless.

#### 5.4 When to Optimize Which Metric

| Task | Prioritize | Reason |
|---|---|---|
| **Cancer diagnosis** | **Recall** | FN (missed cancer) is catastrophic |
| **Spam filtering** | **Precision** | FP (good email deleted) is unacceptable |
| **Fraud detection** | **F1** | Balance — both FP (blocked customer) and FN (missed fraud) matter |
| **Balanced dataset** | **Accuracy** | OK when both classes have equal cost |

#### 5.5 Classification Report in Python

```python
from sklearn.metrics import classification_report, confusion_matrix

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Full classification report
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nTP: {cm[1,1]}  FP: {cm[0,1]}  FN: {cm[1,0]}  TN: {cm[0,0]}")
```

The `classification_report` outputs:

```
              precision    recall  f1-score   support

           0     0.9823    0.9901    0.9862       810
           1     0.9506    0.9247    0.9375       190

    accuracy                         0.9790      1000
   macro avg     0.9664    0.9574    0.9619      1000
weighted avg     0.9788    0.9790    0.9789      1000
```

| Row | Meaning |
|---|---|
| **macro avg** | Unweighted mean across all classes — treats all classes equally |
| **weighted avg** | Mean weighted by class size — appropriate for imbalanced data |

#### 5.6 Multi-Class Metrics

For problems with more than 2 classes, extend to `macro`, `micro`, and `weighted` averaging:

```python
# Multi-class
print(classification_report(y_test, y_pred, target_names=['Class A', 'Class B', 'Class C']))
```

---

### 6. The ROC Curve & AUC

#### 6.1 Threshold Dependence

Most classifiers output a **probability**, not a hard label. The threshold (default 0.5) converts it:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

The **ROC (Receiver Operating Characteristic) curve** plots:
- **True Positive Rate (TPR = Recall)** on the y-axis.
- **False Positive Rate (FPR = FP/(FP+TN))** on the x-axis.

...at every possible threshold from 0 to 1.

#### 6.2 AUC — Area Under the ROC Curve

$$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$

| AUC | Interpretation |
|---|---|
| 1.0 | Perfect classifier |
| 0.9+ | Excellent |
| 0.7 – 0.9 | Good |
| 0.5 | Random guessing (diagonal line) |
| < 0.5 | Worse than random — flip your predictions! |

> **AUC is threshold-independent.** It measures the model's ability to rank positives above negatives across all thresholds — extremely powerful for imbalanced problems.

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# Get probabilities (not hard predictions)
y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC: {roc_auc:.4f}")

# Plot
import matplotlib.pyplot as plt
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Model').plot()
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

#### 6.3 Precision-Recall Curve

For highly imbalanced datasets, the **Precision-Recall (PR) curve** is more informative than ROC:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.plot(recall, precision, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP={ap:.4f})')
plt.show()
```

> **Use ROC-AUC** when false positives and false negatives have similar cost. **Use PR-AUC (Average Precision)** when positive class is rare and highly important.

---

### 7. Regression Metrics

For regression problems, we measure how far predictions are from the truth:

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mean squared error — penalizes large errors heavily |
| **RMSE** | $\sqrt{MSE}$ | Same units as $y$ — most interpretable |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean absolute error — robust to outliers |
| **R² (coefficient of determination)** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained (1.0 = perfect, 0 = baseline) |

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred_reg = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_reg)
r2   = r2_score(y_test, y_pred_reg)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
```

> **Key distinction**: MSE squares errors, making it sensitive to outliers. MAE treats all errors equally. When your target has outliers (e.g., house prices with mansions), prefer MAE or report both.

---

### 8. Hyperparameter Tuning

#### 8.1 Hyperparameters vs. Parameters

| Type | Definition | Examples |
|---|---|---|
| **Parameters** | Learned from data during training | Weights $w$, biases $b$, tree split values |
| **Hyperparameters** | Set before training, control the learning process | `n_estimators`, `max_depth`, `learning_rate`, `C`, `gamma` |

#### 8.2 Grid Search — Exhaustive Search

Try every combination of hyperparameter values:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,                        # 5-fold cross-validation for each combination
    scoring='f1_weighted',
    n_jobs=-1,                   # use all CPU cores
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score:   {grid_search.best_score_:.4f}")

# Evaluate best model on test set (one time only!)
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy:   {test_score:.4f}")
```

> **Total combinations above**: 3 × 4 × 3 = 36 parameter sets × 5 folds = **180 model fits**. Grid search scales exponentially — can become prohibitive.

#### 8.3 Randomized Search — Smarter Alternative

Instead of trying all combinations, sample $N$ random combinations:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators':      randint(50, 500),
    'max_depth':         [None, 5, 10, 15, 20, 30],
    'min_samples_split': randint(2, 20),
    'max_features':      ['sqrt', 'log2'],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,           # try 50 random combinations (vs. 36+ in grid search)
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score:   {random_search.best_score_:.4f}")
```

> **Research finding (Bergstra & Bengio, 2012)**: Randomized search finds equally good or better hyperparameters as grid search using a fraction of the compute. Prefer it in practice.

---

### 9. The Full Evaluation Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# Build a pipeline — preprocessing + model as a single object
# This prevents data leakage: scaler is fit only on training folds
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
])

# Cross-validate the entire pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv,
    scoring=['accuracy', 'f1_weighted', 'roc_auc'],
    return_train_score=True
)

print("=" * 55)
print(f"{'Metric':<20} {'Train':>10} {'Validation':>12}")
print("=" * 55)
for m in ['accuracy', 'f1_weighted', 'roc_auc']:
    tr = results[f'train_{m}'].mean()
    va = results[f'test_{m}'].mean()
    gap = tr - va
    flag = " ⚠️ overfit" if gap > 0.05 else ""
    print(f"{m:<20} {tr:>10.4f} {va:>12.4f}{flag}")

# Final fit and test evaluation
pipeline.fit(X_train, y_train)
print(f"\nFinal Test Score: {pipeline.score(X_test, y_test):.4f}")
```

> **Pipeline is not just convenience** — it is **correctness**. When you wrap preprocessing inside a Pipeline and use it with cross-validation, the scaler is refit on each training fold independently. Fitting the scaler on all data before splitting is a subtle but real form of data leakage.

---

### 10. Deployment — Model as a Service

#### 10.1 The Deployment Gap

Training a model is step 1. Making it useful to the world requires:

```
Raw Data → Preprocessing → Model → Prediction → Business Action
       ↑                                          ↑
  Data pipeline                          API / UI / Dashboard
```

#### 10.2 Saving and Loading Models

```python
import joblib

# Save the trained pipeline
joblib.dump(pipeline, 'model_pipeline.joblib')
print("Model saved to model_pipeline.joblib")

# Load it later (in production)
loaded_model = joblib.load('model_pipeline.joblib')
y_pred_loaded = loaded_model.predict(X_new)
```

> **Always save the full Pipeline, not just the model.** If you save only the classifier, you'll need to manually reapply the same scaler — a maintenance nightmare.

#### 10.3 Building a Streamlit App (Model-as-a-Service)

```python
# app.py — Streamlit web application
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model_pipeline.joblib')

st.set_page_config(page_title="ML Model API", layout="centered")
st.title("🤖 Customer Churn Prediction")
st.markdown("Enter customer features to get a real-time prediction.")

# --- Input sidebar ---
with st.sidebar:
    st.header("Customer Features")
    avg_order_value = st.number_input("Average Order Value ($)", 5.0, 500.0, 95.0)
    purchase_freq   = st.number_input("Monthly Purchase Frequency", 0.5, 50.0, 5.0)
    tenure_months   = st.number_input("Account Tenure (months)", 1, 120, 24)
    days_since_last = st.number_input("Days Since Last Purchase", 1, 365, 30)
    num_categories  = st.slider("Number of Product Categories", 1, 15, 4)

# Build input vector
X_input = np.array([[avg_order_value, purchase_freq, tenure_months,
                      days_since_last, num_categories]])

# Predict
if st.button("🔍 Predict", type="primary"):
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]

    col1, col2 = st.columns(2)
    with col1:
        label = "🚨 HIGH RISK" if prediction == 1 else "✅ LOW RISK"
        st.metric("Prediction", label)
    with col2:
        st.metric("Confidence", f"{max(probability):.1%}")

    st.progress(float(probability[1]), text=f"Churn probability: {probability[1]:.1%}")
```

Run with: `streamlit run app.py`

#### 10.4 REST API with FastAPI

For production integration (other services can call your model via HTTP):

```python
# api.py — FastAPI REST endpoint
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")
model = joblib.load('model_pipeline.joblib')

class CustomerFeatures(BaseModel):
    avg_order_value: float
    purchase_frequency: float
    account_tenure_months: float
    days_since_last_purchase: float
    num_categories_purchased: int

@app.post("/predict")
def predict(data: CustomerFeatures):
    X = np.array([[
        data.avg_order_value,
        data.purchase_frequency,
        data.account_tenure_months,
        data.days_since_last_purchase,
        data.num_categories_purchased
    ]])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return {
        "prediction": prediction,
        "churn_probability": probability,
        "label": "HIGH_RISK" if prediction == 1 else "LOW_RISK"
    }
```

Run with: `uvicorn api:app --host 0.0.0.0 --port 8000`

Call it from anywhere:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"avg_order_value": 95, "purchase_frequency": 5.0, "account_tenure_months": 24,
       "days_since_last_purchase": 30, "num_categories_purchased": 4}'
```

---

### 11. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Accuracy | $(TP + TN) / (TP + TN + FP + FN)$ |
| Precision | $TP / (TP + FP)$ |
| Recall (Sensitivity) | $TP / (TP + FN)$ |
| Specificity | $TN / (TN + FP)$ |
| F1-Score | $2 \cdot \frac{Precision \times Recall}{Precision + Recall}$ |
| K-Fold CV Score | $\frac{1}{K}\sum_{i=1}^{K} \text{score}_i$ |
| R² | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ |
| Total Error | $\text{Bias}^2 + \text{Variance} + \text{Noise}$ |

---

### 12. Summary

1. **Generalization is the goal** — train accuracy is meaningless; only unseen test performance matters.
2. **Bias-Variance Tradeoff**: Simple models underfit (high bias), complex models overfit (high variance). The sweet spot minimizes total error.
3. **The test set is sacred** — look at it exactly once. Use cross-validation for all model selection decisions.
4. **K-Fold Cross-Validation** gives a robust estimate of generalization by averaging performance across $K$ splits. Use **Stratified K-Fold** for classification.
5. **Accuracy is not enough** — use Precision, Recall, F1 for classification; RMSE and R² for regression.
6. **AUC-ROC** measures ranking ability independent of threshold — essential for imbalanced problems.
7. **Pipelines prevent data leakage** — always bundle preprocessing with the model.
8. **Grid/Random Search with CV** finds the best hyperparameters without touching the test set.
9. **Deployment completes the loop** — a model in a notebook helps no one; a model behind an API serves the world.

---

### 13. Bridge to the Tutorial

In the tutorial session, you will:
- Build a complete evaluation pipeline from scratch on a real-world churn dataset.
- Compute and interpret the full confusion matrix and classification report.
- Plot ROC and Precision-Recall curves for multiple models.
- Implement K-Fold cross-validation and compare Train vs. Validation scores to diagnose overfitting.
- Run a RandomizedSearchCV to tune a GradientBoosting classifier.
- Save the best model with `joblib` and wrap it in a Streamlit app for live prediction.

**Come to the tutorial. A model you can't measure is a model you can't trust.**
