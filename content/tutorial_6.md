# Week 6 Tutorial: Ensemble Methods — Fraud Detection
## "Your First Forest"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a fraud detection system using Decision Trees, Random Forests, and XGBoost on a highly imbalanced dataset, mastered feature importance, and compared all models from Weeks 4-6.

---

### 1. What We Are Building

Last week, we classified balanced datasets (Pass/Fail). Today, we face a **real-world challenge**: detecting fraud in financial transactions where only ~1-2% of transactions are fraudulent.

**By the end of this session, you will have:**
- Built and visualized a Decision Tree classifier.
- Trained a Random Forest and understood its advantages over a single tree.
- Used XGBoost with hyperparameter tuning.
- Handled severe class imbalance using `scale_pos_weight` and class weighting.
- Compared all models: Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost.
- Extracted and visualized feature importance.
- Applied cross-validation for robust evaluation.

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# XGBoost — install with: pip install xgboost
from xgboost import XGBClassifier

# Verify versions
print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")

print("\nReady to detect fraud!")
```

---

### 3. Creating Our Dataset

We'll generate a synthetic fraud detection dataset with realistic class imbalance.

```python
np.random.seed(42)
n = 5000  # Number of transactions

# Generate features
data = {
    'amount': np.concatenate([
        np.random.exponential(50, int(n * 0.98)),    # Normal: small amounts
        np.random.exponential(500, int(n * 0.02))    # Fraud: larger amounts
    ]),
    'hour': np.concatenate([
        np.random.normal(14, 4, int(n * 0.98)).clip(0, 23),   # Normal: daytime
        np.random.normal(3, 2, int(n * 0.02)).clip(0, 23)     # Fraud: late night
    ]),
    'distance_from_home': np.concatenate([
        np.random.exponential(10, int(n * 0.98)),    # Normal: close to home
        np.random.exponential(100, int(n * 0.02))    # Fraud: far from home
    ]),
    'num_transactions_last_hour': np.concatenate([
        np.random.poisson(1, int(n * 0.98)),         # Normal: few transactions
        np.random.poisson(5, int(n * 0.02))          # Fraud: many rapid transactions
    ]),
    'is_international': np.concatenate([
        np.random.binomial(1, 0.05, int(n * 0.98)),  # Normal: rarely international
        np.random.binomial(1, 0.6, int(n * 0.02))    # Fraud: often international
    ]),
}

df = pd.DataFrame(data)

# Add noise features (should be ignored by good models)
df['day_of_week'] = np.random.randint(0, 7, n)
df['user_age'] = np.random.normal(35, 12, n).clip(18, 80)

# Labels: first 98% are normal (0), last 2% are fraud (1)
df['is_fraud'] = np.concatenate([np.zeros(int(n * 0.98)), np.ones(int(n * 0.02))]).astype(int)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['is_fraud'].value_counts())
print(f"\nFraud rate: {df['is_fraud'].mean():.1%}")
print(f"\nFirst 5 rows:")
df.head()
```

---

### 4. Step 1 — Exploration

#### 4.1 Understand the Imbalance

```python
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nFraud distribution:")
print(df['is_fraud'].value_counts())
print(f"\nFraud rate: {df['is_fraud'].mean():.2%}")

# Visualize class imbalance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of class distribution
counts = df['is_fraud'].value_counts()
axes[0].bar(['Normal', 'Fraud'], counts.values, color=['#4ade80', '#ef4444'], alpha=0.8)
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# Feature distributions by class
for label, color, name in [(0, '#4ade80', 'Normal'), (1, '#ef4444', 'Fraud')]:
    mask = df['is_fraud'] == label
    axes[1].hist(df.loc[mask, 'amount'], bins=50, alpha=0.5, color=color, label=name, density=True)
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig('fraud_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
```

#### 4.2 Feature Comparison: Normal vs. Fraud

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features = ['amount', 'hour', 'distance_from_home',
            'num_transactions_last_hour', 'is_international', 'user_age']

for ax, feat in zip(axes.flatten(), features):
    for label, color, name in [(0, '#4ade80', 'Normal'), (1, '#ef4444', 'Fraud')]:
        mask = df['is_fraud'] == label
        ax.hist(df.loc[mask, feat], bins=30, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(f'{feat}')
    ax.legend()

plt.suptitle('Feature Distributions: Normal vs. Fraud', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: Which features show clear separation between Normal and Fraud? Which features look like noise?

---

### 5. Step 2 — Decision Tree

#### 5.1 Train a Decision Tree

```python
# Prepare data
feature_cols = ['amount', 'hour', 'distance_from_home',
                'num_transactions_last_hour', 'is_international',
                'day_of_week', 'user_age']

X = df[feature_cols]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {len(X_train)} samples ({y_train.mean():.2%} fraud)")
print(f"Test:     {len(X_test)} samples  ({y_test.mean():.2%} fraud)")

# Train a single Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print(f"\nDecision Tree Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_dt):.4f}")
```

#### 5.2 Visualize the Tree

```python
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=feature_cols, class_names=['Normal', 'Fraud'],
          filled=True, rounded=True, ax=ax, fontsize=8,
          proportion=True, impurity=True)
plt.title('Decision Tree for Fraud Detection', fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree_viz.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 6. Step 3 — Random Forest

```python
# Train a Random Forest (200 trees)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,          # let trees grow fully
    max_features='sqrt',     # sqrt(7) ≈ 2-3 features per split
    class_weight='balanced', # handle imbalance
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"\n{classification_report(y_test, y_pred_rf, target_names=['Normal', 'Fraud'])}")
```

---

### 7. Step 4 — XGBoost

#### 7.1 Handle Imbalance with scale_pos_weight

```python
# Calculate imbalance ratio for scale_pos_weight
n_normal = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale_ratio = n_normal / n_fraud

print(f"Normal: {n_normal}, Fraud: {n_fraud}")
print(f"Imbalance ratio: {scale_ratio:.1f}:1")
print(f"scale_pos_weight = {scale_ratio:.1f}")

# Train XGBoost with imbalance handling
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_ratio,   # handle class imbalance
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print(f"\n{classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Fraud'])}")
```

#### 7.2 Feature Importance from XGBoost

```python
# Feature importance (built-in Gini importance)
importances = xgb.feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importances)
ax.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx], color='#3b82f6', alpha=0.8)
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nFeature Importance Ranking:")
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<30} {imp:.4f}")
```

**Discussion**: Do the important features match your intuition? Are the "noise" features (`day_of_week`, `user_age`) ranked low?

---

### 8. Step 5 — Full Model Comparison

```python
# Define all models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced'))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                             scale_pos_weight=scale_ratio, eval_metric='logloss', random_state=42),
}

# Train and evaluate all models
print("=" * 75)
print("FULL MODEL COMPARISON — FRAUD DETECTION")
print("=" * 75)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 77)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get probabilities for AUC (if supported)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', model), 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}
    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")
```

---

### 9. Step 6 — ROC Curves & Confusion Matrices

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PLOT 1: ROC Curves
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — All Models')
axes[0].legend(fontsize=8)

# PLOT 2: F1 Score Comparison
model_names = list(results.keys())
f1_scores = [results[n]['f1'] for n in model_names]
colors = ['#3b82f6', '#8b5cf6', '#ef4444', '#4ade80', '#f59e0b', '#06b6d4']
bars = axes[1].barh(model_names, f1_scores, color=colors, alpha=0.8)
axes[1].set_xlabel('F1 Score')
axes[1].set_title('F1 Score Comparison')
axes[1].set_xlim(0, 1.1)
for bar, val in zip(bars, f1_scores):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 10. Step 7 — Cross-Validation

```python
# Cross-validation for more robust evaluation
print("=" * 55)
print("5-FOLD CROSS-VALIDATION (F1 Score)")
print("=" * 55)

cv_models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                             scale_pos_weight=scale_ratio, eval_metric='logloss', random_state=42),
}

for name, model in cv_models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"{name:<25} F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

### 11. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Analysis</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Remove `class_weight='balanced'` and `scale_pos_weight` from all models. How does this affect Recall on the fraud class? Does accuracy still look "good"?</li>
        <li><strong>Experiment B:</strong> Tune XGBoost hyperparameters: try `max_depth` values of 3, 5, 8, 12. Plot train vs. test F1 to find the sweet spot before overfitting.</li>
        <li><strong>Experiment C:</strong> Increase fraud rate to 10% and 30%. How does model performance change? At what fraud rate does class weighting stop helping?</li>
        <li><strong>Experiment D:</strong> Download the real Kaggle Credit Card Fraud dataset and apply this same pipeline. Compare results to our synthetic data.</li>
    </ul>
</div>

---

### 12. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that:
    1. Loads an imbalanced dataset (Kaggle Credit Card Fraud, or our synthetic data with ≤5% positive class).
    2. Trains and compares Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
    3. Handles class imbalance using at least two techniques (`class_weight`, `scale_pos_weight`, or oversampling).
    4. Evaluates using Precision, Recall, F1, AUC-ROC, and the Confusion Matrix (not just accuracy!).
    5. Visualizes feature importance and ROC curves for all models.
    6. Uses 5-fold cross-validation for the final comparison.
*   **Report**: Write a brief paragraph (3-5 sentences) explaining why accuracy is misleading on imbalanced datasets, with a numerical example.
*   **Bonus**: Implement a simple hyperparameter search for XGBoost using `GridSearchCV` or `RandomizedSearchCV`. Which hyperparameters improve F1 the most?

**See you at the Lecture!**
