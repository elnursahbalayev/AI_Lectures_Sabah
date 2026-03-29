# Week 8 Tutorial: Model Evaluation & Deployment
## "Can You Actually Trust Your Model?"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a complete evaluation and deployment pipeline — diagnosing overfitting with learning curves, computing the full suite of classification metrics, plotting ROC and Precision-Recall curves, running a randomized hyperparameter search with cross-validation, and wrapping your best model in a Streamlit web application that serves live predictions.

---

### 1. What We Are Building

In previous weeks, we trained many models. Today we ask the harder question: **how do we know they work?**

**By the end of this session, you will have:**
- Diagnosed overfitting using learning curves and train/validation gap analysis.
- Built a complete confusion matrix visualization with TP, FP, FN, TN annotations.
- Computed Precision, Recall, F1-Score, and AUC-ROC for multiple models.
- Plotted and compared ROC curves for Logistic Regression, Random Forest, and GradientBoosting.
- Implemented Stratified K-Fold cross-validation and interpreted mean ± std results.
- Run `RandomizedSearchCV` to find optimal hyperparameters.
- Saved the best model pipeline and deployed it as a Streamlit web app.

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    learning_curve, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score
)
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings('ignore')

print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")
print("\nReady to evaluate and deploy models!")
```

---

### 3. Creating Our Dataset — Customer Churn

We'll build a synthetic **customer churn dataset** — a classic binary classification problem with class imbalance (important customers don't churn that often).

```python
np.random.seed(42)
n = 5000

# --- Generate realistic customer churn features ---
# Feature engineering mirrors what you'd find in a real CRM system

account_tenure   = np.random.exponential(scale=24, size=n).clip(1, 120)      # months
monthly_spend    = np.random.lognormal(mean=4.5, sigma=0.8, size=n).clip(5, 500)
support_tickets  = np.random.poisson(lam=2, size=n).clip(0, 15)
login_frequency  = np.random.exponential(scale=8, size=n).clip(0.5, 50)      # per month
num_products     = np.random.randint(1, 8, size=n)
contract_type    = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])  # 0=month-to-month, 1=1yr, 2=2yr
payment_failures = np.random.poisson(lam=0.8, size=n).clip(0, 5)

# --- Churn probability — based on domain knowledge ---
# Long tenure, more products, long contracts → less churn
# Many support tickets, payment failures → more churn
churn_logit = (
    -0.05 * account_tenure
    - 0.002 * monthly_spend
    + 0.4  * support_tickets
    - 0.1  * login_frequency
    - 0.3  * num_products
    - 0.8  * contract_type        # month-to-month churns most
    + 0.9  * payment_failures
    + np.random.normal(0, 1.2, size=n)   # noise
)

churn_prob = 1 / (1 + np.exp(-churn_logit))
y = (churn_prob > 0.55).astype(int)   # ~22% churn rate — realistic imbalance

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'account_tenure':   account_tenure.round(1),
    'monthly_spend':    monthly_spend.round(2),
    'support_tickets':  support_tickets,
    'login_frequency':  login_frequency.round(1),
    'num_products':     num_products,
    'contract_type':    contract_type,    # 0, 1, 2 — already numeric
    'payment_failures': payment_failures,
    'churned':          y
})

feature_cols = [c for c in df.columns if c != 'churned']
X = df[feature_cols].values
y = df['churned'].values

print(f"Dataset shape:    {df.shape}")
print(f"Churn rate:       {y.mean():.1%}")
print(f"Non-churn (0):    {(y==0).sum()}")
print(f"Churn (1):        {(y==1).sum()}")
print(f"\nFeature summary:")
df[feature_cols].describe().round(2)
```

---

### 4. Step 1 — Proper Train/Validation/Test Split

```python
# Step 1: Hold out 20% as the sacred test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 2: From the remaining 80%, reserve 25% (= 20% of total) as validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Total dataset:  {len(y):>6} samples")
print(f"  Train:        {len(y_train):>6} ({len(y_train)/len(y):.0%})")
print(f"  Validation:   {len(y_val):>6} ({len(y_val)/len(y):.0%})")
print(f"  Test:         {len(y_test):>6} ({len(y_test)/len(y):.0%})")
print(f"\nChurn rate in each split:")
print(f"  Train:      {y_train.mean():.1%}")
print(f"  Validation: {y_val.mean():.1%}")
print(f"  Test:       {y_test.mean():.1%}")
```

> **Verify stratification**: All three splits should show nearly identical churn rates. If they differ significantly, your stratification failed.

---

### 5. Step 2 — Building Pipelines for Three Models

We wrap each model in a `Pipeline` with `StandardScaler` to prevent data leakage in cross-validation.

```python
# Define three competing models
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),   # doesn't affect trees, but keeps pipeline consistent
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                           max_depth=4, random_state=42))
    ]),
}
```

---

### 6. Step 3 — Cross-Validation: Diagnosing Overfitting

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ['accuracy', 'f1_weighted', 'roc_auc']

cv_results = {}

print("=" * 65)
print(f"{'Model':<25} {'Metric':<18} {'Train':>8} {'Val':>8} {'Gap':>8}")
print("=" * 65)

for name, pipe in pipelines.items():
    result = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring=metrics,
        return_train_score=True
    )
    cv_results[name] = result

    for m in metrics:
        tr = result[f'train_{m}'].mean()
        va = result[f'test_{m}'].mean()
        gap = tr - va
        flag = " ← overfit" if gap > 0.07 else ""
        print(f"{name:<25} {m:<18} {tr:>8.4f} {va:>8.4f} {gap:>+8.4f}{flag}")
    print("-" * 65)
```

#### 6.1 Visualize the Train vs. Validation Gap

```python
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors = {'Logistic Regression': '#3b82f6',
          'Random Forest': '#4ade80',
          'Gradient Boosting': '#f59e0b'}

for ax, metric in zip(axes, metrics):
    x = np.arange(len(pipelines))
    width = 0.35

    for i, (name, result) in enumerate(cv_results.items()):
        tr_mean = result[f'train_{metric}'].mean()
        tr_std  = result[f'train_{metric}'].std()
        va_mean = result[f'test_{metric}'].mean()
        va_std  = result[f'test_{metric}'].std()

        ax.bar(i - width/2, tr_mean, width, label='Train' if i == 0 else "_",
               color=colors[name], alpha=0.9, yerr=tr_std, capsize=4)
        ax.bar(i + width/2, va_mean, width, label='Validation' if i == 0 else "_",
               color=colors[name], alpha=0.4, yerr=va_std, capsize=4,
               hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in pipelines.keys()], fontsize=9)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(['Train (solid)', 'Validation (hatched)'], fontsize=9)
plt.suptitle('Cross-Validation: Train vs. Validation Scores', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 7. Step 4 — Learning Curves: How Much Data Do We Need?

Learning curves plot model performance as training size grows — revealing whether adding more data would help.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
train_sizes_frac = np.linspace(0.1, 1.0, 10)

for ax, (name, pipe) in zip(axes, pipelines.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train,
        train_sizes=train_sizes_frac,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    ax.plot(train_sizes, tr_mean, 'b-o', label='Train', linewidth=2, markersize=5)
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.1, color='blue')

    ax.plot(train_sizes, va_mean, 'r-o', label='Validation', linewidth=2, markersize=5)
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.1, color='red')

    # Gap annotation
    gap = tr_mean[-1] - va_mean[-1]
    ax.annotate(f'Gap: {gap:.3f}', xy=(train_sizes[-1], va_mean[-1]),
                xytext=(-60, -20), textcoords='offset points',
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('F1-Score (Weighted)')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Learning Curves — Does More Data Help?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: 
- If train and validation scores both converge to a similar low value → **underfitting** — more data won't help, need a more complex model.
- If train score is high but validation score is low and the gap stays wide → **overfitting** — adding data *will* help.
- If both scores converge to a high value → model is well-fitted, ready for production.

---

### 8. Step 5 — Confusion Matrix & Classification Report

```python
# Fit all models and collect predictions
fitted_models = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

# Full evaluation for each model on the VALIDATION set (not test yet!)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, pipe) in zip(axes, fitted_models.items()):
    y_pred = pipe.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    # Normalized confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # Labels: [TN, FP, FN, TP]
    labels_text = [
        [f'TN\n{cm[0,0]}\n({cm_norm[0,0]:.0%})', f'FP\n{cm[0,1]}\n({cm_norm[0,1]:.0%})'],
        [f'FN\n{cm[1,0]}\n({cm_norm[1,0]:.0%})', f'TP\n{cm[1,1]}\n({cm_norm[1,1]:.0%})']
    ]

    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, labels_text[i][j], ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNo Churn', 'Predicted\nChurn'])
    ax.set_yticklabels(['Actual\nNo Churn', 'Actual\nChurn'])
    ax.set_title(name, fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrices (Validation Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()
```

```python
# Print classification reports
for name, pipe in fitted_models.items():
    y_pred = pipe.predict(X_val)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_val, y_pred,
                                 target_names=['No Churn', 'Churn'],
                                 digits=4))
```

**Discussion — Think carefully about FN vs. FP in this context:**
- A **False Negative** (predicted "no churn" but customer churned) → missed retention opportunity, lost revenue.
- A **False Positive** (predicted "churn" but customer didn't) → unnecessary retention campaign cost.
- Which error is more expensive? This depends on your business case. If retention campaigns are cheap and churn is costly, **maximize Recall**. If campaigns are expensive, **balance with F1**.

---

### 9. Step 6 — ROC Curves & AUC Comparison

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

line_styles = ['-', '--', '-.']
model_colors = ['#3b82f6', '#4ade80', '#f59e0b']

# --- ROC Curves ---
for (name, pipe), ls, color in zip(fitted_models.items(), line_styles, model_colors):
    y_prob = pipe.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, ls, color=color, linewidth=2.2,
                 label=f'{name} (AUC={roc_auc:.4f})')

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=11)
axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
axes[0].set_title('ROC Curve — All Models', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_xlim([-0.01, 1.01])
axes[0].set_ylim([-0.01, 1.05])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add "operating point" at threshold=0.5
for (name, pipe), color in zip(fitted_models.items(), model_colors):
    y_prob = pipe.predict_proba(X_val)[:, 1]
    y_pred_50 = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred_50)
    fpr_pt = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    tpr_pt = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    axes[0].scatter(fpr_pt, tpr_pt, color=color, s=80, zorder=5, marker='o',
                    edgecolors='black', linewidth=1)

# --- Precision-Recall Curves ---
for (name, pipe), ls, color in zip(fitted_models.items(), line_styles, model_colors):
    y_prob = pipe.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)
    axes[1].plot(recall, precision, ls, color=color, linewidth=2.2,
                 label=f'{name} (AP={ap:.4f})')

baseline = y_val.mean()
axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Baseline (AP={baseline:.4f})')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curve — All Models', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_xlim([0, 1.01])
axes[1].set_ylim([0, 1.05])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('ROC & Precision-Recall Curves (Validation Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: 
- The **dots on the ROC curves** mark the operating point at threshold=0.5. Moving the threshold changes where you operate on this curve.
- The **PR curve baseline** (dashed line) shows the performance of random guessing — any model must beat this.
- For the churn use case: which curve is more informative? Why?

---

### 10. Step 7 — Threshold Optimization

The default threshold of 0.5 is rarely optimal. Let's find the best threshold for our business objective:

```python
# Focus on Gradient Boosting (usually the strongest)
gb_pipe  = fitted_models['Gradient Boosting']
y_prob   = gb_pipe.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.02)
f1_scores  = []
precision_scores = []
recall_scores    = []

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    # Avoid division by zero if all predictions are one class
    if y_pred_thresh.sum() == 0 or y_pred_thresh.sum() == len(y_pred_thresh):
        f1_scores.append(0); precision_scores.append(0); recall_scores.append(0)
        continue
    f1_scores.append(f1_score(y_val, y_pred_thresh, zero_division=0))
    from sklearn.metrics import precision_score, recall_score
    precision_scores.append(precision_score(y_val, y_pred_thresh, zero_division=0))
    recall_scores.append(recall_score(y_val, y_pred_thresh, zero_division=0))

best_thresh = thresholds[np.argmax(f1_scores)]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(thresholds, f1_scores,        'b-', linewidth=2.2, label='F1-Score')
ax.plot(thresholds, precision_scores, 'g--', linewidth=1.8, label='Precision')
ax.plot(thresholds, recall_scores,    'r--', linewidth=1.8, label='Recall')
ax.axvline(x=best_thresh, color='black', linestyle=':', linewidth=1.5,
           label=f'Best threshold = {best_thresh:.2f}')
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1,
           label='Default threshold = 0.50')
ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Threshold Analysis — Gradient Boosting on Validation Set', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim([0.05, 0.95])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\n--- Default threshold (0.50) ---")
y_pred_default = (y_prob >= 0.50).astype(int)
print(classification_report(y_val, y_pred_default, target_names=['No Churn', 'Churn'], digits=4))

print(f"\n--- Optimal threshold ({best_thresh:.2f}) ---")
y_pred_optimal = (y_prob >= best_thresh).astype(int)
print(classification_report(y_val, y_pred_optimal, target_names=['No Churn', 'Churn'], digits=4))
```

---

### 11. Step 8 — Hyperparameter Tuning with RandomizedSearchCV

```python
from scipy.stats import randint, uniform

param_dist = {
    'clf__n_estimators':      randint(50, 400),
    'clf__learning_rate':     uniform(0.01, 0.3),
    'clf__max_depth':         randint(2, 8),
    'clf__min_samples_split': randint(2, 20),
    'clf__subsample':         uniform(0.6, 0.4),   # 0.6 to 1.0
}

gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(random_state=42))
])

random_search = RandomizedSearchCV(
    gb_pipeline,
    param_dist,
    n_iter=40,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Running RandomizedSearchCV (40 candidates × 5 folds = 200 fits)...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k:<35} {v}")
print(f"\nBest CV AUC-ROC:   {random_search.best_score_:.4f}")

# Compare to validation set
best_model = random_search.best_estimator_
y_prob_best = best_model.predict_proba(X_val)[:, 1]
val_auc = auc(*roc_curve(y_val, y_prob_best)[:2])
print(f"Validation AUC-ROC: {val_auc:.4f}")
```

```python
# Show how different hyperparameter settings performed
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df[['param_clf__n_estimators', 'param_clf__learning_rate',
                           'param_clf__max_depth', 'mean_test_score', 'std_test_score']]
results_df.columns = ['n_estimators', 'learning_rate', 'max_depth', 'mean_auc', 'std_auc']
results_df = results_df.sort_values('mean_auc', ascending=False)

print("\nTop 10 hyperparameter configurations:")
print(results_df.head(10).round(4).to_string(index=False))

# Scatter: learning rate vs. AUC
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sc = axes[0].scatter(results_df['learning_rate'], results_df['mean_auc'],
                      c=results_df['n_estimators'], cmap='plasma', s=60, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label='n_estimators')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Mean CV AUC-ROC')
axes[0].set_title('Learning Rate vs. CV AUC', fontsize=11, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

sc2 = axes[1].scatter(results_df['max_depth'], results_df['mean_auc'],
                       c=results_df['learning_rate'], cmap='viridis', s=60, alpha=0.7)
plt.colorbar(sc2, ax=axes[1], label='Learning Rate')
axes[1].set_xlabel('Max Depth')
axes[1].set_ylabel('Mean CV AUC-ROC')
axes[1].set_title('Max Depth vs. CV AUC', fontsize=11, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('RandomizedSearchCV Results', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hyperparameter_search.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 12. Step 9 — THE Final Test Evaluation (Once!)

> **This is the moment of truth.** We have made all our decisions — chosen the model, tuned hyperparameters, optimized threshold — using only train and validation data. Now we evaluate exactly once on the held-out test set.

```python
# Final model: best GradientBoosting from RandomizedSearch
final_model = random_search.best_estimator_
final_model.fit(X_train, y_train)   # refit on full training set

y_test_prob = final_model.predict_proba(X_test)[:, 1]
y_test_pred_default = final_model.predict(X_test)
y_test_pred_optimal = (y_test_prob >= best_thresh).astype(int)

print("=" * 55)
print("  FINAL TEST SET EVALUATION (UNSEEN DATA)")
print("=" * 55)

fpr_t, tpr_t, _ = roc_curve(y_test, y_test_prob)
test_auc = auc(fpr_t, tpr_t)
print(f"\nAUC-ROC (test):  {test_auc:.4f}")
print(f"AUC-ROC (val):   {val_auc:.4f}")
print(f"Gap (overfit?):  {val_auc - test_auc:+.4f}")

print(f"\n--- Classification Report (threshold = {best_thresh:.2f}) ---")
print(classification_report(y_test, y_test_pred_optimal,
                             target_names=['No Churn', 'Churn'], digits=4))

# Side-by-side confusion matrices (validation vs. test)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (split_name, y_true, y_pred) in zip(axes, [
    ('Validation Set (used for decisions)', y_val, (y_prob_best >= best_thresh).astype(int)),
    ('Test Set (final truth)', y_test, y_test_pred_optimal)
]):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04)

    labels_text = [
        [f'TN\n{cm[0,0]}\n({cm_norm[0,0]:.0%})', f'FP\n{cm[0,1]}\n({cm_norm[0,1]:.0%})'],
        [f'FN\n{cm[1,0]}\n({cm_norm[1,0]:.0%})', f'TP\n{cm[1,1]}\n({cm_norm[1,1]:.0%})']
    ]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, labels_text[i][j], ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No Churn', 'Pred: Churn'])
    ax.set_yticklabels(['Actual: No Churn', 'Actual: Churn'])
    ax.set_title(split_name, fontsize=10, fontweight='bold')

plt.suptitle('Validation vs. Test Set Confusion Matrices', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('final_evaluation.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 13. Step 10 — Save the Model Pipeline

```python
import joblib

# Save the full pipeline (scaler + model)
model_path = 'churn_model.joblib'
joblib.dump(final_model, model_path)
print(f"Model saved to: {model_path}")

# Verify: load and predict
loaded_model = joblib.load(model_path)
test_prediction = loaded_model.predict(X_test[:5])
test_proba      = loaded_model.predict_proba(X_test[:5])[:, 1]

print("\nVerification — first 5 test predictions:")
for i, (pred, prob) in enumerate(zip(test_prediction, test_proba)):
    label = "CHURN RISK" if pred == 1 else "RETAIN"
    print(f"  Customer {i+1}: {label} (p={prob:.3f})")
```

---

### 14. Step 11 — Streamlit Deployment App

Save the following as **`app.py`** in the same directory as your notebook:

```python
# churn_app.py — Streamlit Churn Prediction Web App
import streamlit as st
import joblib
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')

model = load_model()
OPTIMAL_THRESHOLD = 0.40  # Set to the best_thresh you found

# --- App Header ---
st.title("🔮 Customer Churn Prediction System")
st.markdown("*Powered by Gradient Boosting — Real-time churn risk assessment*")
st.divider()

# --- Input Form ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Customer Features")
    account_tenure   = st.slider("Account Tenure (months)", 1, 120, 24)
    monthly_spend    = st.number_input("Monthly Spend ($)", 5.0, 500.0, 95.0, step=5.0)
    support_tickets  = st.number_input("Support Tickets (last 3 months)", 0, 15, 2)
    login_frequency  = st.slider("Login Frequency (per month)", 0.5, 50.0, 8.0)
    num_products     = st.slider("Number of Products Subscribed", 1, 7, 3)
    contract_type    = st.selectbox("Contract Type",
                                    options=[0, 1, 2],
                                    format_func=lambda x: {0: "Month-to-Month", 1: "1 Year", 2: "2 Year"}[x])
    payment_failures = st.number_input("Payment Failures (last 12 months)", 0, 5, 0)

    predict_btn = st.button("🔍 Assess Churn Risk", type="primary", use_container_width=True)

# --- Prediction ---
with col2:
    if predict_btn:
        X_input = np.array([[account_tenure, monthly_spend, support_tickets,
                              login_frequency, num_products, contract_type, payment_failures]])
        probability = model.predict_proba(X_input)[0][1]
        prediction  = int(probability >= OPTIMAL_THRESHOLD)

        st.subheader("Prediction Result")

        risk_level = "HIGH" if prediction == 1 else "LOW"
        risk_color = "🔴" if prediction == 1 else "🟢"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Risk Level", f"{risk_color} {risk_level}")
        col_b.metric("Churn Probability", f"{probability:.1%}")
        col_c.metric("Threshold Used", f"{OPTIMAL_THRESHOLD:.0%}")

        st.progress(float(probability), text=f"Churn Probability: {probability:.1%}")

        if prediction == 1:
            st.error("⚠️ **High Churn Risk Detected** — Recommend immediate retention action.")
            st.markdown("""
            **Suggested Actions:**
            - 📞 Proactive outreach from account team
            - 🎁 Loyalty discount or free upgrade offer
            - 🔄 Contract renewal incentive
            - 📋 Review open support tickets
            """)
        else:
            st.success("✅ **Low Churn Risk** — Customer appears stable.")
            st.markdown("""
            **Suggested Actions:**
            - 📈 Upsell opportunity assessment
            - ⭐ Candidate for referral program
            """)

        # Feature importance table
        st.subheader("Input Summary")
        feature_names = ['account_tenure', 'monthly_spend', 'support_tickets',
                         'login_frequency', 'num_products', 'contract_type', 'payment_failures']
        import pandas as pd
        input_df = pd.DataFrame(X_input, columns=feature_names)
        st.dataframe(input_df.round(2), use_container_width=True)
    else:
        st.info("👈 Adjust customer features and click **Assess Churn Risk** to get a prediction.")

        # Show model info
        st.subheader("Model Information")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Algorithm** | Gradient Boosting Classifier |
        | **Tuning** | RandomizedSearchCV (40 configs × 5 folds) |
        | **Primary Metric** | AUC-ROC |
        | **Deployment** | Streamlit + joblib |
        """)
```

**Run with:**
```bash
streamlit run churn_app.py
```

Your model is now a live web app accessible at `http://localhost:8501`.

---

### 15. Full Pipeline Summary

```python
# Print a full summary of everything we built
print("=" * 65)
print("WEEK 8 TUTORIAL — COMPLETE PIPELINE SUMMARY")
print("=" * 65)

from sklearn.metrics import roc_auc_score

summary = {
    "Dataset size":                   f"{n} customers",
    "Feature count":                  len(feature_cols),
    "Churn rate (imbalanced)":        f"{y.mean():.1%}",
    "Split strategy":                 "60% train / 20% val / 20% test (stratified)",
    "Cross-validation":               "StratifiedKFold, K=5",
    "Models compared":                ", ".join(pipelines.keys()),
    "Best model":                     "Gradient Boosting (RandomizedSearchCV)",
    "Hyperparameter configs tried":   "40 × 5 folds = 200 fits",
    "Best CV AUC-ROC":                f"{random_search.best_score_:.4f}",
    "Validation AUC-ROC":             f"{val_auc:.4f}",
    "Test AUC-ROC (final)":           f"{test_auc:.4f}",
    "Optimal threshold (F1-max)":     f"{best_thresh:.2f}",
    "Deployment":                     "joblib model + Streamlit web app",
}

for key, val in summary.items():
    print(f"  {key:<40} {val}")
```

---

### 16. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Analysis</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A — Threshold Sensitivity:</strong> Change the `OPTIMAL_THRESHOLD` in `app.py` from 0.40 to 0.60. How does this change the number of flagged customers? In a dataset of 10,000 customers, what is the cost difference if each retention campaign costs $50?</li>
        <li><strong>Experiment B — Imbalance Handling:</strong> Use `class_weight='balanced'` in the Gradient Boosting classifier (or try SMOTE oversampling). Does Recall for the churn class improve? Does overall AUC change?</li>
        <li><strong>Experiment C — Feature Engineering:</strong> Create new features: `spend_per_product = monthly_spend / num_products` and `ticket_rate = support_tickets / account_tenure`. Rerun the full pipeline. Do these derived features improve AUC?</li>
        <li><strong>Experiment D — Calibration:</strong> Use `sklearn.calibration.CalibratedClassifierCV` to calibrate the model's probability outputs. Plot a calibration curve (reliability diagram). Are the raw probabilities well-calibrated? Does calibration improve the threshold analysis?</li>
    </ul>
</div>

---

### 17. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that:
    1. Loads a classification dataset of your choice (e.g., Kaggle Telco Customer Churn, Heart Disease UCI, or Credit Card Fraud).
    2. Implements a correct train/validation/test split with **no data leakage** (use Pipelines).
    3. Trains at least **3 different model types** and compares them using 5-Fold Cross-Validation reporting mean ± std of AUC-ROC and F1.
    4. Plots Learning Curves for the best model and interprets whether the model is over- or under-fitting.
    5. Runs `RandomizedSearchCV` (at least 30 iterations) on the best model and reports the best hyperparameters.
    6. Evaluates the final model on the test set with a full classification report, ROC curve, and Precision-Recall curve.
    7. Saves the model with `joblib` and writes a Streamlit app (`app.py`) that accepts feature inputs and outputs a prediction with probability.

*   **Report**: Write a 5-sentence paragraph answering: *"What is data leakage, how does it happen in a preprocessing pipeline, and how does using `sklearn.pipeline.Pipeline` prevent it?"*

*   **Bonus**: Deploy your Streamlit app to **Streamlit Community Cloud** (free at share.streamlit.io) so it has a public URL. Submit the URL with your assignment.

**See you at the next Lecture! A model without an evaluation is just expensive guessing.**
