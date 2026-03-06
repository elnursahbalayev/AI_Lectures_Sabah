# Week 5 Tutorial: Binary Classification — Pass/Fail Prediction
## "Your First Classifier"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a Logistic Regression classifier from scratch AND with Scikit-learn, trained SVMs, visualized decision boundaries, and evaluated using Precision, Recall, F1, and the Confusion Matrix.

---

### 1. What We Are Building

Last week, we predicted continuous numbers (housing prices). Today, we predict **discrete labels** — does a student Pass or Fail based on their study hours and class attendance?

**By the end of this session, you will have:**
- Implemented Logistic Regression from scratch using NumPy.
- Compared your implementation to Scikit-learn's `LogisticRegression`.
- Trained SVM classifiers with different kernels.
- Visualized decision boundaries for both models.
- Evaluated using Accuracy, Precision, Recall, F1, and the Confusion Matrix.
- Experimented with threshold tuning and observed its effect on predictions.

---

### 2. Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Verify versions
print(f"Pandas:  {pd.__version__}")
print(f"NumPy:   {np.__version__}")

print("\nReady to build our first classifier!")
```

---

### 3. Creating Our Dataset

We'll generate a realistic student performance dataset with known relationships so we can verify our classifier learns correctly.

```python
np.random.seed(42)
n = 400  # Number of students

# Generate features
data = {
    'study_hours': np.random.uniform(0, 12, n),          # Hours studied per week
    'attendance': np.random.uniform(40, 100, n),          # Attendance percentage
    'prev_grade': np.random.uniform(20, 100, n),          # Previous semester grade
    'sleep_hours': np.random.uniform(4, 10, n),           # Average sleep per night
}

df = pd.DataFrame(data)

# Generate labels with known formula + noise
# TRUE RELATIONSHIP (what the model should learn):
#   score = 4*study_hours + 0.3*attendance + 0.2*prev_grade + 2*sleep_hours + noise
#   Pass if score > 50
score = (
    4 * df['study_hours'] +
    0.3 * df['attendance'] +
    0.2 * df['prev_grade'] +
    2 * df['sleep_hours'] +
    np.random.normal(0, 5, n)  # Random noise
)
df['passed'] = (score > 50).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['passed'].value_counts())
print(f"\nPass rate: {df['passed'].mean():.1%}")
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

#### 4.2 Visualize Class Distribution

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plots: each feature colored by class
features = ['study_hours', 'attendance', 'prev_grade', 'sleep_hours']
colors = {0: '#ef4444', 1: '#4ade80'}

for ax, feat in zip(axes.flatten(), features):
    for label in [0, 1]:
        mask = df['passed'] == label
        ax.scatter(df.loc[mask, feat], df.loc[mask, 'passed'],
                   alpha=0.4, s=15, color=colors[label],
                   label=f"{'Pass' if label else 'Fail'}")
    ax.set_xlabel(feat)
    ax.set_ylabel('Passed')
    ax.set_title(f'Pass/Fail vs {feat}')
    ax.legend()

plt.tight_layout()
plt.savefig('classification_features.png', dpi=100, bbox_inches='tight')
plt.show()
```

#### 4.3 Feature vs. Feature (2D View)

```python
fig, ax = plt.subplots(figsize=(10, 7))

for label, color, name in [(0, '#ef4444', 'Fail'), (1, '#4ade80', 'Pass')]:
    mask = df['passed'] == label
    ax.scatter(df.loc[mask, 'study_hours'], df.loc[mask, 'attendance'],
               alpha=0.5, s=25, color=color, label=name)

ax.set_xlabel('Study Hours per Week')
ax.set_ylabel('Attendance (%)')
ax.set_title('Pass/Fail Distribution')
ax.legend()
plt.tight_layout()
plt.savefig('class_distribution_2d.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Discussion**: Can you visually see a boundary that separates Pass from Fail? Is the boundary linear?

---

### 5. Step 2 — Logistic Regression from Scratch

Let's implement logistic regression with gradient descent to understand the mechanics.

#### 5.1 Prepare Data

```python
# Use study_hours and attendance for 2D visualization
X_simple = df[['study_hours', 'attendance']].values
y = df['passed'].values

# Train-test split
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Scale features (critical for gradient descent convergence)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_s)
X_test_sc = scaler.transform(X_test_s)

print(f"Training: {len(X_train_s)} samples")
print(f"Test:     {len(X_test_s)} samples")
print(f"Pass rate (train): {y_train.mean():.1%}")
print(f"Pass rate (test):  {y_test.mean():.1%}")
```

#### 5.2 Sigmoid + Gradient Descent Implementation

```python
def sigmoid(z):
    """The sigmoid function — squashes any value to [0, 1]."""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000):
    """Train logistic regression using Gradient Descent."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights to zero
    b = 0.0                    # Initialize bias to zero
    history = []               # Track loss over time

    for epoch in range(epochs):
        # Forward pass: compute probabilities
        z = X @ w + b
        y_pred = sigmoid(z)

        # Compute Binary Cross-Entropy loss
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        history.append(loss)

        # Compute gradients
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

        # Log progress
        if epoch % 200 == 0:
            acc = np.mean((y_pred >= 0.5) == y)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    return w, b, history

# Train
w, b, history = logistic_regression_gd(X_train_sc, y_train, lr=0.1, epochs=1000)
print(f"\nFinal weights: {w}")
print(f"Final bias: {b:.4f}")
```

#### 5.3 Visualize Training Progress

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history, color='#3b82f6', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].set_title('Training Loss Curve')

# Decision boundary visualization
h = 0.02
x_min, x_max = X_train_sc[:, 0].min() - 1, X_train_sc[:, 0].max() + 1
y_min, y_max = X_train_sc[:, 1].min() - 1, X_train_sc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = sigmoid(grid @ w + b).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['#ef4444', '#4ade80'])
axes[1].contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
for label, color, name in [(0, '#ef4444', 'Fail'), (1, '#4ade80', 'Pass')]:
    mask = y_test == label
    axes[1].scatter(X_test_sc[mask, 0], X_test_sc[mask, 1],
                    color=color, s=25, alpha=0.7, label=name, edgecolors='white', linewidth=0.5)
axes[1].set_xlabel('Study Hours (scaled)')
axes[1].set_ylabel('Attendance (scaled)')
axes[1].set_title('Decision Boundary (from scratch)')
axes[1].legend()

plt.tight_layout()
plt.savefig('logistic_from_scratch.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

### 6. Step 3 — Compare with Scikit-learn

#### 6.1 Fit Scikit-learn Model

```python
# Scikit-learn Logistic Regression
sk_model = LogisticRegression()
sk_model.fit(X_train_sc, y_train)

print(f"Scikit-learn weights = {sk_model.coef_[0]}")
print(f"Scikit-learn bias    = {sk_model.intercept_[0]:.4f}")
print(f"\nOur model    weights = {w}")
print(f"Our model    bias    = {b:.4f}")
```

#### 6.2 Evaluate Both Models

```python
# Our model predictions
y_pred_ours = (sigmoid(X_test_sc @ w + b) >= 0.5).astype(int)

# Scikit-learn predictions
y_pred_sk = sk_model.predict(X_test_sc)

# Compare metrics
print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(f"\n{'Metric':<12} {'Our Model':>15} {'Scikit-learn':>15}")
print("-" * 44)
print(f"{'Accuracy':<12} {accuracy_score(y_test, y_pred_ours):>15.4f} "
      f"{accuracy_score(y_test, y_pred_sk):>15.4f}")
print(f"{'Precision':<12} {precision_score(y_test, y_pred_ours):>15.4f} "
      f"{precision_score(y_test, y_pred_sk):>15.4f}")
print(f"{'Recall':<12} {recall_score(y_test, y_pred_ours):>15.4f} "
      f"{recall_score(y_test, y_pred_sk):>15.4f}")
print(f"{'F1':<12} {f1_score(y_test, y_pred_ours):>15.4f} "
      f"{f1_score(y_test, y_pred_sk):>15.4f}")
```

**Discussion**: Are the models similar? Why might they differ slightly?

---

### 7. Step 4 — Support Vector Machines

Now let's train SVMs with different kernels and compare.

```python
# Train SVMs with different kernels
kernels = ['linear', 'rbf', 'poly']
svm_results = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, kernel in zip(axes, kernels):
    # Train SVM
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_train_sc, y_train)
    
    y_pred_svm = svm.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred_svm)
    f1 = f1_score(y_test, y_pred_svm)
    svm_results[kernel] = {'accuracy': acc, 'f1': f1}
    
    # Plot decision boundary
    h = 0.02
    x_min, x_max = X_train_sc[:, 0].min() - 1, X_train_sc[:, 0].max() + 1
    y_min, y_max = X_train_sc[:, 1].min() - 1, X_train_sc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, colors=['#ef4444', '#4ade80'])
    for label, color, name in [(0, '#ef4444', 'Fail'), (1, '#4ade80', 'Pass')]:
        mask = y_test == label
        ax.scatter(X_test_sc[mask, 0], X_test_sc[mask, 1],
                   color=color, s=25, alpha=0.7, label=name, edgecolors='white', linewidth=0.5)
    ax.set_title(f'SVM ({kernel})\nAcc={acc:.3f} | F1={f1:.3f}')
    ax.set_xlabel('Study Hours (scaled)')
    ax.legend()

axes[0].set_ylabel('Attendance (scaled)')
plt.tight_layout()
plt.savefig('svm_kernels.png', dpi=100, bbox_inches='tight')
plt.show()

# Summary table
print("\n--- SVM Kernel Comparison ---")
print(f"{'Kernel':<10} {'Accuracy':>10} {'F1 Score':>10}")
print("-" * 32)
for kernel, scores in svm_results.items():
    print(f"{kernel:<10} {scores['accuracy']:>10.4f} {scores['f1']:>10.4f}")
```

---

### 8. Step 5 — Full Model Comparison (All Features)

Now let's use **all features** and compare all models side-by-side.

```python
# Prepare all features
X = df.drop('passed', axis=1)
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ]),
    'SVM (Linear)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='linear'))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ]),
}

# Train and evaluate all models
print("=" * 60)
print("FULL MODEL COMPARISON (ALL FEATURES)")
print("=" * 60)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 67)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"{name:<25} {accuracy_score(y_test, y_pred):>10.4f} "
          f"{precision_score(y_test, y_pred):>10.4f} "
          f"{recall_score(y_test, y_pred):>10.4f} "
          f"{f1_score(y_test, y_pred):>10.4f}")
```

---

### 9. Step 6 — Confusion Matrix & Detailed Analysis

```python
# Use the best model (Logistic Regression with all features)
best_pipeline = models['Logistic Regression']
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix heatmap
im = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Fail', 'Pass'])
axes[0].set_yticklabels(['Fail', 'Pass'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=20, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[0])

# Metrics bar chart
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
}
colors = ['#3b82f6', '#8b5cf6', '#ef4444', '#f59e0b']
bars = axes[1].bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Classification Metrics')
for bar, val in zip(bars, metrics.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# Full classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
```

---

### 10. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>Extend Your Analysis</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Change the decision threshold from 0.5 to 0.3 and 0.7. How does this affect Precision vs. Recall?</li>
        <li><strong>Experiment B:</strong> Try different values of C in SVM (0.01, 1, 100). How does regularization strength affect the decision boundary?</li>
        <li><strong>Experiment C:</strong> Make the dataset imbalanced (e.g., 90% pass, 10% fail) and observe how accuracy becomes misleading.</li>
        <li><strong>Experiment D:</strong> Find a real-world binary classification dataset on Kaggle (e.g., Titanic survival, heart disease) and apply the same pipeline.</li>
    </ul>
</div>

---

### 11. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook that:
    1. Loads a dataset (provided or from Kaggle — try the Titanic dataset!).
    2. Explores the data with visualizations and class distribution analysis.
    3. Implements Logistic Regression from scratch using the sigmoid function and gradient descent.
    4. Trains a Scikit-learn `LogisticRegression` and SVM, and compares results.
    5. Evaluates using Accuracy, Precision, Recall, F1, and the Confusion Matrix.
    6. Visualizes decision boundaries for at least two models.
*   **Report**: Write a brief paragraph (3-5 sentences) explaining when you would prioritize Precision over Recall (and vice versa), using a real-world example.
*   **Bonus**: Implement ROC curve plotting and compute AUC (Area Under the Curve) for your models. Which model has the highest AUC?

**See you at the Lecture!**
