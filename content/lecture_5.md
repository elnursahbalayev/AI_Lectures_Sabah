# Week 5 Lecture: Classification Algorithms
## "Drawing the Boundary — This Time for Real"

*Module 2: Applied Machine Learning | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **Last week we drew lines to predict numbers (Regression). This week we draw lines to separate categories (Classification). The math is surprisingly similar — but the interpretation is fundamentally different.**

In Week 4, we predicted continuous values ($342,500, $185,000...). Today, we predict **discrete labels** — Spam or Not Spam, Pass or Fail, Cat or Dog.

**We will cover:**
1. From Regression to Classification — What Changes?
2. Logistic Regression — The Sigmoid Function
3. The Decision Boundary — Where the Line Is Drawn
4. The Loss Function — Binary Cross-Entropy
5. Gradient Descent for Classification
6. Support Vector Machines (SVM) — Maximum Margin
7. Multi-Class Classification — Beyond Binary
8. Evaluation Metrics — Accuracy, Precision, Recall, F1
9. The Confusion Matrix — Reading Your Model's Report Card
10. Practical Implementation with Scikit-learn

---

### 1. From Regression to Classification

#### 1.1 The Two Types of Prediction (Revisited)

| Type | Output | Example | Week |
|---|---|---|---|
| **Regression** | A continuous number | Predict house price: $342,500 | Week 4 |
| **Classification** | A discrete label | Predict spam or not: "Spam" | **This Week** |

#### 1.2 Why Can't We Just Use Linear Regression?

Imagine predicting "Pass" (1) or "Fail" (0) based on study hours:

**Problem 1: Unbounded Output**
Linear regression gives $\hat{y} = wx + b$, which can output any value: -5, 0.5, 1.7, 100. But we need outputs between 0 and 1 (probabilities).

**Problem 2: Threshold Sensitivity**
If we use a threshold ($\hat{y} > 0.5$ → Pass), outliers in the data can shift the line and completely change all predictions.

**Solution**: We need a function that squashes any input into the range [0, 1]. That function is the **Sigmoid**.

---

### 2. Logistic Regression

#### 2.1 The Sigmoid Function

The sigmoid (or logistic) function maps any real number to a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = wx + b$ (the same linear combination as regression!).

Properties of the Sigmoid:
- $\sigma(0) = 0.5$ — the "undecided" point.
- $\sigma(z) \to 1$ as $z \to +\infty$ — very confident positive.
- $\sigma(z) \to 0$ as $z \to -\infty$ — very confident negative.
- The output is always between 0 and 1 — it's a **probability**.

#### 2.2 The Logistic Regression Model

The model is a two-step process:

**Step 1**: Compute the linear combination (same as regression):
$$z = wx + b$$

**Step 2**: Squash through the sigmoid:
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-(wx + b)}}$$

Now $\hat{y}$ is between 0 and 1, and we interpret it as $P(\text{class} = 1 | x)$ — the **probability** that the input belongs to class 1.

#### 2.3 Making Predictions

We use a threshold (typically 0.5):

$$\text{prediction} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

> **Key Insight**: Logistic Regression doesn't output a class — it outputs a *probability*. The threshold converts probability to a class label. Changing the threshold changes the predictions.

---

### 3. The Decision Boundary

#### 3.1 What is a Decision Boundary?

The decision boundary is the line (or surface) where the model switches from predicting class 0 to class 1. For logistic regression:

$$\hat{y} = 0.5 \quad \Leftrightarrow \quad \sigma(z) = 0.5 \quad \Leftrightarrow \quad z = 0 \quad \Leftrightarrow \quad wx + b = 0$$

So the decision boundary is simply the line $wx + b = 0$ — the same equation as linear regression, but now it acts as a **separator** between classes.

#### 3.2 Linear vs. Non-Linear Boundaries

| Boundary Type | Shape | When to Use |
|---|---|---|
| **Linear** | Straight line / flat plane | Classes are linearly separable (can be separated by a line) |
| **Non-Linear** | Curves, circles, arbitrary shapes | Classes overlap or have complex boundaries |

Logistic Regression draws a **linear** boundary. For non-linear boundaries, we'll need SVMs with kernels or polynomial features.

#### 3.3 Multiple Features

With two features ($x_1$, $x_2$), the decision boundary is a line in 2D:

$$w_1 x_1 + w_2 x_2 + b = 0$$

With three features, it's a plane in 3D. With $p$ features, it's a **hyperplane** in $p$-dimensional space.

---

### 4. The Loss Function — Binary Cross-Entropy

#### 4.1 Why Not MSE?

MSE worked for regression because the relationship between predictions and targets was linear. For classification:
- MSE creates a **non-convex** loss surface with many local minima.
- Gradient Descent can get stuck and fail to converge.

We need a new loss function designed for probabilities.

#### 4.2 Binary Cross-Entropy (Log Loss)

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Where:
- $y_i \in \{0, 1\}$ is the true label.
- $\hat{y}_i \in (0, 1)$ is the predicted probability.

#### 4.3 Why This Works

| When $y = 1$ | Loss = $-\log(\hat{y})$ | High $\hat{y}$ → low loss. Low $\hat{y}$ → **very** high loss. |
|---|---|---|
| When $y = 0$ | Loss = $-\log(1 - \hat{y})$ | Low $\hat{y}$ → low loss. High $\hat{y}$ → **very** high loss. |

The log function makes the penalty **exponentially harsh** for confident wrong predictions. If you say "99% spam" and it's actually ham, the loss is enormous.

#### 4.4 Worked Example

```python
import numpy as np

# True labels and predicted probabilities
y_true = [1, 0, 1, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.6, 0.3]

# Binary Cross-Entropy for each sample
for y, p in zip(y_true, y_pred):
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"y={y}, ŷ={p:.1f} → loss = {loss:.4f}")

# Total BCE loss
bce = -np.mean([y * np.log(p) + (1-y) * np.log(1-p) 
                 for y, p in zip(y_true, y_pred)])
print(f"\nBCE Loss = {bce:.4f}")
```

---

### 5. Gradient Descent for Classification

#### 5.1 The Gradients

The gradients for logistic regression look almost identical to linear regression:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

The only difference from Week 4: $\hat{y}_i = \sigma(w x_i + b)$ instead of $\hat{y}_i = w x_i + b$.

#### 5.2 The Training Loop

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    b = 0.0                    # Initialize bias to zero
    n = len(X)
    
    for epoch in range(epochs):
        # Forward pass: compute probabilities
        z = X @ w + b
        y_pred = sigmoid(z)
        
        # Compute gradients
        dw = (1/n) * X.T @ (y_pred - y)
        db = (1/n) * np.sum(y_pred - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
        # Print progress every 200 epochs
        if epoch % 200 == 0:
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1 - y_pred + 1e-8))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return w, b
```

> **Note the `1e-8`**: We add a tiny number inside `log()` to prevent `log(0)` which gives $-\infty$. This is a standard numerical trick.

---

### 6. Support Vector Machines (SVM)

#### 6.1 The Core Idea

While Logistic Regression finds **any** separating boundary, SVM finds the **best** one — the boundary with the **maximum margin**.

**Margin**: The distance between the decision boundary and the closest data points from each class. These closest points are called **support vectors**.

#### 6.2 Why Maximum Margin?

A boundary with a larger margin is more robust:

| Small Margin | Large Margin |
|---|---|
| The boundary is close to data points from both classes | The boundary has maximum breathing room |
| Small perturbations can flip predictions | More tolerant to noise |
| Likely to overfit | Better generalization |

#### 6.3 The SVM Optimization Problem

SVM solves:

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i(w \cdot x_i + b) \geq 1 \quad \forall i$$

This says: Find the weights that have the smallest norm (widest margin) while correctly classifying all training points.

#### 6.4 The Kernel Trick

What if data isn't linearly separable? SVM uses **kernels** to map data to a higher-dimensional space where it becomes separable:

| Kernel | Effect | Use Case |
|---|---|---|
| **Linear** | No transformation | Data is already linearly separable |
| **RBF (Gaussian)** | Maps to infinite-dimensional space | Most problems — default choice |
| **Polynomial** | Adds polynomial feature combinations | Moderate non-linearity |

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear')

# RBF SVM (default — most powerful)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
```

#### 6.5 Hyperparameter C — Regularization

| C Value | Effect |
|---|---|
| **Small C** (e.g., 0.01) | Large margin, more misclassifications allowed → **generalizes better** |
| **Large C** (e.g., 100) | Small margin, fewer misclassifications → **may overfit** |

> **C is the "strictness" knob**: Low C says "it's okay to misclassify some training points for a wider margin." High C says "classify everything correctly, even if the margin is razor-thin."

---

### 7. Multi-Class Classification

#### 7.1 Beyond Two Classes

Both Logistic Regression and SVM are inherently **binary** classifiers. For multiple classes (e.g., Cat / Dog / Bird), we use strategies:

| Strategy | How It Works | # Classifiers |
|---|---|---|
| **One-vs-Rest (OvR)** | Train one classifier per class: "Is it class $k$ or not?" | $K$ classifiers |
| **One-vs-One (OvO)** | Train one classifier per pair of classes: "Is it class $i$ or class $j$?" | $K(K-1)/2$ classifiers |

Scikit-learn handles this automatically.

#### 7.2 Softmax — Multi-Class Logistic Regression

For multi-class problems, the sigmoid is replaced by the **Softmax** function:

$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Softmax outputs a probability distribution over all $K$ classes (they sum to 1).

---

### 8. Evaluation Metrics for Classification

#### 8.1 Accuracy — Necessary But Not Sufficient

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**The problem**: If 95% of emails are "Not Spam," a model that always predicts "Not Spam" gets 95% accuracy — while catching zero spam. This is the **accuracy paradox**.

#### 8.2 Precision, Recall, and F1-Score

| Metric | Formula | Question It Answers |
|---|---|---|
| **Precision** | $\frac{TP}{TP + FP}$ | Of all items predicted positive, how many are actually positive? |
| **Recall** | $\frac{TP}{TP + FN}$ | Of all actually positive items, how many did we catch? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall |

Where:
- **TP** = True Positive (correctly predicted positive)
- **FP** = False Positive (incorrectly predicted positive — "false alarm")
- **FN** = False Negative (missed a positive — "missed detection")
- **TN** = True Negative (correctly predicted negative)

#### 8.3 Precision vs. Recall Tradeoff

| Scenario | Prioritize | Why |
|---|---|---|
| **Spam filter** | **Precision** | Don't want to mark a real email as spam |
| **Cancer detection** | **Recall** | Don't want to miss a patient who has cancer |
| **Fraud detection** | **F1** | Balance between catching fraud and avoiding false alarms |

> **You can't maximize both**: Increasing precision usually hurts recall, and vice versa. The threshold controls this tradeoff.

---

### 9. The Confusion Matrix

#### 9.1 Reading the Matrix

The confusion matrix shows all four outcomes:

|  | **Predicted: Positive** | **Predicted: Negative** |
|---|---|---|
| **Actual: Positive** | True Positive (TP) | False Negative (FN) |
| **Actual: Negative** | False Positive (FP) | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[3, 1],    ← Row 0: Actual Negatives (3 correct, 1 false alarm)
#  [1, 5]]    ← Row 1: Actual Positives (1 missed, 5 caught)

print(classification_report(y_true, y_pred))
```

---

### 10. Practical Implementation with Scikit-learn

#### 10.1 Logistic Regression

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data: students with study hours & attendance → Pass/Fail
np.random.seed(42)
n = 300
hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(50, 100, n)

# True rule: pass if (5*hours + 0.5*attendance + noise) > 50
score = 5 * hours + 0.5 * attendance + np.random.normal(0, 5, n)
labels = (score > 50).astype(int)

X = np.column_stack([hours, attendance])
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fit model
model = LogisticRegression()
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
```

#### 10.2 SVM

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Pipeline: Scale → SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm, target_names=['Fail', 'Pass']))
```

---

### 11. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Sigmoid Function | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| Logistic Regression Model | $\hat{y} = \sigma(wx + b)$ |
| Binary Cross-Entropy | $-\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Gradient (w.r.t. $w$) | $\frac{1}{n}\sum(\hat{y}_i - y_i)x_i$ |
| Gradient (w.r.t. $b$) | $\frac{1}{n}\sum(\hat{y}_i - y_i)$ |
| SVM Objective | $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w \cdot x_i + b) \geq 1$ |
| Precision | $\frac{TP}{TP + FP}$ |
| Recall | $\frac{TP}{TP + FN}$ |
| F1-Score | $2 \cdot \frac{P \cdot R}{P + R}$ |
| Softmax | $P(y=k|x) = \frac{e^{z_k}}{\sum_j e^{z_j}}$ |

---

### 12. Summary

1. **Logistic Regression** is linear regression + sigmoid. It outputs probabilities, not numbers.
2. **The Sigmoid** function squashes any value into [0, 1] — turning a linear model into a classifier.
3. **The Decision Boundary** is where $\sigma(z) = 0.5$, i.e., $wx + b = 0$. It's a line (or hyperplane) that separates classes.
4. **Binary Cross-Entropy** replaces MSE as the loss function — it harshly penalizes confident wrong predictions.
5. **SVM** finds the maximum-margin boundary — the most robust separator. Kernels handle non-linear data.
6. **Accuracy alone is dangerous** — use Precision, Recall, and F1-Score, especially on imbalanced datasets.
7. **The Confusion Matrix** tells the full story: TP, FP, FN, TN.

---

### 13. Bridge to the Tutorial

In the tutorial session, you will:
- Generate a binary classification dataset (Pass/Fail based on study hours and attendance).
- Implement Logistic Regression from scratch (sigmoid + gradient descent).
- Compare your implementation against Scikit-learn's `LogisticRegression`.
- Train and compare SVM against Logistic Regression.
- Visualize decision boundaries for both models.
- Evaluate using the Confusion Matrix, Precision, Recall, and F1-Score.
- Experiment with different SVM kernels (linear, RBF, polynomial).

**Come to the tutorial. A classifier that never sees data never learns to decide.**
