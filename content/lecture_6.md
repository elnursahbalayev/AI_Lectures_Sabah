# Week 6 Lecture: Ensemble Methods
## "The Wisdom of Crowds — Why Many Weak Models Beat One Strong Model"

*Module 2: Applied Machine Learning | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **A single model is fragile — it overfits, underfits, or gets stuck in one perspective. Ensemble methods combine many models to cancel out each other's mistakes, producing predictions that are more accurate and more robust than any individual model.**

In Weeks 4-5, we trained single models: Linear Regression, Logistic Regression, SVM. Today, we ask: *What if we trained hundreds of models and let them vote?*

**We will cover:**
1. The Wisdom of Crowds — Why Ensembles Work
2. Decision Trees — The Building Block
3. Bagging — Reducing Variance with Bootstrap
4. Random Forests — Bagging + Feature Randomness
5. Boosting — Learning from Mistakes
6. Gradient Boosting Machines (GBM)
7. XGBoost & LightGBM — Industry-Grade Boosting
8. The Bias-Variance Tradeoff — The Central Tension of ML
9. Feature Importance — What the Model Learned
10. Practical Implementation with Scikit-learn & XGBoost

---

### 1. The Wisdom of Crowds

#### 1.1 The Idea

In 1906, Francis Galton observed that the **average** of 787 people's guesses about the weight of an ox was more accurate than any individual guess — including experts. This is the *Wisdom of Crowds*.

The same principle applies to machine learning:

| Approach | Strategy | Result |
|---|---|---|
| **Single Model** | One model makes all predictions | Prone to overfitting or systematic errors |
| **Ensemble** | Many models vote / average their predictions | Errors cancel out → better performance |

#### 1.2 Why Ensembles Work — Error Cancellation

If we have $N$ independent models, each with error rate $\epsilon < 0.5$, the probability that the **majority** is wrong drops exponentially with $N$:

$$P(\text{majority wrong}) = \sum_{k > N/2}^{N} \binom{N}{k} \epsilon^k (1-\epsilon)^{N-k}$$

For $N = 21$ models each with $\epsilon = 0.3$ (70% accuracy individually), the ensemble achieves **~97% accuracy** through majority voting.

> **Key Insight**: Ensembles work best when the individual models are **diverse** — they make different mistakes. If all models make the same errors, voting doesn't help.

#### 1.3 Types of Ensembles

| Type | Strategy | Key Method |
|---|---|---|
| **Bagging** | Train models in parallel on different data subsets | Random Forests |
| **Boosting** | Train models sequentially, each fixing the previous one's errors | XGBoost, LightGBM |
| **Stacking** | Train a meta-model on the outputs of several base models | Advanced technique |

---

### 2. Decision Trees — The Building Block

#### 2.1 What Is a Decision Tree?

A decision tree makes predictions by asking a sequence of yes/no questions about the features, splitting the data at each step until it reaches a prediction.

```
                 Is study_hours > 5?
                /                    \
              Yes                     No
              /                        \
     Is attendance > 70?          Predict: FAIL
        /            \
      Yes             No
      /                \
  Predict: PASS    Predict: FAIL
```

Each internal node is a **question** about a feature. Each leaf is a **prediction**.

#### 2.2 How Splits Are Chosen

At each node, the tree picks the feature and threshold that best separates the classes. The quality of a split is measured by **impurity reduction**:

**Gini Impurity** (most common):

$$\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is the proportion of class $k$ in the set $S$.

| Scenario | Gini Value |
|---|---|
| Perfect split (all one class) | $0$ — pure |
| Worst case (50/50 binary) | $0.5$ — maximum impurity |

**Entropy** (alternative):

$$\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

#### 2.3 The Problem with Decision Trees

| Strength | Weakness |
|---|---|
| Easy to interpret ("if X > 5 and Y < 3...") | **Overfits** easily — deep trees memorize noise |
| Handles non-linear boundaries | **High variance** — small data changes → completely different tree |
| No feature scaling needed | Often outperformed by other algorithms |

> **This is exactly what ensembles fix.** A single tree is fragile. A *forest* of trees is robust.

---

### 3. Bagging — Bootstrap Aggregating

#### 3.1 The Bootstrap

**Bootstrap sampling**: Draw $n$ samples from the training set **with replacement**. Some samples appear multiple times, others are left out (~37% are excluded on average).

Each bootstrap sample creates a slightly different dataset → a slightly different model → **diversity**.

#### 3.2 Bagging Algorithm

1. Create $B$ bootstrap samples from the training data.
2. Train one model (typically a decision tree) on each bootstrap sample.
3. For classification: **majority vote**. For regression: **average predictions**.

$$\hat{y}_{\text{ensemble}} = \text{mode}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B)$$

#### 3.3 Why Bagging Reduces Variance

If each model has variance $\sigma^2$ and they are independent, the variance of the average is:

$$\text{Var}(\bar{y}) = \frac{\sigma^2}{B}$$

More trees → lower variance → less overfitting. **Bagging cannot increase bias** — it only reduces variance.

---

### 4. Random Forests

#### 4.1 The Key Innovation

Random Forest = Bagging + **Random Feature Selection**.

At each split, instead of considering **all** features, a Random Forest only considers a random subset of $m$ features:

$$m \approx \sqrt{p} \quad (\text{classification}) \qquad m \approx p/3 \quad (\text{regression})$$

Where $p$ is the total number of features.

#### 4.2 Why Random Feature Selection?

Without it, if one feature is very strong (e.g., "income" for predicting credit risk), **every tree** will split on that feature first → all trees look similar → low diversity → weak ensemble.

Random feature selection **decorrelates** the trees, forcing them to find different patterns.

#### 4.3 The Random Forest Algorithm

```
For each tree t = 1 to B:
    1. Draw a bootstrap sample Dₜ from training data.
    2. Grow a decision tree on Dₜ:
       - At each node, randomly select m features.
       - Find the best split among those m features.
       - Split the node.
       - Repeat until stopping criteria (max depth, min samples).
    3. Do NOT prune the tree — let it overfit!

Prediction: Majority vote of all B trees.
```

> **Why let trees overfit?** Each individual tree has low bias but high variance. Averaging many high-variance models reduces variance while keeping bias low.

#### 4.4 Hyperparameters

| Parameter | What It Controls | Typical Values |
|---|---|---|
| `n_estimators` | Number of trees | 100-1000 |
| `max_depth` | Maximum tree depth | None (full), 10-30 |
| `max_features` | Features per split ($m$) | `'sqrt'`, `'log2'` |
| `min_samples_split` | Minimum samples to split a node | 2-10 |
| `min_samples_leaf` | Minimum samples in a leaf | 1-5 |

---

### 5. Boosting — Learning from Mistakes

#### 5.1 The Core Idea

While bagging trains models **in parallel** and averages them, boosting trains models **sequentially**, where each new model focuses on the mistakes of the previous ones.

| Bagging | Boosting |
|---|---|
| Models trained independently | Models trained sequentially |
| Each model sees a random subset of data | Each model focuses on hard examples |
| Reduces **variance** | Reduces **bias** (and sometimes variance) |
| Robust to overfitting | Can overfit if not regularized |

#### 5.2 AdaBoost — The Original Boosting Algorithm

1. Initialize all sample weights equally: $w_i = 1/n$.
2. For each round $t = 1, \ldots, T$:
   - Train a **weak learner** (e.g., a decision stump — tree with depth 1) on weighted data.
   - Compute the weighted error: $\epsilon_t = \sum_{i: \hat{y}_i \neq y_i} w_i$
   - Compute model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$
   - **Increase weights** on misclassified samples, **decrease weights** on correct ones.
3. Final prediction: weighted vote of all weak learners.

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

> **Key Insight**: AdaBoost turns many "slightly better than random" classifiers into one strong classifier. A decision stump alone might be 55% accurate — but 500 boosted stumps can reach 99%.

---

### 6. Gradient Boosting Machines (GBM)

#### 6.1 The Gradient Boosting Framework

Gradient Boosting generalizes boosting by framing it as **gradient descent in function space**.

Instead of adjusting sample weights (like AdaBoost), Gradient Boosting trains each new tree to predict the **residuals** (errors) of the current ensemble:

$$F_0(x) = \text{initial prediction (e.g., mean)}$$
$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

Where $h_t(x)$ is a tree trained on the **negative gradient** (residuals) of the loss function, and $\eta$ is the learning rate.

#### 6.2 Step-by-Step for Regression

1. Start with a constant prediction: $F_0(x) = \bar{y}$ (the mean).
2. For round $t = 1, \ldots, T$:
   - Compute residuals: $r_i = y_i - F_{t-1}(x_i)$
   - Fit a tree $h_t$ to the residuals.
   - Update the model: $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$
3. Final model: $F(x) = F_0(x) + \eta \sum_{t=1}^{T} h_t(x)$

#### 6.3 The Learning Rate $\eta$

| $\eta$ Value | Effect |
|---|---|
| **Large** (e.g., 0.3) | Learns fast but may overshoot → overfitting |
| **Small** (e.g., 0.01) | Learns slowly but more precisely → needs more trees |

> **Rule of Thumb**: Use a small learning rate ($\eta = 0.01$ to $0.1$) with many trees. This is called **shrinkage** and is the most important regularizer in gradient boosting.

---

### 7. XGBoost & LightGBM

#### 7.1 Why XGBoost Dominates Competitions

XGBoost (eXtreme Gradient Boosting) adds several improvements over basic Gradient Boosting:

| Feature | What It Does |
|---|---|
| **Regularization** | L1 and L2 penalties on tree weights prevent overfitting |
| **Second-order gradients** | Uses Hessian (second derivative) for better split finding |
| **Column subsampling** | Like Random Forest — samples features at each tree/split |
| **Handling missing values** | Learns the optimal direction for missing values |
| **Parallel computation** | Parallelizes split finding within each tree |
| **Built-in cross-validation** | Native CV support for early stopping |

#### 7.2 XGBoost Key Hyperparameters

| Parameter | What It Controls | Typical Range |
|---|---|---|
| `n_estimators` | Number of boosting rounds | 100-3000 |
| `learning_rate` ($\eta$) | Step size for updates | 0.01-0.3 |
| `max_depth` | Maximum tree depth | 3-10 |
| `subsample` | Fraction of data per tree | 0.6-1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.6-1.0 |
| `reg_alpha` (L1) | L1 regularization | 0-10 |
| `reg_lambda` (L2) | L2 regularization | 0-10 |
| `min_child_weight` | Minimum sum of instance weight in a leaf | 1-10 |

#### 7.3 LightGBM — Even Faster

LightGBM (by Microsoft) uses clever tricks for massive speedup:

| Technique | Effect |
|---|---|
| **Leaf-wise growth** | Grows the leaf with largest loss reduction (vs. XGBoost's level-wise) |
| **Histogram-based splitting** | Bins continuous features into histograms → fewer split candidates |
| **GOSS** (Gradient-based One-Side Sampling) | Keeps all large-gradient instances, samples from small-gradient ones |

> **When to use which**: XGBoost is the safe default. LightGBM is faster on large datasets (100k+ rows). Both typically outperform Random Forests on tabular data.

---

### 8. The Bias-Variance Tradeoff

#### 8.1 The Central Tension of Machine Learning

Every model's error can be decomposed as:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Component | Meaning | Analogy |
|---|---|---|
| **Bias** | How far off the average prediction is from truth | Consistently aiming left of the target |
| **Variance** | How much predictions change with different training data | Shots scattered all over |
| **Noise** | Inherent randomness in the data | Wind blowing the arrows |

#### 8.2 How Ensembles Address the Tradeoff

| Method | Trees | Effect on Bias | Effect on Variance |
|---|---|---|---|
| **Single Deep Tree** | 1 | Low bias | **High variance** (overfits) |
| **Random Forest** (Bagging) | Many | Low bias (unchanged) | **Low variance** (averaging) |
| **Gradient Boosting** | Many (sequential) | **Low bias** (corrects errors) | Can increase if not regularized |

#### 8.3 Overfitting vs. Underfitting

| Symptom | Diagnosis | Solution |
|---|---|---|
| Training accuracy ≈ 99%, Test accuracy ≈ 75% | **Overfitting** (high variance) | Reduce model complexity, add regularization, get more data |
| Training accuracy ≈ 70%, Test accuracy ≈ 68% | **Underfitting** (high bias) | Increase model complexity, add features, reduce regularization |
| Training accuracy ≈ 92%, Test accuracy ≈ 90% | **Good fit** | Ship it! |

---

### 9. Feature Importance

#### 9.1 How Trees Measure Importance

Every tree-based model can tell you **which features matter most**:

**Gini Importance** (Mean Decrease in Impurity):
For each feature, sum the total reduction in impurity across all splits in all trees that used that feature.

$$\text{Importance}(f) = \sum_{\text{trees}} \sum_{\text{splits on } f} \Delta \text{Gini}$$

#### 9.2 Permutation Importance

A more robust approach: randomly shuffle one feature and measure how much the model's accuracy drops.

$$\text{Importance}(f) = \text{Score}_{\text{original}} - \text{Score}_{\text{shuffled } f}$$

If shuffling a feature doesn't hurt performance → the feature is unimportant.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]:<25} {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")
```

---

### 10. Practical Implementation

#### 10.1 Random Forest with Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create and train a Random Forest
rf = RandomForestClassifier(
    n_estimators=200,       # 200 trees
    max_depth=None,         # unlimited depth (each tree overfits)
    max_features='sqrt',    # sqrt(p) features per split
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

#### 10.2 XGBoost

```python
from xgboost import XGBClassifier

# Create and train XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
```

#### 10.3 Comparing All Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())]),
    'SVM (RBF)': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf'))]),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric='logloss', random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<25} Accuracy: {acc:.4f}")
```

---

### 11. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Gini Impurity | $1 - \sum p_k^2$ |
| Entropy | $-\sum p_k \log_2 p_k$ |
| Bagging Prediction | $\hat{y} = \text{mode}(\hat{y}_1, \ldots, \hat{y}_B)$ |
| Random Forest Features per Split | $m \approx \sqrt{p}$ |
| Gradient Boosting Update | $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$ |
| AdaBoost Model Weight | $\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$ |
| Bias-Variance Decomposition | $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$ |

---

### 12. Summary

1. **Decision Trees** are intuitive but overfit easily — they have low bias but high variance.
2. **Bagging** (Bootstrap Aggregating) trains many models on bootstrap samples and averages them — this reduces variance without increasing bias.
3. **Random Forests** add random feature selection to bagging, decorrelating the trees for even better performance.
4. **Boosting** trains models sequentially, each correcting the errors of the previous — this reduces bias.
5. **Gradient Boosting** frames boosting as gradient descent in function space, fitting trees to residuals.
6. **XGBoost** and **LightGBM** are optimized gradient boosting implementations that dominate tabular data competitions.
7. The **Bias-Variance Tradeoff** is the fundamental tension: complex models overfit (high variance), simple models underfit (high bias). Ensembles navigate this tradeoff.
8. **Feature Importance** reveals what the model learned — critical for interpretability and debugging.

---

### 13. Bridge to the Tutorial

In the tutorial session, you will:
- Build a fraud detection system using an imbalanced dataset.
- Train a Decision Tree from scratch and visualize it.
- Implement a Random Forest and compare it to a single tree.
- Train XGBoost and tune its hyperparameters.
- Handle class imbalance using `scale_pos_weight` and SMOTE.
- Compare all models: Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost.
- Extract and visualize feature importance.
- Use cross-validation with early stopping.

**Come to the tutorial. A single model is a gamble — an ensemble is a strategy.**
