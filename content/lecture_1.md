# Week 1 Lecture: Optimization & Search
## "The Language of Learning"

*Module 1: Algorithmic Foundations | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's session has one central thesis:

> **Machine Learning is, at its core, an optimization problem.**

Everything else — the architectures, the datasets, the metrics — is in service of that one idea. If you understand this lecture deeply, you understand the skeleton of all of AI.

**We will cover:**
1. What does it mean for a machine to "learn"?
2. How we formalize learning as mathematics.
3. The Loss Function — what we are actually minimizing.
4. Naive search strategies and why they fail.
5. Hill Climbing — a step in the right direction.
6. **Gradient Descent** — the algorithm that powers all of modern AI.
7. The Learning Rate — the critical hyperparameter.
8. Convex vs. Non-Convex landscapes.

---

### 1. What Does It Mean to "Learn"?

The word **learning** is overloaded. Let's be precise.

**Classical Programming:**
```
Input Data + Hand-written Rules → Output
```
*Example: if temperature > 37.5 degrees: classify as "Fever"*

The programmer encodes human knowledge directly as `if/else` logic. The rules are explicit and brittle.

**Machine Learning:**
```
Input Data + Output Labels → Algorithm → Rules (Model)
```
*Example: from 100,000 patient records with outcomes, derive the rules automatically.*

The programmer writes an algorithm that **finds** the rules from data. The rules are implicit in the model's parameters.

**The Fundamental Shift:** We stopped encoding knowledge and started encoding the *process of acquiring knowledge*.

But what is that process? It is **search** — specifically, search over the space of possible rule sets (models) to find the one that best explains the observed data.

---

### 2. Formalizing Learning as Optimization

Let's be mathematical. Suppose we have:

- A dataset $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$
- A model $h_\theta(x)$ parameterized by $\theta$ (a vector of weights)
- A **Loss Function** $\mathcal{L}(\theta)$ that measures how wrong our model is

Our **goal** is:
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

*"Find the parameter vector θ that minimizes the loss."*

This is precisely an **optimization problem**. The "learning" is the iterative process of finding $\theta^*$. Once we found it, the model "knows" how to make predictions.

**Key Insight:** The particular algorithm we use to find $\theta^*$ is what distinguishes different ML training approaches. Gradient Descent is the most important one.

---

### 3. The Loss Function (Cost Function)

The Loss Function is the mathematical **definition of failure**. It quantifies how far our model's predictions are from the true values.

**For Regression (continuous output):**

Mean Squared Error (MSE):
$$\mathcal{L}_{MSE}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)^2$$

*"The average squared difference between our predictions and the truth."*

**For Classification (discrete output):**

Cross-Entropy Loss:
$$\mathcal{L}_{CE}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Critical Properties of a Good Loss Function:**
1. **Differentiable** — We need to compute gradients. Non-smooth losses break Gradient Descent.
2. **Sensitive to errors** — A small wrong prediction should produce a non-zero loss.
3. **Minimizable** — The minimum should correspond to the best possible model.

---

### 4. The Loss Landscape

Imagine plotting $\mathcal{L}(\theta)$ as a geometric surface over all possible values of $\theta$.

| Model Complexity | Parameter Count | Landscape Visualization |
|---|---|---|
| 1 parameter | 1D curve (line) | A 2D curve |
| 2 parameters | 2D surface | Mountains and valleys in 3D |
| Simple Neural Net | ~10,000 | A 10,000-dimensional space |
| GPT-4 | ~1.76 trillion | Unvisualizable |

**The goal is always the same:** navigate this landscape to find the lowest valley.

For today, we work in 1D. Our "Loss Landscape" is the simple quadratic:
$$f(x) = x^2 - 4x + 6$$

This function has a single, reachable minimum at $x = 2$, $f(2) = 2$. Our job is to find it programmatically.

---

### 5. Naive Search: Why Brute Force Fails

Before the elegant solution, let's appreciate why simple approaches don't work.

**Strategy A: Grid Search / Brute Force**
- Divide the parameter space into a grid.
- Evaluate $\mathcal{L}(\theta)$ at every grid point.
- Return the minimum.

**Problem:** The **Curse of Dimensionality**.
If we evaluate 100 points per dimension:
- 1 parameter: 100 evaluations ✓
- 10 parameters: $100^{10} = 10^{20}$ evaluations ✗
- 1,000 parameters: computationally impossible

Grid search is dead on arrival for real ML.

**Strategy B: Random Search**
- Randomly sample $\theta$ values from the parameter space.
- Keep the best one.
- Better than brute force, but still inefficient — no memory of where it has been.

---

### 6. Hill Climbing

Hill Climbing improves on random search by using *local information*. The algorithm:

```
Initialize: x₀ = random starting point
Repeat:
    1. Generate a neighbor: x' = x_current + random small perturbation
    2. If f(x') < f(x_current):  # Found something better
           x_current = x'        # Accept the move
       Else:
           Stay at x_current     # Reject the move
Until: convergence or max iterations
```

**Intuition:** Like a blindfolded hiker on a hilly landscape. They feel the ground around their feet in all directions, and take a step toward whichever direction goes down.

**Advantages:**
- Simple to implement.
- Works on non-differentiable functions.
- Requires no gradient computation.

**Disadvantages:**
- **Gets stuck at local minima.** Once at a local minimum, all neighbors are uphill, so the algorithm stops — even if the global minimum is far away.
- **No sense of direction.** The perturbation is random. It might explore uphill directions 50% of the time, wasting evaluations.
- **Slow.** Many iterations produce no improvement.

For smooth, differentiable functions (which all neural network losses are), we can do much better.

---

### 7. Gradient Descent

The key insight missing from Hill Climbing: **if the function is differentiable, we can compute the exact direction of steepest descent**.

**The Gradient** $\nabla f(x)$ (or $f'(x)$ in 1D) is a vector pointing in the direction of **steepest ascent**.

- If $f'(x) > 0$: the function slopes **up** to the right. To go downhill, step **left** (negative direction).
- If $f'(x) < 0$: the function slopes **down** to the right. To go downhill, step **right** (positive direction).
- If $f'(x) = 0$: we are at a flat point — possibly the minimum.

This leads to the **Golden Rule of Gradient Descent:**

> *Always step in the direction **opposite** to the gradient.*

#### 7.1 The Update Rule

$$x_{t+1} = x_t - \eta \cdot f'(x_t)$$

Or in vector (multi-dimensional) form:
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- $x_t$ (or $\theta_t$): current position in parameter space
- $\eta$ (eta): the **Learning Rate** — how far we step
- $f'(x_t)$ (or $\nabla \mathcal{L}$): the gradient at the current position

#### 7.2 Why Does This Work? (Intuition from Taylor's Theorem)

By the first-order Taylor expansion:
$$f(x + \delta) \approx f(x) + \delta \cdot f'(x)$$

If we set $\delta = -\eta \cdot f'(x)$ (stepping opposite to gradient):
$$f(x - \eta f'(x)) \approx f(x) - \eta \cdot [f'(x)]^2$$

Since $\eta > 0$ and $[f'(x)]^2 \geq 0$, the new value $f(x_{t+1})$ is always **less than or equal to** $f(x_t)$ for small enough $\eta$.

**This is the mathematical guarantee: gradient descent always decreases the loss.**

#### 7.3 The Algorithm in Code

For our example function $f(x) = x^2 - 4x + 6$:
- Derivative: $f'(x) = 2x - 4$

```python
def f(x):
    return x**2 - 4*x + 6

def df(x):
    return 2*x - 4    # The gradient (derivative)

def gradient_descent(start_x, learning_rate, epochs):
    x = start_x
    for epoch in range(epochs):
        gradient = df(x)              # 1. Compute gradient at current x
        x = x - learning_rate * gradient  # 2. Step opposite to gradient
        print(f"Epoch {epoch+1}: x={x:.4f}, f(x)={f(x):.4f}")
    return x

# Example run
final_x = gradient_descent(start_x=0.0, learning_rate=0.1, epochs=20)
# Expected: converges toward x ≈ 2.0
```

---

### 8. The Learning Rate $\eta$ — The Critical Hyperparameter

The Learning Rate is arguably the single most important decision in training any ML model. It controls the **step size** in parameter space.

**Case 1: Too Small ($\eta \approx 0.001$)**
$$x_{t+1} = x_t - 0.001 \cdot f'(x_t)$$
- Steps are tiny. Convergence takes thousands of epochs.
- Risk: terminates early (by time/budget) before reaching the minimum.
- Benefit: stable, never overshoots.

**Case 2: Just Right ($\eta \approx 0.1$ for our example)**
$$x_{t+1} = x_t - 0.1 \cdot f'(x_t)$$
- Efficient convergence in ~20-30 epochs.
- This is the "Goldilocks" region.

**Case 3: Too Large ($\eta \approx 1.0$)**
$$x_{t+1} = x_t - 1.0 \cdot f'(x_t)$$
- Update overshoots the minimum.
- The ball bounces back and forth across the valley.
- At $\eta > 1.0$: the steps grow in size each iteration → **divergence** (loss explodes to infinity).

**Mathematical Condition for Convergence:**
For $f(x) = x^2$, GD converges if and only if $\eta < 1.0$ (specific to this function's curvature).

#### 8.1 Adaptive Learning Rates (Preview for Module 3)

In practice, we don't manually tune $\eta$ forever. Adaptive optimizers adjust it automatically:

- **Momentum**: Accumulates past gradients to build speed in consistent directions (like a ball with inertia).
- **Adam (Adaptive Moment Estimation)**: Different learning rates for each parameter, based on gradient history. The industry standard for deep learning.

---

### 9. Convex vs. Non-Convex Functions

This is the most theoretically important distinction in all of optimization.

#### 9.1 Convex Functions

**Definition:** A function $f$ is convex if for any two points $a$ and $b$:
$$f(\lambda a + (1-\lambda) b) \leq \lambda f(a) + (1-\lambda) f(b), \quad \forall \lambda \in [0,1]$$

*Informally: The line connecting any two points on the function lies **above** the curve.*

**Key Property:** A convex function has **exactly one critical point**, which is the **global minimum**.

**Implication for GD:** Starting from **any** point, Gradient Descent is **guaranteed** to find the global minimum.

Example (convex): $f(x) = x^2 - 4x + 6$ — a perfect bowl. One minimum at $x = 2$.

#### 9.2 Non-Convex Functions

Non-convex functions have:
- **Multiple local minima** — valleys that GD can get trapped in.
- **Saddle points** — flat regions where $\nabla f = 0$ but it's not a minimum.
- **Local maxima** — peaks (GD never gets trapped here in practice).

Example (non-convex): $f(x) = x^4 - 2x^2 + x$

This function has:
- A **local minimum** near $x \approx 0.82$, $f \approx -0.07$
- A **global minimum** near $x \approx -1.07$, $f \approx -2.05$

Starting at $x = 1.0$ → converges to the **local minimum** at $x \approx 0.82$.
Starting at $x = -1.5$ → converges to the **global minimum** at $x \approx -1.07$.

**The starting point determines the outcome.** This is the core challenge in non-convex optimization.

#### 9.3 Why Does Deep Learning Work Despite Non-Convexity?

This seems catastrophic. Real neural networks have billions of parameters and **highly non-convex** loss landscapes. Yet they work. Why?

Modern research (Choromanska et al., 2015; Goodfellow et al.) provides several partial answers:

1. **High-dimensional geometry:** In very high dimensions, local minima are extremely rare. Most critical points in high-dimensional non-convex functions are **saddle points**, which momentum-based optimizers escape easily.

2. **Equivalent optima:** Local minima that exist in high-dimensional networks tend to have **similar loss values** to the global minimum. Finding any "good" local minimum is sufficient.

3. **Over-parameterization:** When a model has far more parameters than data points (common in deep learning), the loss landscape becomes smoother and easier to navigate.

4. **Flat minima generalize better:** Interestingly, the "wide, flat" minima that GD tends to find often **generalize better** to new data than sharp global minima would.

---

### 10. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Goal of ML Training | $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$ |
| MSE Loss | $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (h_\theta(x_i) - y_i)^2$ |
| GD Update Rule (1D) | $x_{t+1} = x_t - \eta \cdot f'(x_t)$ |
| GD Update Rule (General) | $\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$ |
| Our Lecture Function | $f(x) = x^2 - 4x + 6$, minimum at $x=2$ |
| Its Gradient | $f'(x) = 2x - 4$ |
| Tutorial Challenge | $f(x) = x^4 - 2x^2 + x$, $f'(x) = 4x^3 - 4x + 1$ |

---

### 11. Summary

1. **Learning = Optimization.** Finding model parameters $\theta$ that minimize a Loss Function.
2. **The Loss Function** encodes what "wrong" means mathematically. It must be differentiable.
3. **Naive Search** (brute force, random) fails due to the Curse of Dimensionality.
4. **Hill Climbing** uses local search but has no sense of direction and gets stuck.
5. **Gradient Descent** exploits the derivative to always step toward lower loss. It is the engine of all of modern AI.
6. **The Learning Rate** controls step size. Too small: slow. Too large: diverges.
7. **Convex functions** guarantee a global minimum from any start. **Non-convex** (real ML) does not — but in practice, modern deep learning handles this remarkably well.

---

### 12. Bridge to the Tutorial

In the tutorial session (with code), you will:
- Implement `f(x)`, `f'(x)`, and the gradient descent loop in Python from scratch.
- Visualize the loss landscape and the path the optimizer takes.
- Experiment with learning rates: too small, just right, and diverging.
- Tackle the challenge: a non-convex function with two minima.

**Come to the tutorial. The math is empty without the code.**
