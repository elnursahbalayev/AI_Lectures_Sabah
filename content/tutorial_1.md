# Week 1 Tutorial: Practical Optimization
## "Into the Code"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Goal**: By Week 15, you will have built an entire AI pipeline.

---

### 1. Environment Sanity Check

Before we optimize anything, let's optimize your setup.

**Task 1**: Open your Jupyter Notebook / Colab.
**Task 2**: Run the standard imports.

```python
import numpy as np
import matplotlib.pyplot as plt

# If this fails, raise your hand immediately.
print(f"Numpy Version: {np.__version__}")
```

---

### 2. The Task: Find the Minimum

We are given a function:
$$ f(x) = x^2 - 4x + 6 $$

Analytically (using Calculus 101), we know the minimum is at $x=2$.
*   $f'(x) = 2x - 4$
*   Set $2x - 4 = 0 \rightarrow x = 2$.

**But** our computer doesn't know Calculus. It has to "search" for it.

#### Visualization
Let's first see what we are dealing with.

```python
# Define the function
def f(x):
    return x**2 - 4*x + 6

# Generate data
x_vals = np.linspace(-1, 5, 100)
y_vals = f(x_vals)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='f(x) = x^2 - 4x + 6')
plt.title("The Cost Landscape")
plt.grid(True)
plt.show()
```

---

### 3. Implementing Gradient Descent

To perform Gradient Descent, we need the **derivative** (gradient).

$$ \frac{df}{dx} = 2x - 4 $$

```python
def df(x):
    return 2*x - 4
```

Now, the algorithm loops:
1.  Start at a random `current_x`.
2.  Calculate gradient `grad = df(current_x)`.
3.  Take a step: `current_x = current_x - learning_rate * grad`.
4.  Repeat.

#### The Code

```python
def gradient_descent(start_x, learning_rate, epochs):
    x = start_x
    history = [] # To store the path
    
    for i in range(epochs):
        grad = df(x)
        history.append(x) # Save current position
        
        # The Update Rule
        x = x - (learning_rate * grad)
        
        print(f"Epoch {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")
        
    return x, history
```

---

### 4. Running the Experiment

Let's try to start at $x = 0$ (The left side of the bowl). We want to reach $x = 2$.

```python
# Parameters
start_x = 0.0
lr = 0.1
epochs = 20

final_x, path = gradient_descent(start_x, lr, epochs)

print(f"\nConverged to x = {final_x:.4f} (True min is 2.0)")
```

**Discussion Point**: Look at the print statements. Does the change in `x` get smaller or larger as we approach the target? Why? (Hint: derivative gets smaller).

---

### 5. Interactive Visualization: The Path Taken

Let's visualize exactly how the "ball" rolled down the hill.

```python
# Re-plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', label='Function')

# Plot the path
path_y = [f(x) for x in path]
plt.plot(path, path_y, 'ro--', label='Gradient Descent Path')

plt.title(f"Optimization Path (LR={lr})")
plt.legend()
plt.show()
```

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>🧪 Laboratory Experiment</h3>
    <p><strong>Try these values in your notebook and report back:</strong></p>
    <ul>
        <li><strong>Case A:</strong> <code>lr = 0.01</code> (What happens to the speed?)</li>
        <li><strong>Case B:</strong> <code>lr = 0.9</code> (What happens to the path?)</li>
        <li><strong>Case C:</strong> <code>lr = 1.1</code> (WARNING: What happens now?)</li>
    </ul>
</div>

---

### 6. Challenge: The Non-Convex Functions

Real AI problems aren't simple bowls.

**Task**: Change your function `f(x)` to:
$$ f(x) = x^4 - 2x^2 + x $$

1.  Calculate the derivative by hand (or WolframAlpha).
2.  Update the `df(x)` function.
3.  Run Gradient Descent starting at `x = -1.5` and `x = 1.0`.

**Outcome**: You will likely end up in *different* locations depending on where you start. This is the **Local Minima** problem.

---

### 7. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook where you implement Gradient Descent for the "Challenge function" above.
*   **Bonus**: Implement "Momentum" (Look it up!). It helps the ball roll through shallow local minima.

**See you at the Lecture!**
