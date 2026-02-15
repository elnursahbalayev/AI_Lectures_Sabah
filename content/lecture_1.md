# Week 1 Lecture: Optimization & Search
## The Engine of Artificial Intelligence

---

### 1. Introduction: What does "Learning" actually mean?

In the context of Machine Learning (ML), "learning" is not about memorizing facts. It is the mathematical process of **Parameter Optimization**.

Imagine an AI model as a complex box with thousands (or billions) of adjustable knobs (parameters, weights: $w$).
*   **Input**: Data (Images, Text, Numbers).
*   **Output**: Prediction.
*   **Goal**: Adjust the knobs ($w$) so that the Prediction matches the Reality.

The "unhappiness" of the model with its current settings is measured by a **Loss Function** ($J(w)$).
*   **High Loss**: Model is wrong (Unhappy).
*   **Low Loss**: Model is correct (Happy).

**Learning = Minimizing the Loss Function.**

> "We don't code the solution. We code the optimization process that *finds* the solution."

![Concept: Model interacting with Loss Function](https://dummyimage.com/600x400/000/fff&text=Loss+Minimization+Diagram)

---

### 2. Search Strategies: Finding the Bottom of the Valley

Imagine you are dropped on a foggy mountain range at night. You want to reach the lowest point (the valley) to find water. You cannot see the landscape. You can only feel the slope under your feet.

#### Strategy A: Hill Climbing (Blind Search)
*   Look at immediate neighbors.
*   Step to the one that is lower.
*   **Problem**: You might get stuck in a small dip (Local Minimum) and never find the ocean (Global Minimum). It doesn't use the *smoothness* of the terrain efficiently.

#### Strategy B: Gradient Descent (Guided Search)
*   **Calculus to the rescue**: If we have the mathematical function of the terrain, we can calculate the **Gradient** ($\nabla$).
*   The Gradient points in the direction of the **steepest ascent** (uphill).
*   **Strategy**: Calculate Gradient $\rightarrow$ Walk in the *opposite* direction.

---

### 3. The Mathematics of Gradient Descent

This is the most important formula you will learn in this course. It powers everything from Linear Regression to GPT-4.

$$ \theta_{new} = \theta_{old} - \eta \cdot \nabla J(\theta_{old}) $$

Where:
*   $\theta$ (Theta): The parameter (weight) we are updating.
*   $\eta$ (Eta): The **Learning Rate** (Step size). How big of a step do we take?
*   $\nabla J(\theta)$: The **Gradient** of the Cost Function. The slope.

#### Intuition Check
*   If Slope is positive (+), we go Left (-).
*   If Slope is negative (-), we go Right (+).
*   We always slide *down* the hill.

---

### 4. The Learning Rate ($\eta$)

The Learning Rate is a **Hyperparameter**. It determines the speed of convergence.

#### Scenario 1: Learning Rate too Small
*   **Behavior**: Tiny baby steps.
*   **Result**: Takes forever to converge. Computationally expensive.

#### Scenario 2: Learning Rate too Large
*   **Behavior**: Giant leaps.
*   **Result**: You might step *over* the valley and land on the other side. You oscillate or even diverge (climb up the other side!).

#### Scenario 3: Just Right
*   **Behavior**: Components take large steps initially, then smaller steps as the slope gets flatter (because $\nabla J$ approaches 0).

<!-- Interactive Graph Placeholder -->
<div class="interactive-graph" id="learning-rate-demo">
  <p>[INTERACTIVE: Slider for Learning Rate (0.001 to 1.5). Graph shows a parabola y=x^2 with a ball rolling down based on the selected rate.]</p>
</div>

---

### 5. The Landscape: Convex vs. Non-Convex

#### Convex Functions
*   **Shape**: A perfect bowl.
*   **Properties**: Only ONE minimum (Global Minimum).
*   **Implication**: Gradient Descent is **guaranteed** to find the optimal solution (eventually).
*   **Examples**: Linear Regression Mean Squared Error.

#### Non-Convex Functions
*   **Shape**: An egg carton, or rugged terrain.
*   **Properties**: Many **Local Minima**, Saddle Points, and one Global Minimum.
*   **Implication**: Gradient Descent might get stuck in a "good enough" Local Minimum instead of the best one.
*   **Examples**: Deep Neural Networks.

> **Discussion**: Why do Neural Networks work if they are non-convex? (Preview: High dimensionality makes getting stuck less likely than you think!)

---

### 6. Summary for Week 1
1.  **Objective**: Minimize Loss.
2.  **Method**: Iteratively update weights opposite to the gradient.
3.  **Key Challenge**: Choosing the right Learning Rate.
4.  **Reality**: Most interesting problems are Non-Convex.

**Next Class (Tutorial)**: We will code this from scratch in Python. No libraries doing the work for us!
