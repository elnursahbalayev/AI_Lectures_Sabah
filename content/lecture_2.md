# Week 2 Lecture: Game Theory & Decision Making
## "Thinking About the Enemy"

*Module 1: Algorithmic Foundations | Elnur Shahbalayev*

---

### 0. Lecture Roadmap

Today's central thesis:

> **Intelligence isn't just about optimizing alone — it's about optimizing when someone is actively trying to beat you.**

Last week, we searched for the minimum of a function. The landscape didn't fight back. This week, the landscape has a mind of its own.

**We will cover:**
1. From Optimization to Adversarial Thinking
2. Game Trees — Representing Sequential Decisions
3. The Minimax Algorithm — Optimal Play Under Opposition
4. Alpha-Beta Pruning — Making Minimax Practical
5. Evaluation Functions — When You Can't See the End
6. Introduction to Agents — Sense, Think, Act

---

### 1. From Optimization to Adversarial Environments

Last week, we minimized $f(x) = x^2 - 4x + 6$. The function was static — it didn't change its shape based on our moves. This is the world of **single-agent optimization**.

But the real world is full of **adversarial environments**:
- Chess: your opponent actively blocks your winning moves.
- Cybersecurity: attackers adapt to your defenses.
- Business: competitors react to your strategy.
- GANs (Week 13): one neural network tries to fool another.

In these settings, the "landscape" changes with every move your opponent makes. We need algorithms that reason about **what the opponent will do in response to our actions**.

**Key Shift:**
```
Week 1: argmin_θ L(θ)           — Minimize loss, alone.
Week 2: argmax_me min_opponent   — Maximize my outcome, assuming
                                    the opponent minimizes it.
```

This is the core of **Game Theory** applied to AI.

---

### 2. Formalizing Games

#### 2.1 What is a "Game" in CS?

A formal game (in the AI sense) is defined by:
- **Players**: Typically two — MAX (us) and MIN (opponent).
- **States**: All possible configurations of the game (e.g., all possible board positions).
- **Actions**: Legal moves from any given state.
- **Transition Function**: $S' = \text{Result}(S, a)$ — the new state after taking action $a$ in state $S$.
- **Terminal Test**: Is the game over? $\text{Terminal}(S) \in \{\text{True}, \text{False}\}$
- **Utility Function**: $U(S)$ — a numeric score assigned to terminal states.
  - WIN → $+1$ (or $+10$, $+\infty$, etc.)
  - LOSS → $-1$
  - DRAW → $0$

#### 2.2 The Game Tree

A **Game Tree** is a tree structure where:
- The **root** is the current game state.
- Each **edge** represents a legal move.
- Each **node** is a resulting game state.
- **Levels alternate** between MAX's moves and MIN's moves.
- **Leaves** are terminal states with known utility values.

**Example: A tiny Tic-Tac-Toe fragment**

```
         [X| | ]     ← MAX's turn (X plays)
         [ | | ]
         [ | | ]
        /    |    \
  [X|O| ]  [X| | ]  [X| | ]   ← MIN responds (O plays)
  [ | | ]  [ |O| ]  [ | | ]
  [ | | ]  [ | | ]  [O| | ]
     ...     ...      ...
```

For Tic-Tac-Toe, the full game tree has approximately **255,168** possible games. For Chess, it's approximately $10^{120}$ — more than the atoms in the observable universe.

---

### 3. The Minimax Algorithm

Minimax is the foundational algorithm for adversarial search. The logic is beautifully simple:

> **MAX** assumes **MIN** will always play optimally (worst case for MAX).
> MAX then chooses the move that **maximizes** the outcome under this worst-case assumption.

#### 3.1 The Recursive Definition

$$\text{minimax}(S) = \begin{cases} U(S) & \text{if } S \text{ is terminal} \\ \max_{a} \text{minimax}(\text{Result}(S, a)) & \text{if it's MAX's turn} \\ \min_{a} \text{minimax}(\text{Result}(S, a)) & \text{if it's MIN's turn} \end{cases}$$

#### 3.2 Worked Example

Consider a game tree where MAX moves first, choosing between actions A, B, and C. Each leads to a node where MIN makes a choice, eventually reaching terminal nodes with utility values.

```
                    MAX
                 /   |   \
               A     B     C
              /     / \     \
           MIN   MIN   MIN   MIN
           /|    /|    / \    |\
          3  5  2  9  1   7  4  6
```

**Step 1 — MIN evaluates** (picks minimum):
- Left MIN node: $\min(3, 5) = 3$
- Center-left MIN node: $\min(2, 9) = 2$
- Center-right MIN node: $\min(1, 7) = 1$
- Right MIN node: $\min(4, 6) = 4$

**Step 2 — MAX evaluates** (picks maximum of MIN's choices):
- Action A → 3
- Action B → $\max(2, 1) = 2$
- Action C → 4

**MAX chooses C** with value **4**.

The critical insight: MAX doesn't pick the branch with the highest terminal value (9 is in branch B!) — because MIN would never let MAX reach it.

#### 3.3 The Algorithm in Pseudocode

```python
def minimax(state, is_maximizing):
    # Base case: game is over
    if is_terminal(state):
        return utility(state)

    if is_maximizing:
        best = -infinity
        for action in get_actions(state):
            child = result(state, action)
            value = minimax(child, False)  # Opponent's turn
            best = max(best, value)
        return best
    else:
        best = +infinity
        for action in get_actions(state):
            child = result(state, action)
            value = minimax(child, True)   # Our turn
            best = min(best, value)
        return best
```

#### 3.4 Complexity Analysis

- **Time**: $O(b^m)$ where $b$ = branching factor, $m$ = maximum depth.
- **Space**: $O(b \cdot m)$ (depth-first search).

| Game | Branching Factor ($b$) | Depth ($m$) | $b^m$ |
|---|---|---|---|
| Tic-Tac-Toe | ~5 (avg) | 9 | ~2 million |
| Chess | ~35 | ~80 | $\approx 10^{120}$ |
| Go | ~250 | ~150 | $\approx 10^{360}$ |

Tic-Tac-Toe is solvable. Chess is not (with brute-force Minimax). We need a way to **cut branches** without losing optimality.

---

### 4. Alpha-Beta Pruning

Alpha-Beta Pruning is one of the most elegant ideas in computer science. It produces the **exact same result** as Minimax, but skips evaluating branches that **cannot possibly influence the final decision**.

#### 4.1 The Key Insight

During search, we maintain two values:
- $\alpha$ = the best (highest) value MAX can guarantee so far (along the current path).
- $\beta$ = the best (lowest) value MIN can guarantee so far (along the current path).

**Pruning Rule:**
> If at any point $\alpha \geq \beta$, stop evaluating the current subtree. The remaining branches are irrelevant — one player already has a better option elsewhere.

#### 4.2 Walkthrough Example

```
                      MAX (α=-∞, β=+∞)
                    /         \
                  A             B
                MIN           MIN
               / \           / \
              3    5        2    ?
```

1. Evaluate left subtree of A: value = 3. Update MIN's best to 3.
2. Evaluate right subtree of A: value = 5. MIN still picks 3 ($\min(3,5) = 3$).
3. A returns 3 to MAX. MAX updates $\alpha = 3$.
4. Enter B (MIN node). Evaluate left child: value = 2.
5. MIN would pick at most 2 (since MIN minimizes). But MAX already has $\alpha = 3$.
6. Since $2 < 3$, MAX would **never choose B**. **Prune the remaining children of B.**
7. The `?` node is never evaluated.

#### 4.3 The Pruned Algorithm

```python
def alpha_beta(state, alpha, beta, is_maximizing):
    if is_terminal(state):
        return utility(state)

    if is_maximizing:
        value = -infinity
        for action in get_actions(state):
            child = result(state, action)
            value = max(value, alpha_beta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # β cutoff — MIN has a better path
        return value
    else:
        value = +infinity
        for action in get_actions(state):
            child = result(state, action)
            value = min(value, alpha_beta(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # α cutoff — MAX has a better path
        return value

# Initial call:
best_value = alpha_beta(root_state, -infinity, +infinity, True)
```

#### 4.4 How Much Does It Save?

With **perfect move ordering** (best moves evaluated first):
- Minimax evaluates $O(b^m)$ nodes.
- Alpha-Beta evaluates $O(b^{m/2})$ nodes.

This is a massive improvement! It effectively **doubles the searchable depth** for the same computation budget.

| Game | Minimax nodes | Alpha-Beta nodes (optimal) |
|---|---|---|
| Tic-Tac-Toe | ~500,000 | ~5,000 |
| Chess (6 moves deep) | ~1.8 billion | ~42,000 |

---

### 5. Evaluation Functions: When You Can't See the End

For complex games like Chess, even Alpha-Beta Pruning can't reach terminal states within a time budget. The solution: **cut off the search at a fixed depth** and estimate the value of non-terminal states using an **Evaluation Function** $\text{Eval}(S)$.

#### 5.1 What Makes a Good Evaluation Function?

1. **Fast to compute** — it's called millions of times.
2. **Correlated with true utility** — high eval should mean a likely win.
3. **Orders states correctly** — $\text{Eval}(S_1) > \text{Eval}(S_2)$ should mean $S_1$ is truly better for MAX.

#### 5.2 Example: Chess Evaluation

A classic (simplified) chess evaluation function:

$$\text{Eval}(S) = w_1 \cdot \text{Material} + w_2 \cdot \text{Mobility} + w_3 \cdot \text{King Safety} + w_4 \cdot \text{Center Control}$$

Where:
- **Material** = piece values (Queen=9, Rook=5, Bishop/Knight=3, Pawn=1). Sum your pieces minus opponent's.
- **Mobility** = number of legal moves available.
- **King Safety** = is the king exposed?
- **Center Control** = how many central squares do you control?

**Key realization**: this is just a weighted sum — the same structure as a linear model from ML! In fact, modern chess engines (Stockfish NNUE, AlphaZero) replaced hand-crafted evaluation functions with neural networks trained on millions of games. **Game AI and Machine Learning converge.**

#### 5.3 Depth-Limited Minimax

```python
def minimax_depth_limited(state, depth, is_maximizing):
    if is_terminal(state):
        return utility(state)
    if depth == 0:
        return evaluate(state)  # Heuristic estimate

    if is_maximizing:
        best = -infinity
        for action in get_actions(state):
            child = result(state, action)
            value = minimax_depth_limited(child, depth - 1, False)
            best = max(best, value)
        return best
    else:
        best = +infinity
        for action in get_actions(state):
            child = result(state, action)
            value = minimax_depth_limited(child, depth - 1, True)
            best = min(best, value)
        return best
```

---

### 6. Introduction to Agents

An **Agent** is an entity that:
1. **Perceives** its environment through sensors.
2. **Decides** on an action using some internal logic.
3. **Acts** on the environment through actuators.

```
Environment → [Sensors] → Agent Logic → [Actuators] → Environment
                              ↑
                          (Minimax, ML model, Rule-based, etc.)
```

#### 6.1 Types of Agents

| Agent Type | How It Decides | Example |
|---|---|---|
| **Reflex** | `if condition → action` | Thermostat |
| **Model-Based** | Maintains internal state | Self-driving car |
| **Goal-Based** | Plans toward a goal | Minimax game player |
| **Utility-Based** | Maximizes a utility function | Recommendation system |
| **Learning** | Improves from experience | AlphaGo, ChatGPT |

Our Tic-Tac-Toe bot in the tutorial is a **Goal-Based Agent**: its goal is to win (or not lose), and it uses Minimax to plan actions.

#### 6.2 From Games to the Real World

| Game Property | Games (Today) | Real-World AI |
|---|---|---|
| Full observability | We see the whole board | Partial: fog-of-war, sensor noise |
| Deterministic | No randomness | Stochastic: dice, network latency |
| Discrete actions | Finite moves | Continuous: steering angle, dosage |
| Two players | MAX vs MIN | Multi-agent systems |
| Zero-sum | One wins, other loses | Cooperative/Mixed-motive |

Each relaxation of these constraints requires more sophisticated algorithms — Expectiminimax for randomness, POMDP for partial observability, MCTS for huge search spaces (Week 14 preview).

---

### 7. Historical Context: Games as AI's Testing Ground

Games have served as the benchmark for AI progress since the very beginning:

| Year | Milestone | Algorithm |
|---|---|---|
| 1950 | Shannon proposes chess programming | Minimax concept |
| 1997 | Deep Blue defeats Kasparov | Alpha-Beta + custom hardware |
| 2016 | AlphaGo defeats Lee Sedol | Monte Carlo Tree Search + Deep RL |
| 2017 | AlphaZero masters Chess, Go, Shogi | Self-play + Neural Network eval |
| 2019 | OpenAI Five defeats Dota 2 pros | Deep Reinforcement Learning |

**Pattern**: Each breakthrough combined better **search** with better **evaluation** — precisely the two concepts from today's lecture.

---

### 8. Key Equations — Quick Reference

| Concept | Formula |
|---|---|
| Minimax (terminal) | $\text{minimax}(S) = U(S)$ |
| Minimax (MAX turn) | $\text{minimax}(S) = \max_a \text{minimax}(\text{Result}(S,a))$ |
| Minimax (MIN turn) | $\text{minimax}(S) = \min_a \text{minimax}(\text{Result}(S,a))$ |
| Alpha-Beta Prune Condition | Prune when $\alpha \geq \beta$ |
| Time Complexity (Minimax) | $O(b^m)$ |
| Time Complexity (Alpha-Beta) | $O(b^{m/2})$ best case |
| Evaluation Function | $\text{Eval}(S) = \sum_i w_i \cdot \text{feature}_i(S)$ |

---

### 9. Summary

1. **Adversarial environments** require reasoning about opponent responses — static optimization isn't enough.
2. **Game Trees** represent all possible sequences of moves. They grow exponentially with depth.
3. **Minimax** finds the optimal move by assuming the opponent plays perfectly. It guarantees the best worst-case outcome.
4. **Alpha-Beta Pruning** is Minimax with intelligence — it skips branches that can't affect the decision, doubling the effective search depth.
5. **Evaluation Functions** estimate state quality when we can't search to the end. They are the bridge between game-playing and Machine Learning.
6. **Agents** are the unifying framework: sense → think → act. Our game bot is a goal-based agent.

---

### 10. Bridge to the Tutorial

In the tutorial session, you will:
- Build a Tic-Tac-Toe game engine from scratch in Python.
- Implement the Minimax algorithm to create an **unbeatable** AI player.
- Add Alpha-Beta Pruning and measure the speedup.
- Play against your own AI and witness perfect play.

**Come to the tutorial. The algorithm becomes real when you can't beat your own code.**
