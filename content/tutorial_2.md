# Week 2 Tutorial: Bot Battle
## "Building an Unbeatable Opponent"

---

### 0. Instructor Introduction: Elnur Shahbalayev

*   **Background**: AI Engineer @ Bayraktar Technologies.
*   **Academia**: UTP (Malaysia) -> Warwick/ASOIU (Masters).
*   **Philosophy**: "Theory is empty without code. Code is blind without theory."
*   **Today's Goal**: By the end of this tutorial, you will have built a Tic-Tac-Toe AI that **cannot be beaten**.

---

### 1. What We Are Building

Last week, we wrote code that found the bottom of a mathematical valley. Today, we write code that **thinks strategically** — an AI agent that plays Tic-Tac-Toe perfectly using the Minimax algorithm.

**By the end of this session, you will have:**
- A complete Tic-Tac-Toe game engine.
- A Minimax AI that evaluates **every possible future** before making a move.
- An Alpha-Beta Pruned version that does the same thing faster.
- Proof that your AI is unbeatable (try it yourself!).

---

### 2. Environment Setup

```python
import math
import time

# We only need the standard library today.
# No numpy, no matplotlib — pure logic.
print("Ready to build an unbeatable AI!")
```

---

### 3. The Game Engine

Before we build the brain (AI), we need the body (game engine). A Tic-Tac-Toe board is a 3×3 grid. We'll represent it as a list of 9 elements.

#### 3.1 Board Representation

```python
# Board positions:
# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8

EMPTY = ' '
X = 'X'
O = 'O'

def create_board():
    """Create a fresh, empty board."""
    return [EMPTY] * 9
```

#### 3.2 Display the Board

```python
def print_board(board):
    """Display the board in a human-readable format."""
    for row in range(3):
        cells = []
        for col in range(3):
            cells.append(board[row * 3 + col])
        print(f" {cells[0]} | {cells[1]} | {cells[2]} ")
        if row < 2:
            print("-----------")
    print()
```

#### 3.3 Game Logic

```python
# All possible winning lines (indices)
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

def check_winner(board):
    """Return 'X', 'O', or None."""
    for line in WIN_LINES:
        a, b, c = line
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    return None

def get_available_moves(board):
    """Return list of empty cell indices."""
    return [i for i in range(9) if board[i] == EMPTY]

def is_terminal(board):
    """Is the game over?"""
    return check_winner(board) is not None or len(get_available_moves(board)) == 0

def make_move(board, position, player):
    """Return a NEW board with the move applied."""
    new_board = board.copy()
    new_board[position] = player
    return new_board
```

---

### 4. The Minimax Algorithm

Now for the brain. Recall from the lecture:

$$\text{minimax}(S) = \begin{cases} U(S) & \text{if terminal} \\ \max_a \text{minimax}(\text{Result}(S, a)) & \text{if MAX's turn} \\ \min_a \text{minimax}(\text{Result}(S, a)) & \text{if MIN's turn} \end{cases}$$

Let's set:
- X = MAX (AI, tries to maximize score)
- O = MIN (Human, tries to minimize score)
- Utility: X wins → +1, O wins → -1, Draw → 0

```python
def utility(board):
    """Score a terminal board state."""
    winner = check_winner(board)
    if winner == X:
        return +1   # AI wins
    elif winner == O:
        return -1   # Human wins
    else:
        return 0    # Draw

def minimax(board, is_maximizing):
    """Return the minimax value of this board state."""
    # Base case: game is over
    if is_terminal(board):
        return utility(board)

    if is_maximizing:  # X's turn (AI)
        best_value = -math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, X)
            value = minimax(child, False)
            best_value = max(best_value, value)
        return best_value
    else:  # O's turn (Human)
        best_value = math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, O)
            value = minimax(child, True)
            best_value = min(best_value, value)
        return best_value
```

#### 4.1 Choosing the Best Move

```python
def find_best_move(board):
    """Return the optimal move for X (AI) using Minimax."""
    best_value = -math.inf
    best_move = None

    for move in get_available_moves(board):
        child = make_move(board, move, X)
        value = minimax(child, False)  # After AI moves, it's human's turn
        if value > best_value:
            best_value = value
            best_move = move

    return best_move
```

**Discussion Point**: How many nodes does this explore on an empty board? For the first move, the tree has about 9! = 362,880 leaf nodes. That's small enough to solve completely.

---

### 5. Play Against Your AI

```python
def play_game():
    """Human (O) vs AI (X). AI goes first."""
    board = create_board()
    print("=== TIC-TAC-TOE: You (O) vs AI (X) ===")
    print("Enter positions 0-8:\n")
    print_board(board)

    current_player = X  # AI starts

    while not is_terminal(board):
        if current_player == X:
            print("AI is thinking...")
            start = time.time()
            move = find_best_move(board)
            elapsed = time.time() - start
            board = make_move(board, move, X)
            print(f"AI plays position {move} ({elapsed:.3f}s)")
        else:
            valid = False
            while not valid:
                try:
                    move = int(input("Your move (0-8): "))
                    if move in get_available_moves(board):
                        valid = True
                    else:
                        print("That cell is taken!")
                except ValueError:
                    print("Enter a number 0-8.")
            board = make_move(board, move, O)

        print_board(board)
        current_player = O if current_player == X else X

    # Game Over
    winner = check_winner(board)
    if winner == X:
        print("AI wins! (As expected — it plays perfectly.)")
    elif winner == O:
        print("You won?! That shouldn't be possible...")
    else:
        print("It's a draw! That's the best you can do against perfect play.")

# Uncomment to play:
# play_game()
```

---

### 6. Alpha-Beta Pruning: Making It Faster

The Minimax AI works, but it explores every single branch. Let's add Alpha-Beta Pruning to skip irrelevant branches.

```python
def alpha_beta(board, alpha, beta, is_maximizing):
    """Minimax with Alpha-Beta Pruning."""
    if is_terminal(board):
        return utility(board)

    if is_maximizing:
        value = -math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, X)
            value = max(value, alpha_beta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # β cutoff: MIN already has a better option
        return value
    else:
        value = math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, O)
            value = min(value, alpha_beta(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # α cutoff: MAX already has a better option
        return value

def find_best_move_ab(board):
    """Return the optimal move for X using Alpha-Beta."""
    best_value = -math.inf
    best_move = None

    for move in get_available_moves(board):
        child = make_move(board, move, X)
        value = alpha_beta(child, -math.inf, math.inf, False)
        if value > best_value:
            best_value = value
            best_move = move

    return best_move
```

---

### 7. Measuring the Speedup

Let's count how many nodes each algorithm visits:

```python
# Global counter
node_count = 0

def minimax_counted(board, is_maximizing):
    """Minimax with node counting."""
    global node_count
    node_count += 1

    if is_terminal(board):
        return utility(board)

    if is_maximizing:
        best = -math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, X)
            best = max(best, minimax_counted(child, False))
        return best
    else:
        best = math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, O)
            best = min(best, minimax_counted(child, True))
        return best

def alpha_beta_counted(board, alpha, beta, is_maximizing):
    """Alpha-Beta with node counting."""
    global node_count
    node_count += 1

    if is_terminal(board):
        return utility(board)

    if is_maximizing:
        value = -math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, X)
            value = max(value, alpha_beta_counted(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move in get_available_moves(board):
            child = make_move(board, move, O)
            value = min(value, alpha_beta_counted(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

# Compare on empty board
board = create_board()

node_count = 0
minimax_counted(board, True)
minimax_nodes = node_count

node_count = 0
alpha_beta_counted(board, -math.inf, math.inf, True)
ab_nodes = node_count

print(f"Minimax explored:    {minimax_nodes:,} nodes")
print(f"Alpha-Beta explored: {ab_nodes:,} nodes")
print(f"Pruned:              {minimax_nodes - ab_nodes:,} nodes ({(1 - ab_nodes/minimax_nodes)*100:.1f}% reduction)")
```

**Expected Output (approximately):**
```
Minimax explored:    549,946 nodes
Alpha-Beta explored: 18,297 nodes
Pruned:              531,649 nodes (96.7% reduction)
```

---

### 8. Experiment Ideas

<!-- Interactive Exploration -->
<div class="interactive-panel">
    <h3>🧪 Extend Your Bot</h3>
    <p><strong>Try these modifications:</strong></p>
    <ul>
        <li><strong>Experiment A:</strong> Let the Human go first instead of the AI. Does the AI still never lose?</li>
        <li><strong>Experiment B:</strong> Add a "depth" counter to minimax. What is the maximum depth reached from an empty board?</li>
        <li><strong>Experiment C:</strong> Create a "dumb AI" that picks random moves. Pit it against your Minimax AI over 100 games. What's the win rate?</li>
    </ul>
</div>

---

### 9. Assignment for Next Week

*   **Coding**: Submit a Jupyter Notebook containing your Tic-Tac-Toe AI with both Minimax and Alpha-Beta implementations.
*   **Report**: Include the node count comparison (Section 7) and a brief explanation of why Alpha-Beta is so much faster.
*   **Bonus**: Extend your AI to play **Connect-4** (6×7 grid, 4 in a row). Since the tree is too large, implement depth-limited search with a simple evaluation function.

**See you at the Lecture!**
