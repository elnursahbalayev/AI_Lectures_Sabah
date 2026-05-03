# Final Examination
## Modern Artificial Intelligence & Deep Learning
**ASOIU — Spring 2026 | Bachelor 3rd Year**
**Instructor:** Elnur Shahbalayev
**Duration:** 2 Hours | **Total Points:** 80

---

> **Instructions**
> - Answer all 40 questions.
> - For MCQ, circle the single best answer unless the question says *"select all that apply."*
> - For open-ended questions, write concisely — 2 to 5 sentences is usually sufficient. No full derivations are required.
> - Point values are shown next to each question.
> - Difficulty: 🟢 Easy · 🟡 Medium · 🔴 Hard

---

## Section 1 — Optimization & Search (Weeks 1–2)

---

**Q1** 🟢 *(1 pt) — MCQ*

Gradient descent updates the model parameters by moving in which direction?

- A) The direction of the gradient
- B) The direction opposite to the gradient
- C) A random direction to explore the loss surface
- D) The direction of the second derivative of the loss

---

**Q2** 🟢 *(1 pt) — MCQ*

In the Minimax algorithm, the MAX player always tries to:

- A) Minimize the opponent's score
- B) Minimize its own score to confuse the opponent
- C) Maximize its own score
- D) Randomly select a move to avoid being predicted

---

**Q3** 🟡 *(2 pts) — MCQ*

A learning rate that is set too large will most likely cause:

- A) Slow but guaranteed convergence to the global minimum
- B) The model weights to converge to exactly zero
- C) The loss to oscillate or diverge and never converge
- D) The model to underfit the training data

---

**Q4** 🟡 *(2 pts) — MCQ*

Alpha-Beta pruning improves the Minimax algorithm by:

- A) Changing the final decision made by the algorithm to be more optimal
- B) Pruning branches that cannot possibly influence the final decision
- C) Increasing the maximum search depth within the same time budget
- D) Replacing the heuristic evaluation function with an exact one

---

**Q5** 🟡 *(2 pts) — Short Open-Ended*

What is the difference between a convex and a non-convex loss function? Why does this distinction matter when applying gradient descent?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q6** 🟡 *(2 pts) — Short Open-Ended*

A Minimax agent assumes its opponent always plays optimally. What could go wrong in practice if the real opponent does not play optimally? Is this a weakness or could it ever be an advantage?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q7** 🔴 *(3 pts) — Multi-Select MCQ*

Which of the following statements about gradient descent are correct? *(Select all that apply)*

- A) Mini-batch gradient descent uses exactly one training sample per weight update
- B) The gradient vector points in the direction of steepest ascent of the loss
- C) Setting the learning rate to zero means no learning takes place
- D) Stochastic gradient descent can help the optimizer escape shallow local minima
- E) Gradient descent always finds the global minimum regardless of initialization

---

**Q8** 🔴 *(3 pts) — Long Open-Ended*

Explain why Alpha-Beta pruning never changes the result produced by standard Minimax. Describe the best-case scenario for how many nodes can be pruned, and what tree structure leads to that best case.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

## Section 2 — Data Engineering & Regression (Weeks 3–4)

---

**Q9** 🟢 *(1 pt) — MCQ*

The main reason we normalize features before training a machine learning model is:

- A) To increase the number of available training samples
- B) To ensure all features contribute on an equal scale, preventing large-scale features from dominating the gradient
- C) To automatically remove irrelevant features from the dataset
- D) To convert categorical features into numerical representations

---

**Q10** 🟢 *(1 pt) — MCQ*

Mean Squared Error (MSE) is computed as:

- A) The sum of absolute differences between predictions and targets
- B) The maximum single error between any prediction and its target
- C) The average of squared differences between predictions and targets
- D) The square root of the average squared differences

---

**Q11** 🟡 *(2 pts) — MCQ*

One-Hot Encoding is the correct technique to apply when:

- A) You need to scale a numerical feature to the range [0, 1]
- B) A categorical feature has no natural ordinal relationship between its values
- C) You want to fill missing values in a numerical column
- D) A numerical feature has too many outliers

---

**Q12** 🟡 *(2 pts) — MCQ*

A linear regression model achieves very low error on the training set but much higher error on new, unseen houses. This is most likely an example of:

- A) Underfitting — the model is too simple
- B) Data leakage — test data contaminated the training process
- C) Overfitting — the model memorized the training data including its noise
- D) Normalization failure — features were not scaled correctly

---

**Q13** 🟡 *(2 pts) — Short Open-Ended*

What is data leakage? Give one concrete example of how it can accidentally happen during a preprocessing pipeline.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q14** 🟡 *(2 pts) — Short Open-Ended*

In a linear regression model `ŷ = w₁x₁ + w₂x₂ + b`, what does the bias term `b` represent conceptually? What would happen to the model's expressive power if `b` was removed?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q15** 🔴 *(3 pts) — Multi-Select MCQ*

Which of the following preprocessing steps must be fit **only on the training set** and then applied to the validation/test sets — never fit on all data at once? *(Select all that apply)*

- A) StandardScaler — computing mean and standard deviation for normalization
- B) Splitting the dataset into train and test splits
- C) PCA — computing principal components for dimensionality reduction
- D) One-Hot Encoding of a known fixed set of categories
- E) Imputing missing values using the column mean

---

**Q16** 🔴 *(3 pts) — Short Open-Ended*

A regression model reports a training MSE of 0.3 and a test MSE of 11.4. What does this tell you about the model's condition? Name and briefly explain two different techniques that could address this problem.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

## Section 3 — Classification & Ensemble Methods (Weeks 5–6)

---

**Q17** 🟢 *(1 pt) — MCQ*

The Sigmoid activation function always produces an output in the range:

- A) (−∞, +∞)
- B) (−1, 1)
- C) (0, 1)
- D) [0, 100]

---

**Q18** 🟢 *(1 pt) — MCQ*

A Random Forest reduces overfitting compared to a single deep Decision Tree primarily because it:

- A) Uses much deeper individual trees to capture more detail
- B) Trains many trees on different random subsets of data and features, then averages predictions
- C) Assigns higher weights to correctly classified training samples
- D) Removes all features with low importance before training

---

**Q19** 🟡 *(2 pts) — MCQ*

A Support Vector Machine (SVM) selects the decision boundary that:

- A) Minimizes training error to exactly zero on all training points
- B) Passes through the mean of each class in feature space
- C) Maximizes the margin — the distance to the nearest data points of each class
- D) Minimizes the total number of support vectors used

---

**Q20** 🟡 *(2 pts) — MCQ*

In gradient boosting (e.g., XGBoost), each new tree added to the ensemble is trained to:

- A) Independently learn to predict the original target labels from scratch
- B) Correct the residual errors made by the current ensemble
- C) Reduce the maximum depth of the previous tree
- D) Average its weights with all previously trained trees

---

**Q21** 🟡 *(2 pts) — Short Open-Ended*

Logistic Regression is a classification algorithm, yet it has "regression" in its name. Explain what it actually regresses, and how the output is converted into a class prediction.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q22** 🟡 *(2 pts) — Short Open-Ended*

Explain the key difference between **bagging** and **boosting** as ensemble strategies. Which one primarily reduces variance, and which one primarily reduces bias?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q23** 🔴 *(3 pts) — Multi-Select MCQ*

Which of the following statements about Support Vector Machines are correct? *(Select all that apply)*

- A) The kernel trick allows SVMs to find non-linear decision boundaries in the original feature space
- B) The final decision boundary is determined by all training data points equally
- C) Support vectors are the training points that lie closest to the decision boundary
- D) Increasing the regularization parameter C always reduces overfitting regardless of dataset size
- E) SVMs can be applied to both classification and regression tasks

---

**Q24** 🔴 *(3 pts) — Long Open-Ended*

Explain the Bias-Variance Tradeoff. What does a model with high bias look like in practice? What does a model with high variance look like? Explain specifically how Random Forest addresses the variance problem.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

## Section 4 — Unsupervised Learning (Week 7)

---

**Q25** 🟢 *(1 pt) — MCQ*

Unlike most supervised learning algorithms, K-Means clustering requires the user to specify:

- A) The true cluster label for each training data point
- B) The number of clusters K before training begins
- C) A distance threshold that separates clusters
- D) The initial feature importance weights

---

**Q26** 🟡 *(2 pts) — MCQ*

Which is a key advantage DBSCAN has over K-Means?

- A) DBSCAN always converges faster on large datasets
- B) DBSCAN requires fewer hyperparameters to tune
- C) DBSCAN can find clusters of arbitrary shape and automatically label noisy points as outliers
- D) DBSCAN always finds the globally optimal cluster assignment

---

**Q27** 🟡 *(2 pts) — Short Open-Ended*

What does PCA (Principal Component Analysis) do to a dataset? Give one practical reason why you might apply PCA before training a machine learning model.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q28** 🔴 *(3 pts) — Multi-Select MCQ*

Which of the following statements about K-Means clustering are correct? *(Select all that apply)*

- A) The final result of K-Means can differ depending on the random initialization of centroids
- B) K-Means is guaranteed to find the globally optimal clustering if run long enough
- C) K-Means tends to perform poorly when clusters have very different sizes or densities
- D) K-Means minimizes the within-cluster sum of squared distances to the centroid
- E) K-Means handles categorical features natively without any preprocessing

---

## Section 5 — Model Evaluation & Deployment (Week 8)

---

**Q29** 🟢 *(1 pt) — MCQ*

Precision is formally defined as:

- A) TP / (TP + FN)
- B) TP / (TP + FP)
- C) (TP + TN) / (TP + TN + FP + FN)
- D) 2 × Precision × Recall / (Precision + Recall)

---

**Q30** 🟡 *(2 pts) — MCQ*

In a medical system designed to screen patients for cancer, which metric should be prioritized, and why?

- A) Precision — to avoid diagnosing healthy patients as having cancer
- B) Accuracy — because it gives the most complete picture of overall performance
- C) Recall — because missing a patient who actually has cancer is the most dangerous outcome
- D) F1-Score — because it always balances precision and recall equally

---

**Q31** 🟡 *(2 pts) — Short Open-Ended*

What does the AUC-ROC score measure? What does an AUC of 0.5 mean, and what does an AUC of 1.0 mean?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q32** 🔴 *(3 pts) — Long Open-Ended*

A fraud detection model achieves 99% accuracy on a dataset where 99% of all transactions are legitimate. Explain why this result is misleading. What metrics would you use instead to properly evaluate this model, and why are they more appropriate?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

## Section 6 — Neural Networks (Week 9)

---

**Q33** 🟢 *(1 pt) — MCQ*

What is the output of ReLU(−3)?

- A) −3
- B) 3
- C) 0
- D) 0.05 (small positive value)

---

**Q34** 🟡 *(2 pts) — MCQ*

The primary reason ReLU is preferred over Sigmoid for hidden layers in deep networks is:

- A) ReLU outputs probabilities between 0 and 1, which are easier to interpret
- B) ReLU does not saturate for positive inputs, so its gradient is 1 — avoiding vanishing gradients
- C) ReLU is differentiable at every point including zero
- D) ReLU always produces a perfectly sparse representation regardless of input

---

**Q35** 🟡 *(2 pts) — Short Open-Ended*

What is the vanishing gradient problem in deep neural networks? Which activation functions cause it and why does it make training difficult?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q36** 🔴 *(3 pts) — Multi-Select MCQ*

Which of the following statements about backpropagation are correct? *(Select all that apply)*

- A) Backpropagation computes gradients layer by layer using the Chain Rule of calculus
- B) When using Softmax output with Cross-Entropy loss, the gradient at the output layer simplifies to ŷ − y
- C) Backpropagation updates weights by propagating errors in the forward direction
- D) The ReLU activation contributes a gradient of zero for any neuron with a negative pre-activation
- E) Backpropagation requires storing the activations from the forward pass in order to compute gradients

---

## Section 7 — Computer Vision & CNNs (Week 10)

---

**Q37** 🟢 *(1 pt) — MCQ*

The primary purpose of a pooling layer in a Convolutional Neural Network is to:

- A) Introduce learnable parameters to increase model capacity
- B) Reduce the spatial dimensions of feature maps and provide limited translation invariance
- C) Apply the activation function after the convolution
- D) Normalize the feature map statistics across the batch

---

**Q38** 🟡 *(2 pts) — MCQ*

Residual (skip) connections in ResNet were specifically introduced to solve:

- A) The excessive number of parameters in deep convolutional networks
- B) The degradation problem — where adding more layers made training accuracy worse
- C) The slow convergence caused by Batch Normalization
- D) The lack of translation invariance in fully-connected layers

---

**Q39** 🟡 *(2 pts) — Short Open-Ended*

What is transfer learning in the context of CNNs? When would you choose **feature extraction** (frozen backbone) over **full fine-tuning**, and when would you choose the opposite?

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________

---

**Q40** 🔴 *(3 pts) — Long Open-Ended*

Explain why a Convolutional Neural Network is better suited for image classification than a fully connected MLP. In your answer, address at least **two** structural properties that CNNs exploit which MLPs do not.

*Answer:*
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

---

# Answer Key

*For instructor use only.*

---

| Q | Answer | Type | Difficulty | Points |
|---|--------|------|------------|--------|
| 1 | B | MCQ | 🟢 Easy | 1 |
| 2 | C | MCQ | 🟢 Easy | 1 |
| 3 | C | MCQ | 🟡 Medium | 2 |
| 4 | B | MCQ | 🟡 Medium | 2 |
| 5 | *See below* | Open-ended | 🟡 Medium | 2 |
| 6 | *See below* | Open-ended | 🟡 Medium | 2 |
| 7 | B, C, D | Multi-select | 🔴 Hard | 3 |
| 8 | *See below* | Open-ended | 🔴 Hard | 3 |
| 9 | B | MCQ | 🟢 Easy | 1 |
| 10 | C | MCQ | 🟢 Easy | 1 |
| 11 | B | MCQ | 🟡 Medium | 2 |
| 12 | C | MCQ | 🟡 Medium | 2 |
| 13 | *See below* | Open-ended | 🟡 Medium | 2 |
| 14 | *See below* | Open-ended | 🟡 Medium | 2 |
| 15 | A, C, E | Multi-select | 🔴 Hard | 3 |
| 16 | *See below* | Open-ended | 🔴 Hard | 3 |
| 17 | C | MCQ | 🟢 Easy | 1 |
| 18 | B | MCQ | 🟢 Easy | 1 |
| 19 | C | MCQ | 🟡 Medium | 2 |
| 20 | B | MCQ | 🟡 Medium | 2 |
| 21 | *See below* | Open-ended | 🟡 Medium | 2 |
| 22 | *See below* | Open-ended | 🟡 Medium | 2 |
| 23 | A, C, E | Multi-select | 🔴 Hard | 3 |
| 24 | *See below* | Open-ended | 🔴 Hard | 3 |
| 25 | B | MCQ | 🟢 Easy | 1 |
| 26 | C | MCQ | 🟡 Medium | 2 |
| 27 | *See below* | Open-ended | 🟡 Medium | 2 |
| 28 | A, C, D | Multi-select | 🔴 Hard | 3 |
| 29 | B | MCQ | 🟢 Easy | 1 |
| 30 | C | MCQ | 🟡 Medium | 2 |
| 31 | *See below* | Open-ended | 🟡 Medium | 2 |
| 32 | *See below* | Open-ended | 🔴 Hard | 3 |
| 33 | C | MCQ | 🟢 Easy | 1 |
| 34 | B | MCQ | 🟡 Medium | 2 |
| 35 | *See below* | Open-ended | 🟡 Medium | 2 |
| 36 | A, B, D, E | Multi-select | 🔴 Hard | 3 |
| 37 | B | MCQ | 🟢 Easy | 1 |
| 38 | B | MCQ | 🟡 Medium | 2 |
| 39 | *See below* | Open-ended | 🟡 Medium | 2 |
| 40 | *See below* | Open-ended | 🔴 Hard | 3 |

**Total: 80 points**

---

## Model Answers — Open-Ended Questions

**Q5** — A convex loss has a single global minimum, so gradient descent will always converge to the optimal solution regardless of starting point. A non-convex loss (like those in neural networks) has multiple local minima and saddle points, so gradient descent can get stuck and the solution depends on initialization and learning rate.

**Q6** — If the opponent plays suboptimally, the Minimax agent will still make at least as good a move as needed — it cannot perform worse than expected. However, it may miss opportunities to exploit the opponent's mistakes, since it always defends against the worst case rather than trying to maximize against a weaker player. This is a weakness in competitive games but not a correctness issue.

**Q8** — Alpha-Beta pruning never changes the result because it only skips branches it can prove will not be chosen by a rational player — they are already dominated by a previously found option. In the best case, when nodes are ordered from best to worst, every other branch at each level can be pruned, reducing the effective branching factor from b to √b and allowing search to go twice as deep in the same time.

**Q13** — Data leakage is when information from outside the training set influences the model during training, leading to unrealistically optimistic evaluation. Example: fitting a StandardScaler on the entire dataset (train + test) before splitting — the scaler's mean and variance now encode information about the test set, which the model should never have seen.

**Q14** — The bias term b is the intercept — it allows the model to predict a non-zero value even when all input features are zero. Without b, the regression hyperplane is forced to pass through the origin, which severely limits the model's ability to fit data whose true relationship does not pass through zero.

**Q16** — The large gap between training and test MSE is a clear sign of overfitting — the model memorized the training data including noise and fails to generalize. Two fixes: (1) **Regularization** (L1 or L2) — adds a penalty on large weights, preventing the model from fitting noise. (2) **Collecting more training data** — more examples reduce the model's ability to memorize individual samples and force it to learn the true underlying pattern.

**Q21** — Logistic Regression regresses the log-odds (logit) of the probability that a sample belongs to the positive class, which is a linear function of the input features. The output is passed through the Sigmoid function to squash it to a probability between 0 and 1. A threshold (typically 0.5) then converts this probability into a discrete class label.

**Q22** — Bagging (e.g., Random Forest) trains many models independently on random subsets of data and averages predictions — this reduces variance, since averaging many high-variance models cancels out random errors. Boosting (e.g., XGBoost) trains models sequentially, each correcting the errors of the previous one — this reduces bias, since each iteration pushes the ensemble toward fitting the remaining errors.

**Q24** — Bias is systematic error from overly simple assumptions (e.g., fitting a line to non-linear data) — training and test error are both high. Variance is sensitivity to the specific training set (e.g., a deep tree that perfectly fits every sample) — training error is low but test error is high. Random Forest reduces variance by training many trees on random bootstrap samples and random feature subsets, then averaging predictions. Individual trees have high variance, but their errors are uncorrelated — averaging uncorrelated errors drives variance toward zero.

**Q27** — PCA finds the directions (principal components) in feature space that explain the most variance, and projects the data onto a lower-dimensional subspace. You would apply it before training to reduce dimensionality — removing redundant or low-information features — which speeds up training, reduces memory usage, and can reduce overfitting in high-dimensional settings.

**Q31** — AUC-ROC measures the model's ability to rank a positive sample above a negative one across all possible classification thresholds — it is threshold-independent. An AUC of 0.5 means the model performs no better than random guessing (equivalent to flipping a coin). An AUC of 1.0 means the model perfectly separates all positive and negative samples at every threshold.

**Q32** — A model that always predicts "not fraud" achieves 99% accuracy by exploiting the class imbalance without detecting any fraud at all. Accuracy counts all correct predictions equally — so correct predictions on the vast majority class dominate. Instead, use **Precision** (what fraction of flagged transactions are truly fraud), **Recall** (what fraction of actual fraud was caught), **F1-Score** (harmonic mean balancing both), and **AUC-ROC** (ranking ability across thresholds) — these measure performance on the minority class that actually matters.

**Q35** — The vanishing gradient problem occurs when gradients become extremely small as they are propagated back through many layers, causing early layers to receive almost no learning signal and stop updating. Sigmoid and Tanh cause this because their derivatives are always less than 0.25 — multiplying many such small numbers together over dozens of layers produces a gradient approaching zero. ReLU avoids this because its gradient is exactly 1 for positive inputs.

**Q39** — Transfer learning uses weights trained on a large dataset (e.g., ImageNet) as a starting point for a new task, rather than training from random initialization. Use **feature extraction** (frozen backbone) when your dataset is small and visually similar to ImageNet — only the classification head is trained. Use **full fine-tuning** when you have a larger dataset or your domain is very different from ImageNet — all weights are updated, but with a much lower learning rate to preserve the pretrained features.

**Q40** — CNNs are better suited for images because: (1) **Local connectivity** — conv filters look at small local patches, exploiting the fact that meaningful patterns (edges, textures) are spatially local rather than global. An MLP ignores this and connects every input pixel to every neuron. (2) **Weight sharing** — the same filter is applied at every spatial position, so the network can detect a feature (e.g., a vertical edge) anywhere in the image without needing separate weights for each location. This reduces parameters from hundreds of millions to thousands. These two properties together give CNNs translation invariance and far better parameter efficiency than MLPs on image data.
