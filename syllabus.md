**Modern Artificial Intelligence & Deep Learning**

Target Audience: Sabah group Bachelor 3rd Year (Computer Engineering & Information
Security)

Prerequisites: Linear Algebra, Calculus (Derivatives), Proficiency in Python.

Tools: Python, NumPy, Pandas, Scikit-learn, PyTorch (or TensorFlow).

**1. Course Description**

This course provides a rigorous introduction to modern Artificial Intelligence with a primary
focus on Machine Learning (ML) and Deep Learning (DL). Unlike traditional AI surveys, this
curriculum accelerates through classical algorithms to focus on the statistical and
computational methods used in modern industry. Students will move from mathematical
foundations (optimization and probability) to implementing production-grade Machine
Learning models and state-of-the-art Neural Networks, including Transformers and
Generative AI.

**2. Learning Objectives**

By the end of this course, students will be able to:

1. **Formulate** AI problems as optimization tasks and solve them using Gradient
    Descent.
2. **Engineer** data pipelines to clean, normalize, and prepare real-world datasets for
    training.
3. **Implement** and deploy classical ML models (Regression, SVM, Random Forests)
    using Scikit-learn.
4. **Architect** Deep Neural Networks (CNNs, RNNs, Transformers) using
    PyTorch/TensorFlow.
5. **Evaluate** model performance using industry metrics (F1-Score, AUC-ROC) and
    detect overfitting.


**3. Weekly Schedule**

**Module 1: Algorithmic Foundations (Weeks 1-3)**

```
Week Topic Key Concepts
```
```
Lab / Practical
Assignment
```
## 1

```
Optimization &
Search
```
```
The "Learning" in ML. Hill
Climbing vs. Gradient
Descent. Learning Rates.
Convex vs. Non-convex
functions.
```
```
Visualizing
Optimization:
Implement Gradient
Descent from scratch to
find the minimum of a
given function.
```
## 2

```
Game Theory &
Decision
Making
```
```
Adversarial environments.
Minimax Algorithm. Alpha-
Beta Pruning. Introduction
to Agents.
```
```
Bot Battle: Create a
Python agent that plays
Tic-Tac-Toe or Connect- 4
against a human
opponent.
```
## 3

```
Data
Engineering for
AI
```
```
Probability review (Bayes
Theorem). Data
Preprocessing:
Normalization, One-Hot
Encoding, Handling
missing data.
```
```
Data Cleaning: Using
Pandas to clean a "dirty"
dataset (e.g., system logs
or broken CSVs) before
training.
```

**Module 2: Applied Machine Learning (Weeks 4-8)**

```
Week Topic Key Concepts Lab / Practical Assignment
```
## 4

```
Regression
Analysis
```
```
Linear Regression. Loss
Functions (MSE). The
concept of Weights and
Biases.
```
```
Predicting Metrics:
Predict CPU
temperature or
Housing Prices based
on historical data
features.
```
## 5

```
Classification
Algorithms
```
```
Logistic Regression. Support
Vector Machines (SVM).
Decision Boundaries. The
Sigmoid function.
```
```
Binary Classification:
Predicting "Pass/Fail"
or "Spam/Ham" on a
dataset.
```
## 6

```
Ensemble
Methods
```
```
Decision Trees. Random
Forests. Gradient Boosting
(XGBoost/LightGBM). Bias-
Variance Tradeoff.
```
```
Fraud Detection:
Using XGBoost to
identify fraudulent
transactions in a
highly imbalanced
dataset.
```
## 7

```
Unsupervised
Learning
```
```
Clustering (K-Means,
DBSCAN). Dimensionality
Reduction (PCA). Anomaly
Detection concepts.
```
```
Customer
Segmentation:
Grouping users by
behavior without
labeled data.
```
```
8 Model Eval &
Deployment
```
```
Overfitting/Underfitting.
Cross-Validation (K-Fold).
Precision, Recall, F1-Score.
API Deployment.
```
```
Model-as-a-Service:
Wrap a trained Scikit-
learn model in a
Streamlit wrapper to
```

**Week Topic Key Concepts Lab / Practical Assignment**

```
serve predictions via
HTTP.
```

**Module 3: Deep Learning & Neural Networks (Weeks 9-13)**

```
Week Topic Key Concepts Lab / Practical Assignment
```
## 9

```
Neural
Networks
Intuition
```
```
The Multi-Layer
Perceptron (MLP).
Activation Functions
(ReLU, Softmax).
Backpropagation &
Chain Rule.
```
```
Numpy Only: Build a
simple Neural Network
without PyTorch/TensorFlow
to classify handwritten
digits (MNIST).
```
```
10 Computer
Vision (CNNs)
```
```
Convolutions, Kernels,
Padding, Pooling. CNN
Architectures (ResNet,
VGG). Transfer Learning.
```
```
Image Classifier: Fine-
tune a pre-trained ResNet
model to classify specific
objects (e.g., Cars vs.
Bikes).
```
## 11

```
Sequence
Models
(RNNs)
```
```
Recurrent Neural
Networks. Vanishing
Gradient Problem.
LSTMs and GRUs. Time-
series data.
```
```
Forecasting: Predict the
next character in a text
sequence or next value in a
stock price chart.
```
```
12 Transformers
& LLMs
```
```
The Attention
Mechanism ("Attention
is all you need").
Encoder-Decoder
architecture. BERT &
GPT overview.
```
```
Sentiment Analysis: Fine-
tuning a BERT model to
analyze the sentiment of
movie reviews.
```
```
13 Generative AI
```
```
Autoencoders.
Generative Adversarial
Networks (GANs).
```
```
Image Generation: Train a
simple GAN to generate
fake handwritten digits or
anime faces.
```

**Week Topic Key Concepts Lab / Practical Assignment**

```
Diffusion Models
intuition.
```

**Module 4: Advanced Applications (Weeks 14-15)**

```
Week Topic Key Concepts
```
```
Lab / Practical
Assignment
```
```
14 Reinforcement
Learning
```
```
Agents, Environments,
Rewards. Q-Learning.
Exploration vs.
Exploitation. Deep Q-
Networks (DQN).
```
```
OpenAI Gym: Train an
agent to balance a pole
(CartPole) or land a
Lunar Lander.
```
```
15 Ethics & Future
Trends
```
```
Algorithmic Bias.
Explainable AI (SHAP
values). AI Security
(Adversarial Attacks). Final
Presentations.
```
```
Project Showcase:
Group presentations of
final Capstone
Projects.
```

**4. Grading Breakdown**

```
Component Weight Description
```
```
Lab
Assignments 40%^ Weekly/Bi-weekly coding tasks (Python/Jupyter).^
```
```
Midterm Exam 25%
```
```
Theory validation: Math behind Gradient Descent,
calculating Loss, Probability, Search Trees.
```
```
Final Project 35%
```
```
A comprehensive group project building a full ML
pipeline (Data to Model to Deploy).
```
**5. Recommended Resources**

**Textbooks:**

- _Primary:_ **"Hands-On Machine Learning with Scikit-Learn and Pytorch"** by
    Aurélien Géron (Practical).
- _Secondary:_ **"Deep Learning"** by Goodfellow, Bengio, and Courville
    (Theoretical/Math-heavy).

**Tools & Datasets:**

- **Kaggle:** For datasets and notebook environments.
- **Google Colab:** For free GPU access (essential for Weeks 9-13).
- **Hugging Face:** For Transformer models (Week 12).


