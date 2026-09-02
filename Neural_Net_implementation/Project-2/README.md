# Breast Cancer Classification with Neural Networks

A binary classification project that uses a feed-forward neural network built with TensorFlow/Keras to predict whether a breast tumor is **malignant** or **benign** using the Scikit-learn Breast Cancer Wisconsin dataset.

## 📌 Project Overview

This project demonstrates a complete machine learning workflow for binary classification:

1. Load the Breast Cancer Wisconsin dataset.
2. Convert the dataset into a Pandas DataFrame.
3. Separate features (`X`) and target (`y`).
4. Split the data into training and testing sets.
5. Standardize the input features.
6. Build a feed-forward neural network using TensorFlow/Keras.
7. Train the model using binary cross-entropy and the Adam optimizer.
8. Evaluate the model on unseen test data.
9. Generate binary predictions.
10. Analyze predictions using a confusion matrix.

The project is intended as a practical exercise in understanding how neural networks can be applied to a real-world binary classification problem.

---

## 🧠 Problem Statement

The goal is to classify breast tumors into two categories:

* **0 — Malignant**
* **1 — Benign**

The model receives measurements computed from digitized images of breast mass cell nuclei and learns a nonlinear relationship between these measurements and the tumor diagnosis.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset** provided by Scikit-learn through:

```python
from sklearn.datasets import load_breast_cancer
```

The dataset contains:

* **569 samples**
* **30 numerical features**
* **1 binary target variable**

The features describe characteristics of cell nuclei obtained from breast mass images.

Examples include:

* Mean radius
* Mean texture
* Mean perimeter
* Mean area
* Mean smoothness
* Mean compactness
* Mean concavity
* Mean symmetry
* Mean fractal dimension

The dataset is loaded directly through Scikit-learn, so no manual dataset download is required.

---

## 🏗️ Neural Network Architecture

The model is a fully connected feed-forward neural network:

```text
Input Layer
    │
    │ 30 features
    ▼
Dense Layer
    │
    │ 16 neurons
    │ ReLU
    ▼
Dense Layer
    │
    │ 8 neurons
    │ ReLU
    ▼
Output Layer
    │
    │ 1 neuron
    │ Sigmoid
    ▼
Binary Prediction
```

### Architecture

| Layer  | Units | Activation |
| ------ | ----: | ---------- |
| Input  |    30 | —          |
| Dense  |    16 | ReLU       |
| Dense  |     8 | ReLU       |
| Output |     1 | Sigmoid    |

The final sigmoid neuron outputs a probability between 0 and 1.

For example:

```text
0.12 → predicted class 0
0.87 → predicted class 1
```

A threshold of **0.5** is used to convert probabilities into binary predictions.

---

## ⚙️ Machine Learning Pipeline

### 1. Load the Dataset

```python
cancer = load_breast_cancer()
```

The dataset is loaded from Scikit-learn.

It is then converted into a Pandas DataFrame:

```python
df = pd.DataFrame(
    data=cancer.data,
    columns=cancer.feature_names
)

df["target"] = cancer.target
```

---

### 2. Separate Features and Target

The input features are stored in `X`, while the target labels are stored in `y`.

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Therefore:

```text
X → 30 input features
y → binary target
```

---

### 3. Train/Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The split is:

* **80% training data**
* **20% testing data**

`random_state=42` ensures reproducibility.

---

### 4. Feature Scaling

Because the input features have different numerical ranges, StandardScaler is used:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted **only on the training data** and then applied to the test data.

This prevents information from the test set from influencing the preprocessing stage.

---

## 🧮 Neural Network

The model is implemented using Keras Sequential API:

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

### ReLU Activation

The hidden layers use the Rectified Linear Unit:

$$
f(z) = \max(0,z)
$$

ReLU introduces nonlinearity, allowing the network to learn nonlinear decision boundaries.

### Sigmoid Activation

The output layer uses the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output is interpreted as a probability:

$$
0 \leq \hat{y} \leq 1
$$

---

## 📉 Loss Function

The model uses **Binary Cross-Entropy**:

```python
BinaryCrossentropy()
```

The binary cross-entropy loss is:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

where:

* $y^{(i)}$ is the actual label
* $\hat{y}^{(i)}$ is the predicted probability
* $m$ is the number of training examples

The objective is to minimize this loss during training.

---

## ⚡ Optimizer

The Adam optimizer is used:

```python
optimizer=Adam()
```

Adam combines ideas from momentum-based optimization and adaptive learning rates to efficiently update the neural network parameters.

---

## 📏 Evaluation Metric

The model uses binary accuracy:

```python
metrics=[BinaryAccuracy()]
```

Binary accuracy measures the proportion of correctly classified examples:

$$
Accuracy =
\frac{\text{Number of Correct Predictions}}
{\text{Total Number of Predictions}}
$$

---

## 🏋️ Training

The network is trained for 100 epochs:

```python
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

### Training Configuration

| Parameter        |                Value |
| ---------------- | -------------------: |
| Epochs           |                  100 |
| Batch Size       |                   32 |
| Validation Split |                  20% |
| Optimizer        |                 Adam |
| Loss             | Binary Cross-Entropy |
| Metric           |      Binary Accuracy |

The training data is further divided internally into:

```text
Training Set
├── 80% → Actual training
└── 20% → Validation
```

---

## 🧪 Model Evaluation

After training, the model is evaluated on the previously unseen test set:

```python
evaluation = model.evaluate(
    X_test_scaled,
    y_test,
    return_dict=True
)
```

This provides the final test loss and binary accuracy.

Example output structure:

```text
Test Results:
{
    'loss': ...,
    'binary_accuracy': ...
}
```

The exact values may vary slightly depending on the TensorFlow/Keras environment and training behavior.

---

## 🔮 Predictions

The trained network first produces probabilities:

```python
probabilities = model.predict(X_test_scaled)
```

The probabilities are then converted into binary predictions using a threshold of 0.5:

```python
if p >= 0.5:
    predictions.append(1)
else:
    predictions.append(0)
```

Mathematically:

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1|x) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 📊 Confusion Matrix

The project evaluates classification performance using a confusion matrix:

```python
cm = confusion_matrix(y_test, predictions)
```

The matrix has the form:

```text
                 Predicted
                0       1
Actual  0      TN      FP
        1      FN      TP
```

Where:

* **TN — True Negative:** Correctly predicted class 0
* **FP — False Positive:** Class 0 incorrectly predicted as class 1
* **FN — False Negative:** Class 1 incorrectly predicted as class 0
* **TP — True Positive:** Correctly predicted class 1

The confusion matrix provides more information than accuracy alone because it shows the types of classification errors being made.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **Scikit-learn** — Dataset, preprocessing, train/test split, and evaluation
* **TensorFlow/Keras** — Neural network implementation
* **NumPy** — Numerical operations through the underlying ML stack

---

## 📁 Project Structure

A simple repository structure can be:

```text
breast-cancer-neural-network/
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

### `main.py`

Contains the complete machine learning pipeline.

### `README.md`

Project documentation.



---

## 🎯 Learning Objectives

This project demonstrates several important concepts in machine learning and deep learning:

* Loading datasets with Scikit-learn
* Data manipulation with Pandas
* Train/test splitting
* Feature standardization
* Neural network architecture design
* Dense/fully connected layers
* ReLU activation
* Sigmoid activation
* Binary classification
* Binary cross-entropy
* Adam optimization
* Model training with Keras
* Validation data
* Model evaluation
* Probability-based predictions
* Confusion matrix analysis

---

## 📚 Key Concepts

The main learning progression demonstrated by this project is:

```text
Raw Dataset
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Neural Network
     ↓
Forward Propagation
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Adam Parameter Updates
     ↓
Trained Model
     ↓
Predictions
     ↓
Confusion Matrix
```

This makes the project a useful practical example of the complete supervised learning workflow for binary classification.

---

