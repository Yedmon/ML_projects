# Student Pass Prediction — Neural Network with TensorFlow

A simple binary classification project that uses a **neural network built with TensorFlow/Keras** to predict whether a student will **pass or fail** based on their study hours and sleep hours.

The project uses a synthetically generated dataset and demonstrates a complete basic machine learning workflow: data generation, visualization, train/test splitting, neural network construction, training, evaluation, and prediction.

---

## 📌 Project Overview

The goal of this project is to build a neural network that learns the relationship between:

* **Study Hours**
* **Sleep Hours**

and predicts:

* `0` → Fail
* `1` → Pass

The synthetic target is generated using the following relationship:

$$
\text{score} = 0.7 \cdot \text{study_hours} + 0.3 \cdot \text{sleep_hours} + \text{noise}
$$

A student is labeled as passing when:

$$
y =
\begin{cases}
1 & \text{if score} > 5 \\
0 & \text{otherwise}
\end{cases}
$$

The neural network then learns this relationship from the generated training data.

---

## 🧠 Machine Learning Approach

This is a **binary classification** problem.

The neural network consists of:

```text
Input Layer
    ↓
Dense Layer — 4 neurons, ReLU
    ↓
Output Layer — 1 neuron, Sigmoid
    ↓
Pass / Fail
```

### Model Architecture

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

### Why these activations?

#### ReLU

The hidden layer uses the Rectified Linear Unit:

$$
\text{ReLU}(z) = \max(0,z)
$$

ReLU introduces non-linearity, allowing the network to learn relationships that are more complex than a simple linear model.

#### Sigmoid

The output layer uses sigmoid:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This produces a value between 0 and 1, which can be interpreted as the probability that the student will pass.

For example:

```text
0.92 → 92% probability of passing
0.18 → 18% probability of passing
```

The prediction is converted into a class using a threshold of `0.5`:

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 📊 Dataset

The project generates **500 synthetic student samples**.

Each sample contains two features:

| Feature     | Description          | Range |
| ----------- | -------------------- | ----- |
| Study Hours | Hours spent studying | 0–10  |
| Sleep Hours | Hours of sleep       | 4–9   |

The dataset is generated using NumPy.

### Example

```text
Study Hours    Sleep Hours    Result
-------------------------------------
2.5            5.8            Fail
7.2            7.1            Pass
4.1            6.3            Fail
8.7            8.2            Pass
```

The labels are not manually assigned. They are generated from the underlying mathematical relationship plus random noise.

---

## 🔀 Train/Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This gives approximately:

```text
Training samples: 400
Testing samples: 100
```

### Why use `stratify=y`?

`stratify=y` helps preserve approximately the same proportion of passing and failing students in both the training and testing sets.

---

## ⚙️ Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[BinaryAccuracy(name="accuracy")]
)
```

### Optimizer

**Adam** is used to update the neural network's weights during training.

### Loss Function

Because this is binary classification, the model uses **Binary Cross-Entropy**:

$$
L = -\left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]
$$

where:

* $y$ = actual label
* $\hat{y}$ = predicted probability

The training process attempts to minimize this loss.

---

## 🚀 Training

The model is trained for **80 epochs**:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    verbose=1
)
```

During training, TensorFlow updates the network's weights to reduce the binary cross-entropy loss.

The training history is stored in:

```python
history.history
```

This allows the training loss and accuracy to be visualized after training.

---

## 📈 Training Visualization

The project generates two plots.

### 1. Training Loss

The loss plot shows how the binary cross-entropy loss changes during training.

Ideally:

```text
Loss
 │\
 │ \
 │  \
 │   \____
 │
 └──────────── Epochs
```

A decreasing loss generally indicates that the model is learning from the training data.

### 2. Training Accuracy

The accuracy plot shows the percentage of training examples classified correctly during each epoch.

---

## 🧪 Model Evaluation

After training, the model is evaluated on previously unseen test data:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
```

The project reports:

```text
Test Loss: ...
Test Accuracy: ...
```

The **test set** is important because it provides an estimate of how well the trained model generalizes to data that it did not see during training.

---

## 🔮 Making Predictions

The model can also predict outcomes for new students.

Example:

```python
new_study_hours = np.array([2, 6, 8])
new_sleep_hours = np.array([5, 7, 8])
```

These values are combined into:

```text
Student 1 → 2 hours study, 5 hours sleep
Student 2 → 6 hours study, 7 hours sleep
Student 3 → 8 hours study, 8 hours sleep
```

The model first produces probabilities:

```python
probabilities = model.predict(X_new)
```

For example:

```text
0.12
0.68
0.95
```

These probabilities are then converted into class predictions:

```python
predictions = (probabilities >= 0.5).astype(int)
```

Result:

```text
0 → Fail
1 → Pass
```

---

## 🛠️ Tools Used

* **Python**
* **NumPy** 
* **Matplotlib** 
* **Scikit-learn** 
* **TensorFlow**
* **Keras** — neural network construction and training

---

## 📁 Project Structure

```text
student-pass-prediction/
│
├── student_pass_prediction.py
├── README.md
├── requirements.txt
└── .gitignore
```

A `requirements.txt` file can contain:

```text
numpy
matplotlib
scikit-learn
tensorflow
```

---

## 💡 Key Concepts Demonstrated

This project provides hands-on practice with several fundamental deep learning concepts:

* Synthetic dataset generation
* Feature matrices and target vectors
* Binary classification
* Train/test splitting
* Neural network architecture
* Dense/Fully Connected layers
* ReLU activation
* Sigmoid activation
* Binary cross-entropy
* Adam optimization
* Model training
* Epochs
* Training loss
* Training accuracy
* Model evaluation
* Probability-based predictions
* Classification thresholds
* TensorFlow/Keras workflow

---
