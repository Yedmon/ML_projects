# Breast Cancer Classification using Logistic Regression (Scikit-learn)

## Project Overview

This project demonstrates how to build a **Binary Classification** model using **Logistic Regression** with the **Breast Cancer Wisconsin Dataset** provided by Scikit-learn.

The objective is to predict whether a breast tumor is:

* **Malignant** (Cancerous)
* **Benign** (Non-cancerous)

based on 30 numerical features extracted from digitized images of breast cell nuclei.

This project was built as part of my Scikit-learn learning journey to understand the complete machine learning workflow for classification problems.

---

# Dataset

This project uses the built-in **Breast Cancer Wisconsin Dataset** from Scikit-learn.

Dataset characteristics:

* **569 samples (patients)**
* **30 numerical features**
* **Binary classification**
* **No missing values**
* **Well-suited for beginners learning classification**

Target Classes:

| Label | Meaning   |
| ----- | --------- |
| 0     | Malignant |
| 1     | Benign    |

---

# Project Workflow

The project follows the standard Scikit-learn machine learning pipeline:

```text
Load Dataset
      ↓
Create Pandas DataFrame
      ↓
Separate Features and Target
      ↓
Train/Test Split
      ↓
Feature Scaling (StandardScaler)
      ↓
Train Logistic Regression Model
      ↓
Make Predictions
      ↓
Predict Class Probabilities
      ↓
Evaluate Model
      ↓
Predict New Samples
```

---

# Concepts Practiced

This project helped reinforce the following machine learning concepts:

* Binary Classification
* Logistic Regression
* Train/Test Split
* Feature Scaling using `StandardScaler`
* Model Training with Scikit-learn
* Model Coefficients and Intercept
* Class Prediction
* Prediction Probabilities (`predict_proba()`)
* Accuracy Score
* Confusion Matrix
* Predicting New Samples

---

# Libraries Used

* NumPy
* Pandas
* Scikit-learn

---

# Evaluation Metrics

The model is evaluated using:

* **Accuracy Score**

  * Measures the percentage of correctly classified samples.

* **Confusion Matrix**

  * Shows the number of:

    * True Positives
    * True Negatives
    * False Positives
    * False Negatives

* **Prediction Probabilities**

  * Shows the probability that each sample belongs to each class.

---

# What I Learned

Through this project, I learned how to:

* Work with a real dataset provided by Scikit-learn.
* Convert dataset objects into Pandas DataFrames.
* Separate features and target variables.
* Apply feature scaling correctly without data leakage.
* Train and evaluate a Logistic Regression model.
* Interpret model coefficients and prediction probabilities.
* Evaluate classification performance using accuracy and the confusion matrix.
* Predict outcomes for new, unseen samples using the trained model.

---
