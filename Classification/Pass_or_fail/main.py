"""
====================================================
Project: Binary Classification with Logistic Regression
Goal: Predict whether a student passes an exam
      based on the number of study hours.
====================================================
"""

# ============================
# Import Required Libraries
# ============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================================
# Step 1 : Create the Dataset
# ==========================================================
# X contains the input feature(s).
# Each row represents one training example.
#
# Shape = (number_of_samples, number_of_features)
#
# Here:
# 10 students
# 1 feature (Study Hours)

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# Target labels
#
# 0 = Fail
# 1 = Pass

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1
])


# ==========================================================
# Step 2 : Explore the Dataset
# ==========================================================

print("========== Dataset Information ==========")
print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)
print()


# ==========================================================
# Step 3 : Visualize the Dataset
# ==========================================================
# Unlike Linear Regression,
# our outputs are categorical (0 or 1).
#
# We therefore visualize the data as two classes.

plt.figure(figsize=(8,5))

plt.scatter(
    X,
    y,
    color="royalblue",
    s=80
)

plt.xlabel("Study Hours")
plt.ylabel("Pass (1) / Fail (0)")
plt.title("Student Exam Dataset")
plt.grid(True)

plt.show()


# ==========================================================
# Step 4 : Split the Dataset
# ==========================================================
#
# Training data:
# Used for learning the relationship.
#
# Testing data:
# Used only to evaluate the model.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# Step 5 : Feature Scaling
# ==========================================================
#
# Logistic Regression performs better when features
# have similar scales.
#
# IMPORTANT:
#
# Fit ONLY on the training data.
#
# Never fit on the testing data because that would
# leak information from the test set into training.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================================
# Step 6 : Train the Logistic Regression Model
# ==========================================================
#
# Logistic Regression learns:
#
#     z = wx + b
#
# Then transforms it into a probability using
# the Sigmoid Function:
#
#     P = 1 / (1 + e^(-z))
#
# The model predicts:
#
# Pass if Probability >= 0.5
# Fail otherwise.

model = LogisticRegression()

model.fit(X_train_scaled, y_train)


# ==========================================================
# Step 7 : Inspect the Learned Parameters
# ==========================================================
#
# coef_
# ------
# Weight assigned to each feature.
#
# A positive coefficient means:
# Increasing study hours increases the probability
# of passing.
#
# intercept_
# ----------
# Bias term.

print("========== Learned Parameters ==========")
print("Coefficient :", model.coef_)
print("Intercept   :", model.intercept_)
print()


# ==========================================================
# Step 8 : Make Predictions
# ==========================================================
#
# predict()
#
# Returns the predicted class
#
# 0 = Fail
# 1 = Pass

y_pred = model.predict(X_test_scaled)


# ==========================================================
# Step 9 : Predict Probabilities
# ==========================================================
#
# predict_proba()
#
# Returns TWO probabilities.
#
# Column 0:
# Probability of class 0 (Fail)
#
# Column 1:
# Probability of class 1 (Pass)

probabilities = model.predict_proba(X_test_scaled)


print("========== Test Predictions ==========")

print("Actual Labels    :", y_test)
print("Predicted Labels :", y_pred)

print()

print("Prediction Probabilities")
print(probabilities)

print()


# ==========================================================
# Step 10 : Evaluate the Model
# ==========================================================
#
# Accuracy
#
# Percentage of correct predictions.

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print()

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))


# ==========================================================
# Step 11 : Predict New Students
# ==========================================================

new_students = np.array([
    [2],
    [4.5],
    [6],
    [9]
])

# Scale using the SAME scaler learned from
# the training data.

new_students_scaled = scaler.transform(new_students)

predictions = model.predict(new_students_scaled)

prediction_probabilities = model.predict_proba(
    new_students_scaled
)

