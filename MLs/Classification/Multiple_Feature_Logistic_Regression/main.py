"""
====================================================================
Project: Multiple Feature Logistic Regression (Binary Classification)

Goal:
Predict whether a student will PASS or FAIL based on:

1. Study Hours
2. Attendance Percentage
3. Number of Assignments Completed

Target Classes

0 -> Fail
1 -> Pass

This project demonstrates the standard Scikit-learn workflow for
binary classification.
====================================================================
"""

# ================================================================
# Import Required Libraries
# ================================================================

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ================================================================
# Step 1 : Create the Dataset
# ================================================================
#
# Each row represents ONE student.
#
# Features:
#
# Column 1 -> Study Hours
# Column 2 -> Attendance Percentage
# Column 3 -> Assignments Completed
#
# Shape of X:
#
# (Number of Students, Number of Features)
#
# In this project:
#
# 20 students
# 3 features

X = np.array([
    [1.0, 45, 2],
    [2.0, 55, 3],
    [2.5, 60, 4],
    [3.0, 58, 5],
    [3.5, 65, 5],
    [4.0, 70, 6],
    [4.5, 72, 6],
    [5.0, 75, 7],
    [5.5, 78, 7],
    [6.0, 80, 8],
    [6.5, 82, 8],
    [7.0, 85, 9],
    [7.5, 88, 9],
    [8.0, 90, 10],
    [8.5, 92, 10],
    [9.0, 95, 10],
    [4.0, 85, 9],
    [7.5, 60, 4],
    [5.0, 65, 4],
    [3.0, 90, 9]
])

# Target labels
#
# 0 -> Fail
# 1 -> Pass

y = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1])

# ================================================================
# Step 2 : Split the Dataset
# ================================================================
#
# The training set is used to teach the model.
#
# The testing set is kept separate and is only used to
# evaluate how well the model generalizes to unseen data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ================================================================
# Step 3 : Feature Scaling
# ================================================================
#
# Logistic Regression performs better when features have
# similar numerical scales.
#
# Since Study Hours, Attendance and Assignments are measured
# in different units, we standardize them.
#
# StandardScaler transforms every feature into:
#
# Mean = 0
# Standard Deviation = 1
#
# IMPORTANT
#
# The scaler is fitted ONLY on the training data.
#
# The test data and future data are transformed using
# the SAME scaler to prevent data leakage.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ================================================================
# Step 4 : Train the Logistic Regression Model
# ================================================================
#
# Logistic Regression first computes
#
# z = w1x1 + w2x2 + ... + wnxn + b
#
# It then passes z through the Sigmoid Function
#
# Probability = 1 / (1 + e^(-z))
#
# If Probability >= 0.5
#
# Predict PASS
#
# Otherwise
#
# Predict FAIL

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# ================================================================
# Step 5 : Inspect the Learned Parameters
# ================================================================
#
# coef_
#
# One coefficient is learned for EACH feature.
#
# Since this dataset has three features,
# three coefficients are learned.
#
# Positive coefficient
#
# Increasing that feature increases the probability
# of passing.
#
# Negative coefficient
#
# Increasing that feature decreases the probability
# of passing.

feature_names = [
    "Study Hours",
    "Attendance",
    "Assignments"
]

print("\n========== Learned Parameters ==========")

for name, coef in zip(feature_names, model.coef_[0]):
    print(f"{name:<15}: {coef:.3f}")

print(f"Intercept      : {model.intercept_[0]:.3f}")

# ================================================================
# Step 6 : Predict the Test Set
# ================================================================
#
# predict()
#
# Returns the predicted class.
#
# 0 -> Fail
# 1 -> Pass

y_pred = model.predict(X_test_scaled)

# ================================================================
# Step 7 : Predict Probabilities
# ================================================================
#
# predict_proba()
#
# Returns the probability of every class.
#
# Column 0
#
# Probability of FAIL
#
# Column 1
#
# Probability of PASS

y_prob = model.predict_proba(X_test_scaled)

print("\n========== Test Predictions ==========")

print("Actual Labels      :", y_test)

print("Predicted Labels   :", y_pred)

print()

print("Prediction Probabilities")

print(y_prob)


# ================================================================
# Step 8 : Evaluate the Model
# ================================================================
#
# Accuracy
#
# Percentage of predictions that are correct.
#
# Formula
#
# Correct Predictions / Total Predictions

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)


# ================================================================
# Confusion Matrix
# ================================================================
#
# Shows HOW the model made its predictions.
#
#                Predicted
#
#              Fail   Pass
#
# Actual Fail   TN      FP
#
# Actual Pass   FN      TP

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")

print(cm)


# ================================================================
# Step 9 : Predict New Students
# ================================================================
#
# IMPORTANT
#
# New data MUST be transformed using the SAME scaler.
#
# Never call fit_transform() on new data.
#
# Always use transform().

new_students = np.array([
    [0, 0, 0],
    [9, 99, 8],
    [2, 98, 9]
])

new_students_scaled = scaler.transform(new_students)

predictions = model.predict(new_students_scaled)

probabilities = model.predict_proba(new_students_scaled)

print("\n========== New Student Predictions ==========")

for student, prediction, probability in zip(new_students,predictions,probabilities):

    print(f"\nStudent Features : {student}")

    print(f"Probability Fail : {probability[0]:.3f}")

    print(f"Probability Pass : {probability[1]:.3f}")

    print(f"Prediction       : {'Pass' if prediction == 1 else 'Fail'}")