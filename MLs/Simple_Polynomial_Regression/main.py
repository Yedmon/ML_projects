"""
Project:
Polynomial Regression (Degree 2)

Goal:
Predict exam scores from study hours using
Linear Regression with Polynomial Features.

Concepts:
- Feature Engineering
- PolynomialFeatures
- Linear Regression
- Train/Test Split
- Model Evaluation
"""

# ======================================================
# Imports
# ======================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

# ======================================================
# Dataset
# ======================================================

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

y = np.array([18, 30, 47, 69, 96, 127, 164, 205, 252, 304])

# ======================================================
# Train/Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# Polynomial Feature Engineering
# ======================================================

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# ======================================================
# Model Training
# ======================================================

model = LinearRegression()

model.fit(X_train_poly, y_train)

print("Learned Coefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

# ======================================================
# Prediction
# ======================================================

y_pred = model.predict(X_test_poly)

print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

# ======================================================
# Evaluation
# ======================================================

print("\nEvaluation Metrics")
print("------------------")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"MSE : {mean_squared_error(y_test, y_pred):.3f}")
print(f"R²  : {r2_score(y_test, y_pred):.6f}")

# ======================================================
# Predict a New Student
# ======================================================

new_student = np.array([[7.5]])

new_student_poly = poly_features.transform(new_student)

predicted_score = model.predict(new_student_poly)

print(f"\nPredicted Score for 7.5 Study Hours: {predicted_score[0]:.2f}")

# ======================================================
# Visualization
# ======================================================

# Generate many x-values for a smooth curve
X_curve = np.linspace(1, 10, 200).reshape(-1, 1)

X_curve_poly = poly_features.transform(X_curve)

y_curve = model.predict(X_curve_poly)

plt.figure(figsize=(8, 5))

# Original data
plt.scatter(X, y, label="Training Data")

# Polynomial regression curve
plt.plot(X_curve, y_curve, label="Polynomial Regression")

plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()

plt.show()