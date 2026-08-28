"""
Project:
Comparing Polynomial Regression Models

Goal:
Study how different polynomial degrees affect
the regression model.

New Concepts
------------
- Polynomial Feature Engineering
- Model Complexity
- Underfitting
- Overfitting
"""

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score
)

# ==========================================================
# 2. CREATE DATASET
# ==========================================================

X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15]
])

y = np.array([
    20, 31, 45, 68, 94,
    126, 165, 207, 250, 306,
    368, 430, 510, 596, 688
])

# ==========================================================
# 3. TRAIN / TEST SPLIT
# ==========================================================

# The model should learn only from the training set.
# The testing set is reserved for evaluation.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# 4. CHOOSE POLYNOMIAL DEGREES
# ==========================================================

# We will compare several polynomial models.

degrees = [1, 2, 3, 5]

# ==========================================================
# 5. CREATE FIGURE
# ==========================================================

# One figure that will contain ALL fitted curves.

plt.figure(figsize=(10, 6))

# Plot original data

plt.scatter(X, y, color="black", label="Training Data")

# ==========================================================
# 6. TRAIN EACH MODEL
# ==========================================================

for degree in degrees:

    print("=" * 60)
    print(f"Polynomial Degree = {degree}")
    print("=" * 60)
    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------
    # PolynomialFeatures is a TRANSFORMER.
    # It creates new polynomial features.
    # Example:
    # x
    # becomes
    # x, x², x³ ...
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    # fit_transform()
    # Creates polynomial features
    # for the training data.
    '''fit() means Learn something from data.'''
    X_train_poly = poly.fit_transform(X_train)
    '''transform() Use what was already configured to transform new data.'''
    # Uses the SAME transformation
    # on the testing data.
    X_test_poly = poly.transform(X_test)
    ''' takes x and converts it into x, x², x³ depending on the chosen degree.
    No learning happens here. '''
    # ------------------------------------------------------
    # Model Training
    # ------------------------------------------------------
    model = LinearRegression()
    # fit()
    #
    # Learns the coefficients
    # and intercept.
    model.fit(X_train_poly, y_train)
    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------
    y_pred = model.predict(X_test_poly)
    # predict()
    #
    # Uses the learned model
    # to estimate unseen outputs.
    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    print("Coefficients:")
    print(model.coef_)

    print("\nIntercept:")
    print(model.intercept_)

    print("\nMAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("R²  :", r2_score(y_test, y_pred))

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------

    # Generate many x-values so that
    # the regression curve looks smooth.

    X_curve = np.linspace(X.min(), X.max(),300).reshape(-1, 1)
    # Apply the SAME feature engineering.

    X_curve_poly = poly.transform(X_curve)

    # Predict every point on the curve.

    y_curve = model.predict(X_curve_poly)

    plt.plot(X_curve, y_curve, label=f"Degree {degree}")

# ==========================================================
# 7. FINAL GRAPH
# ==========================================================

plt.title("Polynomial Regression Comparison")

plt.xlabel("Study Hours")

plt.ylabel("Exam Score")

plt.legend()

plt.grid(True)

plt.show()