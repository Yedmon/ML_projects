"""
Project: Multiple Linear Regression with Feature Scaling

Goal:
Predict house prices using multiple features.

Features (X):
- Size (m²)
- Number of Bedrooms
- Age of House (years)

Target (y):
- House Price (in thousands of dollars)

New Concepts:
1. Multiple Linear Regression
2. Train/Test Split
3. Feature Scaling using StandardScaler
4. Model Training
5. Prediction
6. Model Evaluation
"""

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import numpy as np
from sklearn.model_selection import train_test_split
# Standardizes numerical features
from sklearn.preprocessing import StandardScaler
# Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
# Regression evaluation metrics
from sklearn.metrics import (mean_absolute_error,mean_squared_error, r2_score)

# ==========================================================
# 2. CREATE THE DATASET
# ==========================================================
# X contains the INPUT FEATURES.
#
# Shape:
# (number_of_samples, number_of_features)
#
# Here:
# 10 samples(examples)
# 3 features
#
# Feature 1 -> House Size
# Feature 2 -> Number of Bedrooms
# Feature 3 -> House Age

X = np.array([
    [50, 1, 20],
    [65, 2, 15],
    [80, 2, 10],
    [95, 3, 8],
    [110, 3, 5],
    [125, 4, 4],
    [140, 4, 3],
    [160, 5, 2],
    [180, 5, 1],
    [200, 6, 1]
])

# y contains the TARGET values.
#
# Each value corresponds to the house price
# of the matching row in X.

y = np.array([
    189,
    220,
    260,
    300,
    340,
    390,
    430,
    500,
    560,
    620
])


# ==========================================================
# 3. SPLIT THE DATASET
# ==========================================================

# WHY SPLIT?
#
# We want to evaluate the model on data
# that it has NEVER seen before.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ==========================================================
# 4. FEATURE SCALING
# ==========================================================
# StandardScaler is a TRANSFORMER.
#
# Unlike LinearRegression,
# it DOES NOT learn to predict.
#
# Instead it learns how to transform
# the feature values.
#
# Specifically it learns:
#
# Mean of every feature
# Standard deviation of every feature
#
# Then each feature is standardized using
#
# z = (x - mean) / standard deviation
#
# This makes every feature approximately:
#
# Mean = 0
# Standard Deviation = 1

scaler = StandardScaler()

# ----------------------------------------------------------
# fit_transform()
# ----------------------------------------------------------
#
# fit()
#   Learn the mean and standard deviation
#   from the TRAINING DATA ONLY.
#
# transform()
#   Apply that scaling.
#
# fit_transform()
# simply combines both operations.

X_train_scaled = scaler.fit_transform(X_train)

# ----------------------------------------------------------
# transform()
# ----------------------------------------------------------
#
# DO NOT call fit() here.
#
# We DO NOT want the scaler to learn
# anything from the test set.
#
# We simply reuse the scaling learned
# from the training data.
#
# This prevents DATA LEAKAGE.

X_test_scaled = scaler.transform(X_test)


# ==========================================================
# 5. CREATE THE MODEL
# ==========================================================

# LinearRegression is an ESTIMATOR.
#
# Estimators learn relationships
# between features (X)
# and targets (y).

model = LinearRegression()


# ==========================================================
# 6. TRAIN THE MODEL
# ==========================================================
# model.fit()
#
# The regression model now learns:
# Weight (coefficient) for every feature and Intercept (bias)
# Since there are three features, we expect three learned coefficients.

model.fit(X_train_scaled, y_train)
# ==========================================================
# 7. INSPECT THE LEARNED PARAMETERS
# ==========================================================

print("Learned Weights:")
print(model.coef_)

print()

print("Learned Bias:")
print(model.intercept_)
# ==========================================================
# 8. MAKE PREDICTIONS
# ==========================================================

# The model predicts prices
# for houses it has never seen.

y_pred = model.predict(X_test_scaled)

print("\nActual Prices:")
print(y_test)

print("\nPredicted Prices:")
print(y_pred)


# ==========================================================
# 9. EVALUATE THE MODEL
# ==========================================================

print("\nMean Absolute Error")
print(mean_absolute_error(y_test, y_pred))


print("\nMean Squared Error")
print(mean_squared_error(y_test, y_pred))

print("\nR² Score")
print(r2_score(y_test, y_pred))


# ==========================================================
# 10. PREDICT A NEW HOUSE
# ==========================================================

# IMPORTANT
#
# New data MUST go through
# the SAME preprocessing pipeline.
#
# Never send raw values directly to the model if the model was trained on scaled features.

new_house = np.array([
    [150, 4, 5]
])

# Use ONLY transform().
#
# Never fit() on new data.

new_house_scaled = scaler.transform(new_house)
prediction = model.predict(new_house_scaled)

print("\nPredicted Price for New House:")
print(prediction)