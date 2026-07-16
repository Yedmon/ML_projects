"""
================================================================================
Project: California Housing Price Prediction using Linear Regression
================================================================================

Goal
----
Predict the median house value of California districts using several numerical
features such as income, house age, number of rooms, population, and location.

Machine Learning Problem
------------------------
This is a Supervised Learning - Regression problem because:

    Input  (X): Numerical features describing each district.
    Output (y): Median house value (continuous numerical value).

Workflow
--------
1. Import libraries
2. Load dataset
3. Prepare features and target
4. Split into training and testing sets
5. Scale the features
6. Train a Linear Regression model
7. Inspect learned parameters
8. Make predictions
9. Evaluate the model
10. Analyze prediction errors (Residual Analysis)
11. Visualize model performance
================================================================================
"""

# ==============================================================================
# 1. Import Required Libraries
# ==============================================================================

# Numerical computations
import numpy as np

# Data manipulation
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt

# Built-in California Housing dataset
from sklearn.datasets import fetch_california_housing

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Standardize numerical features
from sklearn.preprocessing import StandardScaler

# Linear Regression model
from sklearn.linear_model import LinearRegression

# Regression evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==============================================================================
# 2. Load the Dataset
# ==============================================================================

"""
The `fetch_california_housing()` function loads the California Housing dataset
provided by Scikit-learn.

It returns a `Bunch` object, which is similar to a Python dictionary and
contains several pieces of information about the dataset, including:

    • data          -> Feature matrix (input variables)
    • target        -> Target values (house prices)
    • feature_names -> Names of the feature columns
    • DESCR         -> Description of the dataset
    • target_names  -> Name(s) of the target variable

The feature matrix (`housing.data`) is stored as a NumPy array, meaning it
contains only numerical values and has no column labels.

Example (housing.data):

    [
        [8.3252, 41.0, 6.98, ...],
        [8.3014, 21.0, 6.23, ...],
        ...
    ]

The feature names are stored separately in:

    housing.feature_names

which contains:

    [
        'MedInc',
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup',
        'Latitude',
        'Longitude'
    ]

Since NumPy arrays do not have column names, we convert the feature matrix into
a Pandas DataFrame. This creates a labeled table that is much easier to inspect,
analyze, and manipulate.

`housing.data` becomes the rows of the DataFrame, while
`housing.feature_names` becomes the column headers.
"""

# Load the dataset
housing = fetch_california_housing()

# Convert the feature matrix (NumPy array) into a Pandas DataFrame
# using the feature names as column labels.
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add the target values (median house price) as a new column.
# This combines both the features and target into one DataFrame,
# making the dataset easier to explore before separating it into
# X (features) and y (target) for model training.
df["MedHouseValue"] = housing.target

print("=" * 70)
print("First Five Rows")
print("=" * 70)
print(df.head())

# ==============================================================================
# 3. Separate Features (X) and Target (y)
# ==============================================================================

"""
Features (X)
------------
These are the variables used to predict house prices.

Target (y)
----------
This is the value we want the model to learn.

Mathematically,

            X  ----->  y

The model learns a function

            f(X) = y
"""

X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape         :", y.shape)

# ==============================================================================
# 4. Split the Dataset
# ==============================================================================

"""
The model must NOT be evaluated on the same data it was trained on.

Training Set
------------
Used for learning the model parameters.

Testing Set
-----------
Used ONLY to evaluate model performance on unseen data.

80% -> Training
20% -> Testing
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==============================================================================
# 5. Feature Scaling
# ==============================================================================

"""
Different features have different scales.

Example

Population  : 3  -> 35682
Latitude    : 32 -> 42
Income       : 0.5 -> 15

StandardScaler standardizes every feature using

        z = (x - mean) / standard deviation

After scaling:

Mean ≈ 0
Standard Deviation ≈ 1

IMPORTANT

Only the TRAINING data is used to compute the mean and standard deviation.

This prevents DATA LEAKAGE.
"""

scaler = StandardScaler()

# Learn scaling parameters from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to testing data
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 6. Train the Linear Regression Model
# ==============================================================================

"""
Linear Regression learns

ŷ = w₁x₁ + w₂x₂ + ... + w₈x₈ + b

where

w = coefficients
b = intercept

The model chooses the values of w and b that minimize
the sum of squared prediction errors.
"""

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# ==============================================================================
# 7. Inspect the Learned Model
# ==============================================================================

"""
Each coefficient tells us how much the predicted house value changes
for a one standard deviation increase in that feature,
while holding all other features constant.

Positive coefficient
    Increasing the feature tends to increase the prediction.

Negative coefficient
    Increasing the feature tends to decrease the prediction.
"""

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n")
print("=" * 70)
print("Learned Coefficients")
print("=" * 70)
print(coefficients)

print("\nIntercept:", model.intercept_)

# ==============================================================================
# 8. Make Predictions
# ==============================================================================

"""
The trained model predicts house prices for the unseen testing data.
"""

y_pred = model.predict(X_test_scaled)

# ==============================================================================
# 9. Evaluate the Model
# ==============================================================================

"""
Regression Metrics

MAE
----
Average prediction error.

MSE
----
Average squared prediction error.

RMSE
-----
Square root of MSE.
Expressed in the same units as the target.

R²
---
Measures how much of the variation in the target
is explained by the model.

R² = 1  -> Perfect prediction

R² = 0  -> Same as predicting the mean

R² < 0  -> Worse than predicting the mean
"""

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n")
print("=" * 70)
print("Model Evaluation")
print("=" * 70)

print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# ==============================================================================
# 10. Compare Actual vs Predicted Values
# ==============================================================================

"""
Residual = Actual - Predicted

Positive residual
-----------------
The model underestimated the price.

Negative residual
-----------------
The model overestimated the price.
"""

residuals = y_test - y_pred

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": residuals
})

print("\n")
print("=" * 70)
print("Sample Predictions")
print("=" * 70)
print(comparison.head(10))

# ==============================================================================
# 11. Visualization 1: Actual vs Predicted
# ==============================================================================

"""
A perfect model would place every point on the red diagonal line.

Points close to the line indicate good predictions.

Points far away indicate larger prediction errors.
"""

plt.figure(figsize=(7, 7))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted")

plt.show()

# ==============================================================================
# 12. Visualization 2: Residual Plot
# ==============================================================================

"""
A good Linear Regression model should produce residuals that
are randomly scattered around zero.

Patterns may indicate that the model is missing nonlinear
relationships in the data.
"""

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals, alpha=0.5)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted House Value")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()

# ==============================================================================
# 13. Visualization 3: Residual Distribution
# ==============================================================================

"""
This histogram shows how prediction errors are distributed.

Ideally:
- Centered around zero.
- Approximately symmetric.
- Few very large errors.
"""

plt.figure(figsize=(8, 5))

plt.hist(residuals, bins=30)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")

plt.show()

# ==============================================================================
# 14. Visualization 4: Coefficient Plot
# ==============================================================================

"""
Since the features were standardized, the coefficient magnitudes are
more directly comparable.

This plot shows the direction (positive/negative) and magnitude of
each learned coefficient.

NOTE:
These are regression coefficients, not feature importance scores.
Highly correlated features can affect coefficient values.
"""

coefficients = coefficients.sort_values(by="Coefficient")

plt.figure(figsize=(8, 5))

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.xlabel("Coefficient")
plt.title("Linear Regression Coefficients")

plt.show()

# ==============================================================================
# End of Project
# ==============================================================================

"""
Project Summary
---------------

✓ Loaded a real-world dataset.
✓ Converted it into a Pandas DataFrame.
✓ Prepared features and target.
✓ Split into training and testing sets.
✓ Standardized numerical features.
✓ Trained a Linear Regression model.
✓ Examined learned coefficients.
✓ Evaluated model performance using:
      - MAE
      - MSE
      - RMSE
      - R²
✓ Analyzed residuals.
✓ Visualized predictions and model diagnostics.

This serves as a strong baseline model. In subsequent projects,
we can compare it against Polynomial Regression, Ridge Regression,
Lasso Regression, and other regression techniques to determine
whether they improve predictive performance.
"""