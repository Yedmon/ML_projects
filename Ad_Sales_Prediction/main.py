# ==============================================================================
# Advertising Sales Prediction using Multiple Linear Regression
#
# Goal:
# Predict product sales based on the amount of money spent on
# TV, Radio, and Newspaper advertisements.
#
# Workflow:
# 1. Import libraries
# 2. Load dataset
# 3. Separate features and target
# 4. Split the data
# 5. Scale the features
# 6. Train the Linear Regression model
# 7. Inspect learned coefficients
# 8. Make predictions
# 9. Evaluate the model
# 10. Visualize the results
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==============================================================================
# Load the Dataset
# ==============================================================================

# Read the CSV file into a Pandas DataFrame.
df = pd.read_csv("Advertising.csv")

# Uncomment these whenever you want to inspect the dataset.
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())

# ==============================================================================
# Separate Features (X) and Target (y)
# ==============================================================================

# Features used to predict sales.
X = df.drop("sales", axis=1)

# Target variable.
y = df["sales"]

# ==============================================================================
# Split the Dataset
# ==============================================================================

# 80% for training
# 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================================================================
# Feature Scaling
# ==============================================================================

# Standardize the features.
#
# Although Linear Regression does not require feature scaling,
# we use it here because many future machine learning algorithms
# benefit from standardized data.

scaler = StandardScaler()

# Learn the scaling parameters from the training set.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the testing set.
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# Train the Linear Regression Model
# ==============================================================================

model = LinearRegression()

# Learn the relationship between the advertising budget
# and product sales.
model.fit(X_train_scaled, y_train)

# ==============================================================================
# Inspect the Learned Model
# ==============================================================================

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nLearned Coefficients")
print(coef_df)

print("\nIntercept (Bias):")
print(model.intercept_)

# ==============================================================================
# Make Predictions
# ==============================================================================

# Predict sales for the testing data.
y_pred = model.predict(X_test_scaled)

# ==============================================================================
# Evaluate the Model
# ==============================================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ==============================================================================
# Compare Actual and Predicted Values
# ==============================================================================

# Residual = Actual - Predicted
#
# Positive residual:
#     The model underestimated the sales.
#
# Negative residual:
#     The model overestimated the sales.

residuals = y_test - y_pred

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": residuals
})

print("\nFirst 10 Predictions")
print(comparison.head(10))

# ==============================================================================
# Visualization 1 : Actual vs Predicted
# ==============================================================================

plt.figure(figsize=(7, 6))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.show()

# ==============================================================================
# Visualization 2 : Residual Plot
# ==============================================================================

plt.figure(figsize=(7, 6))

plt.scatter(y_pred, residuals)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual")

plt.show()

# ==============================================================================
# Visualization 3 : Residual Distribution
# ==============================================================================

plt.figure(figsize=(7, 6))

plt.hist(residuals, bins=15)

plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.show()

# ==============================================================================
# Visualization 4 : Coefficient Plot
# ==============================================================================

coef_df = coef_df.sort_values(by="Coefficient")

plt.figure(figsize=(7, 5))

plt.barh(
    coef_df["Feature"],
    coef_df["Coefficient"]
)

plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient")

plt.show()