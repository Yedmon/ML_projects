# California Housing Price Prediction using Linear Regression

## Project Overview

This project demonstrates how to build a **Linear Regression** model using **Scikit-learn** to predict the median house value of California districts. The project follows a complete machine learning workflow, from loading a real-world dataset to evaluating and visualizing the model's performance.

Unlike simple examples that use manually created NumPy arrays, this project works with a real dataset and introduces the standard workflow used in practical machine learning applications.

---

# Objectives

* Learn how to work with a real-world dataset.
* Understand the complete Scikit-learn regression workflow.
* Build a baseline Linear Regression model.
* Evaluate the model using common regression metrics.
* Interpret model coefficients.
* Visualize model predictions and residuals.

---

# Dataset

**Dataset:** California Housing Dataset

The dataset is provided directly by Scikit-learn through:

```python
from sklearn.datasets import fetch_california_housing
```

The dataset contains **20,640** samples, where each sample represents a California district.

### Features

| Feature    | Description                   |
| ---------- | ----------------------------- |
| MedInc     | Median income in the district |
| HouseAge   | Median house age              |
| AveRooms   | Average number of rooms       |
| AveBedrms  | Average number of bedrooms    |
| Population | District population           |
| AveOccup   | Average household occupancy   |
| Latitude   | Geographic latitude           |
| Longitude  | Geographic longitude          |

### Target

**MedHouseValue**

The median house value for each district.

---

# Machine Learning Workflow

The project follows these steps:

1. Import required libraries.
2. Load the California Housing dataset.
3. Convert the dataset into a Pandas DataFrame.
4. Separate the features (`X`) and target (`y`).
5. Split the dataset into training and testing sets.
6. Standardize the feature values using `StandardScaler`.
7. Train a `LinearRegression` model.
8. Generate predictions on the test set.
9. Evaluate model performance.
10. Analyze prediction errors.
11. Visualize model performance.

---

# Libraries Used

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# Model

The project uses **Ordinary Least Squares (OLS) Linear Regression** implemented by Scikit-learn.

Mathematically, the model learns

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Where:
* $x_i$: input features
* $w_i$: learned coefficients
* $b$: intercept
* $\hat{y}$: predicted house value



---

# Evaluation Metrics

The model is evaluated using the following regression metrics:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Coefficient of Determination (R² Score)**

These metrics provide different perspectives on the model's prediction performance.

---

# Visualizations

The project includes several diagnostic visualizations:

* Actual vs Predicted Values
* Residual Plot
* Residual Distribution (Histogram)
* Linear Regression Coefficients

These plots help evaluate the model's predictive performance and identify potential limitations.

---

# Project Structure

```text
California-Housing-Linear-Regression/
│
├── california_housing_linear_regression.py
├── README.md
└── requirements.txt
```

---

# Learning Outcomes

After completing this project, you should understand how to:

* Load datasets provided by Scikit-learn.
* Work with Pandas DataFrames.
* Prepare data for machine learning.
* Perform train/test splitting.
* Apply feature scaling correctly.
* Train a Linear Regression model.
* Evaluate regression models using multiple metrics.
* Interpret model coefficients.
* Analyze residuals.
* Visualize regression results.

---

# Future Improvements

Possible extensions to this project include:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Cross-Validation
* Hyperparameter Tuning
* Scikit-learn Pipelines

---

# Requirements

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# Author

This project was developed as part of a hands-on learning journey to master **Scikit-learn** and build a strong foundation in practical machine learning using regression models.
