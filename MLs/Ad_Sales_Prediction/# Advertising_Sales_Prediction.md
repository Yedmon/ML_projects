# Advertising Sales Prediction using Multiple Linear Regression

## Project Overview

This project demonstrates how to build a **Multiple Linear Regression** model using **Scikit-learn** to predict product sales based on advertising expenditures across different media channels.

Unlike the previous project, which used a dataset provided directly by Scikit-learn, this project introduces working with an external **CSV dataset** using **Pandas**. It reinforces the standard machine learning workflow while introducing real-world dataset loading.

---

# Project Objective

The objective is to predict the **Sales** of a product using the advertising budgets allocated to:

* TV
* Radio
* Newspaper

This is a **supervised machine learning regression problem** because:

* The input consists of multiple numerical features.
* The target variable (**Sales**) is continuous.

---

# Dataset

**Dataset:** Advertising Dataset

The dataset contains approximately **200 observations** and four variables.

| Feature   | Description                           |
| --------- | ------------------------------------- |
| TV        | Advertising budget spent on TV        |
| Radio     | Advertising budget spent on Radio     |
| Newspaper | Advertising budget spent on Newspaper |
| Sales     | Product sales (Target Variable)       |

---

# Machine Learning Workflow

This project follows a complete regression workflow:

1. Import required libraries.
2. Load the dataset using Pandas.
3. Explore the dataset.
4. Separate features and target.
5. Split the dataset into training and testing sets.
6. Standardize the feature values.
7. Train a Multiple Linear Regression model.
8. Inspect the learned coefficients.
9. Make predictions.
10. Evaluate model performance.
11. Visualize the results.

---

# Libraries Used

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# Model

The project uses **Multiple Linear Regression**.

The model learns the relationship

[
\hat{y}=w_1(TV)+w_2(Radio)+w_3(Newspaper)+b
]

where:

* (TV), (Radio), and (Newspaper) are the advertising budgets.
* (w_1,w_2,w_3) are the learned coefficients.
* (b) is the intercept.
* (\hat{y}) is the predicted sales.

The model estimates these parameters using the Ordinary Least Squares (OLS) method.

---

# Model Evaluation

The trained model is evaluated using four common regression metrics:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Coefficient of Determination (R² Score)**

These metrics provide insight into the model's prediction accuracy and overall performance.

---

# Visualizations

The project includes several visualizations to better understand the model:

* **Actual vs Predicted Sales**
* **Residual Plot**
* **Residual Distribution (Histogram)**
* **Regression Coefficient Bar Chart**

These plots help evaluate prediction quality and identify potential modeling issues.

---

# Skills Practiced

Through this project, the following concepts were reinforced:

* Reading CSV files with Pandas (`pd.read_csv`)
* Working with DataFrames
* Feature and target separation
* Train/Test splitting
* Feature scaling using `StandardScaler`
* Training a Multiple Linear Regression model
* Model evaluation using regression metrics
* Residual analysis
* Basic regression visualization

---

# Project Structure

```text
Advertising-Linear-Regression/
│
├── Advertising.csv
├── advertising_linear_regression.py
├── README.md
└── requirements.txt
```

---

# Requirements

Install the required libraries before running the project:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# Learning Outcomes

After completing this project, you should be able to:

* Load an external CSV dataset into a Pandas DataFrame.
* Build a Multiple Linear Regression model using Scikit-learn.
* Interpret learned regression coefficients.
* Evaluate regression models using multiple performance metrics.
* Analyze prediction errors using residuals.
* Create visualizations to assess regression model performance.

---

# Future Improvements

Possible extensions to this project include:

* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Feature Selection
* Cross-Validation
* Scikit-learn Pipelines
* Comparing multiple regression models

---

# Author

This project was completed as part of a structured journey to master **Scikit-learn** through hands-on machine learning projects, progressing from simple regression models to real-world datasets and practical workflows.
