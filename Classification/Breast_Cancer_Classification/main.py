import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Load the dataset
# ==========================================================

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# ==========================================================
# Separate Features (X) and Target (y)
# ==========================================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================================
# Split the dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# Train Logistic Regression Model
# ==========================================================

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ==========================================================
# Inspect Learned Parameters
# Since this dataset has 30 features, we expect 30 learned coefficients.
# ==========================================================

print("\n" + "=" * 50)
print("LEARNED COEFFICIENTS")
print("=" * 50)

for feature, coef in zip(cancer.feature_names, model.coef_[0]):
    print(f"{feature:<30} : {coef:.4f}")

print("\nIntercept (Bias):", model.intercept_[0])


# ==========================================================
# Make Predictions
# ==========================================================

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)


print("\n" + "=" * 50)
print("TEST PREDICTIONS")
print("=" * 50)

print("Actual Labels:")
print(y_test.values)

print("\nPredicted Labels:")
print(y_pred)

print("\nPrediction Probabilities (First 5 Samples)")
print(y_prob[:5])


# ==========================================================
# Evaluate the Model
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy Score : {accuracy:.4f}")

print("\nConfusion Matrix")
print(cm)

# ==========================================================
# Predict on New Patients
# ==========================================================

print("\n" + "=" * 50)
print("PREDICTION ON NEW PATIENTS")
print("=" * 50)

# Using the first three patients from the dataset

new_patients = X.iloc[:3]

new_patients_scaled = scaler.transform(new_patients)

new_predictions = model.predict(new_patients_scaled)

new_probabilities = model.predict_proba(new_patients_scaled)

for i in range(len(new_patients)):
    print(f"\nPatient {i+1}")

    print("Actual Label      :", y.iloc[i])
    print("Predicted Label   :", new_predictions[i])

    print(f"Probability Malignant : {new_probabilities[i][0]:.4f}")
    print(f"Probability Benign    : {new_probabilities[i][1]:.4f}")