import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ===============================================================
# Step 1 : Load Dataset
# ===============================================================

df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows")
print(df.head())

# ===============================================================
# Step 2 : Separate Features and Target
# ===============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================================================
# Step 3 : Split Dataset
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)


# ===============================================================
# Step 4 : Feature Scaling
# ===============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# Step 5 : Train Logistic Regression Model
# ===============================================================

model = LogisticRegression()

model.fit(X_train_scaled, y_train)


# ===============================================================
# Step 6 : Inspect Learned Parameters
# ===============================================================

print("\n" + "=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)

for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:<28}: {coef:.4f}")

print(f"\nIntercept (Bias): {model.intercept_[0]:.4f}")


# ===============================================================
# Step 7 : Make Predictions
# ===============================================================
y_predict = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)

# ===============================================================
# Step 8 : Evaluate Model
# ===============================================================

accuracy = accuracy_score(y_test, y_predict)

cm = confusion_matrix(y_test, y_predict)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy Score : {accuracy:.4f}")

print("\nConfusion Matrix")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

# ===============================================================
# Step 9 : Predict New Patients
# ===============================================================

print("\n" + "=" * 60)
print("PREDICTING NEW PATIENTS")
print("=" * 60)

new_patients = np.array([
    [0, 85, 70, 20, 80, 23.5, 0.25, 24],     # Low Risk
    [2, 118, 76, 25, 110, 29.4, 0.45, 35],   # Moderate Risk
    [5, 165, 90, 35, 220, 38.7, 1.10, 52],   # High Risk
    [1, 132, 78, 28, 130, 31.2, 0.60, 29],   # Borderline
    [7, 185, 95, 40, 300, 42.8, 1.35, 60]    # Very High Risk
])

# Convert to DataFrame so feature names are preserved

new_patients_df = pd.DataFrame(new_patients, columns=X.columns)

# Scale using the SAME scaler

new_patients_scaled = scaler.transform(new_patients_df)

predictions = model.predict(new_patients_scaled)

probabilities = model.predict_proba(new_patients_scaled)

for i, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):

    print(f"\nPatient {i}")
    print("-" * 30)

    print(new_patients_df.iloc[i - 1])

    print(f"\nPrediction : {'Diabetic' if pred == 1 else 'Not Diabetic'}")

    print(f"Probability Not Diabetic : {prob[0]:.3f}")
    print(f"Probability Diabetic     : {prob[1]:.3f}")