import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import BinaryAccuracy


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)


# ============================================================
# 2. Generate Synthetic Dataset
# ============================================================

n_samples = 500

study_hours = np.random.uniform(0, 10, n_samples)
sleep_hours = np.random.uniform(4, 9, n_samples)

# Underlying relationship used to generate the labels
score = (0.7 * study_hours + 0.3 * sleep_hours + np.random.normal(0, 0.5, n_samples))

# Binary classification target
# 0 = Fail
# 1 = Pass
y = (score > 5).astype(int)

# Combine features into a single matrix
X = np.column_stack((study_hours, sleep_hours))


# ============================================================
# 3. Inspect Dataset
# ============================================================

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 samples:")
print(X[:5])

print("\nFirst 5 labels:")
print(y[:5])


# ============================================================
# 4. Visualize Dataset
# ============================================================

plt.scatter(study_hours, sleep_hours, c=y, cmap="coolwarm", alpha=0.7)

plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Synthetic Student Dataset")
plt.show()


# ============================================================
# 5. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 6. Build Neural Network
# ============================================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])


# ============================================================
# 7. Inspect Model
# ============================================================

model.summary()


# ============================================================
# 8. Compile Model
# ============================================================

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[BinaryAccuracy(name="accuracy")])


# ============================================================
# 9. Train Model
# ============================================================

history = model.fit(X_train, y_train, epochs=80, verbose=1)


# ============================================================
# 10. Visualize Training History
# ============================================================

training_loss = history.history["loss"]
training_accuracy = history.history["accuracy"]

plt.plot(training_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(training_accuracy)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# ============================================================
# 11. Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 12. Make Predictions on New Data
# ============================================================

new_study_hours = np.array([2, 6, 8])
new_sleep_hours = np.array([5, 7, 8])

X_new = np.column_stack((new_study_hours, new_sleep_hours))

# Model returns probabilities
probabilities = model.predict(X_new, verbose=0)

# Convert probabilities into class predictions
predictions = (probabilities >= 0.5).astype(int)


# ============================================================
# 13. Display Predictions
# ============================================================

for i in range(len(X_new)):
    print(
        f"Study: {X_new[i, 0]:.1f} hours | "
        f"Sleep: {X_new[i, 1]:.1f} hours | "
        f"Probability of Pass: {probabilities[i, 0]:.4f} | "
        f"Prediction: {predictions[i, 0]}"
    )