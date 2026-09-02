import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy


# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

cancer = load_breast_cancer()

df = pd.DataFrame(
    data=cancer.data,
    columns=cancer.feature_names
)

df["target"] = cancer.target


# ============================================================
# STEP 2: PREPARE X AND y
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# STEP 3: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# STEP 4: FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# STEP 5: BUILD THE NEURAL NETWORK
# ============================================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.summary()


# ============================================================
# STEP 6: COMPILE THE MODEL
# ============================================================

model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(),
    metrics=[BinaryAccuracy()]
)


# ============================================================
# STEP 7: TRAIN THE MODEL
# ============================================================

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)


# ============================================================
# STEP 8: EVALUATE THE MODEL
# ============================================================

evaluation = model.evaluate(
    X_test_scaled,
    y_test,
    return_dict=True
)

print("\nTest Results:")
print(evaluation)


# ============================================================
# STEP 9: MAKE PREDICTIONS
# ============================================================

probabilities = model.predict(X_test_scaled)

predictions = []
for p in probabilities:
    if p >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0)



# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)
