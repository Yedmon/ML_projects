import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)

x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([40, 45, 50, 55, 60, 65, 70, 75, 80, 85])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2,  random_state= 42)
model = LinearRegression()
model.fit(x_train, y_train)
print(f"The learned weight(slope): {model.coef_}")
print(f"The learned bias: {model.intercept_}")

y_predicted = model.predict(x_test)
print("Actual scores: ", y_test)
print("predicted scores: ", y_predicted)

print(f"Mean Absolute Error(MAE): {mean_absolute_error(y_test, y_predicted)}")
print(f"Mean Squared Error(MSE): {mean_squared_error(y_test, y_predicted)}")
print(f"r2_score: {r2_score(y_test, y_predicted)}")

print(f"predicted score for 14 hrs {model.predict(np.array([[14]]))}")

plt.figure(figsize=(8,10))
plt.scatter(x,y, label = "Data")
plt.plot(x, model.predict(x), linewidth=2, label="Regression Line")
plt.xlabel("number of studied hours")
plt.ylabel("exam score")
plt.title("simple linear regression")
plt.legend()
plt.show()