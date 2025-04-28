# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Input dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Experience
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])                # Salary

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation & training
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Visualization
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# Display model coefficients
print(f"Intercept (theta_1): {model.intercept_}")
print(f"Coefficient (theta_2): {model.coef_[0]}")
