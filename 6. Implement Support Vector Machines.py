import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# Linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

# Show unclassified data
plt.figure(1)
plt.scatter(X, y)
plt.title('Unclassified Data')
plt.show()

# Shaping data for training the model
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

# Define the model
clf = svm.SVC(kernel='linear', C=1.0)

# Train the model
clf.fit(training_X, training_y)

# Get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]

# Get the y-offset for the linear equation
a = -w[0] / w[1]

# Make the x-axis space for the data points
XX = np.linspace(0, 13)

# Get the y-values to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]

# Plot the decision boundary
plt.figure(2)
plt.plot(XX, yy, 'k-', label='Decision Boundary')  # Add label for legend

# Show the plot visually with legend
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
plt.title('Decision Boundary')
plt.legend()  # Show legend with labels
plt.show()