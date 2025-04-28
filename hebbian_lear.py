import numpy as np

# Bipolar input patterns (X1, X2, Bias)
inputs = np.array([
    [1, 1, 1],
    [1, -1, 1],
    [-1, 1, 1],
    [-1, -1, 1]
])

# Corresponding target outputs for AND logic
targets = np.array([1, -1, -1, -1])

# Initialize weights to 0
weights = np.zeros(3)

# Hebbian Learning Rule
for i in range(len(inputs)):
    weights += inputs[i] * targets[i]

print("Final Weights after Hebbian Learning:", weights)

# Testing the network
print("\nTesting the learned model:")
for i in range(len(inputs)):
    activation = np.dot(inputs[i], weights)
    output = 1 if activation > 0 else -1
    print(f"Input: {inputs[i][:2]} | Activation: {activation} | Output: {output} | Target: {targets[i]}")
