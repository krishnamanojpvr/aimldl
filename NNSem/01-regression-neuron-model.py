# y = 3*x1 + 2*x2 + noise

import numpy as np

np.random.seed(42)

# Number of samples
no_of_samples = 1000

# Input data: random numbers
X = np.random.rand(no_of_samples, 2)

# True weights
true_weights = np.array([3, 2])

# Random noise
noise = np.random.randn(no_of_samples)

# Compute true target values with noise
yhat = np.dot(X, true_weights.T) + noise

# Number of epochs and learning rate
epochs = 1000
alpha = 0.1

# Initialize weights and bias randomly
weights = np.random.randn(2)
bias = np.random.randn(1)

# Training loop
for epoch in range(epochs):
    # Forward pass: predict output
    ypred = np.dot(X, weights) + bias

    # Compute error
    error = ypred - yhat

    # Compute gradients
    dw = (2 / no_of_samples) * np.dot(X.T, error)
    db = (2 / no_of_samples) * np.sum(error)

    # Update weights and bias
    weights -= alpha * dw
    bias -= alpha * db

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(np.abs(error))}")

# Print final learned weights and bias
print("\nFinal weights:", weights)
print("Final bias:", bias)
