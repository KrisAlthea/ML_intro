import numpy as np
import matplotlib.pyplot as plt

# Load data from files
x = np.loadtxt('ex4x.dat')
y = np.loadtxt('ex4y.dat')

# Add a bias term to x
x_bias = np.c_[np.ones((x.shape[0], 1)), x]


def normalize_data(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_norm = (x - mean) / std
    return x_norm


x_norm = normalize_data(x)
x_bias_norm = np.c_[np.ones((x_norm.shape[0], 1)), x_norm]


# Perceptron algorithm
def perceptron_algorithm(x, y, learning_rate=0.01, epochs=100):
    # Initialize weights to zeros
    weights = np.zeros(x.shape[1])

    for epoch in range(epochs):
        for i in range(x.shape[0]):
            # print(f"Epoch: {epoch}, i: {i}, weights: {weights}")
            prediction = np.dot(weights, x[i])
            # print(f"Prediction: {prediction}, y[i]: {y[i]}")
            update = learning_rate * (y[i] - prediction)
            # print(f"Update: {update}, x[i]: {x[i]}")
            weights += update * x[i]
            # print(f"Weights after update: {weights}")

    return weights


# Train the perceptron model
weights = perceptron_algorithm(x_bias_norm, y)

# Plotting the decision boundary
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')

# Decision boundary equation: w0 + w1 * x1 + w2 * x2 = 0
# Solving for x2 gives: x2 = (-w0 - w1 * x1) / w2
x1_decision_boundary = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
x2_decision_boundary = (-weights[0] - weights[1] * x1_decision_boundary) / weights[2]
plt.plot(x1_decision_boundary, x2_decision_boundary, color='black', linestyle='--', linewidth=2)

plt.title('Perceptron Decision Boundary')
plt.show()
