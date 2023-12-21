import numpy as np
import matplotlib.pyplot as plt

# Load data from files
x = np.loadtxt('ex4x.dat')
y = np.loadtxt('ex4y.dat')


# Normalize data
def normalize_data(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_norm = (x - mean) / std
    return x_norm


x_norm = normalize_data(x)
x_bias_norm = np.c_[np.ones((x_norm.shape[0], 1)), x_norm]


# Perceptron algorithm for binary classification
def perceptron_binary(x, y, learning_rate=0.01, epochs=100):
    weights = np.zeros(x.shape[1])
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            prediction = np.dot(weights, x[i])
            update = learning_rate * (y[i] - prediction)
            weights += update * x[i]
    return weights


# Train the binary perceptron model
weights_binary = perceptron_binary(x_bias_norm, y)

# Plotting the decision boundary for binary classification
plt.figure(1)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
x1_decision_boundary = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
x2_decision_boundary = (-weights_binary[0] - weights_binary[1] * x1_decision_boundary) / weights_binary[2]
plt.plot(x1_decision_boundary, x2_decision_boundary, color='black', linestyle='--', linewidth=2)
plt.title('Binary Perceptron Decision Boundary')

# Convert binary labels to multiclass labels
y_multiclass = y.astype(int) + 1

# Initialize weights matrix for multiclass
num_classes = len(np.unique(y_multiclass))
weights_multiclass = np.zeros((num_classes, x_bias_norm.shape[1]))


# Perceptron algorithm for multiclass
def perceptron_multiclass(x, y, weights, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            scores = np.dot(weights, x[i])
            predicted_class = np.argmax(scores)
            true_class = y[i] - 1
            if predicted_class != true_class:
                weights[true_class] += learning_rate * x[i]
                weights[predicted_class] -= learning_rate * x[i]
    return weights


# Train the multiclass perceptron model
trained_weights = perceptron_multiclass(x_bias_norm, y_multiclass, weights_multiclass)

# Plotting the decision boundaries for multiclass
plt.figure(2)
plt.scatter(x[:, 0], x[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
for i in range(num_classes):
    decision_boundary = (-trained_weights[i, 0] - trained_weights[i, 1] * x[:, 0]) / trained_weights[i, 2]
    plt.plot(x[:, 0], decision_boundary, label=f'Class {i + 1}', linestyle='--', linewidth=2)
plt.legend()
plt.title('Multiclass Perceptron Decision Boundaries')
plt.show()
