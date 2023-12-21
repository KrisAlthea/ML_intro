import numpy as np
import matplotlib.pyplot as plt

# Load data from files
x = np.loadtxt('ex4x.dat')
y_binary = np.loadtxt('ex4y.dat')

# Convert binary labels to multiclass labels
y_multiclass = y_binary.astype(int) + 1  # Assuming 1-indexed classes

# Add a bias term to x
x_bias = np.c_[np.ones((x.shape[0], 1)), x]

# Initialize weights matrix for multiclass
num_classes = len(np.unique(y_multiclass))
weights = np.zeros((num_classes, x_bias.shape[1]))


# Perceptron algorithm for multiclass
def perceptron_multiclass(x, y, weights, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            scores = np.dot(weights, x[i])
            predicted_class = np.argmax(scores)
            true_class = y[i] - 1  # Convert to 0-indexed
            if predicted_class != true_class:
                weights[true_class] += learning_rate * x[i]
                weights[predicted_class] -= learning_rate * x[i]

    return weights


# Train the multiclass perceptron model
trained_weights = perceptron_multiclass(x_bias, y_multiclass, weights)

# Plotting the decision boundaries
plt.scatter(x[:, 0], x[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')

# Plot decision boundaries for each class
for i in range(num_classes):
    decision_boundary = (-trained_weights[i, 0] - trained_weights[i, 1] * x[:, 0]) / trained_weights[i, 2]
    plt.plot(x[:, 0], decision_boundary, label=f'Class {i + 1}', linestyle='--', linewidth=2)

plt.legend()
plt.title('Multiclass Perceptron Decision Boundaries')
plt.show()
