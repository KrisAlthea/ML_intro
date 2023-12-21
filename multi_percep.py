import numpy as np
import matplotlib.pyplot as plt


class MultiClassPerceptron:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.weights = np.zeros((num_classes, num_features))

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                scores = np.dot(self.weights, X[i])
                predicted_class = np.argmax(scores)
                true_class = y[i]
                if predicted_class != true_class:
                    self.weights[true_class] += learning_rate * X[i]
                    self.weights[predicted_class] -= learning_rate * X[i]

    def predict(self, X):
        scores = np.dot(self.weights, X)
        return np.argmax(scores, axis=0)


# Load data from files
x = np.loadtxt('ex4x.dat')
y = np.loadtxt('ex4y.dat')

# Convert binary labels to multiclass labels
y_multiclass = y.astype(int)

# Add a bias term to x
x_bias = np.c_[np.ones((x.shape[0], 1)), x]

# Initialize weights matrix for multiclass
num_classes = len(np.unique(y_multiclass))
num_features = x_bias.shape[1]
perceptron = MultiClassPerceptron(num_classes, num_features)

# Train the multiclass perceptron model
perceptron.train(x_bias, y_multiclass)

# Plotting the decision boundaries
plt.scatter(x[:, 0], x[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')

# Plot decision boundaries for each class
x_values = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
for i in range(num_classes):
    y_values = (-perceptron.weights[i, 0] - perceptron.weights[i, 1] * x_values) / perceptron.weights[i, 2]
    plt.plot(x_values, y_values, label=f'Class {i}', linestyle='--', linewidth=2)

plt.legend()
plt.title('Multiclass Perceptron Decision Boundaries')
plt.show()
