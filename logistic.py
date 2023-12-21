import matplotlib.pyplot as plt
import numpy as np

# 导入数据ex4x.dat和ex4y.dat
ex4x = np.loadtxt('ex4x.dat')
ex4y = np.loadtxt('ex4y.dat')


# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the cost function for logistic regression
def cost_function(theta, X, y):
    m = len(y)
    theta_flat = theta.flatten()
    h = sigmoid(X @ theta_flat)
    cost = 1 / m * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
    return cost


# Define the gradient function
def gradient(theta, X, y):
    m = len(y)
    theta_flat = theta.flatten()
    h = sigmoid(X @ theta_flat)
    grad = 1 / m * (X.T @ (h - y))
    return grad.reshape(theta.shape)


# Define the Hessian matrix function for Newton's method
def hessian(theta, X, y):
    m = len(y)
    theta_flat = theta.flatten()
    h = sigmoid(X @ theta_flat)
    H = 1 / m * (X.T @ np.diag(h) @ np.diag(1 - h) @ X)
    return H


# Implement gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        J_history.append(cost_function(theta, X, y))
    return theta, J_history


# Implement stochastic gradient descent
def stochastic_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        for j in range(m):
            theta = theta - alpha * gradient(theta, X[j].reshape(1, -1), np.array([y[j]]))
        J_history.append(cost_function(theta, X, y))
    return theta, J_history


# Implement Newton's method
def newtons_method(X, y, theta, num_iters):
    J_history = []
    for i in range(num_iters):
        theta = theta - np.linalg.inv(hessian(theta, X, y)) @ gradient(theta, X, y)
        J_history.append(cost_function(theta, X, y))
    return theta, J_history


# Plot the cost function
def plot_cost(iters, J_history, alpha, method):
    plt.figure()
    plt.plot(range(gd_iters), J_history_gd, label='Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.title(f'alpha: {gd_alpha}, iters: {gd_iters}')
    plt.show()


# Set initial parameters theta
theta = np.zeros((ex4x.shape[1], 1))

# Set the number of iterations
gd_iters = 2500
sgd_iters = 2500
nm_iters = 4

# Set the learning rate alpha
gd_alpha = 1e-4
sgd_alpha = 1e-6

# Calculate and plot the results
theta_gd, J_history_gd = gradient_descent(ex4x, ex4y, theta, gd_alpha, gd_iters)
theta_sgd, J_history_sgd = stochastic_gradient_descent(ex4x, ex4y, theta, sgd_alpha, sgd_iters)
theta_nm, J_history_nm = newtons_method(ex4x, ex4y, theta, nm_iters)

# Plot the cost function of gradient descent
plot_cost(gd_iters, J_history_gd, gd_alpha, 'Gradient Descent')

# Plot the cost function of stochastic gradient descent
plot_cost(sgd_iters, J_history_sgd, sgd_alpha, 'Stochastic Gradient Descent')

# Plot the cost function of Newton's method
plot_cost(nm_iters, J_history_nm, 0, "Newton's Method")


# Plot the decision boundary
plt.figure()
plt.scatter(ex4x[:, 0], ex4x[:, 1], c=ex4y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('Training set')
x1_min = ex4x[:, 0].min()
x1_max = ex4x[:, 0].max()
x2_min = ex4x[:, 1].min()
x2_max = ex4x[:, 1].max()
x1 = np.linspace(x1_min, x1_max, 100)
x2 = np.linspace(x2_min, x2_max, 100)
xx1, xx2 = np.meshgrid(x1, x2)

# Standardize xx1 and xx2
xx1_norm = (xx1 - np.mean(ex4x[:, 0])) / np.std(ex4x[:, 0])
xx2_norm = (xx2 - np.mean(ex4x[:, 1])) / np.std(ex4x[:, 1])

X = np.column_stack((np.ones_like(xx1.flatten()), xx1_norm.flatten(), xx2_norm.flatten()))
Z = sigmoid(X @ theta_gd.flatten())
Z = Z.reshape(xx1.shape)
plt.contour(xx1, xx2, Z, [0.5], colors='r')
plt.show()

# Predict the probability of admission for a student with scores 45 and 85
x_test = np.array([1, 45, 85])
prob = sigmoid(x_test @ theta_nm.flatten())
print(f'For a student with scores 45 and 85, we predict an admission probability of {prob:.4f}')

# Compute the accuracy on the training set
p = sigmoid(ex4x @ theta_nm.flatten())
p = np.where(p >= 0.5, 1, 0)
accuracy = np.mean(p == ex4y) * 100

print(f'Train Accuracy: {accuracy:.2f}%')
