import numpy as np
import matplotlib.pyplot as plt

# Load dataset (Assume X and y are loaded properly as numpy arrays)
# X: features matrix, y: target vector (prices)
# For simplicity, let's assume X is already normalized and bias term is added

# Hyperparameters
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function
def cost_function(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((hypothesis(X, theta) - y) ** 2)

# Gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradient = (1 / m) * np.dot(X.T, (hypothesis(X, theta) - y))
        theta = theta - alpha * gradient
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history

# Train the model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Predictions
predictions = hypothesis(X, theta)

# Plotting the cost function history
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
