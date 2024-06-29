import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import normalize, denormalize, on_key



#hypothesis function: returns estimated price according to mileage
#theta0 doit etre positif et theta1 negatif
def hypothesis(mileage, theta0, theta1):
    estimatedPrice =  theta0 + (theta1 * mileage)
    return estimatedPrice

def read_csv_to_3d_array(filename):
    try:
        data = pd.read_csv(filename)

        data = data.dropna()        # Drop the missing values

        num_rows = len(data)
        Xmileage_array = np.array(data.loc[:,"km"][0:num_rows]).reshape(-1, 1)  #Reshapes the data arrays into column vector
        Yprice_array = np.array(data.loc[0:, "price"]).reshape(-1, 1)
        X_normalized = normalize(Xmileage_array)
        Y_normalized = normalize(Yprice_array)
        return X_normalized, Y_normalized, Xmileage_array, Yprice_array

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


class LinearRegression:
   
    def __init__(self, X, Y, m): 
        self.T0 = 0
        self.T1 = 0
        self.M = m
        self.X = X
        self.Y = Y
        self.iterations = 0
        self.max_iterations = 10000
        self.learning_rate = 0.07

        self.prev_cost = -1
        self.delta_cost = -1
        self.curr_cost = -1

    def iterations_remaining(self):
            if self.max_iterations == 0 :
                return True
            return self.iterations < self.max_iterations
    
    def mean_absolute_error_cost_function(self):
        self.prev_cost = self.curr_cost

        predictions = hypothesis(self.X, self.T0, self.T1)
        cost = np.mean((self.Y - predictions) ** 2)
    
        return cost 

        
    def gradient_descent(self):
        while self.iterations_remaining():
            sumErrorsT0 = 0.0
            sumErrorsT1 = 0.0
            for i in range(m):
                error = hypothesis(self.X[i], self.T0, self.T1)  - self.Y[i]
                sumErrorsT0 += error
                sumErrorsT1 += error * self.X[i]
            self.T0 = self.T0 - self.learning_rate * (sumErrorsT0 / self.M)
            self.T1 = self.T1 - self.learning_rate * (sumErrorsT1 / self.M)

            self.curr_cost = self.mean_absolute_error_cost_function()
            self.delta_cost = self.prev_cost - self.curr_cost
            if self.prev_cost > 0 and self.delta_cost < 0.0000000001:
                break
            self.iterations += 1
        print("ITERATIONS=", self.iterations)
        return self.T0, self.T1

def denormalize_thetas(theta, X_orig, Y_orig):
        X_min, X_max = np.min(X_orig), np.max(X_orig)
        Y_min, Y_max = np.min(Y_orig), np.max(Y_orig)
        
        theta1_normalized = theta[1]
        theta1 = theta1_normalized * (Y_max - Y_min) / (X_max - X_min)
        
        theta0_normalized = theta[0]
        theta0 = Y_min + theta0_normalized * (Y_max - Y_min) - theta1 * X_min / (X_max - X_min)
        
        return float(theta0.item()), float(theta1.item())
    
filename = 'data.csv'
X_norm, Y_norm, X_orig, Y_orig = read_csv_to_3d_array(filename)
m = len(X_norm) #iterations of the training program

linear_reg = LinearRegression(X_norm, Y_norm, m)

theta0, theta1 = linear_reg.gradient_descent()
print("THETA0= ", theta0, "--- THETA1= ", theta1)
 # Initialize figure and axis for animation 
fig, ax = plt.subplots() 

ax.scatter(X_orig, Y_orig, marker='o', color='green', label='Training Data')


x_vals = np.linspace(min(X_orig), max(X_orig), 100).reshape(-1, 1)
x_vals_normalized = normalize(x_vals)
y_predict_normalized = hypothesis(x_vals_normalized, theta0, theta1)
y_vals = denormalize(y_predict_normalized, Y_orig)
ax.plot(x_vals, y_vals, color='red', label='Regression Line')

def record_thetas_in_file(theta0, theta1, X_orig, Y_orig):
    T0_denorm, T1_denorm = denormalize_thetas([theta0, theta1], X_orig, Y_orig)
    print("denormalize(theta0) = ", T0_denorm, "denormalize(theta1) = ", T1_denorm)
    thetas = {
        "theta0": T0_denorm,
        "theta1": T1_denorm,
    }
    try:
        with open("thetas.json", "w") as json_file:
            json.dump(thetas, json_file, indent=4)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
        sys.exit(-1)

record_thetas_in_file(theta0, theta1, X_orig, Y_orig)
# Adding labels and title
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Car Price Prediction')
plt.legend()


# Connect the escape key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the plot
plt.show()
