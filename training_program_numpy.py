import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import normalize, denormalize, on_key
from output import record_thetas_in_file

def hypothesis(mileage, theta0, theta1):
    """
    Returns estimated price according to mileage. Theta0 is the Y-intercept (positive), Theta1 is the slope (negative)
    """
    estimatedPrice =  theta0 + (theta1 * mileage)
    return estimatedPrice

def read_csv_to_array(filename):
    """
    Reads data from csv file, reshapes it into numpy arrays, and normalizes the data.
    """
    try:
        data = pd.read_csv(filename)
        data = data.dropna()
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
    """
    Performs the linear regression.
    """

    def __init__(self, X, Y, iterations): 
        self.T0 = 0
        self.T1 = 0
        self.M = len(X)
        self.X = X
        self.Y = Y
        self.iterations = 0      
        self.max_iterations = iterations if iterations != 0 else 10000
        self.learning_rate = 0.05

        self.prev_cost = -1
        self.delta_cost = -1
        self.curr_cost = -1

    def iterations_remaining(self):
            if self.max_iterations == 0 :
                return True
            return self.iterations < self.max_iterations
    
    def mean_error_cost_function(self):
        self.prev_cost = self.curr_cost

        predictions = hypothesis(self.X, self.T0, self.T1)
        cost = np.mean((self.Y - predictions) ** 2)
    
        return cost 

        
    def gradient_descent(self):
        while self.iterations_remaining():
            sumErrorsT0 = 0.0
            sumErrorsT1 = 0.0
            for i in range(self.M):
                error = hypothesis(self.X[i], self.T0, self.T1)  - self.Y[i]
                sumErrorsT0 += error
                sumErrorsT1 += error * self.X[i]
            self.T0 = self.T0 - self.learning_rate * (sumErrorsT0 / self.M)
            self.T1 = self.T1 - self.learning_rate * (sumErrorsT1 / self.M)

            self.curr_cost = self.mean_error_cost_function()
            self.delta_cost = self.prev_cost - self.curr_cost
            if self.prev_cost > 0 and self.delta_cost < 0.0000000001:
                break
            self.iterations += 1
        print("ITERATIONS=", self.iterations)
        return self.T0, self.T1

if __name__ == '__main__' :

    while True:
        try:
            iterations = int(input(f"Specify a number of training iterations (maximum 10 000) or 0 for default: "))
            if iterations >= 0 and iterations <= 10000:
                break
            else:
                print("Please enter a positive number inferior to 10 000.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    filename = 'data.csv'
    X_norm, Y_norm, X_orig, Y_orig = read_csv_to_array(filename)

    linear_reg = LinearRegression(X_norm, Y_norm, iterations)

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
