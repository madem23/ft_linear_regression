import sys
import pandas as pd
import numpy as np
from tools import normalize
from output import record_thetas_in_file, show_plot
import matplotlib.pyplot as plt

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

        data['km'] = pd.to_numeric(data['km'], errors='coerce') #coerce forces the conv of invalid data to NaN, better for cleaning
        data['price'] = pd.to_numeric(data['price'], errors='coerce')

        data = data.dropna()
        data = data[(data['km'] >= 0) & (data['price'] >= 0)]

        if data.empty:
            raise ValueError("The data is empty after cleaning. Please check the CSV file.")

        Xmileage_array = np.array(data.loc[:,"km"][0:num_rows]).reshape(-1, 1)  #Reshapes the data arrays into column vector
        Yprice_array = np.array(data.loc[0:, "price"]).reshape(-1, 1)
        X_normalized = normalize(Xmileage_array)
        Y_normalized = normalize(Yprice_array)
        return X_normalized, Y_normalized, Xmileage_array, Yprice_array

    except KeyError as e:
        print(f"Error: The required column {str(e)} is missing from the CSV file.")
        return None, None, None, None
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None, None, None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filename} is empty.")
        return None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

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
        self.max_iterations = iterations if iterations != 0 else 10001
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
            if self.max_iterations == 10001 and self.prev_cost > 0 and self.delta_cost < 0.0000000001:
                break
            self.iterations += 1
        return self.T0, self.T1

if __name__ == '__main__' :
    filename = 'data.csv'

    X_norm, Y_norm, X_orig, Y_orig = read_csv_to_array(filename)
    if X_norm is None or Y_norm is None or X_orig is None or Y_orig is None:
        sys.exit(1)

    while True:
            try:
                iterations = int(input(f"Specify a number of training iterations (maximum 10 000) or 0 for default: "))
                if iterations >= 0 and iterations <= 10000:
                    break
                else:
                    print("Please enter a positive number inferior to 10 000.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
            except KeyboardInterrupt:
                print("\nProgram interrupted by the user. Exiting...")
                sys.exit(1)
            except EOFError:
                print("\nProgram interrupted by the user. Exiting...")
                sys.exit(1)
    
    linear_reg = LinearRegression(X_norm, Y_norm, iterations)
    theta0, theta1 = linear_reg.gradient_descent()
    record_thetas_in_file(theta0, theta1, X_orig, Y_orig)
    show_plot(X_orig, Y_orig, theta0, theta1, hypothesis)