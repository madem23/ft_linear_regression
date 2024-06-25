import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

#hypothesis function: returns estimated price according to mileage
#theta0 doit etre positif et theta1 negatif
def hypothesis(mileage, theta0, theta1):
    estimatedPrice =  theta0 + (theta1 * mileage)
    # print("ESTIMATED PRICE ==", estimatedPrice, "with theta1", theta1, "et theta0=", theta0)
    return estimatedPrice

def read_csv_to_3d_array(filename):
    try:
        data = pd.read_csv(filename)

        data = data.dropna()        # Drop the missing values

        num_rows = len(data)
        Xmileage_array = np.array(data.loc[:,"km"][0:num_rows]).reshape(-1, 1)  #Reshapes the data arrays into column vector
        Yprice_array = np.array(data.loc[0:, "price"]).reshape(-1, 1)
        Xmileage_array = normalize(Xmileage_array)
        Yprice_array = normalize(Yprice_array)
        return Xmileage_array, Yprice_array

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

# def gradient_descent(theta0, theta1, Xmileage_array, Yprice_array):
#     learningRate = 0.1
#     #for _ in range(iterations):
#     for i in range(m):
#         error = hypothesis(Xmileage_array[i], theta0, theta1) - Yprice_array[i]
#         print("ERROR ============", Xmileage_array[i], Yprice_array[i], error)
#         theta0 -= learningRate * error
#         theta1 -= learningRate * error * Xmileage_array[i]
#         print("theta0 = ", theta0, " | theta1 =", theta1)
#     return theta0, theta1

# def gradient_descent(theta0, theta1, X, Y, learning_rate=0.0001, iterations=1000):
#     m = len(X)
#     print("m =====", m)

#     for _ in range(iterations):
#         allErrors0 = 0.0
#         allErrors1 = 0.0
#         for i in range(m):
#             #print("AVANT theta0 = ", theta0, " | theta1 =", theta1)
#             error = hypothesis(X[i], theta0, theta1) - Y[i]
#             # print("hypothesis = ", hypothesis(X[i], theta0, theta1), "- Y[i]: ", Y[i])
#             # print("allError0 = ", allErrors0, "new error = ", error)
#             allErrors0 += error
#             allErrors1 += error * X[i]
#         # print("pour m = ", m, "| ALL ERRORS 0", allErrors0 /m)
#         # print("ALL ERRORS 1 mean", allErrors1 / m)
#         theta0 = theta0 - learning_rate * (allErrors0 / m)
#         theta1 = theta1 - learning_rate * (allErrors1 / m)
#         # print("theta 0 = ", theta0, "| theta1=", theta1)
#     return theta0, theta1

class LinearRegression:
   
    def __init__(self, X, Y, m): 
        self._T0 = 0
        self._T1 = 0
        self.T0 = 0
        self.T1 = 0
        self.M = m
        self.X = X
        self.Y = Y
        self.iterations = 0
        self.max_iterations = 10000
        self.learning_rate = 0.01

    def condition_to_stop_training(self):
            return self.iterations < self.max_iterations
        
    def gradient_descent(self):
        print("\033[33m{:s}\033[0m".format('TRAINING MODEL :'))
        iterations = 0
        while self.condition_to_stop_training():
            sum1 = 0
            sum2 = 0
            for i in range(m):
                T = self.T0 + self.T1 * self.X[i] - self.Y[i]
                sum1 += T
                sum2 += T * self.X[i]

            self.T0 = self.T0 - self.learning_rate * (sum1 / self.M)
            self.T1 = self.T1 - self.learning_rate * (sum2 / self.M)

            # self.C.append(self.cost())

            # self.prev_mse = self.cur_mse
            # self.cur_mse = self.cost()
            # self.delta_mse = self.cur_mse - self.prev_mse

            self.iterations += 1


            # if self.iterations % 100 == 0 or self.iterations == 1:
            #     self.live_update(output_lines)
            #     if self.live == True:
            #         self.plot_all(self.po, self.pn, self.history)

        # self.live_update(output_lines)

        # self.RMSE_percent()
        # self.MSE_percent()

        return self.T0, self.T1


filename = 'data.csv'
X, Y = read_csv_to_3d_array(filename)
X = np.hstack([np.ones((X.shape[0], 1)), X])
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX=", X)
m = len(X) #iterations of the training program

# linear_reg = LinearRegression(X, Y, m)

# theta0, theta1 = linear_reg.gradient_descent()
# print("theta0 et theta1===", theta0, theta1)
# # Predict car prices
# #predicted_prices = [hypothesis(mileage, theta0, theta1) for mileage in X]

#  # Initialize figure and axis for animation 
# fig, ax = plt.subplots() 
# x_vals = np.linspace(min(X), max(X), 100) 
# y_vals = hypothesis(x_vals, theta0, theta1)
# line = ax.plot(x_vals, y_vals, color='red', label='Regression Line') 
# ax.scatter(X, Y, marker='o', 
#         color='green', label='Training Data') 

# # Adding labels and title
# plt.xlabel('Mileage')
# plt.ylabel('Price')
# plt.title('Car Price Prediction')
# plt.legend()

# # Show the plot
# plt.show()