import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#hypothesis function: returns estimated price according to mileage
def hypothesis(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def read_csv_to_3d_array(filename):
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            data = []

            for row in reader:
                try:
                    km = int(row[0])
                    price = int(row[1])
                    data.append([km, price])
                except ValueError as e:
                    print(f"Error converting row {row}: {e}")
        
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

#data[i][0] = mileage / data[i][1] = actual price
def gradient_descent1():
   
    #for _ in range(iterations):
    for i in range(m):
        error = hypothesis(data_3d_array[i][0], theta0, theta1) - data_3d_array[i][1]
        theta0 -= learningRate * error
        theta1 -= learningRate * error * data_3d_array[i][0]
    return theta0, theta1

def gradient_descent(X, y, theta0, theta1, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        sum_errors = [0, 0]  # Gradient accumulators for theta[0] and theta[1]
        for i in range(m):
            error = hypothesis(X[i], theta0, theta1) - y[i]
            sum_errors[0] += error
            sum_errors[1] += error * X[i]
        
        theta0 -= alpha * sum_errors[0] / m
        theta1 -= alpha * sum_errors[1] / m
    return theta0, theta1

# Example usage
filename = 'data.csv'
data_3d_array = read_csv_to_3d_array(filename)
m = len(data_3d_array) #iterations of the training program
learningRate = 0.1
theta0 = 0
theta1 = 0

if data_3d_array is not None:
       print('data array0 = ', data_3d_array)
       print('data array1 = ', data_3d_array[1])
theta0, theta1 = gradient_descent(data_3d_array[0], data_3d_array[1], theta0, theta1, 0.01, 1000)
print(theta0, theta1)
# Predict car prices
#predicted_prices = [hypothesis(mileage, theta0, theta1) for mileage in X]

 # Initialize figure and axis for animation 
fig, ax = plt.subplots() 
x_vals = np.linspace(min(data_3d_array[0]), max(data_3d_array[0]), 100) 
line = ax.plot(x_vals, hypothesis(x_vals, theta0, theta1), color='red', label='Regression Line') 
#ax.scatter(data_3d_array[0], data_3d_array[1], marker='o', 
        #color='green', label='Training Data') 

# Set y-axis limits to exclude negative values 
ax.set_ylim(0, max(data_3d_array[1]) + 1)

# Adding labels and title
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Car Price Prediction')
plt.legend()

# Show the plot
plt.show()