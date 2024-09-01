from tools import denormalize_thetas,normalize, denormalize, on_key
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def record_thetas_in_file(theta0, theta1, X_orig, Y_orig):
    T0_denorm, T1_denorm = denormalize_thetas([theta0, theta1], X_orig, Y_orig)
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

def show_plot(X_orig, Y_orig, theta0, theta1, hypothesis):
    try:
        fig, ax = plt.subplots() 
        ax.scatter(X_orig, Y_orig, marker='o', color='green', label='Training Data')
        x_vals = np.linspace(min(X_orig), max(X_orig), 100).reshape(-1, 1)
        x_vals_normalized = normalize(x_vals)
        y_predict_normalized = hypothesis(x_vals_normalized, theta0, theta1)
        y_vals = denormalize(y_predict_normalized, Y_orig)
        ax.plot(x_vals, y_vals, color='red', label='Regression Line')

        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Car Price Prediction')
        plt.legend()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
    except KeyboardInterrupt:
        plt.close('all')

