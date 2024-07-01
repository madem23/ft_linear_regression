import matplotlib.pyplot as plt
import numpy as np

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def denormalize(array, original):
    return array * (np.max(original) - np.min(original)) + np.min(original)

def on_key(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)

def denormalize_thetas(theta, X_orig, Y_orig):
    X_min, X_max = np.min(X_orig), np.max(X_orig)
    Y_min, Y_max = np.min(Y_orig), np.max(Y_orig)
    
    theta1_normalized = theta[1]
    theta1 = theta1_normalized * (Y_max - Y_min) / (X_max - X_min)
    
    theta0_normalized = theta[0]
    theta0 = Y_min + theta0_normalized * (Y_max - Y_min) - theta1 * X_min / (X_max - X_min)
    
    return float(theta0.item()), float(theta1.item())
