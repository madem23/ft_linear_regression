import matplotlib.pyplot as plt
import numpy as np

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def denormalize(array, original):
    return array * (np.max(original) - np.min(original)) + np.min(original)

def on_key(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)