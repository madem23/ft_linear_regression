from tools import denormalize_thetas
import json
import sys

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