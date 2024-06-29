import json
import sys


def calculate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def get_thetas():
    try:
        with open("thetas.json", "r") as read_file:
             data = json.load(read_file)
    except Exception as e:
         raise Exception(e) 
    return data["theta0"], data["theta1"]

while True:
    try:
        mileage = int(input("Enter a mileage: "))
        if mileage > 0:
            print(f"Valid mileage entered: {mileage}")
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

try:
    theta0, theta1 = get_thetas()
except:
   theta0 = 0
   theta1 = 0
   print("Please launch training program first for accurate price predictions: py training_program")
estimatedPrice = calculate_price(mileage, theta0, theta1)
print(f'Estimated Price: {estimatedPrice}')
