import json

bold_red = "\033[1m\033[31m"
reset = "\033[0m"
bold_green = "\033[1m\033[32m"
bold_orange = "\033[1m\033[38;2;255;165;0m"
bold = "\033[1m\033["

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
        mileage = int(input(f"ENTER A MILEAGE: "))
        if mileage >= 0:
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
   print(f"\n{bold_red}******  Please launch training program first for accurate price predictions -> py training_program   ******{reset}\n")


estimatedPrice = calculate_price(mileage, theta0, theta1)
if estimatedPrice < 0:
    print(f'{bold_orange}With a mileage of {mileage}, the price will be negative. You should keep the car or give it away.{reset}')
else:
    print(f'Estimated Price for car with a mileage of {mileage} km: {bold_green}{round(estimatedPrice)} euros{reset}')
