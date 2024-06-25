theta0 = 0
theta1 = 0
def calculate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

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

estimatedPrice = calculate_price(mileage, theta0, theta1)
print(f'Estimated Price: {estimatedPrice}')
