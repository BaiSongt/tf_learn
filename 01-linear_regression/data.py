import csv
import random

# Parameters for the linear function y = mx + b
m = 2.5  # slope
b = 4.0  # intercept
num_samples = 200

# Generate data
data = []
for _ in range(num_samples):
    x = random.uniform(0, 100)  # Random x value between 0 and 100
    noise = random.uniform(-0, 0)  # Random noise
    y = m * x + b + noise  # Linear function with noise
    data.append((x, y))

# Write to CSV
output_file = "01-linear_regression/data.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])  # Header
    writer.writerows(data)

print(f"Generated {num_samples} samples and saved to {output_file}")
