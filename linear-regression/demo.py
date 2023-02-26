import matplotlib.pyplot as plt
from linear import LinearRegression
import numpy as np

# Generate some random data for linear regression
n_samples = 100  # Number of data points
x = np.array(range(n_samples)).reshape(n_samples, 1)  # Input feature
y = 3*x + 5 - np.random.uniform(-30, 30, size=(n_samples, 1))  # Output label with some noise

# Fit a linear regression model to the data
Lr = LinearRegression(max_iter=500).fit(x, y)  # Create and train a LinearRegression object
print(f"weights: {Lr.w}, bias: {Lr.b}")  # Print the learned weights and bias

# Plot the data and the learned linear regression model
plt.scatter(x, y)  # Plot the data points
y_predicted = Lr.predict(x)  # Make predictions on the input feature using the trained model
plt.plot(x, y_predicted, c='red')  # Plot the learned linear regression model
plt.show()  # Display the plot
