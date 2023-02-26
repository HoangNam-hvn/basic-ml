import numpy as np


class LinearRegression:
    
    def __init__(self, lr=1e-5, max_iter=100):
        # Constructor function to initialize instance variables
        self.lr = lr           # Learning rate for gradient descent
        self.max_iter = max_iter  # Maximum number of iterations for gradient descent
        self.w = None          # Weights for the linear regression model
        self.b = None          # Bias term for the linear regression model
        
    def fit(self, x_trains, y_trains):
        # Fit the linear regression model to the training data
        n_samples, n_features = x_trains.shape  # Number of samples and features in the training data
        self.w = np.random.uniform(-1, 1, (n_features, 1))  # Initialize weights randomly
        self.b = 0  # Initialize bias term to zero
    
        for _ in range(self.max_iter):
            # Perform gradient descent to update weights and bias term
            fx = x_trains @ self.w + self.b  # Linear combination of features and weights
            dw = (x_trains.T @ (fx - y_trains)) / n_samples
            db = np.sum(fx - y_trains) / n_samples
            self.w -= self.lr * dw  # Update weights
            self.b -= self.lr * db  # Update bias term
    
        return self  # Return the trained model
    
    def predict(self, x_test):
        # Make predictions on new data using the trained model
        y_predicted = x_test @ self.w + self.b  # Linear combination of features and weights
        return y_predicted
