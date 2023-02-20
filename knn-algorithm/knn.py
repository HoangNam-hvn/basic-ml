import numpy as np
from scipy.spatial.distance import cdist

# Define the KNN class
class KNN:
    def __init__(self, k=3) -> None:
        self.k = k   # Set the number of nearest neighbors to consider
        
    # Define the Euclidean distance function
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
        
    # Define the fit method to store the training data and labels
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    # Define the predict method to predict the labels for new test data
    def predict(self, x_test):
        # Compute the distances between test data and training data using the cdist function
        distaces = cdist(x_test, self.x_train)
        
        # Find the indices of the k nearest neighbors for each test data point
        k_nearest_indices = np.argsort(distaces)[:, :self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # Use np.bincount to count the occurrence of each label among the k nearest neighbors,
        # and np.argmax to find the label that occurs most frequently
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=1,
            arr=k_nearest_labels
        )
        
        # Return the predicted labels for the test data
        return y_pred
