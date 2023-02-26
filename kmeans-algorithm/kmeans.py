import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class KMeans:
    
    def __init__(self, k=2, max_iter=10):
        # Initialize KMeans object with number of clusters and maximum iterations
        self.k = k 
        self.max_iter = max_iter
        self.C = [] # list to store centroids
        self.L = [] # list to store labels
        self.centroids = None # initial centroids
        self.labels = None # labels after clustering

    def fit(self, x_trains):
        # fit the data and return the object itself
        self.x_trains = x_trains # training data
        self.centroids = self._init_centroids() # initialize centroids
        self.C.append(self.centroids)
        self.labels = self._assign_labels(self.C[-1]) # assign labels to each data point
        self.L.append(self.labels)
        
        for _ in range(self.max_iter):
            # update centroids and labels iteratively until convergence
            self.C.append(self._update_centroids(self.L[-1]))
            self.L.append(self._assign_labels(self.C[-1]))
            if np.allclose(self.C[-1], self.C[-2]):
                # break if the centroids do not change significantly
                break
        
        # plot the data points and centroids before and after clustering
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1) 
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('before') 
        ax2.set_title('after')
        for i in range(self.k):
            # plot data points for each cluster
            data1 = self.x_trains[self.L[0] == i].T
            data2 = self.x_trains[self.L[-1] == i].T
            ax1.scatter(*data1)
            ax1.scatter(*self.C[0][i], marker='x', c='black', linewidths=4)
            ax2.scatter(*data2)
            ax2.scatter(*self.C[-1][i], marker='x', c='black', linewidths=4)
        plt.show()
        
        # update centroids and labels
        self.centroids = self.C[-1]
        self.labels[self.L[-1]]
        return self
            
    def predict(self, x_tests):
        # predict cluster labels for new data points
        distances = cdist(x_tests, self.centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _init_centroids(self):
        # randomly select the first centroid, then select the remaining ones
        centroids = np.zeros(shape=(self.k, self.x_trains.shape[1]))
        centroids[0] = self.x_trains[np.random.randint(self.x_trains.shape[0])]
        
        for i in range(1, self.k):
            # calculate distances to the nearest centroid for each data point
            distances = cdist(self.x_trains, centroids[:i])
            min_d2 = np.min(distances, axis=1) ** 2
            probs = min_d2 / np.sum(min_d2)
            # select the next centroid with a probability proportional to its distance to the nearest centroid
            next_id = np.random.choice(range(self.x_trains.shape[0]), p=probs)
            centroids[i] = self.x_trains[next_id]
        return centroids
    
    def _update_centroids(self, labels):
        # update centroids by calculating the mean of data points assigned to each cluster
        centroids = np.zeros(shape=(self.k, self.x_trains.shape[1]))
        for i in range(self.k):
            centroids[i] = np.mean(self.x_trains[labels == i], axis=0)
        return centroids
    
    def _assign_labels(self, centroids):
        # assign labels to data points based on their distances to centroids
        distances = cdist(self.x_trains, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
