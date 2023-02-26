from sklearn.datasets import load_iris
from kmeans import KMeans

iris = load_iris()
X = iris.data

x_trains = X[:, :2]
km = KMeans(k=3, max_iter=10).fit(x_trains)
