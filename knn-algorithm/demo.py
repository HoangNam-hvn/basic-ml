from knn import KNN
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
bc_data, bc_target = load_breast_cancer(return_X_y=True)
iris_data, iris_target = load_iris(return_X_y=True)

# Split datasets into training and testing sets
bc_X_train, bc_X_test, bc_y_train, bc_y_test = train_test_split(
    bc_data, bc_target, test_size=0.2, random_state=42)
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris_data, iris_target, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
bc_X_train = scaler.fit_transform(bc_X_train)
bc_X_test = scaler.transform(bc_X_test)
iris_X_train = scaler.fit_transform(iris_X_train)
iris_X_test = scaler.transform(iris_X_test)

# Train and evaluate the models
knn1 = KNN(k=5)
knn2 = KNN(k=3)
for X_train, X_test, y_train, y_test, knn, name in [
    (bc_X_train, bc_X_test, bc_y_train, bc_y_test, knn1, "Breast Cancer"),
    (iris_X_train, iris_X_test, iris_y_train, iris_y_test, knn2, "Iris")
    ]:
    knn.fit(X_train, y_train)
    accuracy = (knn.predict(X_test) == y_test).mean()
    print(f"Accuracy for {name} dataset: {accuracy:.2f}")
