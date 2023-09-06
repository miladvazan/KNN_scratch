# KNN_scratch
Implementing the k-nearest neighbors (KNN) algorithm from scratch

# KNN from scratch
```python
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2) ** 2, axis=1))
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = self.euclidean_distance(self.X_train, x)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[indices]
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return np.array(y_pred)
```
### use scikit-learn's KNN implementation and compare it with our scratch implementation using an example
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Our scratch KNN implementation
```python
knn_scratch = KNN(k=3)
knn_scratch.fit(X_train, y_train)
y_pred_scratch = knn_scratch.predict(X_test)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print("Accuracy (Scratch):", accuracy_scratch)
```
## Scikit-learn's KNN implementation
```python
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Accuracy (scikit-learn):", accuracy_sklearn)
```
## test the K-nearest neighbors (KNN) algorithm on artificial data
```python
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our scratch KNN implementation
knn_scratch = KNN(k=3)
knn_scratch.fit(X_train, y_train)
y_pred_scratch = knn_scratch.predict(X_test)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print("Accuracy (Scratch):", accuracy_scratch)

# Scikit-learn's KNN implementation
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Accuracy (scikit-learn):", accuracy_sklearn)
```
