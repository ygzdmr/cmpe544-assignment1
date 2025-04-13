import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two data points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.
    """
    def __init__(self, k=3):
        """
        Initializes the KNN classifier.
        """
        self.k = k
        self.X_train = None  # Training data features
        self.y_train = None  # Training data labels

    def fit(self, X, y):
        """
        Fits the KNN classifier to the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for the given data points.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.
        """
        # Calculate distances between x and all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]  # Get indices of the k smallest distances

        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Determine the most common class label among the k neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the KNN classifier on the test data.
        """
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy

      