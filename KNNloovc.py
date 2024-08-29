import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

def loocv_knn(X, y, k_values):
    best_k = None
    best_accuracy = 0.0

    for k in k_values:
        knn = KNN(k=k)
        accuracies = []

        for i in range(len(X)):
            X_loocv, y_loocv = np.delete(X, i, axis=0), np.delete(y, i)
            knn.fit(X_loocv, y_loocv)
            prediction = knn.predict(np.array([X[i]]))[0]
            accuracies.append(prediction == y[i])

        accuracy = np.mean(accuracies)
        print(f"K = {k}, Accuracy = {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k

# data
X_0 = np.array([[1, 6], [2, 7], [3, 8], [8, 4], [7, 3], [4, 9]])
y_0 = np.array([0, 0, 0, 0, 0, 0])

X_1 = np.array([[2, 6], [3, 7], [9, 4], [8, 3], [7, 2], [6, 1]])
y_1 = np.array([1, 1, 1, 1, 1, 1])

X_all = np.concatenate((X_0, X_1))
y_all = np.concatenate((y_0, y_1))

# K
possible_k_values = [1, 3, 5, 7, 9, 11]

# optimal K
optimal_k = loocv_knn(X_all, y_all, possible_k_values)

print(f"Optimal K: {optimal_k}")
