import numpy as np


# Train a Hebbian Learning model
class Hebbian:
    def __init__(self, input_size):
        self.weights = np.zeros((input_size, input_size))

    def train(self, X):
        for x in X:
            self.weights += np.outer(x, x)

    def predict(self, X, threshold=0):
        activations = np.dot(X, self.weights)
        return np.where(activations >= threshold, 1, 0)

