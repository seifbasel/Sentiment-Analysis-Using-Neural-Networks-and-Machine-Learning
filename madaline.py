import numpy as np

class MADALINE:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)

    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                activation = np.dot(X[i], self.weights)
                if activation >= 0:
                    prediction = 1
                else:
                    prediction = 0
                error = y[i] - prediction
                self.weights += learning_rate * error * X[i]

    def predict(self, X):
        activations = np.dot(X, self.weights)
        return np.where(activations >= 0, 1, 0)