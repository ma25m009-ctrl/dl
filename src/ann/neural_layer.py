import numpy as np


class DenseLayer:

    def __init__(self, input_dim, output_dim, weight_init="xavier"):
        if weight_init == "xavier":
            limit = np.sqrt(2 / (input_dim + output_dim))
            self.W = np.random.randn(input_dim, output_dim) * limit
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad_output, weight_decay=0.0):
        m = self.X.shape[0]
        self.grad_W = (self.X.T @ grad_output) / m + weight_decay * self.W
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True) / m
        return grad_output @ self.W.T
