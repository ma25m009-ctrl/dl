import numpy as np


class CrossEntropy:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        y_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.mean(np.sum(y_true * np.log(y_clipped), axis=1))

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSE:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / (self.y_true.shape[0] * self.y_true.shape[1])
