
import numpy as np


class NeuralNetwork:

    def __init__(self, layers=None, activations=None, loss_fn=None, optimizer=None, weight_decay=0.0):
        if layers is None:
            self.layers = []
        elif isinstance(layers, list):
            self.layers = layers
        else:
            self.layers = [layers]

        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.weight_decay = weight_decay

        if activations is None or (hasattr(activations, '__len__') and len(activations) == 0):
            self.activations = [_Identity() for _ in self.layers]
        else:
            self.activations = activations if isinstance(activations, list) else [activations]

    def set_weights(self, weights):
        """
        Set weights from flat list/array: [W0, b0, W1, b1, ...]
        Each element must be a numpy array.
        """
        weights = list(weights)
        for i, layer in enumerate(self.layers):
            layer.W      = np.array(weights[2 * i],     dtype=float)
            layer.b      = np.array(weights[2 * i + 1], dtype=float)
            layer.grad_W = np.zeros_like(layer.W)
            layer.grad_b = np.zeros_like(layer.b)

    def get_weights(self):
        out = []
        for layer in self.layers:
            out.append(layer.W)
            out.append(layer.b)
        return out

    def forward(self, X):
        for layer, act in zip(self.layers, self.activations):
            X = layer.forward(X)
            X = act.forward(X)
        return X

    def backward(self, y_true, y_pred):
        grad = self.loss_fn.backward()
        for layer, act in reversed(list(zip(self.layers, self.activations))):
            grad = act.backward(grad)
            grad = layer.backward(grad, self.weight_decay)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_s, y_s = X[perm], y[perm]
            epoch_loss, num_batches = 0.0, 0
            for i in range(0, len(X_s), batch_size):
                xb, yb = X_s[i:i+batch_size], y_s[i:i+batch_size]
                y_pred = self.forward(xb)
                loss   = self.loss_fn.forward(y_pred, yb)
                epoch_loss  += loss
                num_batches += 1
                self.backward(yb, y_pred)
                self.update_weights()
        return epoch_loss / max(num_batches, 1)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


class _Identity:
    def forward(self, x):  return x
    def backward(self, g): return g
