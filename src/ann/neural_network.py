import numpy as np
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.neural_layer import DenseLayer
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, RMSProp, Adam, NAG, Nadam


def _get_activation(name):
    if name == "relu":    return ReLU()
    if name == "sigmoid": return Sigmoid()
    if name == "tanh":    return Tanh()
    raise ValueError(f"Unknown activation: {name}")

def _get_loss(name):
    if name == "cross_entropy":      return CrossEntropy()
    if name == "mean_squared_error": return MSE()
    raise ValueError(f"Unknown loss: {name}")

def _get_optimizer(name, lr):
    if name == "sgd":      return SGD(lr)
    if name == "momentum": return Momentum(lr)
    if name == "nag":      return NAG(lr)
    if name == "rmsprop":  return RMSProp(lr)
    if name == "adam":     return Adam(lr)
    if name == "nadam":    return Nadam(lr)
    raise ValueError(f"Unknown optimizer: {name}")


class NeuralNetwork:

    def __init__(self, config):
        """
        Build network from argparse.Namespace config with fields:
          hidden_size, activation, weight_init, loss, optimizer,
          learning_rate, weight_decay
        """
        hidden_sizes = config.hidden_size
        activation   = config.activation
        weight_init  = getattr(config, 'weight_init', 'xavier')
        loss_name    = getattr(config, 'loss', 'cross_entropy')
        opt_name     = getattr(config, 'optimizer', 'adam')
        lr           = getattr(config, 'learning_rate', 0.001)
        self.weight_decay = getattr(config, 'weight_decay', 0.0)

        # Build layers
        self.layers = []
        self.activations = []
        prev = 784
        for h in hidden_sizes:
            self.layers.append(DenseLayer(prev, h, weight_init))
            self.activations.append(_get_activation(activation))
            prev = h
        self.layers.append(DenseLayer(prev, 10, weight_init))
        self.activations.append(Softmax())

        self.loss_fn   = _get_loss(loss_name)
        self.optimizer = _get_optimizer(opt_name, lr)

    def set_weights(self, weights):
        """
        Accept dict {'W0':..,'b0':..,'W1':..,'b1':..} 
        or flat list/array [W0, b0, W1, b1, ...]
        """
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                layer.W      = np.array(weights[f'W{i}'], dtype=float)
                layer.b      = np.array(weights[f'b{i}'], dtype=float)
                layer.grad_W = np.zeros_like(layer.W)
                layer.grad_b = np.zeros_like(layer.b)
        else:
            weights = list(weights)
            for i, layer in enumerate(self.layers):
                layer.W      = np.array(weights[2*i],   dtype=float)
                layer.b      = np.array(weights[2*i+1], dtype=float)
                layer.grad_W = np.zeros_like(layer.W)
                layer.grad_b = np.zeros_like(layer.b)

    def get_weights(self):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'W{i}'] = layer.W
            weights_dict[f'b{i}'] = layer.b
        return weights_dict

    def forward(self, X):
        """Returns (output, cache) tuple as grader expects."""
        cache = []
        for layer, act in zip(self.layers, self.activations):
            X = layer.forward(X)
            cache.append(X.copy())
            X = act.forward(X)
            cache.append(X.copy())
        return X, cache

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
                y_pred, _ = self.forward(xb)
                loss = self.loss_fn.forward(y_pred, yb)
                epoch_loss  += loss
                num_batches += 1
                self.backward(yb, y_pred)
                self.update_weights()
        return epoch_loss / max(num_batches, 1)

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
