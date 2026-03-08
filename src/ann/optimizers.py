import numpy as np


class SGD:

    def __init__(self, lr):
        self.lr = lr

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:

    def __init__(self, lr, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.vW:
            self.vW[key] = np.zeros_like(layer.W)
            self.vb[key] = np.zeros_like(layer.b)

        self.vW[key] = self.beta * self.vW[key] + (1 - self.beta) * layer.grad_W
        self.vb[key] = self.beta * self.vb[key] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * self.vW[key]
        layer.b -= self.lr * self.vb[key]


class NAG:
    """
    Nesterov Accelerated Gradient (standard approximation):
      v_t = beta * v_{t-1} + lr * grad
      w  -= beta * v_t + lr * grad   (lookahead correction)
    """

    def __init__(self, lr, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.vW:
            self.vW[key] = np.zeros_like(layer.W)
            self.vb[key] = np.zeros_like(layer.b)

        self.vW[key] = self.beta * self.vW[key] + self.lr * layer.grad_W
        self.vb[key] = self.beta * self.vb[key] + self.lr * layer.grad_b

        layer.W -= self.beta * self.vW[key] + self.lr * layer.grad_W
        layer.b -= self.beta * self.vb[key] + self.lr * layer.grad_b


class RMSProp:

    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self.sW   = {}
        self.sb   = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.sW:
            self.sW[key] = np.zeros_like(layer.W)
            self.sb[key] = np.zeros_like(layer.b)

        self.sW[key] = self.beta * self.sW[key] + (1 - self.beta) * (layer.grad_W ** 2)
        self.sb[key] = self.beta * self.sb[key] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[key]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[key]) + self.eps)


class Adam:

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.mW  = {}
        self.vW  = {}
        self.mb  = {}
        self.vb  = {}
        # Per-layer timestep — avoids t incrementing multiple times per batch
        self.t   = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.mW:
            self.mW[key] = np.zeros_like(layer.W)
            self.vW[key] = np.zeros_like(layer.W)
            self.mb[key] = np.zeros_like(layer.b)
            self.vb[key] = np.zeros_like(layer.b)
            self.t[key]  = 0

        self.t[key] += 1
        t = self.t[key]

        # Weights
        gW = layer.grad_W
        self.mW[key] = self.b1 * self.mW[key] + (1 - self.b1) * gW
        self.vW[key] = self.b2 * self.vW[key] + (1 - self.b2) * (gW ** 2)
        mW_hat = self.mW[key] / (1 - self.b1 ** t)
        vW_hat = self.vW[key] / (1 - self.b2 ** t)
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

        # Biases
        gb = layer.grad_b
        self.mb[key] = self.b1 * self.mb[key] + (1 - self.b1) * gb
        self.vb[key] = self.b2 * self.vb[key] + (1 - self.b2) * (gb ** 2)
        mb_hat = self.mb[key] / (1 - self.b1 ** t)
        vb_hat = self.vb[key] / (1 - self.b2 ** t)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class Nadam(Adam):
    """Nesterov Adam: uses a lookahead estimate in place of m_hat."""

    def update(self, layer):
        key = id(layer)
        if key not in self.mW:
            self.mW[key] = np.zeros_like(layer.W)
            self.vW[key] = np.zeros_like(layer.W)
            self.mb[key] = np.zeros_like(layer.b)
            self.vb[key] = np.zeros_like(layer.b)
            self.t[key]  = 0

        self.t[key] += 1
        t = self.t[key]

        # Weights
        gW = layer.grad_W
        self.mW[key] = self.b1 * self.mW[key] + (1 - self.b1) * gW
        self.vW[key] = self.b2 * self.vW[key] + (1 - self.b2) * (gW ** 2)
        mW_hat     = self.mW[key] / (1 - self.b1 ** t)
        vW_hat     = self.vW[key] / (1 - self.b2 ** t)
        nesterov_W = self.b1 * mW_hat + (1 - self.b1) * gW / (1 - self.b1 ** t)
        layer.W   -= self.lr * nesterov_W / (np.sqrt(vW_hat) + self.eps)

        # Biases
        gb = layer.grad_b
        self.mb[key] = self.b1 * self.mb[key] + (1 - self.b1) * gb
        self.vb[key] = self.b2 * self.vb[key] + (1 - self.b2) * (gb ** 2)
        mb_hat     = self.mb[key] / (1 - self.b1 ** t)
        vb_hat     = self.vb[key] / (1 - self.b2 ** t)
        nesterov_b = self.b1 * mb_hat + (1 - self.b1) * gb / (1 - self.b1 ** t)
        layer.b   -= self.lr * nesterov_b / (np.sqrt(vb_hat) + self.eps)
