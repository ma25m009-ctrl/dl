import argparse
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb

from utils.data_loader import load_data
from ann.neural_layer import DenseLayer
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, RMSProp, Adam, NAG, Nadam
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")
    parser.add_argument("-d",   "--dataset",       default="mnist",
                        choices=["mnist", "fashion_mnist", "fashion"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",   "--loss",          default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",   "--optimizer",     default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=2)
    parser.add_argument("-sz",  "--hidden_size",   nargs="+",  type=int, default=[128, 128])
    parser.add_argument("-a",   "--activation",    default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("--experiment", default="general")
    return parser.parse_args()


def get_activation(name):
    if name == "relu":    return ReLU()
    if name == "sigmoid": return Sigmoid()
    if name == "tanh":    return Tanh()
    raise ValueError(f"Unknown activation: {name}")


def get_loss(name):
    if name == "cross_entropy":      return CrossEntropy()
    if name == "mean_squared_error": return MSE()
    raise ValueError(f"Unknown loss: {name}")


def get_optimizer(name, lr):
    if name == "sgd":      return SGD(lr)
    if name == "momentum": return Momentum(lr)
    if name == "nag":      return NAG(lr)
    if name == "rmsprop":  return RMSProp(lr)
    if name == "adam":     return Adam(lr)
    if name == "nadam":    return Nadam(lr)
    raise ValueError(f"Unknown optimizer: {name}")


def build_network(hidden_sizes, activation, weight_init):
    layers, activations = [], []
    prev = 784
    for h in hidden_sizes:
        layers.append(DenseLayer(prev, h, weight_init))
        activations.append(get_activation(activation))
        prev = h
    layers.append(DenseLayer(prev, 10, weight_init))
    activations.append(Softmax())
    return layers, activations


def save_weights(model, path):
    """Save weights as flat numpy object array: [W0, b0, W1, b1, ...]"""
    weights = []
    for layer in model.layers:
        weights.append(layer.W)
        weights.append(layer.b)
    # Save as object array so shapes are preserved
    arr = np.empty(len(weights), dtype=object)
    for i, w in enumerate(weights):
        arr[i] = w
    np.save(path, arr)


def main():
    args = parse_arguments()

    wandb.init(project="da6401-assignment1", name=args.experiment, config=vars(args))

    dataset_name = "fashion_mnist" if args.dataset == "fashion" else args.dataset
    X_train, y_train, X_val, y_val, X_test, y_test_oh, y_test = load_data(dataset_name)

    layers, activations = build_network(args.hidden_size, args.activation, args.weight_init)
    loss_fn   = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model     = NeuralNetwork(layers, activations, loss_fn, optimizer,
                              weight_decay=args.weight_decay)

    best_val_acc = 0.0
    src_dir = os.path.dirname(os.path.abspath(__file__))

    for epoch in range(args.epochs):
        model.train(X_train, y_train, 1, args.batch_size)

        y_pred_train = model.forward(X_train)
        train_loss   = loss_fn.forward(y_pred_train, y_train)
        train_acc    = np.mean(model.predict(X_train) == np.argmax(y_train, axis=1))
        val_acc      = np.mean(model.predict(X_val) == np.argmax(y_val, axis=1))

        print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        wandb.log({
            "epoch":               epoch + 1,
            "loss":                train_loss,
            "train_accuracy":      train_acc,
            "validation_accuracy": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save to src/best_model.npy (where grader looks)
            save_weights(model, os.path.join(src_dir, "best_model.npy"))
            # Also save config
            with open(os.path.join(src_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=4)

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
