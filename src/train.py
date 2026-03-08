import argparse
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb

from utils.data_loader import load_data
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


def main():
    args = parse_arguments()

    wandb.init(project="da6401-assignment1", name=args.experiment, config=vars(args))

    dataset_name = "fashion_mnist" if args.dataset == "fashion" else args.dataset
    X_train, y_train, X_val, y_val, X_test, y_test_oh, y_test = load_data(dataset_name)

    model   = NeuralNetwork(args)
    loss_fn = model.loss_fn

    best_val_acc = 0.0
    src_dir = os.path.dirname(os.path.abspath(__file__))

    for epoch in range(args.epochs):
        model.train(X_train, y_train, 1, args.batch_size)

        y_pred_train, _ = model.forward(X_train)
        train_loss  = loss_fn.forward(y_pred_train, y_train)
        train_acc   = np.mean(model.predict(X_train) == np.argmax(y_train, axis=1))
        val_acc     = np.mean(model.predict(X_val)   == np.argmax(y_val,   axis=1))

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
            np.save(os.path.join(src_dir, "best_model.npy"), model.get_weights())
            with open(os.path.join(src_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=4)

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
