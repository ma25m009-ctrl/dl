import argparse
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from ann.neural_layer import DenseLayer
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.neural_network import NeuralNetwork


def get_activation(name):
    if name == "relu":    return ReLU()
    if name == "sigmoid": return Sigmoid()
    if name == "tanh":    return Tanh()
    raise ValueError(f"Unknown activation: {name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with a saved MLP model")
    parser.add_argument("--model_path",  default="src/best_model.npy")
    parser.add_argument("--config_path", default="src/config.json")
    parser.add_argument("--output_dir",  default="src")
    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.config_path) as f:
        config = json.load(f)

    weights = np.load(args.model_path, allow_pickle=True)

    dataset_name = config.get("dataset", "mnist")
    if dataset_name == "fashion":
        dataset_name = "fashion_mnist"

    _, _, _, _, X_test, _, y_test = load_data(dataset_name)

    layers, activations = [], []
    prev = 784
    for h in config["hidden_size"]:
        layers.append(DenseLayer(prev, h))
        activations.append(get_activation(config["activation"]))
        prev = h
    layers.append(DenseLayer(prev, 10))
    activations.append(Softmax())

    model = NeuralNetwork(layers, activations, None, None)
    model.set_weights(list(weights))

    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = {"accuracy": float(acc), "precision": float(prec),
               "recall": float(rec), "f1_score": float(f1)}
    with open(os.path.join(args.output_dir, "inference_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved.")


if __name__ == "__main__":
    main()
