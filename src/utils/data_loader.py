import numpy as np
import os
import gzip
import urllib.request
from sklearn.model_selection import train_test_split


# ── Fallback download URLs (used only if keras import fails) ──────────────────
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".mnist_data")

URLS = {
    "mnist": {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    }
}


def _download(url, dest_path):
    """Download file only if not already cached."""
    if not os.path.exists(dest_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest_path)


def _load_images(path):
    """Parse IDX3 gz → float64 array (N, 784) normalised to [0, 1]."""
    with gzip.open(path, "rb") as f:
        f.read(16)  # skip header: magic(4) + n_images(4) + rows(4) + cols(4)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 784).astype(np.float64) / 255.0


def _load_labels(path):
    """Parse IDX1 gz → int64 array (N,)."""
    with gzip.open(path, "rb") as f:
        f.read(8)   # skip header: magic(4) + n_labels(4)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def _load_via_fallback(dataset):
    """
    Download and parse raw IDX binary files directly using urllib + numpy.
    Used when keras 3.x is installed and requires tensorflow as backend.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    paths = {}
    for key, url in URLS[dataset].items():
        filename  = url.split("/")[-1]
        dest_path = os.path.join(CACHE_DIR, f"{dataset}_{filename}")
        _download(url, dest_path)
        paths[key] = dest_path

    X_tr = _load_images(paths["train_images"])
    y_tr = _load_labels(paths["train_labels"])
    X_te = _load_images(paths["test_images"])
    y_te = _load_labels(paths["test_labels"])

    return (X_tr, y_tr), (X_te, y_te)


def _load_via_keras(dataset):
    """
    Load via keras.datasets — works when keras 2.x is installed.
    """
    if dataset == "mnist":
        from keras.datasets import mnist
        return mnist.load_data()
    else:
        from keras.datasets import fashion_mnist
        return fashion_mnist.load_data()


def one_hot_encode(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def load_data(dataset="mnist"):
    """
    Load MNIST or Fashion-MNIST.

    Tries keras.datasets first (satisfies the keras>=2.7.0 requirement).
    If keras 3.x is installed and requires tensorflow, falls back to
    downloading the raw IDX files directly via urllib — no tensorflow needed.

    Args:
        dataset: "mnist" or "fashion_mnist"

    Returns:
        X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh, y_test
    """

    if dataset not in ("mnist", "fashion_mnist"):
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    # Try keras first; fall back silently if tensorflow is missing
    try:
        (X_tr, y_tr), (X_te, y_te) = _load_via_keras(dataset)
        # keras returns (N,28,28) — flatten here
        X_tr = X_tr.reshape(-1, 784).astype(np.float64) / 255.0
        X_te = X_te.reshape(-1, 784).astype(np.float64) / 255.0
        y_tr = y_tr.astype(np.int64)
        y_te = y_te.astype(np.int64)
    except Exception:
        # keras 3.x needs tensorflow — use direct urllib fallback instead
        (X_tr, y_tr), (X_te, y_te) = _load_via_fallback(dataset)

    # Combine all 70 000 samples then re-split 80 / 10 / 10
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return (
        X_train,
        one_hot_encode(y_train),
        X_val,
        one_hot_encode(y_val),
        X_test,
        one_hot_encode(y_test),
        y_test,
    )
