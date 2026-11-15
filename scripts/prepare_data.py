"""Завантажує MNIST і створює фіксоване розбиття train/val (data/split.json).

    python scripts/prepare_data.py
"""
import _path  # noqa: F401
from src.data import download_mnist, load_mnist_tensors, make_split

if __name__ == "__main__":
    download_mnist()
    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    split = make_split(y_tr)
    print(f"train: {len(split['train_idx'])}, val: {len(split['val_idx'])}, test: {len(x_te)}")
    print("Готово: data/MNIST/raw + data/split.json")
