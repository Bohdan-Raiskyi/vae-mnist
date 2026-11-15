"""Завантаження MNIST, фіксоване розбиття train/val/test, підвибірки, швидкий GPU-лоадер.

MNIST маленький (60k × 28×28), тому весь датасет тримаємо як тензор на GPU і
нарізаємо батчі вручну — це у рази швидше за стандартний DataLoader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import PROJECT_ROOT, load_json, save_json

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_FILE = DATA_DIR / "split.json"


def download_mnist(root: Path = DATA_DIR) -> None:
    """Завантажує MNIST через torchvision у data/MNIST/raw."""
    from torchvision import datasets
    root.mkdir(parents=True, exist_ok=True)
    # Резервне дзеркало на GitHub (torchvision сам перевіряє MD5 файлів)
    mirror = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    if mirror not in datasets.MNIST.mirrors:
        datasets.MNIST.mirrors = list(datasets.MNIST.mirrors) + [mirror]
    datasets.MNIST(root, train=True, download=True)
    datasets.MNIST(root, train=False, download=True)


def load_mnist_tensors(root: Path = DATA_DIR):
    """Повертає (x_train, y_train, x_test, y_test); x — float32 у [0, 1], форма (N, 1, 28, 28)."""
    from torchvision import datasets
    try:
        tr = datasets.MNIST(root, train=True, download=False)
        te = datasets.MNIST(root, train=False, download=False)
    except RuntimeError as e:
        raise RuntimeError(
            "MNIST не знайдено. Спершу запусти notebooks/01_data.ipynb "
            "або `python scripts/prepare_data.py`.") from e
    x_tr = tr.data.float().div(255.0).unsqueeze(1)
    x_te = te.data.float().div(255.0).unsqueeze(1)
    return x_tr, tr.targets.long(), x_te, te.targets.long()


def stratified_indices(labels: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Стратифіковано обирає n_select індексів (пропорції класів зберігаються)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    exact = n_select * counts / len(labels)
    k = np.floor(exact).astype(int)
    # метод найбільших залишків → сума рівно n_select
    for i in np.argsort(-(exact - k))[: n_select - k.sum()]:
        k[i] += 1
    chosen = []
    for c, kc in zip(classes, k):
        idx = np.where(labels == c)[0]
        chosen.append(rng.choice(idx, size=kc, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return np.sort(out)


def make_split(y_train: torch.Tensor, val_size: int = 5000, seed: int = 42,
               path: Path = SPLIT_FILE, overwrite: bool = False) -> dict:
    """Створює (або читає) фіксоване стратифіковане розбиття 60k → train/val."""
    if path.exists() and not overwrite:
        split = load_json(path)
        if split["val_size"] == val_size and split["seed"] == seed:
            return split
    labels = y_train.numpy()
    val_idx = stratified_indices(labels, val_size, seed)
    train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)
    split = {"seed": seed, "val_size": int(len(val_idx)),
             "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(split, path)
    return split


class TensorLoader:
    """Простий ітератор батчів по тензорах, що вже лежать на потрібному пристрої."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int,
                 shuffle: bool, device: torch.device, seed: int = 0):
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.x)
        if self.shuffle:
            idx = torch.randperm(n, generator=self.gen).to(self.x.device)
        else:
            idx = torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            j = idx[i:i + self.batch_size]
            yield self.x[j], self.y[j]


def infinite(loader: TensorLoader):
    """Нескінченний потік батчів (навчання рахуємо в кроках, а не епохах)."""
    while True:
        yield from loader


def get_data(cfg: dict, device: torch.device) -> dict:
    """Готує всі лоадери згідно з конфігом.

    data.train_fraction < 1 → стратифікована підвибірка train (сід = сід запуску,
    тому розкид між сідами включає і вибір підвибірки).
    """
    x_tr_all, y_tr_all, x_te, y_te = load_mnist_tensors()
    dcfg = cfg["data"]
    split = make_split(y_tr_all, dcfg["val_size"], dcfg["split_seed"])
    tr_idx = np.array(split["train_idx"])
    va_idx = np.array(split["val_idx"])

    frac = float(dcfg.get("train_fraction", 1.0))
    if frac < 1.0:
        sub = stratified_indices(y_tr_all.numpy()[tr_idx], int(len(tr_idx) * frac), cfg["seed"])
        tr_idx = tr_idx[sub]

    bs = cfg["train"]["batch_size"]
    x_tr, y_tr = x_tr_all[tr_idx], y_tr_all[tr_idx]
    x_va, y_va = x_tr_all[va_idx], y_tr_all[va_idx]
    return {
        "train": TensorLoader(x_tr, y_tr, bs, True, device, seed=cfg["seed"]),
        "val": TensorLoader(x_va, y_va, 1000, False, device),
        "test": TensorLoader(x_te, y_te, 1000, False, device),
        "n_train": len(tr_idx),
    }
