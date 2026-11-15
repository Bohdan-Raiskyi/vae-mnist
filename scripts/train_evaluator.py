"""Навчає CNN-класифікатор-оцінювач (потрібен для метрик генерації).

    python scripts/train_evaluator.py --epochs 6
"""
import argparse

import _path  # noqa: F401
from src.train import train_evaluator

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()
    train_evaluator(epochs=args.epochs)
