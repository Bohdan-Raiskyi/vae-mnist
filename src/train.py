"""Цикл навчання генеративних моделей і класифікатора-оцінювача.

Навчання ведемо в КРОКАХ (а не епохах): так експерименти з різною часткою даних
мають однаковий обчислювальний бюджет. Найкращий чекпоінт обирається за val-loss,
тест використовується лише для фінальних метрик (див. metrics.py).
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_data, infinite, load_mnist_tensors, make_split, TensorLoader
from .losses import compute_loss
from .models import build_model, count_params, MNISTClassifier
from .utils import (CHECKPOINTS_DIR, env_info, get_device, save_json, save_yaml,
                    set_seed, setup_logger)


@torch.no_grad()
def evaluate_loss(model, loader, beta: float) -> dict:
    """Середні recon / KL / NELBO / loss на всьому лоадері."""
    model.eval()
    sums, n = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "nelbo": 0.0}, 0
    for x, y in loader:
        out = model(x, y)
        l = compute_loss(out, x, beta, model.variational)
        b = len(x)
        for k in sums:
            sums[k] += l[k].item() * b
        n += b
    model.train()
    return {k: v / n for k, v in sums.items()}


def train_model(cfg: dict, run_dir: Path | None = None, verbose: bool = True,
                data: dict | None = None):
    """Навчає модель за конфігом. Повертає (model, history: list[dict], info: dict).

    Якщо run_dir задано — пише туди config.yaml, history.csv, train.log, best.pt.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(cfg, run_dir / "config.yaml")
        save_json(env_info(), run_dir / "env.json")
    log = setup_logger(run_dir, verbose=verbose)

    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device(cfg.get("device", "auto"))
    data = data or get_data(cfg, device)
    model = build_model(cfg).to(device)
    tcfg = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                           weight_decay=tcfg.get("weight_decay", 0.0))

    steps = int(tcfg["steps"])
    eval_every = int(tcfg["eval_every"])
    beta = float(tcfg["beta"])
    warmup = int(tcfg.get("kl_warmup_steps", 0) or 0)
    grad_clip = tcfg.get("grad_clip")
    steps_per_epoch = len(data["train"])

    log.info(f"model={cfg['model']} params={count_params(model):,} device={device} "
             f"n_train={data['n_train']} steps={steps} (~{steps / steps_per_epoch:.1f} епох)")

    history, best_val, best_step = [], math.inf, 0
    acc = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "nelbo": 0.0}
    acc_n, status = 0, "ok"
    it = infinite(data["train"])
    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = next(it)
        beta_eff = beta * min(1.0, step / warmup) if warmup > 0 else beta
        out = model(x, y)
        losses = compute_loss(out, x, beta_eff, model.variational)

        if not torch.isfinite(losses["loss"]):
            status = "diverged"
            log.warning(f"step {step}: loss = {losses['loss'].item()} — навчання зупинено")
            break

        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        for k in acc:
            acc[k] += losses[k].item()
        acc_n += 1

        if step % eval_every == 0 or step == steps:
            val = evaluate_loss(model, data["val"], beta)
            row = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                   "time_s": round(time.time() - t0, 2), "beta_eff": beta_eff,
                   **{f"train_{k}": v / acc_n for k, v in acc.items()},
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(row)
            acc = {k: 0.0 for k in acc}
            acc_n = 0
            improved = val["loss"] < best_val and step >= warmup
            if improved:
                best_val, best_step = val["loss"], step
                if run_dir is not None:
                    torch.save(model.state_dict(), run_dir / "best.pt")
                else:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log.info(f"step {step:>6} | train loss {row['train_loss']:.2f} "
                     f"(rec {row['train_recon']:.2f}, kl {row['train_kl']:.2f}) | "
                     f"val loss {val['loss']:.2f} {'*' if improved else ''}")
            if run_dir is not None:
                _write_history(history, run_dir / "history.csv")

    train_time = time.time() - t0
    # повертаємо модель з найкращого (за валідацією) кроку
    if best_step > 0:
        if run_dir is not None:
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        else:
            model.load_state_dict(best_state)
    model.eval()
    info = {"status": status, "best_step": best_step, "best_val_loss": best_val,
            "train_time_s": round(train_time, 1), "n_train": data["n_train"],
            "n_params": count_params(model)}
    log.info(f"Готово за {train_time:.0f} с, найкращий крок {best_step}, val loss {best_val:.2f}")
    return model, history, info


def _write_history(history: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


# ----------------------------------------------------------------------------
# Класифікатор-оцінювач
# ----------------------------------------------------------------------------
EVALUATOR_PATH = CHECKPOINTS_DIR / "evaluator.pt"


def train_evaluator(epochs: int = 6, lr: float = 1e-3, batch_size: int = 128,
                    seed: int = 0, path: Path = EVALUATOR_PATH, verbose: bool = True):
    """Навчає CNN-класифікатор на тих самих train-індексах, що й генеративні моделі.

    Легка аугментація (зсуви ±2 px) робить його стійкішим до «неідеальних» генерацій.
    """
    set_seed(seed)
    device = get_device()
    x_all, y_all, x_te, y_te = load_mnist_tensors()
    split = make_split(y_all)
    tr, va = np.array(split["train_idx"]), np.array(split["val_idx"])
    train = TensorLoader(x_all[tr], y_all[tr], batch_size, True, device, seed)
    val = TensorLoader(x_all[va], y_all[va], 1000, False, device)
    test = TensorLoader(x_te, y_te, 1000, False, device)

    clf = MNISTClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(train))
    for ep in range(1, epochs + 1):
        clf.train()
        for x, y in train:
            dx, dy = np.random.randint(-2, 3, size=2)
            x = torch.roll(x, shifts=(int(dx), int(dy)), dims=(2, 3))
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc_val = classifier_accuracy(clf, val)
        if verbose:
            print(f"epoch {ep}: loss {loss.item():.4f}, val acc {acc_val:.4f}")
    acc_test = classifier_accuracy(clf, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), path)
    if verbose:
        print(f"test acc = {acc_test:.4f}; збережено в {path}")
    clf.eval()
    return clf, {"val_acc": acc_val, "test_acc": acc_test}


@torch.no_grad()
def classifier_accuracy(clf, loader) -> float:
    clf.eval()
    correct, n = 0, 0
    for x, y in loader:
        correct += (clf(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


def load_evaluator(device=None, path: Path = EVALUATOR_PATH) -> MNISTClassifier:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не знайдено — спершу навчи класифікатор у notebooks/02_sanity_check.ipynb "
            "або `python scripts/train_evaluator.py`.")
    device = device or get_device()
    clf = MNISTClassifier().to(device)
    clf.load_state_dict(torch.load(path, map_location=device))
    clf.eval()
    return clf
