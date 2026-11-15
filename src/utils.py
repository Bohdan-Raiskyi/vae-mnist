"""Допоміжні функції: конфіги, сіди, пристрій, логування, інформація про середовище."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


# ----------------------------------------------------------------------------
# Відтворюваність
# ----------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Фіксує всі джерела випадковості (python, numpy, torch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def env_info() -> dict[str, Any]:
    """Версії бібліотек, GPU і git-коміт — зберігаються з кожним запуском."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


# ----------------------------------------------------------------------------
# Конфіги
# ----------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict, upd: dict) -> dict:
    """Рекурсивно зливає upd у копію base."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_dotted(d: dict, key: str, value: Any) -> None:
    """set_by_dotted(cfg, "model.latent_dim", 8)"""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_by_dotted(d: dict, key: str, default: Any = None) -> Any:
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def load_config(path: str | Path = CONFIGS_DIR / "base.yaml",
                overrides: dict[str, Any] | None = None) -> dict:
    """Завантажує YAML-конфіг (з підтримкою ключа `base:`) і застосовує overrides."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = load_yaml(path)
    if "base" in cfg:
        base_cfg = load_config(cfg.pop("base"))
        cfg = deep_update(base_cfg, cfg)
    for k, v in (overrides or {}).items():
        set_by_dotted(cfg, k, v)
    return cfg


def config_hash(cfg: dict) -> str:
    """Короткий хеш конфігурації — однакові конфіги = той самий запуск."""
    payload = {k: v for k, v in cfg.items() if k not in ("experiment", "device")}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:6]


def run_name(cfg: dict) -> str:
    """Читабельна назва запуску, напр. vae_conv_z16_b1_s0_3fa2c1."""
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    name = f"{m['type']}_{m['arch']}_z{m['latent_dim']}"
    if m["type"] != "ae":
        name += f"_b{t['beta']:g}"
    if d.get("train_fraction", 1.0) < 1.0:
        name += f"_f{d['train_fraction']:g}"
    name += f"_s{cfg['seed']}_{config_hash(cfg)}"
    return name


# ----------------------------------------------------------------------------
# Логування
# ----------------------------------------------------------------------------
def setup_logger(run_dir: Path | None = None, name: str = "vae",
                 verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"{name}.{run_dir.name if run_dir else 'main'}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
