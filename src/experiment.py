"""Запуск серій експериментів з YAML-файлів у configs/experiments/.

Формат файлу експерименту:
    name: e2_latent_dim
    description: ...
    base: configs/base.yaml        # базовий конфіг
    fixed:                         # (необов'язково) зміни відносно base для всіх запусків
      model.type: vae
    grid:                          # декартів добуток значень
      model.latent_dim: [2, 4, 8]
    seeds: [0, 1, 2]

Кожен запуск зберігається в runs/<run_name>/:
    config.yaml, env.json, history.csv, train.log, best.pt, metrics.json
Запуски з однаковим конфігом мають однакову назву → не перераховуються
(можна переривати й продовжувати; спільні запуски між експериментами переюзаються).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import pandas as pd

from .data import get_data
from .metrics import evaluate_run
from .train import load_evaluator, train_model
from .utils import (CONFIGS_DIR, PROJECT_ROOT, RUNS_DIR, get_device, load_config,
                    load_json, load_yaml, run_name, save_json)

EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"


def list_experiment_files() -> list[Path]:
    return sorted(EXPERIMENTS_DIR.glob("*.yaml"))


def expand_experiment(exp_path: str | Path, overrides: dict | None = None) -> list[dict]:
    """Розгортає файл експерименту в список повних конфігів."""
    exp_path = Path(exp_path)
    if not exp_path.is_absolute():
        exp_path = PROJECT_ROOT / exp_path
    spec = load_yaml(exp_path)
    base = spec.get("base", "configs/base.yaml")
    grid = spec.get("grid", {}) or {}
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    cfgs = []
    for combo in combos:
        for seed in spec.get("seeds", [0]):
            ov = dict(spec.get("fixed", {}) or {})
            ov.update(dict(zip(keys, combo)))
            ov["seed"] = seed
            ov.update(overrides or {})
            cfg = load_config(base, ov)
            cfg["experiment"] = spec["name"]
            cfgs.append(cfg)
    return cfgs


def run_single(cfg: dict, data_cache: dict | None = None, clf=None, force: bool = False,
               verbose: bool = False) -> dict:
    """Навчає + оцінює один конфіг. Якщо metrics.json вже є — просто читає його."""
    name = run_name(cfg)
    run_dir = RUNS_DIR / name
    mpath = run_dir / "metrics.json"
    if mpath.exists() and not force:
        return load_json(mpath)

    device = get_device(cfg.get("device", "auto"))
    clf = clf or load_evaluator(device)
    model, history, info = train_model(cfg, run_dir, verbose=verbose)
    data = get_data(cfg, device)
    metrics = evaluate_run(model, clf, data, cfg) if info["status"] == "ok" else {}
    record = {"run": name, "experiment": cfg.get("experiment"),
              "model_type": cfg["model"]["type"], "arch": cfg["model"]["arch"],
              "latent_dim": cfg["model"]["latent_dim"], "beta": cfg["train"]["beta"],
              "train_fraction": cfg["data"]["train_fraction"], "seed": cfg["seed"],
              "steps": cfg["train"]["steps"], **info, **metrics}
    save_json(record, mpath)
    return record


def run_experiment(exp_path: str | Path, overrides: dict | None = None,
                   force: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Запускає всі конфіги експерименту послідовно. Повертає таблицю результатів."""
    cfgs = expand_experiment(exp_path, overrides)
    clf = load_evaluator()
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs, 1):
        name = run_name(cfg)
        done = (RUNS_DIR / name / "metrics.json").exists() and not force
        print(f"[{i}/{len(cfgs)}] {name} {'(вже є, пропускаю)' if done else ''}", flush=True)
        rec = run_single(cfg, clf=clf, force=force, verbose=verbose)
        rec = dict(rec, experiment=cfg["experiment"])
        rows.append(rec)
        if not done:
            print(f"    status={rec.get('status')} test_nelbo={rec.get('test_nelbo', float('nan')):.2f} "
                  f"fid_prior={_fmt(rec.get('fid_prior'))} time={rec.get('train_time_s')}s", flush=True)
    print(f"Експеримент завершено за {(time.time() - t0) / 60:.1f} хв")
    return pd.DataFrame(rows)


def experiment_runs(exp_path: str | Path, overrides: dict | None = None) -> list[str]:
    return [run_name(c) for c in expand_experiment(exp_path, overrides)]


def collect_results(overrides: dict | None = None) -> pd.DataFrame:
    """Збирає metrics.json усіх експериментів у DataFrame (рядок = запуск × експеримент)."""
    rows = []
    for exp in list_experiment_files():
        spec = load_yaml(exp)
        for name in experiment_runs(exp, overrides):
            p = RUNS_DIR / name / "metrics.json"
            if p.exists():
                rows.append(dict(load_json(p), experiment=spec["name"]))
    return pd.DataFrame(rows)


def load_history(run: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / run / "history.csv")


def load_run_model(run: str, device=None):
    """Відновлює модель запуску з runs/<run>/best.pt. Повертає (model, cfg)."""
    import torch

    from .models import build_model
    cfg = load_yaml(RUNS_DIR / run / "config.yaml")
    device = device or get_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(RUNS_DIR / run / "best.pt", map_location=device))
    model.eval()
    return model, cfg


def _fmt(v):
    return "—" if v is None else f"{v:.2f}"
