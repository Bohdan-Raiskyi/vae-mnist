"""Запуск експериментів з терміналу (зручно для довгих прогонів, напр. на ніч).

    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --exp configs/experiments/e2_latent_dim.yaml
    python scripts/run_experiments.py --all --quick      # швидка перевірка (300 кроків, 1 сід)

Вже завершені запуски (є runs/<name>/metrics.json) пропускаються —
прогін можна перервати (Ctrl+C) і продовжити тією ж командою.
"""
import argparse

import _path  # noqa: F401
from src.experiment import list_experiment_files, run_experiment

QUICK = {"train.steps": 300, "train.eval_every": 100, "eval.n_generated": 2000}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="*", default=[], help="шляхи до yaml експериментів")
    p.add_argument("--all", action="store_true", help="усі експерименти з configs/experiments")
    p.add_argument("--quick", action="store_true", help="швидкий smoke-тест")
    p.add_argument("--force", action="store_true", help="перерахувати навіть завершені запуски")
    p.add_argument("--verbose", action="store_true", help="лог кожного кроку оцінювання")
    args = p.parse_args()

    exps = list_experiment_files() if args.all else args.exp
    if not exps:
        p.error("вкажи --exp <file.yaml> або --all")
    overrides = QUICK if args.quick else None
    for e in exps:
        print(f"\n=== {e} ===")
        run_experiment(e, overrides=overrides, force=args.force, verbose=args.verbose)
