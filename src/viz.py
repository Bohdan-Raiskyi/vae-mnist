"""Графіки та візуалізації для звіту (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .utils import FIGURES_DIR

# Категоріальна палітра (фіксований порядок; не циклимо)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Послідовна синя шкала для впорядкованих параметрів (latent_dim, β, частка даних)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
TEXT, MUTED, GRID = "#1f1e1c", "#52514e", "#e4e3df"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
    })


def ordered_colors(n: int) -> list[str]:
    """n кольорів з синьої шкали (світліший = менше значення параметра)."""
    if n <= len(BLUES):
        idx = np.linspace(0, len(BLUES) - 1, n).round().astype(int)
        return [BLUES[i] for i in idx]
    return [plt.cm.Blues(v) for v in np.linspace(0.35, 0.95, n)]


def save_fig(fig, name: str, folder: Path = FIGURES_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


# ----------------------------------------------------------------------------
# Зображення
# ----------------------------------------------------------------------------
def to_numpy_images(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.reshape(-1, 28, 28)


def image_grid(images, nrow: int, pad: int = 2) -> np.ndarray:
    """Склеює зображення в одну сітку (nrow рядків)."""
    imgs = to_numpy_images(images)
    n = len(imgs)
    ncol = int(np.ceil(n / nrow))
    grid = np.ones((nrow * (28 + pad) + pad, ncol * (28 + pad) + pad))
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncol)
        grid[pad + r * (28 + pad): pad + r * (28 + pad) + 28,
             pad + c * (28 + pad): pad + c * (28 + pad) + 28] = 1 - im
    return grid


def show_grid(images, nrow: int, ax=None, title: str | None = None, ylabels=None):
    """Сітка цифр (чорні цифри на білому фоні)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(0.45 * np.ceil(len(images) / nrow) + 0.5, 0.45 * nrow + 0.4))
    ax.imshow(image_grid(images, nrow), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if ylabels is not None:
        step = 30
        ax.set_yticks([2 + 14 + i * step for i in range(len(ylabels))])
        ax.set_yticklabels(ylabels)
        ax.tick_params(left=False)
    if title:
        ax.set_title(title)
    return ax


# ----------------------------------------------------------------------------
# Криві та залежності
# ----------------------------------------------------------------------------
def plot_curves_band(histories: dict[str, list[pd.DataFrame]], key: str, ax=None,
                     colors=None, x: str = "step", title: str | None = None,
                     ylabel: str | None = None, logy: bool = False):
    """Криві навчання: середнє по сідах (лінія) ± std (смуга). histories: label → [df_seed0, ...]."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    colors = colors or COLORS
    for (label, dfs), color in zip(histories.items(), colors):
        dfs = [d for d in dfs if key in d and len(d)]
        if not dfs:
            continue
        L = min(len(d) for d in dfs)
        arr = np.stack([d[key].values[:L] for d in dfs])
        xs = dfs[0][x].values[:L]
        m, s = arr.mean(0), arr.std(0)
        ax.plot(xs, m, color=color, label=f"{label} (n={len(dfs)})")
        if len(dfs) > 1:
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("крок" if x == "step" else x)
    ax.set_ylabel(ylabel or key)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_metric_vs_param(df: pd.DataFrame, param: str, metric: str, ax=None,
                         group: str | None = None, logx: bool = False,
                         title: str | None = None, ylabel: str | None = None,
                         xlabel: str | None = None, color: str = COLORS[0]):
    """Метрика як функція параметра: середнє ± std по сідах (+ окремі сіди точками)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.4))
    groups = [(None, df)] if group is None else list(df.groupby(group))
    for i, (g, d) in enumerate(groups):
        c = color if group is None else COLORS[i]
        agg = d.groupby(param)[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg[param], agg["mean"], yerr=agg["std"].fillna(0), marker="o",
                    color=c, capsize=3, label=None if g is None else f"{group}={g}")
        ax.scatter(d[param], d[metric], color=c, s=10, alpha=0.35, zorder=3)
    if logx:
        ax.set_xscale("log")
        vals = sorted(df[param].unique())
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{v:g}" for v in vals])
        ax.minorticks_off()
    ax.set_xlabel(xlabel or param)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    if group is not None:
        ax.legend(fontsize=8)
    return ax


# ----------------------------------------------------------------------------
# Латентний простір
# ----------------------------------------------------------------------------
TAB10_SAFE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948", "#52514e", "#8a6d3b"]


def latent_scatter(z: np.ndarray, y: np.ndarray, ax=None, title: str | None = None,
                   max_points: int = 5000):
    """2D-проєкція латентів, колір = справжня цифра (підписи центроїдів — щоб не покладатися лише на колір)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5))
    idx = np.random.default_rng(0).permutation(len(z))[:max_points]
    z, y = z[idx], y[idx]
    for c in range(10):
        m = y == c
        ax.scatter(z[m, 0], z[m, 1], s=3, color=TAB10_SAFE[c], alpha=0.6, label=str(c))
        cx, cy = np.median(z[m, 0]), np.median(z[m, 1])
        ax.text(cx, cy, str(c), fontsize=12, weight="bold", ha="center", va="center",
                color=TEXT, bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=TAB10_SAFE[c], lw=1.5))
    ax.legend(markerscale=4, fontsize=8, ncol=2, title="цифра", loc="best")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    if title:
        ax.set_title(title)
    return ax


@torch.no_grad()
def latent_manifold(model, n: int = 20, lim: float = 2.5, y=None):
    """Декодує рівномірну сітку в 2D-латентному просторі → зображення (n*n, 28, 28).

    Сітка береться за квантилями N(0,1), як у Kingma & Welling (2013).
    """
    from scipy.stats import norm
    device = next(model.parameters()).device
    q = norm.ppf(np.linspace(0.03, 0.97, n))
    zz = np.array([[a, b] for b in q[::-1] for a in q], dtype=np.float32)
    z = torch.from_numpy(zz).to(device)
    labels = None if y is None else torch.full((len(z),), y, device=device, dtype=torch.long)
    return model.sample(len(z), y=labels, z=z)


@torch.no_grad()
def interpolate(model, x1, x2, steps: int = 10, y1=None, y2=None):
    """Лінійна інтерполяція між μ двох зображень."""
    mu1, _ = model.encode(x1[None], None if y1 is None else y1[None])
    mu2, _ = model.encode(x2[None], None if y2 is None else y2[None])
    t = torch.linspace(0, 1, steps, device=mu1.device)[:, None]
    z = (1 - t) * mu1 + t * mu2
    return model.sample(steps, z=z)
