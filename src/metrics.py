"""Метрики якості моделей.

Відновлення / правдоподібність (на ТЕСТІ):
  test_recon   — BCE, нати/зображення (менше = краще)
  test_kl      — KL, нати/зображення
  test_nelbo   — від'ємний ELBO = recon + KL (β = 1), верхня межа −log p(x)
  recon_mse    — MSE на піксель при детермінованому відновленні через μ
Латентний простір:
  active_units — к-сть «активних» вимірів (Cov_x(E[z|x]) > 0.01); мало → posterior collapse
Генерація (10k зразків, оцінюємо класифікатором-оцінювачем):
  fid_prior    — FID між ознаками класифікатора для генерацій з N(0, I) і для тесту
  fid_fitted   — те саме, але z з Гаусіани, підігнаної до латентів train (чесний baseline для AE)
  class_entropy— нормована ентропія розподілу класів серед генерацій (1 = всі цифри порівну)
  mean_conf    — середня впевненість класифікатора (max p)
  ambiguous_share — частка «неоднозначних» генерацій (max p < поріг)
  cond_acc     — лише CVAE: частка генерацій, де класифікатор впізнав задану цифру
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg

from .train import evaluate_loss

_REAL_CACHE: dict = {}


# ----------------------------------------------------------------------------
# Базові функції
# ----------------------------------------------------------------------------
@torch.no_grad()
def classifier_outputs(clf, images: torch.Tensor, bs: int = 1000):
    """Повертає (ознаки (N,128), ймовірності (N,10)) як numpy."""
    feats, probs = [], []
    for i in range(0, len(images), bs):
        x = images[i:i + bs]
        f = clf.features(x)
        feats.append(f.cpu())
        probs.append(torch.softmax(clf.out(f), 1).cpu())
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(probs).numpy()


def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    """Fréchet distance між двома Гаусіанами, підігнаними до ознак (Heusel et al., 2017)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    # tr( (s1 s2)^{1/2} ) через симетричну форму s1^{1/2} s2 s1^{1/2}:
    # стабільно навіть для вироджених коваріацій (ReLU-ознаки часто мають нульові виміри)
    w, v = linalg.eigh(s1)
    sqrt_s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = sqrt_s1 @ s2 @ sqrt_s1
    tr_covmean = np.sqrt(np.clip(linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


@torch.no_grad()
def collect_latents(model, loader):
    """μ (або код z для AE) та мітки для всього лоадера."""
    model.eval()
    mus, ys = [], []
    for x, y in loader:
        mu, _ = model.encode(x, y)
        mus.append(mu)
        ys.append(y)
    return torch.cat(mus), torch.cat(ys)


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> int:
    """Burda et al. (2015): вимір активний, якщо дисперсія E[z|x] по даних > threshold."""
    return int((mu.var(0) > threshold).sum().item())


@torch.no_grad()
def kl_per_dimension(model, loader) -> list[float]:
    model.eval()
    tot, n = None, 0
    for x, y in loader:
        mu, logvar = model.encode(x, y)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(0)
        tot = kl if tot is None else tot + kl
        n += len(x)
    return (tot / n).cpu().tolist()


@torch.no_grad()
def recon_mse(model, loader) -> float:
    model.eval()
    se, n = 0.0, 0
    for x, y in loader:
        se += (model.reconstruct(x, y) - x).pow(2).mean((1, 2, 3)).sum().item()
        n += len(x)
    return se / n


@torch.no_grad()
def generate(model, n: int, labels: torch.Tensor | None = None, mean=None, cov=None,
             bs: int = 1000) -> torch.Tensor:
    """Генерує n зображень. mean/cov задані → z ~ N(mean, cov), інакше z ~ N(0, I)."""
    device = next(model.parameters()).device
    if mean is not None:
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    out = []
    for i in range(0, n, bs):
        m = min(bs, n - i)
        z = dist.sample((m,)) if mean is not None else torch.randn(m, model.latent_dim, device=device)
        y = labels[i:i + m] if labels is not None else None
        out.append(model.sample(m, y=y, z=z))
    return torch.cat(out)


def fit_gaussian(mu: torch.Tensor):
    """Гаусіана, підігнана до латентних кодів (для AE — єдиний спосіб щось генерувати)."""
    mean = mu.mean(0)
    cov = torch.cov(mu.T) + 1e-4 * torch.eye(mu.shape[1], device=mu.device)
    if mu.shape[1] == 1:
        cov = cov.view(1, 1)
    return mean, cov


def generation_stats(probs: np.ndarray, threshold: float) -> dict:
    pred = probs.argmax(1)
    hist = np.bincount(pred, minlength=10) / len(pred)
    ent = -(hist[hist > 0] * np.log(hist[hist > 0])).sum() / math.log(10)
    conf = probs.max(1)
    return {"class_entropy": float(ent), "class_share_min": float(hist.min()),
            "class_share_max": float(hist.max()), "mean_conf": float(conf.mean()),
            "ambiguous_share": float((conf < threshold).mean()),
            "class_hist": hist.tolist()}


def real_reference(clf, test_loader, threshold: float) -> dict:
    """Ознаки та статистики справжнього тесту (кешується)."""
    key = (id(clf), threshold)
    if key not in _REAL_CACHE:
        x = torch.cat([b[0] for b in test_loader])
        feats, probs = classifier_outputs(clf, x)
        _REAL_CACHE[key] = {"feats": feats, "probs": probs,
                            "stats": generation_stats(probs, threshold)}
    return _REAL_CACHE[key]


# ----------------------------------------------------------------------------
# Повна оцінка запуску
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_run(model, clf, data: dict, cfg: dict) -> dict:
    """Обчислює всі метрики для навченої моделі (вибрана за val), на тесті."""
    torch.manual_seed(12345)  # фіксуємо шум генерації → метрики відтворювані
    ecfg = cfg.get("eval", {})
    n_gen = int(ecfg.get("n_generated", 10000))
    thr = float(ecfg.get("ambiguous_threshold", 0.9))
    model.eval()

    test = evaluate_loss(model, data["test"], beta=1.0)
    m: dict = {"test_recon": test["recon"], "test_kl": test["kl"],
               "test_nelbo": test["nelbo"], "recon_mse": recon_mse(model, data["test"])}

    mu_test, _ = collect_latents(model, data["test"])
    if model.variational:
        m["active_units"] = active_units(mu_test)
        m["kl_per_dim"] = kl_per_dimension(model, data["test"])
    else:
        m["active_units"] = None

    ref = real_reference(clf, data["test"], thr)
    m["real_ambiguous_share"] = ref["stats"]["ambiguous_share"]

    device = mu_test.device
    if model.conditional:
        labels = torch.arange(10, device=device).repeat_interleave(math.ceil(n_gen / 10))[:n_gen]
        imgs = generate(model, n_gen, labels=labels)
        feats, probs = classifier_outputs(clf, imgs)
        pred = probs.argmax(1)
        lab = labels.cpu().numpy()
        m["cond_acc"] = float((pred == lab).mean())
        m["cond_acc_per_class"] = [float((pred[lab == c] == c).mean()) for c in range(10)]
        m["fid_prior"] = fid_from_features(feats, ref["feats"])
        m.update({k: v for k, v in generation_stats(probs, thr).items()})
        m["fid_fitted"] = None
    else:
        imgs = generate(model, n_gen)
        feats, probs = classifier_outputs(clf, imgs)
        m["fid_prior"] = fid_from_features(feats, ref["feats"])
        m.update(generation_stats(probs, thr))
        mu_train, _ = collect_latents(model, data["train"])
        mean, cov = fit_gaussian(mu_train)
        imgs_f = generate(model, n_gen, mean=mean, cov=cov)
        feats_f, probs_f = classifier_outputs(clf, imgs_f)
        m["fid_fitted"] = fid_from_features(feats_f, ref["feats"])
        m["ambiguous_share_fitted"] = generation_stats(probs_f, thr)["ambiguous_share"]
        m["cond_acc"] = None
    return m
