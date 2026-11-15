"""Функції втрат.

loss = reconstruction + β · KL
  reconstruction — бінарна крос-ентропія (Bernoulli-правдоподібність пікселів), сума по пікселях
  KL             — KL( q(z|x) || N(0, I) ), сума по вимірах латенту
Обидва доданки — у натах на одне зображення (середнє по батчу).
Від'ємний ELBO (NELBO) = reconstruction + KL (тобто β = 1) — верхня межа −log p(x).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """BCE, сума по пікселях → вектор (B,)."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Аналітичний KL(N(μ, σ²) || N(0, I)) → вектор (B,)."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL по кожному виміру латенту (B, d) — для аналізу posterior collapse."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def compute_loss(out: dict, x: torch.Tensor, beta: float, variational: bool) -> dict:
    rec = reconstruction_loss(out["logits"], x)
    if variational:
        kl = kl_divergence(out["mu"], out["logvar"])
    else:
        kl = torch.zeros_like(rec)
    loss = rec + beta * kl
    return {"loss": loss.mean(), "recon": rec.mean(), "kl": kl.mean(),
            "nelbo": (rec + kl).mean()}
