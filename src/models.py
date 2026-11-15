"""Моделі: AE, VAE, Conditional VAE (MLP- та згорткова архітектури) і класифікатор-оцінювач."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 10
IMG_SHAPE = (1, 28, 28)
IMG_DIM = 28 * 28


# ----------------------------------------------------------------------------
# Енкодери / декодери
# ----------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, out_dim: int, cond_dim: int = 0, hidden=(512, 256)):
        super().__init__()
        layers, d = [], IMG_DIM + cond_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x, c=None):
        h = x.flatten(1)
        if c is not None:
            h = torch.cat([h, c], dim=1)
        return self.head(self.body(h))


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int = 0, hidden=(256, 512)):
        super().__init__()
        layers, d = [], latent_dim + cond_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, IMG_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=1)
        return self.net(z).view(-1, *IMG_SHAPE)  # логіти пікселів


class ConvEncoder(nn.Module):
    """28×28 → 14×14 → 7×7 → FC."""

    def __init__(self, out_dim: int, cond_dim: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7 + cond_dim, 256), nn.ReLU())
        self.head = nn.Linear(256, out_dim)

    def forward(self, x, c=None):
        h = self.conv(x).flatten(1)
        if c is not None:
            h = torch.cat([h, c], dim=1)
        return self.head(self.fc(h))


class ConvDecoder(nn.Module):
    """FC → 7×7 → 14×14 → 28×28 (транспоновані згортки)."""

    def __init__(self, latent_dim: int, cond_dim: int = 0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=1)
        return self.deconv(self.fc(z).view(-1, 64, 7, 7))  # логіти пікселів


# ----------------------------------------------------------------------------
# Генеративна модель
# ----------------------------------------------------------------------------
class GenerativeAE(nn.Module):
    """Одна обгортка для трьох варіантів.

    type = "ae"   — детермінований автоенкодер: x → z → x̂
    type = "vae"  — варіаційний: x → (μ, log σ²) → z ~ N(μ, σ²) → x̂
    type = "cvae" — умовний VAE: енкодер і декодер додатково отримують one-hot мітку
    """

    def __init__(self, model_type: str = "vae", arch: str = "conv", latent_dim: int = 16):
        super().__init__()
        assert model_type in ("ae", "vae", "cvae"), model_type
        assert arch in ("mlp", "conv"), arch
        self.model_type = model_type
        self.arch = arch
        self.latent_dim = latent_dim
        self.variational = model_type in ("vae", "cvae")
        self.conditional = model_type == "cvae"
        cond_dim = N_CLASSES if self.conditional else 0
        enc_out = 2 * latent_dim if self.variational else latent_dim
        Enc, Dec = (ConvEncoder, ConvDecoder) if arch == "conv" else (MLPEncoder, MLPDecoder)
        self.encoder = Enc(enc_out, cond_dim)
        self.decoder = Dec(latent_dim, cond_dim)

    # --- допоміжне --------------------------------------------------------
    def _cond(self, y):
        if not self.conditional:
            return None
        if y is None:
            raise ValueError("CVAE потребує мітки класу y")
        return F.one_hot(y, N_CLASSES).float()

    # --- основне API ------------------------------------------------------
    def encode(self, x, y=None):
        """Повертає (μ, log σ²); для AE log σ² = None, μ = код z."""
        h = self.encoder(x, self._cond(y))
        if self.variational:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar.clamp(-30.0, 20.0)
        return h, None

    @staticmethod
    def reparameterize(mu, logvar):
        """z = μ + σ·ε, ε ~ N(0, I) — трюк репараметризації."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, y=None):
        """Повертає логіти пікселів (sigmoid → яскравість)."""
        return self.decoder(z, self._cond(y))

    def forward(self, x, y=None, sample: bool = True):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar) if (self.variational and sample) else mu
        return {"logits": self.decode(z, y), "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def reconstruct(self, x, y=None):
        """Детерміноване відновлення (через μ) — для візуалізацій і MSE."""
        return torch.sigmoid(self.forward(x, y, sample=False)["logits"])

    @torch.no_grad()
    def sample(self, n: int, y=None, z=None, device=None):
        """Генерація: z ~ N(0, I) → декодер. Для CVAE потрібна мітка y."""
        device = device or next(self.parameters()).device
        if z is None:
            z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z, y))


def build_model(cfg: dict) -> GenerativeAE:
    m = cfg["model"]
    return GenerativeAE(m["type"], m["arch"], int(m["latent_dim"]))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Класифікатор-оцінювач (для метрик якості генерації)
# ----------------------------------------------------------------------------
class MNISTClassifier(nn.Module):
    """Невелика CNN (~99% на MNIST). Передостанній шар (128) — ознаки для FID."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, feat_dim), nn.ReLU())
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(feat_dim, N_CLASSES)

    def features(self, x):
        return self.fc(self.conv(x))

    def forward(self, x):
        return self.out(self.drop(self.features(x)))
