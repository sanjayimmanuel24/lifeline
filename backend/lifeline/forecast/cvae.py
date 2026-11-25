"""Conditional VAE over joint zone x hour incident counts with a negative-binomial likelihood.

Why a CVAE: surges are spatially correlated (rain or an event lifts many zones at once).
The latent z captures that shared state, so samples are *joint* futures. A per-zone
Poisson model only gives independent marginals and under-protects against correlated surges.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LATENT = 8


class CVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, latent: int = LATENT, hidden: int = 256):
        super().__init__()
        self.x_dim, self.c_dim, self.latent = x_dim, c_dim, latent
        self.enc = nn.Sequential(nn.Linear(x_dim + c_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent))
        # decoder = conditional baseline f(c) + latent modulation g(z, c)
        self.base = nn.Sequential(nn.Linear(c_dim, hidden), nn.GELU(), nn.Linear(hidden, x_dim))
        self.dec = nn.Sequential(nn.Linear(latent + c_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, x_dim))
        self.log_r = nn.Parameter(torch.full((x_dim,), 1.5))   # NB dispersion per cell

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.enc(torch.cat([torch.log1p(x), c], -1))
        mu, logvar = h.chunk(2, -1)
        return mu, logvar.clamp(-8, 4)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Returns log-mean of each cell."""
        return self.base(c) + self.dec(torch.cat([z, c], -1))

    def nb_logprob(self, x: torch.Tensor, log_mu: torch.Tensor) -> torch.Tensor:
        r = F.softplus(self.log_r) + 1e-3
        mu = log_mu.exp()
        return (torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
                + r * (torch.log(r) - torch.log(r + mu)) + x * (log_mu - torch.log(r + mu)))

    def loss(self, x, c, beta: float, free_bits: float = 0.05):
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = self.nb_logprob(x, self.decode(z, c)).sum(-1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)        # (B, latent)
        kl_fb = torch.clamp(kl.mean(0), min=free_bits).sum()
        return -(rec.mean()) + beta * kl_fb, -rec.mean().item(), kl.sum(-1).mean().item()

    @torch.no_grad()
    def sample(self, c: np.ndarray, n: int, rng: np.random.Generator | None = None,
               z_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """n joint count samples for ONE condition vector. Returns (counts (n, x_dim), z (n, latent))."""
        g = torch.Generator().manual_seed(int((rng or np.random.default_rng()).integers(1 << 31)))
        cc = torch.as_tensor(c, dtype=torch.float32).expand(n, -1)
        z = torch.randn(n, self.latent, generator=g) * z_scale
        mu = self.decode(z, cc).exp()
        r = F.softplus(self.log_r) + 1e-3
        # NB as Gamma-Poisson mixture
        lam = torch.distributions.Gamma(r.expand_as(mu), (r / mu)).sample()
        counts = torch.poisson(lam, generator=g)
        return counts.numpy(), z.numpy()

    @torch.no_grad()
    def mean(self, c: np.ndarray, n: int = 200) -> np.ndarray:
        cc = torch.as_tensor(c, dtype=torch.float32).expand(n, -1)
        z = torch.randn(n, self.latent)
        return self.decode(z, cc).exp().mean(0).numpy()

    @torch.no_grad()
    def posterior_mean(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        mu, _ = self.encode(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32))
        return mu.numpy()


def train_cvae(Xtr, Ctr, Xva, Cva, epochs: int = 60, lr: float = 2e-3, batch: int = 512,
               seed: int = 0, log=print) -> CVAE:
    torch.manual_seed(seed)
    x_dim, c_dim = Xtr.reshape(len(Xtr), -1).shape[1], Ctr.shape[1]
    model = CVAE(x_dim, c_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt = torch.as_tensor(Xtr.reshape(len(Xtr), -1))
    ct = torch.as_tensor(Ctr)
    xv = torch.as_tensor(Xva.reshape(len(Xva), -1))
    cv = torch.as_tensor(Cva)
    best, best_state = math.inf, None
    for ep in range(epochs):
        model.train()
        beta = min(1.0, (ep + 1) / (epochs * 0.3))                 # KL annealing
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            loss, _, _ = model.loss(xt[b], ct[b], beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss, vrec, vkl = model.loss(xv, cv, 1.0, free_bits=0.0)
        if vloss.item() < best:
            best, best_state = vloss.item(), {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  val -ELBO {vloss.item():8.2f}  rec {vrec:8.2f}  KL {vkl:5.2f}")
    model.load_state_dict(best_state)
    return model
