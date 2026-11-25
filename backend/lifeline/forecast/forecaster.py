"""Runtime forecaster used by the agents: loads the trained CVAE (or the Poisson baseline)."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import torch

from lifeline.config import ARTIFACTS, HORIZON
from lifeline.forecast.cvae import CVAE
from lifeline.forecast.dataset import LAG, condition


class Forecaster:
    kind = "cvae"

    def __init__(self, model: CVAE, stats: dict, Z: int):
        self.model, self.stats, self.Z = model, stats, Z
        self.lat_mean = np.array(stats["latent_mean"])
        self.lat_prec = np.linalg.inv(np.array(stats["latent_cov"]) + 1e-6 * np.eye(len(self.lat_mean)))

    @classmethod
    def load(cls) -> "Forecaster":
        stats = json.loads((ARTIFACTS / "cvae_stats.json").read_text())
        model = CVAE(stats["x_dim"], stats["c_dim"])
        model.load_state_dict(torch.load(ARTIFACTS / "cvae.pt", weights_only=True))
        model.eval()
        return cls(model, stats, stats["x_dim"] // HORIZON)

    def cond(self, ts: pd.Timestamp, lag_counts: np.ndarray, precip: np.ndarray, temp: np.ndarray) -> np.ndarray:
        return condition(ts, np.asarray(lag_counts)[-LAG:], np.asarray(precip)[:HORIZON], np.asarray(temp)[:HORIZON])

    def scenarios(self, c: np.ndarray, n: int, rng: np.random.Generator) -> dict:
        counts, z = self.model.sample(c, n, rng)
        return {"counts": counts.reshape(n, HORIZON, self.Z), "latent": z}

    def ood(self, c: np.ndarray, recent: np.ndarray | None = None) -> dict:
        """Out-of-distribution signals: condition outside the training range, and
        Mahalanobis distance of the encoded recent situation in latent space."""
        lo, hi = np.array(self.stats["c_min"]), np.array(self.stats["c_max"])
        span = np.maximum(hi - lo, 1e-6)
        outside = np.maximum(0, np.maximum(lo - c, c - hi) / span)
        names = self.stats["c_names"]
        flagged = [names[i] for i in np.where(outside > 0.05)[0] if i < len(names)]
        maha = 0.0
        if recent is not None:
            mu = self.model.posterior_mean(recent.reshape(1, -1), c.reshape(1, -1))[0]
            d = mu - self.lat_mean
            maha = float(np.sqrt(d @ self.lat_prec @ d))
        thr = self.stats["maha_p99"]
        score = float(min(1.0, 0.5 * (maha / thr) + 0.5 * min(1.0, outside.max() * 4)))
        return {"score": score, "latent_distance": maha, "latent_p99": thr, "outside_training_range": flagged}


class PoissonForecaster:
    """Ablation: same interface, independent per-cell Poisson samples."""
    kind = "poisson"

    def __init__(self, glm, Z: int):
        self.glm, self.Z = glm, Z

    @classmethod
    def load(cls, Z: int) -> "PoissonForecaster":
        with open(ARTIFACTS / "poisson_glm.pkl", "rb") as f:
            return cls(pickle.load(f), Z)

    def cond(self, ts, lag_counts, precip, temp):
        return condition(ts, np.asarray(lag_counts)[-LAG:], np.asarray(precip)[:HORIZON], np.asarray(temp)[:HORIZON])

    def scenarios(self, c, n, rng):
        return {"counts": self.glm.sample(c, n, rng).reshape(n, HORIZON, self.Z), "latent": np.zeros((n, 2))}

    def ood(self, c, recent=None):
        return {"score": 0.0, "latent_distance": 0.0, "latent_p99": 1.0, "outside_training_range": []}
