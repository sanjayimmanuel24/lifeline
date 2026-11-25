"""Forecast baselines the CVAE is compared against."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import PoissonRegressor


class PoissonGLM:
    """One Poisson regression per (hour slot, zone) on the same condition vector.
    Samples are independent across cells - no joint structure."""

    def fit(self, X: np.ndarray, C: np.ndarray) -> "PoissonGLM":
        n, H, Z = X.shape
        self.shape = (H, Z)
        self.models = [[PoissonRegressor(alpha=1e-3, max_iter=300).fit(C, X[:, h, z]) for z in range(Z)]
                       for h in range(H)]
        return self

    def mean(self, c: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(c)
        return np.array([[m.predict(c)[0] for m in row] for row in self.models]).reshape(-1)

    def sample(self, c: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.mean(c), size=(n, self.mean(c).size)).astype(np.float32)


class SeasonalNaive:
    """Same hours one week earlier, Poisson noise around it."""

    def __init__(self, Y: np.ndarray):
        self.Y = Y

    def mean_at(self, pos: int, horizon: int) -> np.ndarray:
        return self.Y[pos - 168:pos - 168 + horizon].reshape(-1)

    def sample_at(self, pos: int, horizon: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.mean_at(pos, horizon) + 1e-3, size=(n, self.mean_at(pos, horizon).size)).astype(np.float32)
