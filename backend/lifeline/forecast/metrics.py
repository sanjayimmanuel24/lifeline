"""Proper scoring rules for probabilistic forecasts (sample-based)."""
from __future__ import annotations

import numpy as np


def crps(samples: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS over cells. samples (n, d), y (d,). CRPS = E|X-y| - 0.5 E|X-X'|."""
    s = np.sort(samples, axis=0)
    n = s.shape[0]
    term1 = np.abs(s - y).mean(0)
    # E|X-X'| for sorted samples = 2/n^2 * sum_i (2i - n - 1) x_(i)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = (w * s).sum(0) * 2 / n ** 2
    return float((term1 - 0.5 * term2).mean())


def energy_score(samples: np.ndarray, y: np.ndarray) -> float:
    """Multivariate energy score - sensitive to dependence between cells."""
    d1 = np.linalg.norm(samples - y, axis=1).mean()
    half = len(samples) // 2
    d2 = np.linalg.norm(samples[:half] - samples[half:2 * half], axis=1).mean()
    return float(d1 - 0.5 * d2)


def interval_hit(samples_total: np.ndarray, y_total: float, level: float = 0.9) -> bool:
    lo, hi = np.percentile(samples_total, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return bool(lo <= y_total <= hi)


def pit(samples_total: np.ndarray, y_total: float, rng: np.random.Generator) -> float:
    """Randomised PIT for discrete totals."""
    below = (samples_total < y_total).mean()
    at = (samples_total == y_total).mean()
    return float(below + rng.random() * at)
