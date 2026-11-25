"""Fast analytic evaluation of a fleet configuration against one demand scenario.

Used inside MCTS rollouts (~1 ms each) so the LLM is never in the search loop.
Each available unit is busy with probability q (hypercube-style approximation, as in MEXCLP).
For zone z with available units sorted by travel time t_1 <= t_2 <= ...,
    P(served by k-th nearest) = q^(k-1) (1-q)
    E[response_z]  ~= proc + sum_k q^(k-1)(1-q) t_k  (+ tail term)
    P(covered_z)    = 1 - q^(n_z),  n_z = #units reachable within target
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lifeline.config import L1_WITHIN, L2_GINI, L3_KM, RESPONSE_TARGET_S

K_NEAREST = 6
PROC_S = 75.0        # median call-processing delay, overwritten from calibration


@dataclass
class Evaluation:
    reward: float
    mean_rt: float
    within: float
    gini: float
    km: float
    zone_rt: np.ndarray
    zone_cov: np.ndarray
    q: float


def busy_prob(demand_per_hour: float, n_units: int, service_h: float, base_busy: float) -> float:
    """Utilisation over the coming horizon: blend of the forecast load (rho = lambda * service / N)
    and the fraction busy right now (busy units free up during the horizon)."""
    rho = demand_per_hour * service_h / max(n_units, 1)
    return float(np.clip(0.5 * rho + 0.5 * base_busy, 0.05, 0.95))


def evaluate(positions: np.ndarray, demand: np.ndarray, T: np.ndarray, q: float,
             km_moved: float = 0.0, proc_s: float = PROC_S) -> Evaluation:
    """positions: zone index of each available unit; demand: expected calls per zone (Z,);
    T: (Z, Z) travel seconds."""
    Z = T.shape[0]
    if len(positions) == 0:
        zr = np.full(Z, 3600.0)
        return Evaluation(-1e4, 3600, 0.0, 0.0, km_moved, zr, np.zeros(Z), q)
    tt = T[positions][:, :].T                      # (Z, n_units): travel from each unit to each zone
    k = min(K_NEAREST, tt.shape[1])
    near = np.partition(tt, k - 1, axis=1)[:, :k] if tt.shape[1] > k else tt
    near = np.sort(near, axis=1)
    w = (1 - q) * q ** np.arange(k)
    tail = q ** k
    zone_rt = proc_s + near @ w + tail * (near[:, -1] * 1.5 + 600)
    n_cov = (tt + proc_s <= RESPONSE_TARGET_S).sum(1)
    zone_cov = 1 - q ** n_cov
    d = demand / max(demand.sum(), 1e-9)
    mean_rt = float(d @ zone_rt)
    within = float(d @ zone_cov)
    g = _weighted_gini(zone_rt, demand + 1e-3)
    reward = -mean_rt + L1_WITHIN * within - L2_GINI * g - L3_KM * km_moved
    return Evaluation(float(reward), mean_rt, within, g, km_moved, zone_rt, zone_cov, q)


def _weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    if cxw[-1] == 0:
        return 0.0
    lorenz = cxw / cxw[-1]
    prev = np.concatenate([[0], lorenz[:-1]])
    return float(1 - ((prev + lorenz) * (w / cw[-1])).sum())
