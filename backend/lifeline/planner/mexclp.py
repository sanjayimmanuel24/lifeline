"""MEXCLP (Daskin 1983): maximum expected covering location, solved exactly with OR-Tools CP-SAT.

Classical operations-research baseline: a static posting plan from mean demand.
    max  sum_z d_z sum_k (1-q) q^(k-1) y_zk
    s.t. sum_k y_zk <= sum_{p covers z} x_p      for every zone z
         sum_p x_p = N,  x_p integer >= 0,  y_zk binary
"""
from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

from lifeline.config import RESPONSE_TARGET_S


def mexclp(demand: np.ndarray, T: np.ndarray, n_units: int, q: float, proc_s: float = 75.0,
           time_limit_s: float = 10.0) -> list[int]:
    Z = len(demand)
    K = n_units
    cover = (T + proc_s <= RESPONSE_TARGET_S)          # cover[p, z]
    m = cp_model.CpModel()
    x = [m.NewIntVar(0, n_units, f"x{p}") for p in range(Z)]
    kmax = min(K, 12)
    y = [[m.NewBoolVar(f"y{z}_{k}") for k in range(kmax)] for z in range(Z)]
    for z in range(Z):
        m.Add(sum(y[z]) <= sum(x[p] for p in range(Z) if cover[p, z]))
        for k in range(1, kmax):
            m.AddImplication(y[z][k], y[z][k - 1])
    m.Add(sum(x) == n_units)
    w = [[int(round(1e4 * demand[z] * (1 - q) * q ** k)) for k in range(kmax)] for z in range(Z)]
    m.Maximize(sum(w[z][k] * y[z][k] for z in range(Z) for k in range(kmax)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 4
    solver.Solve(m)
    posts = []
    for p in range(Z):
        posts += [p] * int(solver.Value(x[p]))
    return posts


def proportional(demand: np.ndarray, n_units: int) -> list[int]:
    """Status-quo approximation: units spread in proportion to historical demand (largest remainder)."""
    share = demand / demand.sum() * n_units
    base = np.floor(share).astype(int)
    rest = n_units - base.sum()
    base[np.argsort(-(share - base))[:rest]] += 1
    return [p for p, c in enumerate(base) for _ in range(c)]
