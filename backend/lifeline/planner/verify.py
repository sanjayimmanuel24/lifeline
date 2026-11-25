"""Second planning stage: check the best MCTS plans in the full discrete-event simulator."""
from __future__ import annotations

import numpy as np

from lifeline.config import L1_WITHIN, L2_GINI, L3_KM, RESPONSE_TARGET_S
from lifeline.sim.engine import Incident, Simulator, gini


def scenario_incidents(counts: np.ndarray, t0: float, rng: np.random.Generator) -> list[Incident]:
    """counts (H, Z) -> calls with uniform arrival times inside each hour."""
    out = []
    H, Z = counts.shape
    for h in range(H):
        for z in range(Z):
            for _ in range(int(counts[h, z])):
                out.append(Incident(f"s{h}-{z}-{len(out)}", t0 + (h + rng.random()) * 3600, z,
                                    int(rng.choice([1, 2, 3, 4], p=[.05, .25, .35, .35]))))
    return sorted(out, key=lambda i: i.t)


def simulate_plan(sim: Simulator, moves: list[tuple[int, int]], scenarios: np.ndarray,
                  seed: int, n: int = 20) -> dict:
    """Run the plan on n generated futures; returns mean metrics and their spread.
    Common random numbers: the same seed gives every candidate plan identical calls and
    identical service/processing times, so differences between plans are not sampling noise."""
    H = scenarios.shape[1]
    idx = np.random.default_rng(seed).choice(len(scenarios), min(n, len(scenarios)), replace=False)
    res = []
    for s in idx:
        rng = np.random.default_rng(seed * 1000 + int(s))
        f = sim.fork(scenario_incidents(scenarios[s], sim.now, rng), np.random.default_rng(seed * 1000 + int(s) + 1))
        km = 0.0
        for u, p in moves:
            if f.reposition(u, p):
                km += float(f.travel.geo.km[f.units[u].loc, p]) if f.units[u].status == "moving" else 0.0
        f.run_until(sim.now + H * 3600)
        df = f.records_frame()
        if df.empty:
            continue
        rt = df.response.to_numpy()
        zone_mean = df.groupby("zone").response.mean().to_numpy()
        res.append({"mean_rt": rt.mean(), "p90_rt": np.percentile(rt, 90),
                    "within": (rt <= RESPONSE_TARGET_S).mean(), "gini": gini(zone_mean), "km": km})
    if not res:
        return {"reward": 0.0, "n": 0}
    m = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    m["reward"] = -m["mean_rt"] + L1_WITHIN * m["within"] - L2_GINI * m["gini"] - L3_KM * m["km"]
    m["within_sd"] = float(np.std([r["within"] for r in res]))
    m["n"] = len(res)
    return m
