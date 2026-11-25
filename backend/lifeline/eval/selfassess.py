"""Offline self-assessment: build the agent's self-model before deployment.

For many (2025 calibration day, hour) situations - never the 2026 test days:
  1. replay the real day up to that hour (fleet state, recent calls)
  2. forecast 50 futures with the CVAE and plan with MCTS, exactly as in operation
  3. reveal what really happened in the next 2 hours, and score the plan against holding
     positions and against a hindsight oracle (win, regret)
  4. stress test: also score against 3 CVAE futures the planner did not see
Half of the situations build the competence table + isotonic calibration, the other half
measure whether the resulting confidence is honest (reliability diagram, ECE, Brier).

Run:  python -m lifeline.eval.selfassess
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from lifeline.config import ARTIFACTS, HORIZON
from lifeline.memory.selfmodel import SelfModel, bucket_of
from lifeline.planner.mcts import Constraints, MCTSPlanner
from lifeline.runtime.session import Runtime, Session, hindsight

HOURS = list(range(24))
N_STRESS = 5
HOLD_TOLERANCE = 10.0     # a "hold" decision is right if the hindsight oracle could not gain more than this


def realised_counts(sess: Session) -> np.ndarray:
    out = np.zeros(sess.geo.Z, np.float32)
    t0, t1 = sess.sim.now, sess.sim.now + HORIZON * 3600
    for t, _, kind, inc in sess.sim._events:
        if kind == "call" and t0 <= t < t1:
            out[inc.zone] += 1
    return out


def one(rt: Runtime, day: str, hour: int, seed: int) -> list[dict]:
    s = Session(rt, day, start_hour=hour, hours=0.1, headless=True, source="calibration", seed=seed)
    fc = s.forecast()
    truth = realised_counts(s)
    avail, fixed = s.fleet_view()
    T = rt.travel.matrix(hour)
    pl = MCTSPlanner(T, rt.geo.km, rt.cal["service_h"], len(s.sim.units), np.random.default_rng(seed),
                     rt.cal["proc_median_s"])
    held = s.scenarios[-N_STRESS:]
    res = pl.plan(avail, s.scenarios[:-N_STRESS], s.sim.busy_fraction(), Constraints(), iters=400, fixed=fixed)
    before = np.array([z for _, z in avail])
    after = pl._apply(before, res.moves)
    precip = s.weather()["precip"][0]
    bucket = bucket_of(s.sim.timestamp(), precip, fc["surge_prob"] > 0.5)
    rows = []
    for kind, demand in [("real", truth)] + [("stress", h.sum(0)) for h in held]:
        win, regret = hindsight(rt, before, after, fixed, demand, hour, s.sim.busy_fraction(), len(s.sim.units))
        if not res.moves:                 # judged a hold: right if nothing much better was possible
            win = regret <= HOLD_TOLERANCE
        rows.append({"day": day, "hour": hour, "kind": kind, "bucket": bucket, "moves": len(res.moves),
                     "win": win, "regret": regret, "spread": fc["total_cv"], "ood": fc["ood"]["score"],
                     "crps_total": float(np.abs(fc["total_mean"] - demand.sum()))})
    return rows


def ece(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    out, e = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = float(y[m].mean()), float(conf[m].mean())
        e += m.mean() * abs(acc - cf)
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "confidence": cf, "accuracy": acc, "n": int(m.sum())})
    return float(e), out


def main() -> None:
    rt = Runtime()
    days = rt.splits["calibration_days"]
    rows = []
    for i, day in enumerate(days):
        for h in HOURS:
            rows += one(rt, day, h, seed=i * 100 + h)
        print(f"  {day}: {len(rows)} rows")
    df = pd.DataFrame(rows)
    fit_days = set(days[: len(days) // 2])
    fit, hold = df[df.day.isin(fit_days)], df[~df.day.isin(fit_days)]

    # 1) competence table from the fit half, then raw confidence on both halves
    tmp = ARTIFACTS / "memory" / "selfmodel_fit.sqlite"
    tmp.unlink(missing_ok=True)
    sm = SelfModel(tmp)
    sm.iso = None
    for r in fit.itertuples():
        sm.update(r.bucket, r.regret, r.win, r.crps_total, source="selfassess")
    raw = lambda d: np.array([sm.raw_confidence(r.bucket, r.spread, r.ood) for r in d.itertuples()])
    iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(raw(fit), fit.win.astype(float))
    conf_hold = iso.predict(raw(hold))
    e_raw, _ = ece(raw(hold), hold.win.to_numpy().astype(float))
    e_cal, table = ece(conf_hold, hold.win.to_numpy().astype(float))
    brier = float(np.mean((conf_hold - hold.win.astype(float)) ** 2))
    print(f"held-out: ECE raw {e_raw:.3f} -> calibrated {e_cal:.3f}, Brier {brier:.3f}, n={len(hold)}")

    # 2) final self-model from all rows
    final = SelfModel()
    final.db.execute("DELETE FROM competence")
    final.db.commit()
    for r in df.itertuples():
        final.update(r.bucket, r.regret, r.win, r.crps_total, source="selfassess")
    iso_all = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(
        np.array([final.raw_confidence(r.bucket, r.spread, r.ood) for r in df.itertuples()]), df.win.astype(float))
    (ARTIFACTS / "isotonic.pkl").write_bytes(pickle.dumps(iso_all))
    summary = {
        "n_situations": int(len(df)), "n_real": int((df.kind == "real").sum()), "n_stress": int((df.kind == "stress").sum()),
        "win_rate": float(df.win.mean()), "win_rate_real": float(df[df.kind == "real"].win.mean()),
        "share_move_decisions": float((df.moves > 0).mean()),
        "win_rate_moves": float(df[df.moves > 0].win.mean()) if (df.moves > 0).any() else None,
        "win_rate_holds": float(df[df.moves == 0].win.mean()) if (df.moves == 0).any() else None,
        "mean_regret": float(df.regret.mean()),
        "reliability": {"ece_raw": e_raw, "ece_calibrated": e_cal, "brier": brier, "bins": table, "n_holdout": int(len(hold))},
        "by_bucket": df.groupby("bucket").agg(n=("win", "size"), win_rate=("win", "mean"), regret=("regret", "mean"))
                       .reset_index().to_dict(orient="records"),
    }
    (ARTIFACTS / "selfassess.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_bucket"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
