"""Benchmark: replay the 30 held-out 2026 days (never used for training or calibration) under each policy.

Policies
  static            units spread in proportion to historical demand, no repositioning (status quo)
  mexclp            MEXCLP optimal static posts (OR-Tools), no repositioning (classical OR baseline)
  lifeline          full agent system (CVAE + MCTS + Critic + self-model) on top of the status-quo posts
  lifeline_poisson  same, but planning on independent Poisson forecasts instead of CVAE futures
  no_critic         ablation: Critic always accepts
  no_selfmodel      ablation: always fully confident (no wait / no escalation)
  no_equity         ablation: equity term removed from the planner's objective
Tier-3 approvals are auto-approved by a simulated supervisor; LLM steps run offline (deterministic)
so results are reproducible. Every policy sees the same real calls on the same day.

Run:  python -m lifeline.eval.benchmark [--days N] [--policies a,b] [--workers K]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lifeline.config import ARTIFACTS

POLICIES = ["static", "mexclp", "lifeline", "lifeline_poisson", "no_critic", "no_selfmodel", "no_equity"]


def run_day(policy: str, day: str, seed: int) -> dict:
    os.environ["LIFELINE_OFFLINE"] = "1"
    from lifeline.runtime.session import Runtime, Session
    rt = _worker_rt()
    s = Session(rt, day, start_hour=0, hours=24, policy=policy, headless=True, seed=seed)
    s.run()
    df = s.sim.records_frame()
    from lifeline.sim.engine import summarize
    m = summarize(df, rt.geo.Z)
    tiers = pd.Series([r["tier"] for r in s.reviews]).value_counts().to_dict()
    return {"policy": policy, "day": day, **{k: v for k, v in m.items() if not isinstance(v, dict)},
            "zone_p90": m.get("zone_p90", {}), "moves": len(s.sim.moves),
            "km_moved": float(sum(mv["km"] for mv in s.sim.moves)), "reviews": len(s.reviews), "tiers": tiers,
            "critic_rejections": int(sum(max(0, r["critic_rounds"] - 1) for r in s.reviews)),
            "recorded_p90": float(np.nanpercentile([r for r in df.recorded_response if r == r], 90))
            if df.recorded_response.notna().any() else None}


_RT = None


def _worker_rt():
    global _RT
    if _RT is None:
        from lifeline.runtime.session import Runtime
        _RT = Runtime(memory_dir=tempfile.mkdtemp(prefix="lifeline-bench-"))
    return _RT


def paired(df: pd.DataFrame, a: str, b: str, metric: str, rng) -> dict:
    x = df[df.policy == a].set_index("day")[metric]
    y = df[df.policy == b].set_index("day")[metric]
    d = (x - y).dropna()
    if len(d) < 3:
        return {}
    boots = [rng.choice(d.to_numpy(), len(d)).mean() for _ in range(4000)]
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p, "n_days": int(len(d)), "days_better": int((d < 0).sum() if metric != "within_target" else (d > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--ablation-days", type=int, default=10, help="ablation policies run on the first N days only")
    ap.add_argument("--workers", type=int, default=4, help="each worker loads PyTorch; keep low on 16 GB machines")
    args = ap.parse_args()
    days = json.loads((ARTIFACTS.parent / "splits.json").read_text())["test_days"][: args.days]
    policies = args.policies.split(",")
    core = {"static", "mexclp", "lifeline"}
    jobs = [(p, d, i) for i, d in enumerate(days) for p in policies if p in core or i < args.ablation_days]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_day, *j): j for j in jobs}
        for k, f in enumerate(as_completed(futs)):
            try:
                rows.append(f.result())
            except Exception as e:
                print(f"  FAILED {futs[f]}: {e}")
                continue
            r = rows[-1]
            print(f"  [{k + 1}/{len(jobs)}] {r['policy']:16s} {r['day']}  within {r['within_target']:.3f}  "
                  f"mean {r['mean_rt']:.0f}s  p90 {r['p90_rt']:.0f}s  moves {r['moves']}")
    df = pd.DataFrame(rows)
    df.to_json(ARTIFACTS / "benchmark_days.json", orient="records")
    rng = np.random.default_rng(0)
    summary = {"policies": {}, "comparisons": {}, "n_days": len(days), "ablation_days": args.ablation_days,
               "failed_jobs": len(jobs) - len(rows)}
    for p, g in df.groupby("policy"):
        summary["policies"][p] = {"n_days": int(len(g))}
        summary["policies"][p] |= {m: float(g[m].mean()) for m in
                                  ("mean_rt", "p50_rt", "p90_rt", "within_target", "gini_zone_p90", "moves", "km_moved",
                                   "critic_rejections", "reviews")}
        zp = pd.DataFrame(list(g.zone_p90)).mean().to_dict()
        summary["policies"][p]["zone_p90"] = {str(k): float(v) for k, v in zp.items()}
        summary["policies"][p]["tiers"] = pd.DataFrame(list(g.tiers)).sum().to_dict() if g.tiers.astype(bool).any() else {}
    if "lifeline" in policies:
        for b in policies:
            if b == "lifeline":
                continue
            summary["comparisons"][f"lifeline_vs_{b}"] = {m: paired(df, "lifeline", b, m, rng)
                                                          for m in ("mean_rt", "p90_rt", "within_target", "gini_zone_p90")}
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(summary, indent=1, default=float))
    print(pd.DataFrame(summary["policies"]).T[["within_target", "mean_rt", "p90_rt", "gini_zone_p90", "moves"]].round(3))


if __name__ == "__main__":
    main()
