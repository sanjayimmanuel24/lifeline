"""Tune the per-km move cost on 2025 calibration days only (the 2026 test days stay untouched).

Run:  python -m lifeline.eval.tune  -> data/artifacts/tuning.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lifeline.config import ARTIFACTS, DATA_DIR

GRID = [4.0, 1.0, 0.25]
WORKER = r"""
import os, sys, json, tempfile
os.environ["LIFELINE_OFFLINE"] = "1"
from lifeline.runtime.session import Runtime, Session
from lifeline.sim.engine import summarize
rt = Runtime(memory_dir=tempfile.mkdtemp())
out = []
for i, day in enumerate(json.loads(sys.argv[1])):
    s = Session(rt, day, start_hour=0, hours=24, policy="lifeline", headless=True, source="calibration", seed=i)
    s.run()
    m = summarize(s.sim.records_frame(), rt.geo.Z)
    out.append({"day": day, "within": m["within_target"], "mean_rt": m["mean_rt"], "p90": m["p90_rt"], "moves": len(s.sim.moves)})
print("@@" + json.dumps(out))
"""


def main() -> None:
    days = json.loads((DATA_DIR / "splits.json").read_text())["calibration_days"][:6]
    procs = {}
    for l3 in GRID:                                   # one process per setting, run in parallel
        env = dict(os.environ, LIFELINE_L3_KM=str(l3))
        procs[l3] = subprocess.Popen([sys.executable, "-c", WORKER, json.dumps(days)], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    res = {}
    for l3, p in procs.items():
        out = p.communicate()[0]
        rows = json.loads(out.split("@@", 1)[1])
        res[l3] = {k: float(np.mean([r[k] for r in rows])) for k in ("within", "mean_rt", "p90", "moves")}
        print(f"  L3_KM={l3:5.2f}  within {res[l3]['within']:.4f}  mean {res[l3]['mean_rt']:.1f}s  "
              f"p90 {res[l3]['p90']:.0f}s  moves/day {res[l3]['moves']:.1f}")
    best = min(res, key=lambda k: res[k]["mean_rt"] - 120 * res[k]["within"])
    (ARTIFACTS / "tuning.json").write_text(json.dumps({"days": days, "grid": {str(k): v for k, v in res.items()},
                                                       "chosen_L3_KM": best}, indent=1))
    print(f"chosen L3_KM = {best}")


if __name__ == "__main__":
    main()
