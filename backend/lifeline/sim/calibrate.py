"""Calibrate the simulator on 12 real 2025 days, and validate the status quo against the records.

1. Speed curve: per-hour driving speed so simulated mean travel time matches the recorded mean.
2. Effective fleet size: chosen so the simulated status-quo p90 response time matches the recorded p90.
   (Only severity 1-4 calls are simulated, so this is the fleet *effectively available* for them.)

Run:  python -m lifeline.sim.calibrate      -> data/artifacts/calibration.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from lifeline.config import ARTIFACTS, DATA_DIR, FLEET_SIZE, TRAIN_START, VAL_START
from lifeline.geo import Geography, TravelModel, load_geography
from lifeline.planner.mexclp import proportional
from lifeline.sim.engine import Incident, Samplers, Simulator, summarize

BASE_S = 45.0


def day_incidents(df: pd.DataFrame, geo: Geography, day: str) -> list[Incident]:
    d0 = pd.Timestamp(day)
    sub = df[(df.incident_datetime >= d0) & (df.incident_datetime < d0 + pd.Timedelta(days=1))]
    out = []
    for r in sub.itertuples():
        if r.zone not in geo.zones:
            continue
        rec = r.incident_response_seconds_qy if r.valid_incident_rspns_time_indc == "Y" else None
        out.append(Incident(str(r.incident_id), (r.incident_datetime - d0).total_seconds(),
                            geo.index(r.zone), int(r.severity), rec))
    return out


def mean_zone_demand(geo: Geography) -> np.ndarray:
    c = pd.read_parquet(DATA_DIR / "hourly_counts.parquet")
    c = c[(c.h >= TRAIN_START) & (c.h < VAL_START)]
    hours = (pd.Timestamp(VAL_START) - pd.Timestamp(TRAIN_START)).total_seconds() / 3600
    s = c.groupby("zone").n.sum().reindex(geo.zones).fillna(0).to_numpy()
    return s / hours                                                   # calls per hour per zone


def run_days(days, incidents, travel, posts, samplers_src, seed=0) -> pd.DataFrame:
    frames = []
    for i, day in enumerate(days):
        rng = np.random.default_rng(seed + i)
        sim = Simulator(travel, posts, pd.Timestamp(day), Samplers(samplers_src.proc_s, samplers_src.service_s, rng))
        inc = incidents[day]
        sim.add_incidents(inc)
        sim.run_until(86400 + 6 * 3600)
        df = sim.records_frame()
        df["day"] = day
        df["hour"] = ((df.t // 3600) % 24).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    geo = load_geography()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    cal = pd.read_parquet(DATA_DIR / "calibration_incidents.parquet")
    days = splits["calibration_days"]
    incidents = {d: day_incidents(cal, geo, d) for d in days}
    samplers = Samplers.from_incidents(cal, np.random.default_rng(0))

    valid = cal[cal.valid_incident_rspns_time_indc == "Y"]
    rec_rt = valid.incident_response_seconds_qy.dropna().clip(0, 7200).to_numpy()
    rec_p50, rec_p90 = np.percentile(rec_rt, [50, 90])
    ts = pd.read_parquet(DATA_DIR / "travel_stats.parquet")
    rec_travel = (ts.assign(w=ts.travel * ts.n).groupby("hour")[["w", "n"]].sum()
                  .pipe(lambda g: g.w / g.n).reindex(range(24)).interpolate().to_numpy())
    print(f"recorded: p50 {rec_p50:.0f}s  p90 {rec_p90:.0f}s  mean travel {np.nanmean(rec_travel):.0f}s")

    demand = mean_zone_demand(geo)
    speed = np.full(24, 20.0)
    fleets = [FLEET_SIZE] if FLEET_SIZE else list(range(30, 131, 10))
    results = []
    for n in fleets:
        posts = proportional(demand, n)
        sp = speed.copy()
        for _ in range(4):
            travel = TravelModel(geo, sp, BASE_S)
            df = run_days(days, incidents, travel, posts, samplers)
            sim_tr = df.groupby("hour").travel.mean().reindex(range(24)).interpolate().bfill().ffill().to_numpy()
            ratio = np.clip((sim_tr - BASE_S) / np.maximum(rec_travel - BASE_S, 30), 0.5, 2.0)
            sp = np.convolve(np.pad(sp * ratio, 1, mode="wrap"), [0.25, 0.5, 0.25], "valid")
        travel = TravelModel(geo, sp, BASE_S)
        df = run_days(days, incidents, travel, posts, samplers)
        p50, p90 = np.percentile(df.response, [50, 90])
        print(f"fleet {n:3d}: sim p50 {p50:5.0f}  p90 {p90:5.0f}  travel {df.travel.mean():4.0f}  wait {df.wait.mean():5.1f}")
        results.append((abs(p90 - rec_p90), n, sp, df))

    _, n, sp, df = min(results, key=lambda r: r[0])
    p50, p90 = np.percentile(df.response, [50, 90])
    ks = ks_2samp(df.response.to_numpy(), rec_rt)
    service_h = float(np.mean(samplers.service_s) / 3600 + df.travel.mean() / 3600 * 2)
    out = {
        "fleet": n, "speed_by_hour": sp.tolist(), "base_s": BASE_S,
        "proc_median_s": float(np.median(samplers.proc_s)),
        "service_h": service_h,
        "proc_s": samplers.proc_s.tolist()[:4000], "service_s": samplers.service_s.tolist()[:4000],
        "validation": {
            "recorded_p50": float(rec_p50), "recorded_p90": float(rec_p90),
            "sim_p50": float(p50), "sim_p90": float(p90),
            "p50_error": float(p50 / rec_p50 - 1), "p90_error": float(p90 / rec_p90 - 1),
            "ks_stat": float(ks.statistic), "recorded_mean_travel": float(np.nanmean(rec_travel)),
            "sim_mean_travel": float(df.travel.mean()), "days": days,
            "recorded_hist": np.histogram(rec_rt, bins=30, range=(0, 1800))[0].tolist(),
            "sim_hist": np.histogram(df.response, bins=30, range=(0, 1800))[0].tolist(),
        },
        "zone_demand_per_hour": demand.tolist(),
        "sweep": [{"fleet": r[1], "p90_gap": float(r[0])} for r in results],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "calibration.json").write_text(json.dumps(out))
    print(f"chosen fleet {n}: p50 err {out['validation']['p50_error']:+.1%}  p90 err {out['validation']['p90_error']:+.1%}  KS {ks.statistic:.3f}")


def load_calibration() -> dict:
    return json.loads((ARTIFACTS / "calibration.json").read_text())


if __name__ == "__main__":
    main()
