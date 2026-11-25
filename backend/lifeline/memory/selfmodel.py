"""The agent's model of its own competence.

Built offline by self-stress-testing (selfassess.py): the agent plans against CVAE futures,
then one held-back future is revealed as "truth" and the plan is scored against a hindsight
oracle. Per context bucket we keep:
    regret      mean seconds of response time lost vs the hindsight-best plan (EMA online)
    wins/losses Beta posterior for "my plan beat doing nothing"
    crps        forecast error in this bucket
An isotonic regression maps raw confidence to the observed win rate (calibration).
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import threading

import numpy as np
import pandas as pd

from lifeline.config import ARTIFACTS

_lock = threading.Lock()


def bucket_of(ts: pd.Timestamp, precip_mm: float, surge: bool) -> str:
    band = ["night", "morning", "afternoon", "evening"][ts.hour // 6]
    wx = "rain" if precip_mm >= 0.5 else "dry"
    return f"{band}|{wx}|{'surge' if surge else 'normal'}"


class SelfModel:
    def __init__(self, path=None):
        path = path or ARTIFACTS / "memory" / "selfmodel.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS competence(
            bucket TEXT PRIMARY KEY, n INT, regret REAL, wins REAL, losses REAL, crps REAL, source TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions(
            id INTEGER PRIMARY KEY, t TEXT, bucket TEXT, confidence REAL, tier TEXT, outcome_win INT, regret REAL)""")
        self.db.commit()
        cal = ARTIFACTS / "isotonic.pkl"
        self.iso = pickle.loads(cal.read_bytes()) if cal.exists() else None

    def row(self, bucket: str) -> dict:
        r = self.db.execute("SELECT n, regret, wins, losses, crps FROM competence WHERE bucket=?", (bucket,)).fetchone()
        if r is None:
            return {"bucket": bucket, "n": 0, "regret": None, "win_rate": 0.5, "crps": None, "known": False}
        n, regret, w, l, crps = r
        return {"bucket": bucket, "n": n, "regret": regret, "win_rate": (w + 1) / (w + l + 2), "crps": crps,
                "known": n >= 5}

    def table(self) -> list[dict]:
        rows = self.db.execute("SELECT bucket FROM competence ORDER BY bucket").fetchall()
        return [self.row(b) for (b,) in rows]

    def update(self, bucket: str, regret: float, win: bool, crps: float | None = None,
               source: str = "online", alpha: float = 0.2) -> None:
        with _lock:
            r = self.db.execute("SELECT n, regret, wins, losses, crps FROM competence WHERE bucket=?", (bucket,)).fetchone()
            if r is None:
                self.db.execute("INSERT INTO competence VALUES(?,?,?,?,?,?,?)",
                                (bucket, 1, regret, float(win), float(not win), crps, source))
            else:
                n, rg, w, l, c = r
                rg = regret if rg is None else (1 - alpha) * rg + alpha * regret
                c = crps if c is None else (c if crps is None else (1 - alpha) * c + alpha * crps)
                self.db.execute("UPDATE competence SET n=?, regret=?, wins=?, losses=?, crps=? WHERE bucket=?",
                                (n + 1, rg, w + float(win), l + float(not win), c, bucket))
            self.db.commit()

    def raw_confidence(self, bucket: str, spread: float, ood: float) -> float:
        """Win-rate prior for this bucket, discounted for forecast spread and novelty."""
        row = self.row(bucket)
        base = row["win_rate"] if row["known"] else 0.5
        regret_pen = 0.0 if row["regret"] is None else min(0.2, row["regret"] / 400.0)
        return float(np.clip(base * (1 - 0.6 * ood) * (1 - min(0.4, spread)) - regret_pen, 0.01, 0.99))

    def calibrated(self, raw: float) -> float:
        return float(self.iso.predict([raw])[0]) if self.iso is not None else raw

    def log_decision(self, t: str, bucket: str, conf: float, tier: str) -> int:
        with _lock:
            cur = self.db.execute("INSERT INTO decisions(t,bucket,confidence,tier) VALUES(?,?,?,?)", (t, bucket, conf, tier))
            self.db.commit()
            return cur.lastrowid

    def resolve_decision(self, decision_id: int, win: bool, regret: float) -> None:
        with _lock:
            self.db.execute("UPDATE decisions SET outcome_win=?, regret=? WHERE id=?", (int(win), regret, decision_id))
            self.db.commit()

    def reliability(self, bins: int = 10) -> dict:
        """Reliability diagram + ECE from the offline self-assessment holdout."""
        p = ARTIFACTS / "selfassess.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text()).get("reliability", {})
