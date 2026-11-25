"""Discrete-event ambulance simulator.

Events: call arrives -> nearest available unit dispatched (after a processing delay)
-> travel -> on-scene + transport + hospital time -> unit clears -> drives back to its post.
Calls with no available unit wait in a priority queue (severity, then arrival time).
Units that are driving (returning or being repositioned) can still be dispatched.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lifeline.geo import TravelModel

IDLE, BUSY, MOVING = "idle", "busy", "moving"


@dataclass
class Incident:
    id: str
    t: float                 # seconds since simulation start
    zone: int
    severity: int = 3
    recorded_response: float | None = None
    note: str | None = None


@dataclass
class Unit:
    id: int
    post: int
    loc: int
    status: str = IDLE
    dest: int = -1
    depart_t: float = 0.0
    arrive_t: float = 0.0
    free_t: float = 0.0
    incident: str | None = None
    version: int = 0
    km_moved: float = 0.0

    def position(self, t: float) -> int:
        if self.status == MOVING and t >= (self.depart_t + self.arrive_t) / 2:
            return self.dest
        return self.loc


@dataclass
class Record:
    id: str
    t: float
    zone: int
    severity: int
    unit: int
    wait: float
    proc: float
    travel: float
    response: float
    recorded_response: float | None


class Samplers:
    """Empirical distributions fitted on the 2025 calibration days."""

    def __init__(self, proc_s: np.ndarray, service_s: np.ndarray, rng: np.random.Generator):
        self.proc_s = np.asarray(proc_s)
        self.service_s = np.asarray(service_s)
        self.rng = rng

    def proc(self) -> float:
        return float(self.rng.choice(self.proc_s))

    def service(self) -> float:
        return float(self.rng.choice(self.service_s))

    @classmethod
    def from_incidents(cls, df: pd.DataFrame, rng: np.random.Generator) -> "Samplers":
        ok = df[(df.held_indicator == "N") & df.dispatch_response_seconds_qy.notna()]
        proc = ok.dispatch_response_seconds_qy.clip(5, 600).to_numpy()
        service = (df.incident_close_datetime - df.first_on_scene_datetime).dt.total_seconds().dropna()
        service = service[(service > 300) & (service < 4 * 3600)].to_numpy()
        return cls(proc, service, rng)


@dataclass
class Simulator:
    travel: TravelModel
    home_posts: list[int]
    start: pd.Timestamp
    samplers: Samplers
    units: list[Unit] = field(init=False)
    now: float = 0.0

    def __post_init__(self):
        self.units = [Unit(i, p, p) for i, p in enumerate(self.home_posts)]
        self._events: list[tuple[float, int, str, object]] = []
        self._seq = 0
        self.queue: list[tuple[int, float, int, Incident]] = []
        self.records: list[Record] = []
        self.moves: list[dict] = []
        self.calls: list[Incident] = []

    # ---- clock helpers -------------------------------------------------
    def hour(self, t: float | None = None) -> int:
        return (self.start + pd.Timedelta(seconds=self.now if t is None else t)).hour

    def timestamp(self, t: float | None = None) -> pd.Timestamp:
        return self.start + pd.Timedelta(seconds=self.now if t is None else t)

    def _push(self, t: float, kind: str, payload) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, payload))

    # ---- public API ----------------------------------------------------
    def add_incidents(self, incidents: list[Incident]) -> None:
        for inc in incidents:
            self._push(inc.t, "call", inc)

    def run_until(self, t_end: float) -> None:
        while self._events and self._events[0][0] <= t_end:
            t, _, kind, payload = heapq.heappop(self._events)
            self.now = t
            if kind == "call":
                self._on_call(payload)
            elif kind == "clear":
                self._on_clear(*payload)
            elif kind == "arrive":
                self._on_arrive(*payload)
        self.now = max(self.now, t_end)

    def reposition(self, unit_id: int, post: int) -> bool:
        """Change a unit's standby post. Idle/moving units start driving there now."""
        u = self.units[unit_id]
        if u.status == BUSY:
            u.post = post
            return False
        origin = u.position(self.now)
        u.post = post
        if origin == post:
            return True
        self._drive(u, origin, post)
        self.moves.append({"t": self.now, "unit": unit_id, "from": origin, "to": post,
                           "km": float(self.travel.geo.km[origin, post])})
        return True

    def available_positions(self) -> np.ndarray:
        return np.array([u.position(self.now) for u in self.units if u.status != BUSY], dtype=int)

    def busy_fraction(self) -> float:
        return sum(u.status == BUSY for u in self.units) / len(self.units)

    def fork(self, incidents: list[Incident], rng: np.random.Generator) -> "Simulator":
        """Copy of the current fleet state with *hypothetical* calls instead of the real
        future ones (the planner must never see the replayed ground truth)."""
        import copy
        sim = Simulator.__new__(Simulator)
        sim.travel, sim.home_posts, sim.start, sim.now = self.travel, self.home_posts, self.start, self.now
        sim.samplers = Samplers(self.samplers.proc_s, self.samplers.service_s, rng)
        sim.units = copy.deepcopy(self.units)
        sim._events = [e for e in self._events if e[2] != "call"]
        heapq.heapify(sim._events)
        sim._seq = self._seq
        sim.queue = list(self.queue)
        sim.records, sim.moves, sim.calls = [], [], []
        sim.add_incidents(incidents)
        return sim

    # ---- internals -----------------------------------------------------
    def _drive(self, u: Unit, origin: int, dest: int) -> None:
        u.version += 1
        u.status, u.loc, u.dest = MOVING, origin, dest
        u.depart_t = self.now
        u.arrive_t = self.now + self.travel.seconds(origin, dest, self.hour())
        u.km_moved += float(self.travel.geo.km[origin, dest])
        self._push(u.arrive_t, "arrive", (u.id, u.version))

    def _on_arrive(self, unit_id: int, version: int) -> None:
        u = self.units[unit_id]
        if u.version != version or u.status != MOVING:
            return
        u.status, u.loc, u.dest = IDLE, u.dest, -1

    def _nearest(self, zone: int) -> Unit | None:
        best, best_t = None, np.inf
        m = self.travel.matrix(self.hour())
        for u in self.units:
            if u.status == BUSY:
                continue
            tt = m[u.position(self.now), zone]
            if tt < best_t:
                best, best_t = u, tt
        return best

    def _on_call(self, inc: Incident) -> None:
        self.calls.append(inc)
        u = self._nearest(inc.zone)
        if u is None:
            self._seq += 1
            heapq.heappush(self.queue, (inc.severity, inc.t, self._seq, inc))
            return
        self._dispatch(u, inc, wait=0.0)

    def _dispatch(self, u: Unit, inc: Incident, wait: float) -> None:
        proc = self.samplers.proc()
        origin = u.position(self.now)
        travel = self.travel.seconds(origin, inc.zone, self.hour())
        u.version += 1
        u.status, u.loc, u.incident = BUSY, inc.zone, inc.id
        on_scene = self.now + proc + travel
        u.free_t = on_scene + self.samplers.service()
        self._push(u.free_t, "clear", (u.id, u.version))
        self.records.append(Record(inc.id, inc.t, inc.zone, inc.severity, u.id, wait, proc, travel,
                                   wait + proc + travel, inc.recorded_response))

    def _on_clear(self, unit_id: int, version: int) -> None:
        u = self.units[unit_id]
        if u.version != version:
            return
        u.status, u.incident = IDLE, None
        if self.queue:
            _, t0, _, inc = heapq.heappop(self.queue)
            self._dispatch(u, inc, wait=self.now - t0)
            return
        if u.loc != u.post:
            self._drive(u, u.loc, u.post)

    # ---- reporting -----------------------------------------------------
    def records_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def snapshot(self) -> dict:
        return {
            "t": self.now,
            "time": self.timestamp().isoformat(),
            "units": [{"id": u.id, "status": u.status, "post": u.post,
                       "pos": u.position(self.now), "loc": u.loc, "dest": u.dest,
                       "depart_t": u.depart_t, "arrive_t": u.arrive_t} for u in self.units],
            "queue": len(self.queue),
        }


def summarize(records: pd.DataFrame, Z: int, target: float = 480.0) -> dict:
    """Operational metrics used everywhere (dashboard, benchmark, self-model)."""
    if records.empty:
        return {"n": 0}
    rt = records.response.to_numpy()
    by_zone = records.groupby("zone").response.quantile(0.9).reindex(range(Z))
    p90z = by_zone.dropna().to_numpy()
    return {
        "n": int(len(rt)),
        "mean_rt": float(rt.mean()),
        "p50_rt": float(np.percentile(rt, 50)),
        "p90_rt": float(np.percentile(rt, 90)),
        "within_target": float((rt <= target).mean()),
        "mean_travel": float(records.travel.mean()),
        "mean_wait": float(records.wait.mean()),
        "gini_zone_p90": gini(p90z),
        "zone_p90": {int(k): float(v) for k, v in by_zone.dropna().items()},
    }


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
