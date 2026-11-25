"""A dispatch shift: replays a real held-out day through the simulator and runs the agent graph.

The Session object is also the runtime ("rt") that agent nodes use for tools and models.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from langgraph.types import Command

from lifeline.agents import prompts
from lifeline.agents.graph import build_graph
from lifeline.agents.llm import get_llm, usage
from lifeline.config import (ARTIFACTS, COVERAGE_FLOOR, DATA_DIR, EPOCH_MIN, HORIZON, MCTS_ITERS, N_SCENARIOS,
                             RESPONSE_TARGET_S)
from lifeline.forecast.forecaster import Forecaster, PoissonForecaster
from lifeline.geo import TravelModel, load_geography
from lifeline.memory.selfmodel import SelfModel, bucket_of
from lifeline.memory.store import Memory
from lifeline.planner.coverage import busy_prob, evaluate
from lifeline.planner.mcts import Constraints, MCTSPlanner
from lifeline.planner.mexclp import mexclp, proportional
from lifeline.planner.verify import simulate_plan
from lifeline.sim.calibrate import day_incidents, load_calibration
from lifeline.sim.engine import BUSY, Incident, Samplers, Simulator, summarize
from lifeline.tools import guard as guard_tool
from lifeline.tools.sandbox import run_code
from lifeline.tools.weather import get_weather

NOTE_TEMPLATES = {
    1: ["Caller reports patient not breathing, CPR in progress", "Unresponsive adult, bystander on scene"],
    2: ["Chest pain, 64 year old male, sweating", "Difficulty breathing, history of asthma", "Possible stroke, slurred speech"],
    3: ["Fall from ladder, leg injury, conscious", "Diabetic patient confused", "Seizure, now post-ictal"],
    4: ["Abdominal pain for 3 hours", "Allergic reaction, hives, breathing fine", "Minor MVA, neck pain"],
}

# Known venues used for "event" injections: zone -> name
VENUES = {"11217": "Barclays Center", "11224": "Maimonides Park / Coney Island boardwalk", "11211": "Williamsburg waterfront"}


class Runtime:
    """Heavy, shared, read-mostly objects (loaded once per process)."""

    def __init__(self, memory_dir=None):
        """memory_dir: use a private copy of memory + self-model (parallel benchmark workers)."""
        self.geo = load_geography()
        self.cal = load_calibration()
        self.travel = TravelModel(self.geo, np.array(self.cal["speed_by_hour"]), self.cal["base_s"])
        self.forecaster = Forecaster.load()
        if memory_dir is not None:
            import shutil
            from pathlib import Path
            memory_dir = Path(memory_dir)
            memory_dir.mkdir(parents=True, exist_ok=True)
            src = ARTIFACTS / "memory" / "selfmodel.sqlite"
            if src.exists():
                shutil.copy(src, memory_dir / "selfmodel.sqlite")
            self.memory = Memory(memory_dir)
            self.selfmodel = SelfModel(memory_dir / "selfmodel.sqlite")
        else:
            self.memory = Memory()
            self.selfmodel = SelfModel()
        self.fleet = int(self.cal["fleet"])
        self.zone_demand = np.array(self.cal["zone_demand_per_hour"])
        self.test = pd.read_parquet(DATA_DIR / "test_incidents.parquet")
        self.splits = json.loads((DATA_DIR / "splits.json").read_text())
        self.graph = build_graph(None if memory_dir is not None else str(ARTIFACTS / "checkpoints.sqlite"))

    @cached_property
    def calibration(self) -> pd.DataFrame:
        return pd.read_parquet(DATA_DIR / "calibration_incidents.parquet")

    @cached_property
    def poisson(self) -> PoissonForecaster:
        return PoissonForecaster.load(self.geo.Z)

    @cached_property
    def home_posts(self) -> dict[str, list[int]]:
        path = ARTIFACTS / "home_posts.json"
        if path.exists():
            return json.loads(path.read_text())
        q = busy_prob(self.zone_demand.sum(), self.fleet, self.cal["service_h"], 0.0)
        posts = {"proportional": proportional(self.zone_demand, self.fleet),
                 "mexclp": mexclp(self.zone_demand, self.travel.matrix(14), self.fleet, q, self.cal["proc_median_s"])}
        path.write_text(json.dumps(posts))
        return posts


@dataclass
class Session:
    rt: Runtime
    day: str
    start_hour: int = 17
    hours: float = 6.0
    policy: str = "lifeline"          # lifeline | static | mexclp | lifeline_poisson | no_critic | no_selfmodel
    speed: float = 120.0              # simulated seconds per real second
    headless: bool = False
    seed: int = 0
    review_every_min: int = 30
    source: str = "test"              # test (held-out 2026 days) | calibration (2025 days)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.geo, self.memory, self.selfmodel = self.rt.geo, self.rt.memory, self.rt.selfmodel
        self.travel = self.rt.travel
        self.forecaster = self.rt.poisson if self.policy == "lifeline_poisson" else self.rt.forecaster
        # agents start from the demand-proportional posts (the stronger static plan in validation);
        # the MEXCLP baseline uses its own optimised posts
        posts = self.rt.home_posts["mexclp" if self.policy == "mexclp" else "proportional"]
        cal = self.rt.cal
        samplers = Samplers(np.array(cal["proc_s"]), np.array(cal["service_s"]), np.random.default_rng(self.seed + 1))
        self.sim = Simulator(self.travel, list(posts), pd.Timestamp(self.day), samplers)
        incs = day_incidents(self.rt.test if self.source == "test" else self.rt.calibration, self.geo, self.day)
        for inc in incs:                                  # synthetic free-text notes on some calls
            if self.rng.random() < 0.08:
                inc.note = str(self.rng.choice(NOTE_TEMPLATES[inc.severity]))
        self.sim.add_incidents(incs)
        self.t_start = self.start_hour * 3600.0
        self.t_end = min(self.t_start + self.hours * 3600.0, 86400.0)
        self.sim.run_until(self.t_start)                  # warm-up: realistic busy units and recent calls
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self.status = "ready"
        self.paused = False
        self.epoch = 0
        self.last_review = -1e9
        self.last_cov: float | None = None
        self.scenarios: np.ndarray | None = None
        self.last_forecast: dict | None = None
        self.last_latent = None
        self.last_state: dict = {}
        self.injections: list[dict] = []
        self.pending_notes: list[dict] = []
        self.pending_decision: dict | None = None
        self._decision_ready = threading.Event()
        self._decision: dict | None = None
        self.pending_eval: list[dict] = []
        self.reviews: list[dict] = []
        self.next_reflect = (int(self.t_start // 3600) + 1) * 3600.0
        self._seen_calls = len(self.sim.calls)
        self.stopped = False
        self.history: list[dict] = []

    # ================================================================ event bus
    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"i": len(self.events), "kind": kind, "sim_t": self.sim.now,
                                "time": self.sim.timestamp().strftime("%H:%M"), **data})

    def events_since(self, i: int) -> list[dict]:
        with self._lock:
            return self.events[i:]

    # ================================================================ main loop
    def run(self) -> None:
        self.status = "running"
        self.emit("session", status="running", day=self.day, policy=self.policy, fleet=len(self.sim.units))
        try:
            while self.sim.now < self.t_end and not self.stopped:
                if self.paused:
                    time.sleep(0.1)
                    continue
                t_next = min(self.sim.now + 60.0, self.t_end)
                self.sim.run_until(t_next)
                self._frame()
                if not self.headless:
                    time.sleep(60.0 / self.speed)
                if self.sim.now >= self.next_reflect:
                    self.reflect()
                    self.next_reflect += 3600.0
                if int(self.sim.now) % (EPOCH_MIN * 60) == 0:
                    self.epoch_tick()
            self.status = "finished"
            self.emit("session", status="finished", metrics=self.metrics())
        except Exception as e:  # surface errors to the dashboard
            import traceback
            self.status = "error"
            self.emit("error", message=str(e), detail=traceback.format_exc()[-1500:])
            if self.headless:
                raise

    def _frame(self) -> None:
        new = self.sim.calls[self._seen_calls:]
        self._seen_calls = len(self.sim.calls)
        for inc in new:
            if inc.note:
                self.pending_notes.append({"call": inc.id, "zone": self.geo.zones[inc.zone], "text": inc.note})
        if self.headless:
            return
        snap = self.sim.snapshot()
        self.emit("frame", units=snap["units"], queue=snap["queue"],
                  calls=[{"id": c.id, "zone": c.zone, "sev": c.severity} for c in new],
                  metrics=self.metrics(window_s=3600))

    # ================================================================ agent epoch
    def epoch_tick(self) -> None:
        self.epoch += 1
        if self.policy in ("static", "mexclp"):
            return
        triggers = self.triggers()
        if not triggers:
            return
        self.last_review = self.sim.now
        self.run_graph(triggers)
        self.last_cov = self.coverage_now()

    def triggers(self) -> list[str]:
        t = []
        if self.sim.now - self.last_review >= self.review_every_min * 60:
            t.append("scheduled review")
        if self.injections and any(i["t"] >= self.sim.now - EPOCH_MIN * 60 for i in self.injections):
            t.append("new event reported")
        if any(u.status != BUSY for u in self.sim.units) is False:
            t.append("no units available")
        if self.last_forecast is not None and self.sim.now - self.last_review >= 3600:
            obs = self.lag_counts(1).sum()
            lo, hi = self.last_forecast["first_hour_p05"], self.last_forecast["first_hour_p95"]
            if obs < lo or obs > hi:
                t.append(f"forecast drift ({obs:.0f} calls vs {lo:.0f}-{hi:.0f} predicted)")
        cov = self.coverage_now()
        if self.last_cov is not None and cov < self.last_cov - 0.05:
            t.append(f"coverage drop ({self.last_cov:.0%} -> {cov:.0%} expected within 8 min)")
        return t

    def run_graph(self, triggers: list[str]) -> dict:
        notes, self.pending_notes = self.pending_notes, []
        config = {"configurable": {"thread_id": f"{self.id}:{self.epoch}", "rt": self}}
        inp = {"epoch": self.epoch, "sim_time": self.sim.timestamp().isoformat(), "triggers": triggers,
               "notes": notes[-12:], "critic_round": 0, "audit_round": 0}
        self.emit("review_start", epoch=self.epoch, triggers=triggers)
        self.status = "thinking"
        state = self._stream(inp, config)
        while state.get("__interrupt__"):
            payload = state["__interrupt__"]
            self.pending_decision = {"epoch": self.epoch, **payload}
            self.status = "awaiting_approval"
            self.emit("approval", **payload)
            decision = self._wait_decision()
            self.pending_decision = None
            self.status = "thinking"
            state = self._stream(Command(resume=decision), config)
        self.status = "running"
        final = self.rt.graph.get_state(config).values
        self.last_state = final
        review = {"epoch": self.epoch, "time": self.sim.timestamp().strftime("%H:%M"), "triggers": triggers,
                  "tier": final.get("meta", {}).get("tier"), "moves": final.get("executed", {}).get("moves", []),
                  "critic_rounds": final.get("critic_round", 0), "thread_id": config["configurable"]["thread_id"]}
        self.reviews.append(review)
        self.emit("review_end", review=review, state=_public_state(final))
        return final

    def _stream(self, inp, config) -> dict:
        interrupted = None
        for chunk in self.rt.graph.stream(inp, config, stream_mode="updates"):
            for node, upd in chunk.items():
                if node == "__interrupt__":
                    interrupted = upd[0].value
                    continue
                if not self.headless:
                    self.emit("node", node=node, trace=(upd or {}).get("trace", []),
                              update=_public_state({k: v for k, v in (upd or {}).items() if k != "trace"}))
        return {"__interrupt__": interrupted} if interrupted else {}

    def _wait_decision(self) -> dict:
        if self.headless:
            return {"action": "approve", "by": "auto (benchmark)"}
        self._decision_ready.clear()
        while not self._decision_ready.wait(0.2):
            if self.stopped:
                return {"action": "reject", "reason": "session stopped"}
        return self._decision or {"action": "approve"}

    def decide(self, decision: dict) -> None:
        self._decision = decision
        self._decision_ready.set()

    # ================================================================ tools used by agents
    def lag_counts(self, hours: int) -> np.ndarray:
        """(hours, Z) calls per zone in each of the last `hours` hours."""
        out = np.zeros((hours, self.geo.Z), np.float32)
        now = self.sim.now
        for c in reversed(self.sim.calls):
            age = now - c.t
            if age >= hours * 3600:
                break
            out[hours - 1 - int(age // 3600), c.zone] += 1
        return out

    def weather(self) -> dict:
        w = get_weather(self.sim.timestamp(), hours=3)
        for inj in self.injections:
            if inj["kind"] == "rain" and inj["t"] <= self.sim.now < inj["t"] + 3 * 3600:
                w = {**w, "precip": [max(p, 8.0) for p in w["precip"]], "source": w["source"] + "+injected"}
        return w

    def active_events(self) -> list[dict]:
        return [e for e in self.injections if e["kind"] == "event" and e["t"] <= self.sim.now < e["t"] + 3 * 3600]

    def situation_facts(self, notes: list[dict]) -> dict:
        lag = self.lag_counts(1)[0]
        by_zone = {self.geo.zones[z]: int(n) for z, n in enumerate(lag) if n > 0}
        w = self.weather()
        m = self.metrics(window_s=3600)
        busy = sum(u.status == BUSY for u in self.sim.units)
        return {
            "time": self.sim.timestamp().strftime("%a %Y-%m-%d %H:%M"),
            "calls_last_hour": int(lag.sum()),
            "calls_last_hour_by_zone": dict(sorted(by_zone.items(), key=lambda kv: -kv[1])[:8]),
            "fleet": len(self.sim.units), "busy_units": busy, "queue": len(self.sim.queue),
            "last_hour": {k: round(m.get(k, 0), 3) for k in ("mean_rt", "p90_rt", "within_target")},
            "weather": {"precip_next_2h_mm": round(sum(w["precip"][:2]), 1), "temp_c": round(w["temp"][0], 1),
                        "source": w["source"]},
            "events": [{"name": e["name"], "zone": e["zone"], "zones": e["zones"]} for e in self.active_events()],
            "call_notes": [n["text"] for n in notes][:6],
            "units_out_of_service": [i["unit"] for i in self.injections if i["kind"] == "breakdown"],
        }

    def forecast(self) -> dict:
        ts = self.sim.timestamp().ceil("h")
        w = self.weather()
        lag = self.lag_counts(3)
        c = self.forecaster.cond(ts, lag, np.array(w["precip"]), np.array(w["temp"]))
        sc = self.forecaster.scenarios(c, N_SCENARIOS, self.rng)
        counts, latent = sc["counts"], sc["latent"]
        self.scenarios = counts
        totals = counts.sum((1, 2))
        first = counts[:, 0].sum(1)
        hist = self.rt.zone_demand.sum() * HORIZON
        zmean, zstd = counts.sum(1).mean(0), counts.sum(1).std(0)
        unc = np.argsort(-(zstd * np.sqrt(zmean + 1e-3)))[:3]
        ood = self.forecaster.ood(c, lag[-HORIZON:].reshape(-1) if lag.shape[0] >= HORIZON else None)
        self.last_latent = latent.mean(0)
        fc = {
            "model": self.forecaster.kind,
            "total_mean": float(totals.mean()), "total_p05": float(np.percentile(totals, 5)),
            "total_p95": float(np.percentile(totals, 95)), "total_cv": float(totals.std() / max(totals.mean(), 1)),
            "first_hour_p05": float(np.percentile(first, 5)), "first_hour_p95": float(np.percentile(first, 95)),
            "historical_mean": float(hist), "surge_prob": float((totals > 1.25 * hist).mean()),
            "zone_mean": zmean.round(2).tolist(), "zone_std": zstd.round(2).tolist(),
            "uncertain_zones": [self.geo.zones[z] for z in unc],
            "scenario_zone_totals": counts.sum(1).astype(int).tolist(),
            "latent_2d": latent[:, :2].round(3).tolist(), "ood": ood,
        }
        self.last_forecast = fc
        return fc

    def current_bucket(self) -> str:
        w = self.weather()
        surge = bool(self.last_forecast and self.last_forecast["surge_prob"] > 0.5)
        return bucket_of(self.sim.timestamp(), w["precip"][0], surge)

    def frozen_units(self) -> list[int]:
        """Crews within 30 minutes of shift end (two-shift rota, staggered by unit id)."""
        h = self.sim.timestamp().hour + self.sim.timestamp().minute / 60
        out = []
        for u in self.sim.units:
            ends = (7, 19) if u.id % 2 == 0 else (8, 20)
            if any(0 <= (e - h) % 24 <= 0.5 for e in ends):
                out.append(u.id)
        return out

    def run_planner(self, cons: Constraints, multipliers: dict[str, float]) -> dict:
        scen = self.scenarios.copy()
        for z, m in multipliers.items():
            scen[:, :, self.geo.index(z)] *= m
        for inj in self.injections:                       # broken-down units are not available
            if inj["kind"] == "breakdown":
                cons.frozen_units.add(inj["unit"])
        if self.policy == "no_equity":
            cons.equity_boost = 0.0
        avail, fixed = self.fleet_view()
        T = self.travel.matrix(self.sim.hour())
        planner = MCTSPlanner(T, self.geo.km, self.rt.cal["service_h"], len(self.sim.units), self.rng,
                              self.rt.cal["proc_median_s"])
        res = planner.plan(avail, scen, self.sim.busy_fraction(), cons, iters=MCTS_ITERS, fixed=fixed)

        # stage 2: verify the best candidates in the full simulator on fresh futures
        cands = {"mcts-best": res.moves, "hold": []}
        for i, a in enumerate(res.alternatives):
            cands.setdefault(f"alt-{i + 1}", [tuple(m) for m in a["moves"]])
        seed = int(self.rng.integers(1 << 20))
        verified = {name: simulate_plan(self.sim, list(mv), scen, seed, n=16) for name, mv in cands.items()}
        # pick the best candidate in the simulator; holding wins unless a move is clearly better
        chosen = max((k for k in cands if cands[k]), key=lambda k: verified[k]["reward"], default="hold")
        if verified.get(chosen, {}).get("reward", -1e9) <= verified["hold"]["reward"]:
            chosen = "hold"
        chosen_moves = list(cands[chosen])
        root_pos = np.array([z for _, z in avail])
        plan_pos = planner._apply(root_pos, chosen_moves)
        plan_eval = planner._mean_eval(plan_pos, tuple(chosen_moves), cons)
        support = 0
        for s in range(len(scen)):
            r_plan = planner._score(plan_pos, s, planner._km(tuple(chosen_moves)), cons)[0]
            support += int(r_plan > planner.base[s] + 1e-6) if chosen_moves else 0
        moves = [{"unit": int(u), "from": int(planner.origin[u]), "to": int(p),
                  "from_zone": self.geo.zones[planner.origin[u]], "to_zone": self.geo.zones[p],
                  "km": round(float(self.geo.km[planner.origin[u], p]), 2),
                  "route": [self.geo.zones[z] for z in self.geo.route(planner.origin[u], p)[0]]}
                 for u, p in chosen_moves]
        return {"moves": moves, "root_eval": res.root_eval, "plan_eval": plan_eval, "alternatives": res.alternatives,
                "tree": res.tree, "iterations": res.iterations, "candidate_posts": [self.geo.zones[p] for p in res.candidate_posts],
                "verified": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in verified.items()},
                "verification": {"plan": verified.get(chosen, {}), "no_move": verified["hold"]},
                "chosen": chosen, "support": support, "positions_before": root_pos.tolist(),
                "positions_after": plan_pos.tolist(), "fixed": list(fixed), "unit_ids": [u for u, _ in avail]}

    def fleet_view(self) -> tuple[list[tuple[int, int]], list[int]]:
        """(movable units with their current zone, posts of busy units they will return to)."""
        avail = [(u.id, u.position(self.sim.now)) for u in self.sim.units if u.status != BUSY]
        fixed = [u.post for u in self.sim.units if u.status == BUSY]
        return avail, fixed

    def coverage_now(self) -> float:
        avail, fixed = self.fleet_view()
        pos = np.array([z for _, z in avail] + fixed)
        q = busy_prob(self.rt.zone_demand.sum(), len(self.sim.units), self.rt.cal["service_h"], self.sim.busy_fraction())
        return evaluate(pos, self.rt.zone_demand, self.travel.matrix(self.sim.hour()), q, 0,
                        self.rt.cal["proc_median_s"]).within

    def apply_moves(self, moves: list[dict], state: dict, source: str) -> list[dict]:
        applied = []
        for m in moves:
            u, p = int(m["unit"]), int(m["to"]) if "to" in m else self.geo.index(m["to_zone"])
            if self.sim.reposition(u, p):
                applied.append({**m, "unit": u, "to": p})
        plan = state.get("plan", {})
        meta = state.get("meta", {})
        bucket = self.current_bucket()
        did = self.selfmodel.log_decision(self.sim.timestamp().isoformat(), bucket, meta.get("confidence", 0.5),
                                          meta.get("tier", "none"))
        self.pending_eval.append({"t": self.sim.now, "bucket": bucket, "decision_id": did,
                                  "before": plan.get("positions_before", []), "after": plan.get("positions_after", []),
                                  "fixed": plan.get("fixed", []),
                                  "applied": bool(applied), "predicted_within": plan.get("plan_eval", {}).get("within")})
        sit = state.get("situation", {})
        self.memory.remember(f"{self.id}-{self.epoch}", sit.get("headline", "")
                             + " | " + "; ".join(sit.get("concerns", [])), self.last_latent,
                             {"time": self.sim.timestamp().isoformat(), "bucket": bucket, "tier": meta.get("tier"),
                              "moves": len(applied), "source": source,
                              "predicted_within": plan.get("plan_eval", {}).get("within")})
        if source == "override":
            self.memory.add_lesson(self.sim.timestamp().isoformat(), bucket,
                                   f"Supervisor overrode the plan: {state.get('human', {}).get('reason', 'no reason given')}",
                                   "human-override")
        if applied and not self.headless:
            self.emit("moves", moves=applied, tier=meta.get("tier"))
        return applied

    def answer_question(self, question: str, ledger: dict) -> str:
        g = guard_tool.screen(question)
        if g["blocked"]:
            return "That question was blocked by the prompt-injection guardrail."
        res, _ = get_llm().ask(prompts.EXPLAINER_QA, json.dumps({"question": question, "facts": ledger}))
        if res:
            return res
        words = set(question.lower().split())
        hits = [f"{v} [{k}]" for k, v in ledger.items() if words & set(v.lower().split())]
        return " ".join(hits[:3]) or "The facts I have: " + " ".join(f"{v} [{k}]." for k, v in list(ledger.items())[:3])

    # ================================================================ reflection (after-action review)
    def reflect(self) -> None:
        if self.policy in ("static", "mexclp"):
            return
        hour_counts = self.lag_counts(1)[0]
        m = self.metrics(window_s=3600)
        lessons = []
        for pe in [p for p in self.pending_eval if p["t"] >= self.sim.now - 3600 and p["applied"]]:
            if not len(pe["before"]):
                continue
            win, regret = hindsight(self.rt, pe["before"], pe["after"], pe["fixed"], hour_counts,
                                    self.sim.hour(), self.sim.busy_fraction(), len(self.sim.units))
            self.selfmodel.update(pe["bucket"], regret, win)
            self.selfmodel.resolve_decision(pe["decision_id"], win, regret)
            lessons.append((pe, win, regret))
        self.pending_eval = [p for p in self.pending_eval if p["t"] >= self.sim.now - 3600 and not p["applied"]]
        if self.last_forecast:
            lo, hi = self.last_forecast["first_hour_p05"], self.last_forecast["first_hour_p95"]
            obs = hour_counts.sum()
            bias = "under-predicted" if obs > hi else "over-predicted" if obs < lo else "matched"
            text = (f"{self.current_bucket()}: demand {bias} ({obs:.0f} calls vs {lo:.0f}-{hi:.0f}); "
                    f"{m.get('within_target', 0):.0%} reached within 8 min")
            if lessons:
                wins = sum(w for _, w, _ in lessons)
                text += f"; {wins}/{len(lessons)} moves helped, mean regret {np.mean([r for *_, r in lessons]):.0f}s"
            if get_llm().online and not self.headless:
                res, _ = get_llm().ask(prompts.REFLECTOR, text)
                if res:
                    text = f"{text}. Lesson: {res.strip()}"
            self.memory.add_lesson(self.sim.timestamp().isoformat(), self.current_bucket(), text, "reflector")
            if not self.headless:
                self.emit("node", node="reflector", trace=[{"agent": "Reflector", "kind": "reflection", "summary": text}],
                          update={})

    # ================================================================ user actions
    def inject(self, kind: str, text: str | None = None, zone: str | None = None) -> dict:
        now = self.sim.now
        if kind == "rain":
            inj = {"kind": "rain", "t": now, "name": "heavy rain"}
            extra = self._surge(now, {z: 1.25 for z in self.geo.zones})
        elif kind == "event":
            zone = zone if zone in VENUES else "11217"
            i = self.geo.index(zone)
            neigh = [self.geo.zones[j] for j in self.geo.graph[i]]
            mult = {zone: 2.5, **{z: 1.4 for z in neigh}}
            inj = {"kind": "event", "t": now, "zone": zone, "name": f"large event at {VENUES[zone]}",
                   "zones": {zone: 2.0, **{z: 1.3 for z in neigh[:4]}}}
            extra = self._surge(now, mult)
        elif kind == "breakdown":
            idle = [u for u in self.sim.units if u.status != BUSY]
            u = idle[int(self.rng.integers(len(idle)))]
            u.status, u.version = BUSY, u.version + 1
            u.free_t = now + 3 * 3600
            self.sim._push(u.free_t, "clear", (u.id, u.version))
            inj = {"kind": "breakdown", "t": now, "unit": u.id, "name": f"unit {u.id} out of service"}
            extra = 0
        elif kind == "note":
            self.pending_notes.append({"call": "manual", "zone": zone or "", "text": text or ""})
            inj = {"kind": "note", "t": now, "name": "call note", "text": text}
            extra = 0
        else:
            raise ValueError(kind)
        self.injections.append(inj)
        self.emit("injection", injection=inj, extra_calls=extra)
        return inj

    def _surge(self, now: float, mult: dict[str, float]) -> int:
        """Add extra calls on top of the replay for the next 2 hours (the 'truth' the agents must react to)."""
        incs = []
        for z, m in mult.items():
            lam = self.rt.zone_demand[self.geo.index(z)] * (m - 1) * 2
            for k in range(self.rng.poisson(lam)):
                incs.append(Incident(f"x{len(self.injections)}-{z}-{k}", now + self.rng.random() * 7200,
                                     self.geo.index(z), int(self.rng.choice([1, 2, 3, 4], p=[.05, .3, .35, .3]))))
        self.sim.add_incidents(incs)
        return len(incs)

    # ================================================================ reporting
    def metrics(self, window_s: float | None = None) -> dict:
        df = self.sim.records_frame()
        if df.empty:
            return {"n": 0}
        df = df[df.t >= (self.sim.now - window_s if window_s else self.t_start)]
        return summarize(df, self.geo.Z, RESPONSE_TARGET_S) if not df.empty else {"n": 0}

    def state(self) -> dict:
        snap = self.sim.snapshot()
        return {"id": self.id, "day": self.day, "policy": self.policy, "status": self.status, "paused": self.paused,
                "speed": self.speed, "epoch": self.epoch, "time": self.sim.timestamp().isoformat(),
                "t_start": self.t_start, "t_end": self.t_end, "units": snap["units"], "queue": snap["queue"],
                "metrics": self.metrics(), "metrics_hour": self.metrics(3600), "reviews": self.reviews[-20:],
                "pending_decision": self.pending_decision, "injections": self.injections,
                "agent_state": _public_state(self.last_state), "llm": {"providers": get_llm().names(), **usage()},
                "events_count": len(self.events)}


def hindsight(rt: Runtime, before, after, fixed, realised: np.ndarray, hour: int, busy: float,
              fleet: int) -> tuple[bool, float]:
    """Score an executed decision against what actually happened.
    win    = executed configuration beats holding positions on the realised demand
    regret = seconds-equivalent reward lost vs a hindsight oracle that knew the realised demand"""
    T = rt.travel.matrix(hour)
    before, after, fixed = np.array(before, int), np.array(after, int), list(fixed)
    q = busy_prob(realised.sum(), fleet, rt.cal["service_h"], busy)
    allp = lambda p: np.concatenate([p, np.array(fixed, int)])
    proc = rt.cal["proc_median_s"]
    r_b = evaluate(allp(before), realised, T, q, 0, proc).reward
    r_a = evaluate(allp(after), realised, T, q, 0, proc).reward
    pl = MCTSPlanner(T, rt.geo.km, rt.cal["service_h"], fleet, np.random.default_rng(0), proc)
    oracle = pl.plan([(i, int(z)) for i, z in enumerate(before)], realised[None, None, :], busy, iters=150, fixed=fixed)
    r_o = evaluate(allp(pl._apply(before, oracle.moves)), realised, T, q, 0, proc).reward
    return bool(r_a >= r_b - 1e-9), float(max(0.0, r_o - r_a))


def _public_state(s: dict) -> dict:
    """Drop bulky fields before sending state to the browser."""
    out = {}
    for k, v in (s or {}).items():
        if k == "forecast" and isinstance(v, dict):
            out[k] = v
        elif k == "plan" and isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if kk not in ("positions_before", "positions_after", "unit_ids")}
        elif k in ("trace",):
            continue
        else:
            out[k] = v
    return json.loads(json.dumps(out, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)
