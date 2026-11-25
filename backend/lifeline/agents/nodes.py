"""Agent nodes of the LIFELINE graph.

Every node reads the shared AgentState, may call tools / the LLM, and returns a partial update.
Large objects (simulator, scenarios, models) live on the runtime object passed through
config["configurable"]["rt"], so checkpoints stay small and serialisable.
Every LLM agent has a deterministic fallback, so the graph also runs with no API key.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from lifeline.agents import prompts
from lifeline.agents.llm import get_llm
from lifeline.agents.schemas import AgentState, Directive, Rationale, SelfExplanation, SituationSummary
from lifeline.config import (CONFIDENCE_AUTO, COVERAGE_FLOOR, MAX_CRITIC_ROUNDS, MAX_MOVES, N_SCENARIOS,
                             TIER1_MAX_KM, TIER1_MAX_UNITS, TIER2_MAX_UNITS)
from lifeline.memory.selfmodel import bucket_of
from lifeline.planner.coverage import evaluate
from lifeline.planner.mcts import Constraints, MCTSPlanner
from lifeline.planner.verify import simulate_plan
from lifeline.tools.guard import screen


def _rt(config: RunnableConfig):
    return config["configurable"]["rt"]


def _ev(agent: str, kind: str, summary: str, **data) -> dict:
    return {"agent": agent, "kind": kind, "summary": summary, "ts": time.time(), **data}


def _zone_codes(rt, codes) -> list[str]:
    return [c for c in (codes or []) if c in rt.geo.zones]


# ---------------------------------------------------------------------------------------------
def guard(state: AgentState, config: RunnableConfig) -> dict:
    notes = state.get("notes", [])
    results, passed, blocked = [], [], []
    for n in notes:
        r = screen(n["text"])
        results.append({**n, **r})
        (blocked if r["blocked"] else passed).append(n)
    trace = [_ev("Guard", "guardrail", f"screened {len(notes)} call notes, blocked {len(blocked)}",
                 blocked=[b["text"] for b in blocked])]
    for b in blocked:
        trace.append(_ev("Guard", "blocked", f"prompt injection blocked in note for call {b.get('call')}",
                         text=b["text"]))
    return {"guard": {"results": results, "passed": passed, "blocked": blocked}, "trace": trace}


def situation_analyst(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    snap = rt.situation_facts(state.get("guard", {}).get("passed", []))
    result, meta = get_llm().ask(prompts.SITUATION, json.dumps(snap), SituationSummary)
    if result is None:
        result = _fallback_situation(snap)
    s = result.model_dump()
    s["hotspots"] = _zone_codes(rt, s["hotspots"])[:4]
    return {"situation": {**s, "facts": snap},
            "context": {"weather": snap["weather"], "events": snap["events"]},
            "trace": [_ev("Situation Analyst", "llm", s["headline"], provider=meta["provider"],
                          latency_s=meta.get("latency_s"), output=s)]}


def _fallback_situation(snap: dict) -> SituationSummary:
    hot = [z for z, _ in sorted(snap["calls_last_hour_by_zone"].items(), key=lambda kv: -kv[1])[:3]]
    risk = "high" if snap["queue"] or snap["events"] or snap["weather"]["precip_next_2h_mm"] > 4 else \
        "elevated" if snap["busy_units"] > 0.6 * snap["fleet"] else "low"
    concerns = []
    if snap["events"]:
        concerns.append(f"event: {snap['events'][0]['name']}")
    if snap["weather"]["precip_next_2h_mm"] > 1:
        concerns.append("rain expected in the next two hours")
    if snap["queue"]:
        concerns.append(f"{snap['queue']} calls waiting for a unit")
    return SituationSummary(
        headline=f"{snap['calls_last_hour']} calls in the last hour, {snap['busy_units']}/{snap['fleet']} units busy"
                 + (f", {snap['queue']} queued" if snap["queue"] else ""),
        risk_level=risk, hotspots=hot, concerns=concerns or ["normal operations"])


# ---- parallel branch 1 -------------------------------------------------------------------------
def demand_forecaster(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    t0 = time.perf_counter()
    fc = rt.forecast()
    return {"forecast": fc, "trace": [_ev("Demand Forecaster", "tool",
            f"generate_demand_scenarios: {N_SCENARIOS} futures, {fc['total_mean']:.0f} calls expected "
            f"(90% range {fc['total_p05']:.0f}-{fc['total_p95']:.0f}), surge probability {fc['surge_prob']:.0%}",
            tool="generate_demand_scenarios", latency_s=round(time.perf_counter() - t0, 2),
            model=fc["model"], ood=fc["ood"]["score"])]}


# ---- parallel branch 2 -------------------------------------------------------------------------
def memory_retriever(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    sit = state.get("situation", {})
    bucket = rt.current_bucket()
    similar = rt.memory.recall(sit.get("headline", ""), rt.last_latent, k=3)
    lessons = rt.memory.lessons(bucket, k=3) or rt.memory.lessons(None, k=2)
    templates = rt.memory.templates(bucket)
    return {"memory": {"bucket": bucket, "similar": similar, "lessons": lessons, "templates": templates},
            "trace": [_ev("Memory Retriever", "tool",
                          f"recall_similar_situations: {len(similar)} episodes, {len(lessons)} lessons, "
                          f"{len(templates)} posting templates for bucket {bucket}", tool="recall_similar_situations")]}


# ---- parallel branch 3 -------------------------------------------------------------------------
def sop_retriever(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    sit = state.get("situation", {})
    q = " ".join(sit.get("concerns", [])) + " repositioning coverage equity"
    hits = rt.memory.search_sop(q, k=2)
    return {"sop": hits, "trace": [_ev("SOP Retriever", "tool", "search_sop: " + ", ".join(h["id"] for h in hits),
                                       tool="search_sop")]}


# ---------------------------------------------------------------------------------------------
def planner(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    rnd = state.get("critic_round", 0)
    critic = state.get("critic", [])
    last = critic[-1] if critic and not critic[-1]["ok"] else None
    sit, fc = state.get("situation", {}), state.get("forecast", {})

    payload = {
        "situation": {k: sit.get(k) for k in ("headline", "risk_level", "hotspots", "concerns")},
        "context": state.get("context", {}),
        "forecast": {k: fc.get(k) for k in ("total_mean", "total_p05", "total_p95", "surge_prob", "uncertain_zones")},
        "frozen_units_near_shift_end": rt.frozen_units(),
        "lessons": [l["text"] for l in state.get("memory", {}).get("lessons", [])],
        "sop": [h["text"][:400] for h in state.get("sop", [])],
        "critic_feedback": last["violations"] if last else None,
        "previous_directive": state.get("directive") if last else None,
    }
    result, meta = get_llm().ask(prompts.PLANNER, json.dumps(payload, default=str), Directive)
    if result is None:
        result = _fallback_directive(rt, sit, state.get("context", {}), last, state.get("directive"))
    d = result.model_dump()

    # deterministic merge: critic fixes are always applied, so the loop converges
    protect = set(_zone_codes(rt, d["protect_zones"]))
    eq, mm = d["equity_boost"], min(d["max_moves"], MAX_MOVES)
    if last:
        for v in last["violations"]:
            fix = v.get("fix", {})
            protect |= set(fix.get("protect_zones", []))
            eq = max(eq, fix.get("equity_boost", 1.0))
            mm = min(mm, fix.get("max_moves", mm))
    d.update(protect_zones=sorted(protect), equity_boost=round(eq, 2), max_moves=mm,
             demand_multipliers={z: float(np.clip(m, 1.0, 3.0)) for z, m in d.get("demand_multipliers", {}).items()
                                 if z in rt.geo.zones})

    cons = Constraints(frozen_units=set(rt.frozen_units()),
                       protected_zones={rt.geo.index(z): COVERAGE_FLOOR for z in d["protect_zones"]},
                       max_moves=d["max_moves"], equity_boost=d["equity_boost"])
    t0 = time.perf_counter()
    plan = rt.run_planner(cons, d["demand_multipliers"])
    plan_s = time.perf_counter() - t0
    trace = [
        _ev("Planner", "llm", d["reasoning"], provider=meta["provider"], latency_s=meta.get("latency_s"),
            output={k: d[k] for k in ("tasks", "protect_zones", "equity_boost", "max_moves", "demand_multipliers")},
            round=rnd),
        _ev("Planner", "tool", f"mcts_plan: {plan['iterations']} rollouts -> {len(plan['moves'])} moves "
            f"({plan['plan_eval']['within'] - plan['root_eval']['within']:+.1%} within 8 min)",
            tool="mcts_plan", latency_s=round(plan_s, 2)),
        _ev("Planner", "tool", f"simulate_plan: verified {len(plan['verified'])} candidate plans in the simulator; "
            f"picked '{plan['chosen']}'", tool="simulate_plan"),
    ]
    return {"directive": d, "plan": plan, "trace": trace}


def _fallback_directive(rt, sit, context, last, prev) -> Directive:
    protect = list(sit.get("hotspots", []))
    mult = {}
    for e in context.get("events", []):
        for z, m in e.get("zones", {}).items():
            mult[z] = m
            protect.append(z)
    eq = 1.0 if not prev else prev.get("equity_boost", 1.0)
    mm = MAX_MOVES if not prev else prev.get("max_moves", MAX_MOVES)
    tasks = ["assess current coverage", "forecast demand for the next 2 hours",
             "search repositioning moves with MCTS", "verify best plans in the simulator", "monitor and replan"]
    if last:
        tasks.insert(3, "fix the Critic's violations")
    return Directive(tasks=tasks, protect_zones=protect, demand_multipliers=mult, equity_boost=eq, max_moves=mm,
                     reasoning="Rule-based directive (offline mode): protect current hotspots and event zones, "
                               "keep default equity weight unless the Critic asked for more.")


# ---------------------------------------------------------------------------------------------
def critic(state: AgentState, config: RunnableConfig) -> dict:
    """Safety & Equity Critic: deterministic rule engine (check_constraints) over the draft plan."""
    rt = _rt(config)
    plan = state["plan"]
    before, after = plan["root_eval"], plan["plan_eval"]
    v = []
    zc_b, zc_a = np.array(before["zone_cov"]), np.array(after["zone_cov"])
    dropped = [rt.geo.zones[z] for z in np.where((zc_a < COVERAGE_FLOOR) & (zc_b >= COVERAGE_FLOOR - 1e-9))[0]]
    if dropped:
        v.append({"rule": "SOP-2 coverage floor",
                  "evidence": f"zones {', '.join(dropped)} fall below {COVERAGE_FLOOR:.0%} coverage after the plan",
                  "fix": {"protect_zones": dropped}})
    if len(plan["moves"]) > MAX_MOVES:
        v.append({"rule": "SOP-4 max moves", "evidence": f"{len(plan['moves'])} units moved",
                  "fix": {"max_moves": MAX_MOVES}})
    long = [m for m in plan["moves"] if m["km"] > 12]
    if long:
        v.append({"rule": "SOP-4 move length", "evidence": f"unit {long[0]['unit']} would drive {long[0]['km']:.1f} km",
                  "fix": {"max_moves": max(1, len(plan["moves"]) - 1)}})
    frozen = set(rt.frozen_units())
    bad = [m["unit"] for m in plan["moves"] if m["unit"] in frozen]
    if bad:
        v.append({"rule": "SOP-4 crew welfare", "evidence": f"units {bad} are within 30 min of shift end",
                  "fix": {}})
    if after["gini"] > before["gini"] + 0.01:
        v.append({"rule": "SOP-3 equity", "evidence": f"response-time inequality rises from Gini "
                  f"{before['gini']:.3f} to {after['gini']:.3f}", "fix": {"equity_boost": 1.6}})
    ver = plan.get("verification", {})
    if plan["moves"] and ver.get("plan", {}).get("reward", 0) < ver.get("no_move", {}).get("reward", 0) - 2:
        v.append({"rule": "no-harm check", "evidence": "in the full simulator the plan does worse than not moving "
                  f"({ver['plan']['mean_rt']:.0f}s vs {ver['no_move']['mean_rt']:.0f}s mean response)",
                  "fix": {"max_moves": max(0, len(plan["moves"]) - 1)}})
    if rt.policy == "no_critic":                       # ablation: reflection loop switched off
        v = []
    ok = not v
    rnd = state.get("critic_round", 0) + 1
    verdict = {"ok": ok, "round": rnd, "violations": v}
    summary = "plan accepted" if ok else f"plan REJECTED (round {rnd}): " + "; ".join(x["rule"] for x in v)
    return {"critic": [verdict], "critic_round": rnd,
            "trace": [_ev("Safety & Equity Critic", "reflection", summary, verdict=verdict, tool="check_constraints")]}


def route_after_critic(state: AgentState) -> str:
    last = state["critic"][-1]
    if last["ok"]:
        return "meta_reasoner"
    return "planner" if state["critic_round"] < MAX_CRITIC_ROUNDS else "meta_reasoner"


# ---------------------------------------------------------------------------------------------
def meta_reasoner(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    plan, fc = state["plan"], state["forecast"]
    critic_ok = state["critic"][-1]["ok"]
    bucket = rt.current_bucket()
    row = rt.selfmodel.row(bucket)
    spread = fc["total_cv"]
    ood = fc["ood"]["score"]
    raw = rt.selfmodel.raw_confidence(bucket, spread, ood)
    conf = rt.selfmodel.calibrated(raw)
    if rt.policy == "no_selfmodel":                    # ablation: always fully confident, never asks or waits
        raw = conf = 1.0

    moves = plan["moves"]
    km = sum(m["km"] for m in moves)
    gain_rt = plan["root_eval"]["mean_rt"] - plan["plan_eval"]["mean_rt"]
    below_now = plan["root_eval"]["min_zone_cov"] < COVERAGE_FLOOR
    below_after = plan["plan_eval"]["min_zone_cov"] < COVERAGE_FLOOR

    reasons = []
    if not moves:
        tier = "none"
        reasons.append("the best plan is to hold positions")
    elif not critic_ok:
        tier = "tier3"
        reasons.append("the Critic could not approve the plan after the maximum number of revisions")
    elif conf < CONFIDENCE_AUTO and not below_now and gain_rt < 5:
        tier = "wait"                     # value of information: cheap to wait, little to gain now
        reasons.append(f"low confidence ({conf:.0%}) and no zone is under the floor, so waiting one epoch "
                       f"for more data costs little (expected gain only {gain_rt:.1f}s)")
    elif conf < CONFIDENCE_AUTO or ood > 0.6 or below_after or len(moves) > TIER2_MAX_UNITS:
        tier = "tier3"
        reasons.append("confidence is low or the situation is unusual" if conf < CONFIDENCE_AUTO or ood > 0.6
                       else "the plan leaves a zone under the coverage floor")
    elif len(moves) <= TIER1_MAX_UNITS and km <= TIER1_MAX_KM:
        tier = "tier1"
        reasons.append("single short move with good confidence")
    else:
        tier = "tier2"
        reasons.append("up to three moves with good confidence")

    facts = {"bucket": bucket, "competence": row, "forecast_spread_cv": round(spread, 3),
             "ood_score": round(ood, 2), "ood_flags": fc["ood"]["outside_training_range"],
             "confidence": round(conf, 2), "tier": tier, "reasons": reasons}
    res, meta = get_llm().ask(prompts.META, json.dumps(facts, default=str), SelfExplanation)
    if res is None:
        rg = f"regret {row['regret']:.1f}s over {row['n']} cases" if row["regret"] is not None else "no track record yet"
        strength = "strong" if row["win_rate"] >= 0.6 and row["known"] else "weak" if row["known"] else "untested"
        res = SelfExplanation(statement=f"I am {strength} in '{bucket}' situations ({rg}); forecast spread is "
                                        f"{spread:.0%} and novelty {ood:.2f}, so my confidence is {conf:.0%}: "
                                        f"{reasons[0]} ({tier}).")
    out = {**facts, "raw_confidence": round(raw, 3), "self_explanation": res.statement}
    return {"meta": out, "trace": [_ev("Meta-Reasoner", "meta", res.statement, provider=meta["provider"],
                                       tier=tier, confidence=round(conf, 3))]}


# ---------------------------------------------------------------------------------------------
def build_ledger(rt, state: AgentState) -> dict[str, str]:
    plan, fc, meta = state["plan"], state["forecast"], state["meta"]
    b, a = plan["root_eval"], plan["plan_eval"]
    F = {
        "F1": f"{plan['support']} of {N_SCENARIOS} generated futures favour this plan over holding positions",
        "F2": f"share of calls reached within 8 minutes: {b['within']:.0%} now, {a['within']:.0%} with the plan",
        "F3": f"expected mean response time: {b['mean_rt']:.0f}s now, {a['mean_rt']:.0f}s with the plan",
        "F4": f"forecast for the next 2 hours: {fc['total_mean']:.0f} calls (90% range {fc['total_p05']:.0f} to {fc['total_p95']:.0f})",
        "F5": f"lowest-covered zone after the plan: {rt.geo.zones[a['worst_zone']]} at {a['min_zone_cov']:.0%}",
        "F6": f"confidence {meta['confidence']:.0%}, autonomy {meta['tier']}",
    }
    for i, m in enumerate(plan["moves"]):
        F[f"F{7 + i}"] = f"unit {m['unit']} moves from {m['from_zone']} to {m['to_zone']} ({m['km']:.1f} km)"
    if fc.get("uncertain_zones"):
        F[f"F{7 + len(plan['moves'])}"] = "most uncertain zones: " + ", ".join(fc["uncertain_zones"][:3])
    return F


def explainer(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    ledger = build_ledger(rt, state)
    rnd = state.get("audit_round", 0)
    feedback = state.get("explanation", {}).get("audit", {}) if rnd else {}
    user = json.dumps({"facts": ledger, "plan_tasks": state.get("directive", {}).get("tasks"),
                       "audit_feedback": feedback.get("unsupported")})
    res, meta = get_llm().ask(prompts.EXPLAINER, user, Rationale)
    text = res.text if res is not None else None
    if text is None:
        text = _template_rationale(state, ledger)
    return {"explanation": {"text": text, "ledger": ledger, "provider": meta["provider"]},
            "trace": [_ev("Liaison", "llm", text[:160] + ("..." if len(text) > 160 else ""), provider=meta["provider"])]}


def _template_rationale(state, ledger) -> str:
    """Grounded fallback: built only from ledger entries, so it always passes the audit."""
    moves = [k for k in ledger if int(k[1:]) >= 7 and ledger[k].startswith("unit")]
    cite = lambda k, text=None: f"{(text or ledger[k])[:1].upper()}{(text or ledger[k])[1:]} [{k}]." if k in ledger else ""
    if not moves:
        return " ".join(filter(None, ["Hold all positions: no move improves coverage enough.", cite("F3"), cite("F4")]))
    head = f"Recommend {len(moves)} move(s) [" + "][".join(moves) + "]."
    return " ".join(filter(None, [head, cite("F2"), cite("F1"), cite("F6")]))


NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def claims_audit(state: AgentState, config: RunnableConfig) -> dict:
    """Every number in the rationale must appear in the claims ledger (checked in code)."""
    exp = state["explanation"]
    ledger_text = " ".join(exp["ledger"].values()) + " " + " ".join(exp["ledger"].keys())
    allowed = set(NUM.findall(ledger_text)) | {"8", "2", "50"}
    cited = re.sub(r"\[F\d+\]", "", exp["text"])
    unsupported = sorted({n for n in NUM.findall(cited) if n not in allowed})
    rnd = state.get("audit_round", 0) + 1
    ok = not unsupported
    if not ok and rnd >= 2:                        # give up on the LLM text, use the grounded template
        exp = {**exp, "text": _template_rationale(state, exp["ledger"]), "provider": "template-after-audit"}
        ok = True
    audit = {"ok": ok, "unsupported": unsupported, "round": rnd}
    msg = "all numbers grounded in the claims ledger" if not unsupported else \
        f"unsupported numbers {unsupported} -> " + ("rewrite" if not ok else "replaced with grounded template")
    return {"explanation": {**exp, "audit": audit}, "audit_round": rnd,
            "trace": [_ev("Claims Auditor", "reflection", msg, audit=audit, tool="claims_ledger")]}


def route_after_audit(state: AgentState) -> str:
    return "autonomy_gate" if state["explanation"]["audit"]["ok"] else "explainer"


def autonomy_gate(state: AgentState, config: RunnableConfig) -> dict:
    tier = state["meta"]["tier"]
    return {"trace": [_ev("Autonomy Gate", "route", {"tier1": "auto-execute", "tier2": "auto-execute + notify",
                           "tier3": "needs supervisor approval", "wait": "wait one epoch (value of information)",
                           "none": "hold positions"}[tier], tier=tier)]}


def route_after_gate(state: AgentState) -> str:
    return "human_gate" if state["meta"]["tier"] == "tier3" else "executor"


def human_gate(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    plan = state["plan"]
    answer = interrupt({
        "type": "approval",
        "moves": plan["moves"], "alternatives": plan["alternatives"],
        "rationale": state["explanation"]["text"], "ledger": state["explanation"]["ledger"],
        "confidence": state["meta"]["confidence"], "self_explanation": state["meta"]["self_explanation"],
        "critic": state["critic"][-1], "blast_radius": {"units": len(plan["moves"]),
                                                        "km": round(sum(m["km"] for m in plan["moves"]), 1)},
    })
    action = answer.get("action", "approve")
    trace = [_ev("Human Supervisor", "human", f"{action}" + (f": {answer.get('reason') or answer.get('question')}"
                                                             if answer.get("reason") or answer.get("question") else ""))]
    if action == "ask":
        reply = rt.answer_question(answer.get("question", ""), state["explanation"]["ledger"])
        trace.append(_ev("Liaison", "llm", reply))
        return {"human": {"action": "ask", "question": answer.get("question"), "answer": reply}, "trace": trace}
    return {"human": answer, "trace": trace}


def route_after_human(state: AgentState) -> str:
    return "human_gate" if state.get("human", {}).get("action") == "ask" else "executor"


def executor(state: AgentState, config: RunnableConfig) -> dict:
    rt = _rt(config)
    tier = state["meta"]["tier"]
    human = state.get("human") or {}
    moves = state["plan"]["moves"]
    if tier in ("wait", "none"):
        moves = []
    elif tier == "tier3":
        if human.get("action") == "override":
            moves = human.get("moves", [])
        elif human.get("action") == "reject":
            moves = []
    applied = rt.apply_moves(moves, state, source=human.get("action") or tier)
    summary = f"applied {len(applied)} move(s)" if applied else "no moves applied"
    return {"executed": {"moves": applied, "tier": tier, "human": human.get("action")},
            "trace": [_ev("Executor", "action", summary, moves=applied)]}
