"""Agent-layer tests: guardrail, sandbox, critic, claims audit, graph routing, and an end-to-end
review with a human-in-the-loop interrupt (needs the prepared data in data/artifacts)."""
import os
import threading

import pytest

from lifeline.agents import nodes
from lifeline.agents.graph import build_graph, graph_spec
from lifeline.config import ARTIFACTS, MAX_CRITIC_ROUNDS
from lifeline.eval.agent_eval import BENIGN, ESCAPES, INJECTIONS, _RT, critic_cases
from lifeline.tools.guard import screen
from lifeline.tools.sandbox import run_code

os.environ["LIFELINE_OFFLINE"] = "1"
CFG = {"configurable": {"rt": _RT()}}


def test_guard_blocks_injections_and_passes_benign_notes():
    assert all(screen(t)["blocked"] for t in INJECTIONS)
    assert not any(screen(t)["blocked"] for t in BENIGN)


def test_sandbox_rejects_escapes_and_runs_safe_code():
    for code in ESCAPES:
        assert not run_code(code, [], [], [])["ok"]
    out = run_code("result = int(records.response.sum())", [{"response": 3}, {"response": 4}], [], [])
    assert out["ok"] and out["result"] == 7


def test_sandbox_times_out_infinite_loops():
    out = run_code("while True:\n    pass", [], [], [])
    assert not out["ok"] and "timeout" in out["error"]


def test_critic_catches_every_violation_type_and_accepts_good_plans():
    bad, good = critic_cases()
    assert all(not nodes.critic({"plan": p, "critic_round": 0}, CFG)["critic"][0]["ok"] for p in bad)
    assert all(nodes.critic({"plan": p, "critic_round": 0}, CFG)["critic"][0]["ok"] for p in good)


def test_critic_loop_routing_and_round_limit():
    rej = {"ok": False, "violations": [{"rule": "x"}]}
    assert nodes.route_after_critic({"critic": [rej], "critic_round": 1}) == "planner"
    assert nodes.route_after_critic({"critic": [rej], "critic_round": MAX_CRITIC_ROUNDS}) == "meta_reasoner"
    assert nodes.route_after_critic({"critic": [{"ok": True}], "critic_round": 1}) == "meta_reasoner"


def test_claims_audit_rejects_invented_numbers_then_falls_back_to_template():
    ledger = {"F2": "share of calls reached within 8 minutes: 41% now, 47% with the plan",
              "F1": "30 of 50 generated futures favour this plan over holding positions",
              "F6": "confidence 80%, autonomy tier2"}
    st = {"explanation": {"text": "Coverage jumps to 60% [F2].", "ledger": ledger}, "audit_round": 0}
    out = nodes.claims_audit(st, CFG)
    assert not out["explanation"]["audit"]["ok"] and out["explanation"]["audit"]["unsupported"] == ["60"]
    assert nodes.route_after_audit(out) == "explainer"
    st2 = {**st, "audit_round": 1, "plan": {"moves": []}}
    out2 = nodes.claims_audit(st2, CFG)
    assert out2["explanation"]["audit"]["ok"] and out2["explanation"]["provider"] == "template-after-audit"


def test_autonomy_routing():
    assert nodes.route_after_gate({"meta": {"tier": "tier3"}}) == "human_gate"
    for t in ("tier1", "tier2", "wait", "none"):
        assert nodes.route_after_gate({"meta": {"tier": t}}) == "executor"
    assert nodes.route_after_human({"human": {"action": "ask"}}) == "human_gate"
    assert nodes.route_after_human({"human": {"action": "approve"}}) == "executor"


def test_graph_compiles_with_parallel_branches():
    g = build_graph()
    spec = graph_spec()
    assert "planner" in g.get_graph().nodes
    assert sum(e["label"] == "parallel" for e in spec["edges"]) == 3


needs_data = pytest.mark.skipif(not (ARTIFACTS / "cvae.pt").exists() or not (ARTIFACTS / "calibration.json").exists(),
                                reason="run the data + training pipeline first")


@needs_data
def test_end_to_end_review_with_human_interrupt(tmp_path):
    from lifeline.runtime.session import Runtime, Session
    rt = Runtime(memory_dir=tmp_path)
    s = Session(rt, rt.splits["test_days"][0], start_hour=18, hours=0.5, headless=False, speed=1e9)

    def supervisor():                                    # approve anything that waits for a human
        while s.status != "finished" and not s.stopped:
            if s.pending_decision:
                s.decide({"action": "approve"})
            threading.Event().wait(0.05)
    threading.Thread(target=supervisor, daemon=True).start()
    s.run()
    assert s.status == "finished", s.events[-1]
    assert s.reviews, "the agents never reviewed the situation"
    st = s.last_state
    for key in ("situation", "forecast", "memory", "sop", "plan", "critic", "meta", "explanation", "executed"):
        assert key in st
    assert st["explanation"]["audit"]["ok"]
    kinds = {e["kind"] for e in s.events}
    assert {"frame", "node", "review_start", "review_end"} <= kinds
