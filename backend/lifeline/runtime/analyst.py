"""Ops Analyst sub-workflow: question -> guard -> LLM writes pandas code -> sandbox -> (retry) -> answer."""
from __future__ import annotations

import time

from lifeline.agents import prompts
from lifeline.agents.llm import get_llm
from lifeline.agents.schemas import AnalystCode
from lifeline.tools.guard import screen
from lifeline.tools.sandbox import run_code

# used only when no LLM is configured, so the demo still works offline
OFFLINE_TEMPLATES = [
    (("worst", "p90", "zone"), "result = records.groupby('zone').response.quantile(0.9).sort_values(ascending=False)"
                              ".head(5).rename(index=lambda i: zones[i]).round(0)"),
    (("busiest", "most calls", "zone"), "result = calls.zone.map(lambda i: zones[i]).value_counts().head(5)"),
    (("mean", "average", "response"), "result = round(records.response.mean(), 1)"),
    (("within", "8"), "result = round((records.response <= 480).mean() * 100, 1)"),
    (("hour",), "result = records.assign(h=(records.t // 3600) % 24).groupby('h').response.mean().round(0)"),
]


def ask_analyst(session, question: str) -> dict:
    steps = []
    g = screen(question)
    steps.append({"step": "guard", "detail": g})
    if g["blocked"]:
        return {"ok": False, "answer": "Blocked by the prompt-injection guardrail.", "steps": steps}
    df = session.sim.records_frame()
    records = df.to_dict(orient="records") if not df.empty else []
    calls = [{"id": c.id, "t": c.t, "zone": c.zone, "severity": c.severity} for c in session.sim.calls]
    zones = session.geo.zones
    llm = get_llm()
    error = None
    for attempt in range(2):
        t0 = time.perf_counter()
        user = f"Question: {question}"
        if error:
            user += f"\nYour previous code failed with: {error}\nFix it."
        res, meta = llm.ask(prompts.ANALYST, user, AnalystCode)
        if res is None:
            q = question.lower()
            code = next((c for keys, c in OFFLINE_TEMPLATES if sum(k in q for k in keys) >= 1), OFFLINE_TEMPLATES[2][1])
            res = AnalystCode(code=code, explanation="offline template (no LLM configured)")
        steps.append({"step": f"write_code (attempt {attempt + 1})", "provider": meta.get("provider"),
                      "code": res.code, "explanation": res.explanation, "latency_s": round(time.perf_counter() - t0, 2)})
        out = run_code(res.code, records, calls, zones)
        steps.append({"step": "run_code", "detail": out})
        if out.get("ok"):
            return {"ok": True, "answer": out.get("result"), "stdout": out.get("stdout"), "code": res.code,
                    "steps": steps}
        error = out.get("error")
    return {"ok": False, "answer": f"Could not compute an answer: {error}", "steps": steps}
