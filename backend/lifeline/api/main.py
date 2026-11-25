"""FastAPI backend: session control, live event stream (SSE), analyst Q&A and evaluation data.

Run:  uvicorn lifeline.api.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lifeline.agents.graph import graph_spec
from lifeline.agents.llm import get_llm, usage
from lifeline.config import ARTIFACTS
from lifeline.runtime.analyst import ask_analyst
from lifeline.runtime.session import Runtime, Session

app = FastAPI(title="LIFELINE", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_rt: Runtime | None = None
_session: Session | None = None
_thread: threading.Thread | None = None


def rt() -> Runtime:
    global _rt
    if _rt is None:
        _rt = Runtime()
    return _rt


def session() -> Session:
    if _session is None:
        raise HTTPException(404, "no session - start one first")
    return _session


def _artifact(name: str) -> dict:
    p = ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/health")
def health():
    return {"ok": True, "llm_providers": get_llm().names()}


@app.get("/api/meta")
def meta():
    r = rt()
    return {
        "zones": r.geo.zones, "geojson": r.geo.geojson, "centroids": r.geo.centroids.tolist(),
        "hospitals": r.geo.hospitals, "fleet": r.fleet, "test_days": r.splits["test_days"],
        "adjacency": {i: list(v) for i, v in r.geo.graph.items()},
        "zone_demand_per_hour": r.zone_demand.tolist(), "home_posts": r.home_posts,
        "calibration": {k: v for k, v in r.cal.items() if k not in ("proc_s", "service_s")},
        "llm_providers": get_llm().names(), "graph": graph_spec(), "embedder": r.memory.embed.kind,
    }


class StartReq(BaseModel):
    day: str | None = None
    start_hour: int = 17
    hours: float = 6
    policy: str = "lifeline"
    speed: float = 120


@app.post("/api/session/start")
def start(req: StartReq):
    global _session, _thread
    if _session is not None:
        _session.stopped = True
        _session.decide({"action": "reject", "reason": "session replaced"})
        if _thread:
            _thread.join(timeout=5)
    r = rt()
    day = req.day or r.splits["test_days"][3]
    _session = Session(r, day, start_hour=req.start_hour, hours=req.hours, policy=req.policy, speed=req.speed)
    _thread = threading.Thread(target=_session.run, daemon=True)
    _thread.start()
    return _session.state()


class ControlReq(BaseModel):
    action: str             # pause | resume | speed | stop | review
    value: float | None = None


@app.post("/api/session/control")
def control(req: ControlReq):
    s = session()
    if req.action == "pause":
        s.paused = True
    elif req.action == "resume":
        s.paused = False
    elif req.action == "speed" and req.value:
        s.speed = float(req.value)
    elif req.action == "stop":
        s.stopped = True
    elif req.action == "review":
        s.last_review = -1e9          # forces a scheduled review at the next epoch boundary
    else:
        raise HTTPException(400, f"unknown action {req.action}")
    return {"paused": s.paused, "speed": s.speed, "status": s.status}


class InjectReq(BaseModel):
    kind: str               # rain | event | breakdown | note
    text: str | None = None
    zone: str | None = None


@app.post("/api/session/inject")
def inject(req: InjectReq):
    s = session()
    inj = s.inject(req.kind, req.text, req.zone)
    s.last_review = -1e9
    return inj


class DecisionReq(BaseModel):
    action: str             # approve | override | reject | ask
    moves: list[dict] | None = None
    question: str | None = None
    reason: str | None = None


@app.post("/api/session/decision")
def decision(req: DecisionReq):
    s = session()
    if s.pending_decision is None:
        raise HTTPException(409, "nothing is waiting for approval")
    s.decide(req.model_dump(exclude_none=True))
    return {"ok": True}


@app.get("/api/session/state")
def state():
    return session().state()


@app.get("/api/session/events")
async def events(since: int = 0):
    s = session()

    async def gen():
        i = since
        sid = s.id
        while True:
            if _session is None or _session.id != sid:
                yield {"event": "end", "data": "{}"}
                return
            batch = s.events_since(i)
            for e in batch:
                yield {"event": "message", "data": json.dumps(e, default=str)}
            i += len(batch)
            await asyncio.sleep(0.15)

    return EventSourceResponse(gen(), ping=15)


@app.get("/api/session/checkpoints")
def checkpoints(epoch: int | None = None):
    """LangGraph checkpoint history of one review (time travel through the agents' steps)."""
    s = session()
    epochs = [r["epoch"] for r in s.reviews]
    if not epochs:
        return {"epochs": [], "steps": []}
    ep = epoch or epochs[-1]
    cfg = {"configurable": {"thread_id": f"{s.id}:{ep}"}}
    steps = []
    for snap in s.rt.graph.get_state_history(cfg):
        steps.append({"checkpoint_id": snap.config["configurable"]["checkpoint_id"], "next": list(snap.next),
                      "step": snap.metadata.get("step"), "keys": sorted(snap.values.keys())})
    return {"epochs": epochs, "epoch": ep, "steps": steps[::-1]}


class AskReq(BaseModel):
    question: str


@app.post("/api/analyst")
def analyst(req: AskReq):
    return ask_analyst(session(), req.question)


@app.get("/api/selfmodel")
def selfmodel():
    r = rt()
    return {"table": r.selfmodel.table(), "assessment": _artifact("selfassess.json"),
            "lessons": r.memory.lessons(None, 10)}


@app.get("/api/evaluation")
def evaluation():
    return {"forecast": _artifact("forecast_eval.json"), "benchmark": _artifact("benchmark.json"),
            "calibration": {k: v for k, v in rt().cal.items() if k not in ("proc_s", "service_s")},
            "agent_eval": _artifact("agent_eval.json"), "llm_usage": usage()}


@app.get("/api/graph")
def graph():
    return graph_spec()


# ---- serve the built dashboard (docker / production) ------------------------------------------
_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = _dist / path
        return FileResponse(f if f.is_file() else _dist / "index.html")
