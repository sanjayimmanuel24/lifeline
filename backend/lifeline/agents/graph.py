r"""The LIFELINE LangGraph.

START -> guard -> situation_analyst -> [demand_forecaster | memory_retriever | sop_retriever]  (parallel)
      -> planner -> critic --(reject, round<3)--> planner                                     (reflection loop)
                       \--(accept or give up)--> meta_reasoner -> explainer -> claims_audit
      claims_audit --(unsupported numbers)--> explainer                                       (self-verification)
      -> autonomy_gate --(tier3)--> human_gate (interrupt) --(ask)--> human_gate
                     \--(tier1/2/wait/none)--> executor -> END
"""
from __future__ import annotations

import functools
import sqlite3

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from lifeline.agents import nodes as n
from lifeline.agents.schemas import AgentState


def _plain(o):
    """numpy scalars/arrays -> Python types, so checkpoints serialise."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return o


def _clean(fn):
    @functools.wraps(fn)
    def node(state, config: RunnableConfig):
        return _plain(fn(state, config))
    return node


def build_graph(checkpoint_path: str | None = None):
    g = StateGraph(AgentState)
    for name in ("guard", "situation_analyst", "demand_forecaster", "memory_retriever", "sop_retriever", "planner",
                 "critic", "meta_reasoner", "explainer", "claims_audit", "autonomy_gate", "human_gate", "executor"):
        g.add_node(name, _clean(getattr(n, name)))

    g.add_edge(START, "guard")
    g.add_edge("guard", "situation_analyst")
    for branch in ("demand_forecaster", "memory_retriever", "sop_retriever"):
        g.add_edge("situation_analyst", branch)
    g.add_edge(["demand_forecaster", "memory_retriever", "sop_retriever"], "planner")   # join
    g.add_edge("planner", "critic")
    g.add_conditional_edges("critic", n.route_after_critic, ["planner", "meta_reasoner"])
    g.add_edge("meta_reasoner", "explainer")
    g.add_edge("explainer", "claims_audit")
    g.add_conditional_edges("claims_audit", n.route_after_audit, ["explainer", "autonomy_gate"])
    g.add_conditional_edges("autonomy_gate", n.route_after_gate, ["human_gate", "executor"])
    g.add_conditional_edges("human_gate", n.route_after_human, ["human_gate", "executor"])
    g.add_edge("executor", END)

    if checkpoint_path:
        from langgraph.checkpoint.sqlite import SqliteSaver
        saver = SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))
    else:
        saver = InMemorySaver()
    return g.compile(checkpointer=saver)


def graph_spec() -> dict:
    """Nodes and edges for the dashboard's React Flow view."""
    nodes = ["guard", "situation_analyst", "demand_forecaster", "memory_retriever", "sop_retriever", "planner",
             "critic", "meta_reasoner", "explainer", "claims_audit", "autonomy_gate", "human_gate", "executor"]
    edges = [("START", "guard", ""), ("guard", "situation_analyst", ""),
             ("situation_analyst", "demand_forecaster", "parallel"), ("situation_analyst", "memory_retriever", "parallel"),
             ("situation_analyst", "sop_retriever", "parallel"), ("demand_forecaster", "planner", "join"),
             ("memory_retriever", "planner", "join"), ("sop_retriever", "planner", "join"),
             ("planner", "critic", ""), ("critic", "planner", "reject"), ("critic", "meta_reasoner", "accept"),
             ("meta_reasoner", "explainer", ""), ("explainer", "claims_audit", ""),
             ("claims_audit", "explainer", "unsupported"), ("claims_audit", "autonomy_gate", "grounded"),
             ("autonomy_gate", "human_gate", "tier3"), ("autonomy_gate", "executor", "tier1/2"),
             ("human_gate", "human_gate", "ask"), ("human_gate", "executor", "approve/override"),
             ("executor", "END", "")]
    return {"nodes": ["START"] + nodes + ["END"], "edges": [{"source": a, "target": b, "label": l} for a, b, l in edges]}
