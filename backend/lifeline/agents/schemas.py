"""Typed graph state and the structured outputs each LLM agent must return."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class SituationSummary(BaseModel):
    headline: str = Field(description="one sentence describing the operational situation right now")
    risk_level: Literal["low", "elevated", "high"]
    hotspots: list[str] = Field(description="zone codes (5-digit) that need attention, at most 4")
    concerns: list[str] = Field(description="at most 3 short concerns, each under 15 words")


class Directive(BaseModel):
    tasks: list[str] = Field(description="ordered plan-and-execute task list, 3-6 short imperative steps")
    protect_zones: list[str] = Field(description="zone codes that must keep coverage after any move")
    demand_multipliers: dict[str, float] = Field(default_factory=dict,
        description="zone code -> multiplier (1.0-3.0) for known events the forecaster cannot see")
    equity_boost: float = Field(ge=1.0, le=2.5, description="1.0 normal, higher = weigh fairness across zones more")
    max_moves: int = Field(ge=0, le=3)
    reasoning: str = Field(description="2 sentences: why these settings")


class SelfExplanation(BaseModel):
    statement: str = Field(description="first-person, 1-2 sentences, explains the chosen autonomy level "
                                       "using the competence record, uncertainty and novelty given")


class Rationale(BaseModel):
    text: str = Field(description="3-4 sentences for the dispatch supervisor. Cite facts as [F1], [F2]. "
                                  "Use ONLY numbers that appear in the facts.")


class AnalystCode(BaseModel):
    code: str = Field(description="python code using pandas DataFrames `records` and `calls`; "
                                  "must assign the final answer to a variable named `result`")
    explanation: str


class AgentState(TypedDict, total=False):
    epoch: int
    sim_time: str
    triggers: list[str]
    notes: list[dict]
    guard: dict
    situation: dict
    context: dict
    forecast: dict
    memory: dict
    sop: list[dict]
    directive: dict
    plan: dict
    critic: Annotated[list[dict], operator.add]
    critic_round: int
    meta: dict
    explanation: dict
    audit_round: int
    human: dict
    executed: dict
    trace: Annotated[list[dict], operator.add]
