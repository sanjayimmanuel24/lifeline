"""Prompt-injection guardrail for untrusted free text (call notes, analyst questions).

1. Llama Prompt Guard 2 hosted on Groq (if GROQ_API_KEY is set)
2. Pattern-based fallback (always available)
"""
from __future__ import annotations

import os
import re

PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts)",
    r"disregard (the |all )?(rules|instructions|system)",
    r"you are now", r"system prompt", r"act as (an? )?(admin|developer|dispatcher)",
    r"send (all|every) (units|ambulances)", r"override (the )?(policy|critic|supervisor|approval)",
    r"do not (ask|notify) (the )?(human|supervisor)", r"auto[- ]?approve", r"jailbreak",
    r"import os|subprocess|__import__|open\(|eval\(|exec\(",
]
_rx = [re.compile(p, re.I) for p in PATTERNS]
THRESHOLD = 0.5


def screen(text: str) -> dict:
    """Returns {"blocked": bool, "score": float, "method": str, "matched": [...]}"""
    if not text:
        return {"blocked": False, "score": 0.0, "method": "empty", "matched": []}
    if os.getenv("GROQ_API_KEY") and os.getenv("LIFELINE_OFFLINE") != "1":
        try:
            from groq import Groq
            r = Groq().chat.completions.create(model="meta-llama/llama-prompt-guard-2-86m",
                                               messages=[{"role": "user", "content": text}])
            score = float(r.choices[0].message.content)
            return {"blocked": score >= THRESHOLD, "score": score, "method": "llama-prompt-guard-2", "matched": []}
        except Exception:
            pass
    matched = [p.pattern for p in _rx if p.search(text)]
    score = min(1.0, 0.6 * len(matched)) if matched else 0.0
    return {"blocked": score >= THRESHOLD, "score": score, "method": "patterns", "matched": matched}
