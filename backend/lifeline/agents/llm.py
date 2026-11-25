"""Multi-provider LLM layer with automatic fallback, response cache and usage accounting.

Order: Gemini -> Groq -> OpenRouter -> Ollama (whichever have keys / are configured).
If no provider is configured or all fail, agents use their deterministic fallback
("offline mode") so the system still runs end to end - the trace marks those steps.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from lifeline.config import ARTIFACTS


@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    offline: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_provider: dict = field(default_factory=dict)
    latency_s: float = 0.0


USAGE = Usage()
_lock = threading.Lock()


def _providers() -> list[tuple[str, object]]:
    out = []
    timeout = float(os.getenv("LIFELINE_LLM_TIMEOUT", "40"))
    gkey = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gkey:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("LIFELINE_GEMINI_MODEL", "gemini-2.5-flash-lite")
        out.append((f"gemini:{model}", ChatGoogleGenerativeAI(model=model, google_api_key=gkey, temperature=0.2,
                                                              timeout=timeout, max_retries=1)))
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        model = os.getenv("LIFELINE_GROQ_MODEL", "llama-3.3-70b-versatile")
        out.append((f"groq:{model}", ChatGroq(model=model, temperature=0.2, timeout=timeout, max_retries=1)))
    if os.getenv("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("LIFELINE_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        out.append((f"openrouter:{model}", ChatOpenAI(model=model, api_key=os.getenv("OPENROUTER_API_KEY"),
                                                      base_url="https://openrouter.ai/api/v1", temperature=0.2,
                                                      timeout=timeout, max_retries=1)))
    if os.getenv("OLLAMA_HOST") or os.getenv("LIFELINE_OLLAMA_MODEL"):
        from langchain_ollama import ChatOllama
        model = os.getenv("LIFELINE_OLLAMA_MODEL", "qwen2.5:7b")
        out.append((f"ollama:{model}", ChatOllama(model=model, temperature=0.2,
                                                   base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))))
    return out


class LLM:
    def __init__(self):
        self.providers = [] if os.getenv("LIFELINE_OFFLINE") == "1" else _providers()

    @property
    def online(self) -> bool:
        return bool(self.providers)

    def names(self) -> list[str]:
        return [n for n, _ in self.providers]

    def ask(self, system: str, user: str, schema: type[BaseModel] | None = None) -> tuple[object | None, dict]:
        """Returns (result, meta). result is None if every provider failed / none configured."""
        if not self.providers:
            with _lock:
                USAGE.offline += 1
            return None, {"provider": "offline", "latency_s": 0.0}
        for name, model in self.providers:
            t0 = time.perf_counter()
            try:
                if schema is not None:
                    runnable = model.with_structured_output(schema, include_raw=True)
                    out = runnable.invoke([("system", system), ("human", user)])
                    raw, result = out["raw"], out["parsed"]
                    if result is None:
                        raise ValueError(f"unparseable output: {out.get('parsing_error')}")
                else:
                    raw = model.invoke([("system", system), ("human", user)])
                    result = raw.content if isinstance(raw.content, str) else str(raw.content)
                dt = time.perf_counter() - t0
                um = getattr(raw, "usage_metadata", None) or {}
                with _lock:
                    USAGE.calls += 1
                    USAGE.latency_s += dt
                    USAGE.input_tokens += um.get("input_tokens", 0)
                    USAGE.output_tokens += um.get("output_tokens", 0)
                    USAGE.by_provider[name] = USAGE.by_provider.get(name, 0) + 1
                return result, {"provider": name, "latency_s": round(dt, 2),
                                "tokens": um.get("input_tokens", 0) + um.get("output_tokens", 0)}
            except Exception as e:  # rate limit, quota, timeout, parse error -> next provider
                with _lock:
                    USAGE.failures += 1
                last = f"{name}: {e.__class__.__name__}: {str(e)[:160]}"
                continue
        return None, {"provider": "fallback", "error": last, "latency_s": 0.0}


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _setup_cache()
        _llm = LLM()
    return _llm


def _setup_cache() -> None:
    """Identical prompts are answered from a SQLite cache: saves free-tier quota, makes demos repeatable."""
    try:
        from langchain_core.globals import set_llm_cache
        from langchain_community.cache import SQLiteCache
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(ARTIFACTS / "llm_cache.sqlite")))
    except ImportError:
        pass


def usage() -> dict:
    with _lock:
        return {"calls": USAGE.calls, "failures": USAGE.failures, "offline_steps": USAGE.offline,
                "input_tokens": USAGE.input_tokens, "output_tokens": USAGE.output_tokens,
                "by_provider": dict(USAGE.by_provider), "latency_s": round(USAGE.latency_s, 1)}
