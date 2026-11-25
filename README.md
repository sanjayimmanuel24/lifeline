# LIFELINE: an agentic ambulance-command co-pilot

LIFELINE is a multi-agent decision-support system for an EMS (ambulance) control room.
It replays **real NYC EMS calls** (Brooklyn, held-out 2026 days) through a calibrated simulator.
A team of LangGraph agents decides, every 15 simulated minutes, whether and where to reposition idle ambulances.

At each review the agents:

1. **Forecast** 50 joint "possible futures" for the next 2 hours with a **Conditional VAE**.
2. **Plan** repositioning moves with **Monte Carlo Tree Search**, then verify the best plans in the full simulator.
3. **Critique** the plan (a Safety & Equity Critic forces rewrites).
4. **Reason about their own reliability** with a self-model built by self-stress-testing, then act alone, notify, wait, or ask the supervisor.
5. **Explain** every decision with a rationale whose numbers are checked against a claims ledger.

Course: Agentic AI (25DC2011). Domain: decision support & optimisation.
This is a **simulation only**: it is never connected to real dispatch.

---

## Quick start (Windows / macOS / Linux)

```bash
# 1. Python environment (3.11+)
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend

# 2. Data + models (about 15 minutes; all data is free and needs no key)
python -m lifeline.data.ingest        # NYC Open Data, Open-Meteo, OpenStreetMap
python -m lifeline.forecast.train     # CVAE + Poisson GLM + seasonal naive, scored on 2026 hours
python -m lifeline.sim.calibrate      # fleet size + speed curve vs recorded 2025 response times
python -m lifeline.eval.selfassess    # builds the agent's self-model (competence table + calibration)

# 3. Run
uvicorn lifeline.api.main:app --port 8000          # API
cd ../frontend && npm install && npm run dev       # dashboard on http://localhost:5173
```

**LLM keys (optional but recommended):** copy `.env.example` to `.env` and add a free Gemini and/or Groq key.
Without a key every agent uses its deterministic fallback. The top bar shows "offline mode", and the trace marks those steps.

**Docker:** `docker compose up --build`, then open http://localhost:8000. Run step 2 first; the `data/` folder is mounted.
The Ops Analyst's code runs in a separate `sandbox` container. That container has no internet route, a read-only filesystem, 1 CPU and 512 MB.

**Evaluation:**

```bash
python -m lifeline.eval.benchmark      # 30 held-out days x 7 policies (about 40 min)
python -m lifeline.eval.agent_eval     # guardrail / Critic / claims audit / sandbox checks
python -m pytest -q                    # 23 tests incl. an end-to-end review with a human interrupt
```

---

## Architecture

```
          ┌──────────────── React dashboard (MapLibre, Recharts, React Flow) ────────────────┐
          │ Live Ops · 50 Futures · Plan Inspector · Agent Trace · Self-Model · Equity · ...  │
          └───────────────▲ SSE event stream ─────────── REST (start, inject, approve) ───────┘
                          │
FastAPI ── Session (replays a real day in the discrete-event simulator, every 15 min runs:)
                          │
 LangGraph StateGraph (SqliteSaver checkpoints, interrupt() for the human)
   guard ─► situation_analyst ─► ┌ demand_forecaster (CVAE x50) ┐
                                 ├ memory_retriever  (Chroma)   ├─► planner ─► critic ─┐
                                 └ sop_retriever     (RAG)      ┘      ▲   reject (≤3) │
                                                                       └──────────────┘
   critic(accept) ─► meta_reasoner ─► explainer ─► claims_audit ─(unsupported number)─► explainer
   claims_audit ─► autonomy_gate ─► tier3 ─► human_gate (interrupt: approve / override / reject / ask)
                                 └► tier1/tier2/wait ─► executor ─► simulator
 Every hour: Reflector compares predicted vs realised outcomes → lessons + self-model update
```

| Layer | Implementation |
|---|---|
| Agents | LangGraph `StateGraph`: parallel fan-out and join, Critic loop, audit loop, conditional autonomy gate, `interrupt()`/`Command(resume=)` |
| LLMs | Gemini → Groq → OpenRouter → Ollama fallback chain, structured outputs (Pydantic), SQLite response cache; deterministic fallback per agent |
| Generative model | Conditional VAE (PyTorch), negative-binomial likelihood, KL annealing + free bits; 37 zones × 2 hourly slots |
| Planning | Plan-and-execute task list (LLM) → MCTS (UCT + progressive widening) on CVAE futures → simulator verification with common random numbers; A* routing; MEXCLP (OR-Tools CP-SAT) baseline |
| Tools | `generate_demand_scenarios`, `mcts_plan`, `simulate_plan`, `route` (A*), `run_code` (sandbox), `get_weather` (Open-Meteo), `search_sop` (RAG), `recall_similar_situations`, `check_constraints`, `guard_text`, `request_human_approval` |
| Memory | Working state + checkpoints (SqliteSaver); episodic (Chroma, text embedding ⊕ CVAE latent); semantic SOP RAG; procedural templates; lessons; self-model (SQLite) |
| Reflection | Critic rewrite loop; claims-ledger self-verification; self-model with isotonic-calibrated confidence + out-of-distribution score; value-of-information "wait"; hourly Reflector |
| Safety & ethics | Prompt-injection guard (Llama Prompt Guard 2 via Groq, pattern fallback); equity term (Gini) and per-zone p90 monitoring; no personal attributes; human approval for risky actions; overrides become lessons |

## Data (all free)

- **NYC Open Data, EMS Incident Dispatch Data** (`76xm-jjuj`): Brooklyn, severity 1-4, 2022 to Sep 2026 (1.15 M calls).
  - Splits: train 2022-24, calibrate/validate 2025, **test 2026 (30 held-out days)**.
- **NYC Modified ZCTA polygons** (`pri4-ifjk`): the 37 zones. The dataset only records ZIP codes, so locations are zone-level.
- **Open-Meteo**: hourly precipitation and temperature (archive and forecast APIs).
- **OpenStreetMap** (Overpass): hospitals.
- **Simplification:** travel uses A* over the zone-adjacency graph, with an hourly speed curve calibrated on recorded travel seconds. It does not use the street network.

## Rubric map

| Criterion | Where it is shown |
|---|---|
| Generative AI & agent architecture | CVAE world model (50 Futures page, latent scatter, CRPS/energy table, latent ablation); 9 agents + human; single- vs multi-agent ablations |
| Reflection, reasoning & decision-making | Critic loop (Plan Inspector), claims audit, Self-Model page (competence map, reliability diagram), tiers and "wait", Reflector lessons, equity and guardrail |
| Tool use & planning | 11 typed tools; plan-and-execute + MCTS + simulator verification + A*; MCTS tree explorer |
| Agentic workflow design & deployment | LangGraph graph with live node highlighting, checkpoints (time travel), layered memory, Docker compose with isolated sandbox, CI |
| Application, documentation & presentation | Real NYC data, validated simulator, 30-day held-out benchmark with paired tests, ablations, agent-quality tests |

## Five-minute demo

1. Start a shift: a held-out 2026 day at 17:00. Units, calls and coverage are on the map.
2. Click **Barclays Center event**. The Situation Analyst and Planner raise demand around 11217, and **50 Futures** shows the spread.
3. **Plan Inspector**: the MCTS tree, the simulator verification table and the Critic's verdicts.
4. When a Tier-3 card appears, click **Ask why** and then **Approve**. The units drive their A* routes.
5. Send the injected call note. The **Guard** blocks it, and **Agent Trace** shows the step.
6. **Ops Analyst**: "Which zones had the worst 90th-percentile response time?" Show the generated code and the sandbox result.
7. **Self-Model** (calibration) and **Evaluation** (benchmark vs status quo and MEXCLP).
