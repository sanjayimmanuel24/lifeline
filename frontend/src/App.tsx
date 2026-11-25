import { useEffect, useState } from "react";
import { useLive } from "./store";
import LiveOps from "./pages/LiveOps";
import Futures from "./pages/Futures";
import PlanInspector from "./pages/PlanInspector";
import AgentTrace from "./pages/AgentTrace";
import SelfModel from "./pages/SelfModel";
import Equity from "./pages/Equity";
import Analyst from "./pages/Analyst";
import Evaluation from "./pages/Evaluation";

const PAGES = [
  ["live", "Live Ops", "◉"], ["futures", "50 Futures", "◇"], ["plan", "Plan Inspector", "⌗"],
  ["trace", "Agent Trace", "≋"], ["self", "Self-Model", "◎"], ["equity", "Equity", "⚖"],
  ["analyst", "Ops Analyst", "›_"], ["eval", "Evaluation", "▤"],
] as const;
type Page = (typeof PAGES)[number][0];

export default function App() {
  const [page, setPage] = useState<Page>(() => (localStorage.getItem("page") as Page) || "live");
  useEffect(() => { try { localStorage.setItem("page", page); } catch {} }, [page]);
  const { live } = useLive();
  return (
    <div className="app">
      <TopBar />
      <aside className="sidebar">
        <nav className="nav">
          {PAGES.map(([k, label, icon]) => (
            <button key={k} className={page === k ? "active" : ""} onClick={() => setPage(k)}>
              <span style={{ width: 18, textAlign: "center", opacity: 0.8 }}>{icon}</span>{label}
              {k === "live" && live.approval && <span className="badge">1</span>}
            </button>
          ))}
        </nav>
        <div className="faint" style={{ fontSize: 11.5, padding: "16px 12px" }}>
          Decision support only. Replays real NYC EMS data in simulation; never connected to live dispatch.
        </div>
      </aside>
      <main className="main">
        {page === "live" && <LiveOps />}
        {page === "futures" && <Futures />}
        {page === "plan" && <PlanInspector />}
        {page === "trace" && <AgentTrace />}
        {page === "self" && <SelfModel />}
        {page === "equity" && <Equity />}
        {page === "analyst" && <Analyst />}
        {page === "eval" && <Evaluation />}
      </main>
    </div>
  );
}

function TopBar() {
  const { meta, live, start, control, paused, speed } = useLive();
  const [day, setDay] = useState<string>("");
  const [hour, setHour] = useState(17);
  const [policy, setPolicy] = useState("lifeline");
  useEffect(() => { if (meta && !day) setDay(meta.test_days[3]); }, [meta]);
  const status = live.status;
  const pill = status === "awaiting_approval" ? "warn" : status === "thinking" ? "info" : status === "error" ? "bad" : status === "running" ? "ok" : "";
  return (
    <header className="topbar">
      <div className="brand"><div className="pulse">✚</div>LIFELINE</div>
      <span className="faint" style={{ fontSize: 12 }}>EMS co-pilot · Brooklyn</span>
      <div className="spacer" />
      <select value={day} onChange={(e) => setDay(e.target.value)} title="held-out 2026 day to replay">
        {meta?.test_days.map((d) => <option key={d}>{d}</option>)}
      </select>
      <select value={hour} onChange={(e) => setHour(+e.target.value)} title="start hour">
        {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2, "0")}:00</option>)}
      </select>
      <select value={policy} onChange={(e) => setPolicy(e.target.value)} title="policy">
        <option value="lifeline">LIFELINE agents</option>
        <option value="mexclp">MEXCLP static (baseline)</option>
        <option value="static">Status quo (baseline)</option>
      </select>
      <button className="btn primary" onClick={() => start({ day, start_hour: hour, hours: 6, policy, speed: 120 })}>
        {live.sessionId ? "Restart shift" : "Start shift"}
      </button>
      <button className="btn" disabled={!live.sessionId} onClick={() => control(paused ? "resume" : "pause")}>{paused ? "▶ Resume" : "❚❚ Pause"}</button>
      <select value={speed} disabled={!live.sessionId} onChange={(e) => control("speed", +e.target.value)} title="simulation speed">
        {[30, 60, 120, 300, 600].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <span className="mono" style={{ fontSize: 16, minWidth: 48 }}>{live.time || "--:--"}</span>
      <span className={`pill ${pill}`}><span className="dot" />{status.replace("_", " ")}</span>
      <span className="pill" title="LLM providers in the fallback chain">
        {meta?.llm_providers.length ? meta.llm_providers[0].split(":")[0] + (meta.llm_providers.length > 1 ? ` +${meta.llm_providers.length - 1}` : "") : "offline mode"}
      </span>
    </header>
  );
}
