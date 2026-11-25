import { useState } from "react";
import { api } from "../api";
import { useLive } from "../store";

const EXAMPLES = ["Which zones had the worst 90th-percentile response time?", "What share of calls were reached within 8 minutes?",
  "Which zones had the most calls?", "Average response time by hour"];

export default function Analyst() {
  const { live } = useLive();
  const [q, setQ] = useState(EXAMPLES[0]);
  const [log, setLog] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const ask = async (question: string) => {
    setBusy(true);
    try { const r = await api("/analyst", { question }); setLog((l) => [{ question, ...r }, ...l]); }
    catch (e: any) { setLog((l) => [{ question, ok: false, answer: String(e) }, ...l]); }
    finally { setBusy(false); }
  };
  return (
    <div className="grid">
      <div className="card">
        <h3>Ops Analyst: the LLM writes pandas code, a sandbox runs it</h3>
        <div className="row">
          <input style={{ flex: 1, minWidth: 0 }} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)} />
          <button className="btn primary" disabled={busy || !live.sessionId} onClick={() => ask(q)}>{busy ? "Running…" : "Ask"}</button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>{EXAMPLES.map((e) => <button key={e} className="btn" onClick={() => { setQ(e); ask(e); }} disabled={busy || !live.sessionId}>{e}</button>)}</div>
        {!live.sessionId && <div className="faint" style={{ marginTop: 8 }}>Start a shift first: the analyst queries the live shift's call log.</div>}
      </div>
      {log.map((r, i) => (
        <div key={i} className="card">
          <div style={{ fontWeight: 600 }}>{r.question}</div>
          <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", marginTop: 8 }}>
            <div>
              <h3>Steps</h3>
              {(r.steps || []).map((s: any, j: number) => (
                <div key={j} className="ev tool" style={{ marginBottom: 6 }}>
                  <div className="row"><span className="who">{s.step}</span>{s.provider && <span className="tag">{s.provider}</span>}
                    {s.detail?.where && <span className="tag">{s.detail.where}</span>}</div>
                  {s.code && <pre className="code">{s.code}</pre>}
                  {s.detail && s.step === "run_code" && !s.detail.ok && <div style={{ color: "var(--bad)", fontSize: 12 }}>{s.detail.error}</div>}
                  {s.step === "guard" && <div className="faint" style={{ fontSize: 12 }}>injection score {s.detail.score} ({s.detail.method})</div>}
                </div>))}
            </div>
            <div><h3>Answer</h3><Answer v={r.answer} ok={r.ok} /></div>
          </div>
        </div>
      ))}
    </div>
  );
}

function Answer({ v, ok }: { v: any; ok: boolean }) {
  if (!ok) return <div style={{ color: "var(--bad)" }}>{String(v)}</div>;
  if (Array.isArray(v) && v.length && typeof v[0] === "object") {
    const cols = Object.keys(v[0]);
    return <table><thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead><tbody>
      {v.map((row: any, i: number) => <tr key={i}>{cols.map((c) => <td key={c}>{String(row[c])}</td>)}</tr>)}</tbody></table>;
  }
  if (v && typeof v === "object") return <table><tbody>{Object.entries(v).map(([k, x]) => <tr key={k}><td className="mono">{k}</td><td className="num">{String(x)}</td></tr>)}</tbody></table>;
  return <div style={{ fontSize: 28, fontWeight: 700 }}>{String(v)}</div>;
}
