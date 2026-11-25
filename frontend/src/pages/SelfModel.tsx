import { useEffect, useState } from "react";
import { CartesianGrid, Line, ComposedChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct } from "../api";
import { good } from "../geo";
import { useLive } from "../store";

export default function SelfModel() {
  const { live } = useLive();
  const [d, setD] = useState<any>();
  useEffect(() => { api("/selfmodel").then(setD); }, [live.reviews.length]);
  const m = live.agent?.meta;
  if (!d) return <div className="card empty">Loading…</div>;
  const a = d.assessment || {};
  const bands = ["night", "morning", "afternoon", "evening"], ctx = ["dry|normal", "dry|surge", "rain|normal", "rain|surge"];
  const cell = (b: string, c: string) => d.table.find((r: any) => r.bucket === `${b}|${c}`);
  const rel = (a.reliability?.bins || []).map((x: any) => ({ x: x.confidence, y: x.accuracy, n: x.n }));
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h3>What the agent believes about itself right now</h3>
        {m ? <div className="row" style={{ alignItems: "flex-start" }}>
          <div style={{ flex: 1, minWidth: 260 }}>
            <div style={{ fontSize: 16 }}>🧭 {m.self_explanation}</div>
            <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>bucket <span className="mono">{m.bucket}</span> · raw {m.raw_confidence} → calibrated {m.confidence}</div>
          </div>
          <span className={`pill ${m.tier === "tier3" ? "warn" : m.tier === "wait" ? "info" : "ok"}`}>{m.tier}</span>
          <span className="pill">confidence {pct(m.confidence)}</span>
          <span className={`pill ${m.ood_score > 0.6 ? "bad" : "ok"}`}>novelty {m.ood_score}</span>
          <span className="pill">spread {pct(m.forecast_spread_cv)}</span>
        </div> : <div className="faint">Appears after the first agent review.</div>}
      </div>
      <div className="card">
        <h3>Competence map: chance my decision is right</h3>
        <table><thead><tr><th></th>{ctx.map((c) => <th key={c} className="num" style={{ fontSize: 11 }}>{c.replace("|", " · ")}</th>)}</tr></thead><tbody>
          {bands.map((b) => <tr key={b}><td>{b}</td>{ctx.map((c) => { const r = cell(b, c); return (
            <td key={c} className="num" title={r ? `n=${r.n}, regret ${r.regret?.toFixed(1)}` : "never seen"}
              style={{ background: r ? good((r.win_rate - 0.4) / 0.5) + "33" : undefined, color: r ? undefined : "var(--faint)" }}>
              {r ? `${pct(r.win_rate)}` : "—"}<div className="faint" style={{ fontSize: 10 }}>{r ? `n=${r.n} · regret ${r.regret?.toFixed(0)}` : ""}</div></td>); })}</tr>)}
        </tbody></table>
        <div className="faint" style={{ fontSize: 12, marginTop: 8 }}>Built offline by self-stress-testing on {a.n_situations} situations from 2025 (real outcomes + CVAE stress futures), then updated online after every hour.</div>
      </div>
      <div className="card">
        <h3>Is my confidence honest? (held-out reliability)</h3>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart><CartesianGrid stroke="var(--line)" />
            <XAxis type="number" dataKey="x" domain={[0, 1]} stroke="var(--muted)" fontSize={11} name="confidence" />
            <YAxis type="number" dataKey="y" domain={[0, 1]} stroke="var(--muted)" fontSize={11} name="observed" />
            <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
            <Line data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} dataKey="y" stroke="var(--faint)" strokeDasharray="4 4" dot={false} />
            <Scatter data={rel} fill="var(--accent-2)" />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="row"><span className="pill">ECE raw {a.reliability?.ece_raw?.toFixed(3)}</span>
          <span className="pill ok">ECE calibrated {a.reliability?.ece_calibrated?.toFixed(3)}</span>
          <span className="pill">Brier {a.reliability?.brier?.toFixed(3)}</span><span className="faint">n={a.reliability?.n_holdout}</span></div>
      </div>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h3>Lessons in memory (Reflector + supervisor overrides)</h3>
        {d.lessons.length ? <table><tbody>{d.lessons.map((l: any, i: number) => <tr key={i}><td className="mono faint" style={{ whiteSpace: "nowrap" }}>{l.t.slice(11, 16)}</td>
          <td><span className="tag">{l.source}</span></td><td>{l.text}</td></tr>)}</tbody></table> : <div className="faint">Lessons are written after each simulated hour.</div>}
      </div>
    </div>
  );
}
