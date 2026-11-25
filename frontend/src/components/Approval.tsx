import { useState } from "react";
import { pct } from "../api";
import { useLive } from "../store";

export function Approval() {
  const { live, decide } = useLive();
  const a = live.approval;
  const [q, setQ] = useState("Why move these units and not others?");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const lastAnswer = [...live.trace].reverse().find((t) => t.agent === "Liaison" && t.node === "human_gate");
  const act = async (d: any) => { setBusy(true); try { await decide(d); } finally { setBusy(false); } };
  return (
    <div className="card approval">
      <h3 style={{ color: "var(--warn)" }}>⚠ Supervisor approval needed</h3>
      <div className="row" style={{ marginBottom: 8 }}>
        <span className="pill warn">confidence {pct(a.confidence)}</span>
        <span className="pill">{a.blast_radius.units} unit(s) · {a.blast_radius.km} km</span>
        {!a.critic.ok && <span className="pill bad">Critic unresolved</span>}
      </div>
      <table><tbody>
        {a.moves.map((m: any) => (
          <tr key={m.unit}><td>Unit {m.unit}</td><td className="mono">{m.from_zone} → {m.to_zone}</td><td className="num">{m.km} km</td></tr>
        ))}
      </tbody></table>
      <p style={{ margin: "10px 0" }}>{a.rationale}</p>
      <div className="muted" style={{ fontSize: 12.5 }}>🧭 {a.self_explanation}</div>
      {lastAnswer && <div className="code" style={{ marginTop: 8 }}>{lastAnswer.summary}</div>}
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn ok" disabled={busy} onClick={() => act({ action: "approve" })}>Approve</button>
        <button className="btn" disabled={busy} onClick={() => act({ action: "reject", reason: reason || "supervisor rejected" })}>Reject</button>
        {a.alternatives?.[0]?.moves?.length > 0 && (
          <button className="btn" disabled={busy} title="use the runner-up plan from the search tree"
            onClick={() => act({ action: "override", reason: reason || "chose runner-up plan",
              moves: a.alternatives[0].moves.map(([u, p]: number[]) => ({ unit: u, to: p })) })}>Use alternative</button>)}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <input style={{ flex: 1, minWidth: 0 }} value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn" disabled={busy} onClick={() => act({ action: "ask", question: q })}>Ask why</button>
      </div>
      <input style={{ width: "100%", marginTop: 8 }} placeholder="reason (stored as a lesson on reject/override)" value={reason}
        onChange={(e) => setReason(e.target.value)} />
    </div>
  );
}
