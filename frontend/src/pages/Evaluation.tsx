import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmtS, pct } from "../api";

const NAMES: Record<string, string> = { static: "Status quo", mexclp: "MEXCLP (OR)", lifeline: "LIFELINE", lifeline_poisson: "LIFELINE + Poisson",
  no_critic: "– Critic", no_selfmodel: "– Self-model", no_equity: "– Equity term" };
const ORDER = ["static", "mexclp", "lifeline", "lifeline_poisson", "no_critic", "no_selfmodel", "no_equity"];
const tip = { contentStyle: { background: "var(--panel)", border: "1px solid var(--line)" } };

export default function Evaluation() {
  const [d, setD] = useState<any>();
  useEffect(() => { api("/evaluation").then(setD); }, []);
  if (!d) return <div className="card empty">Loading…</div>;
  const b = d.benchmark, f = d.forecast, v = d.calibration?.validation, ae = d.agent_eval;
  const pol = b?.policies ? ORDER.filter((p) => b.policies[p]).map((p) => ({ name: NAMES[p], key: p, ...b.policies[p] })) : [];
  const hist = v ? v.recorded_hist.map((r: number, i: number) => ({ s: `${i}`, recorded: r / v.recorded_hist.reduce((a: number, x: number) => a + x, 0),
    simulated: v.sim_hist[i] / v.sim_hist.reduce((a: number, x: number) => a + x, 0) })) : [];
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h2>Evaluation on {b?.n_days ?? "–"} held-out days of real 2026 Brooklyn calls</h2>
        <div className="muted">Days never used for training or calibration. Every policy replays the same real calls; Tier-3 approvals are auto-approved by a simulated supervisor.</div>
      </div>
      {pol.length > 0 ? <>
        <div className="card">
          <h3>Share of calls reached within 8 minutes</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={pol}><CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--muted)" fontSize={11} interval={0} angle={-15} textAnchor="end" height={50} />
              <YAxis stroke="var(--muted)" fontSize={11} tickFormatter={(x) => pct(x)} domain={["auto", "auto"]} />
              <Tooltip {...tip} formatter={(x: any) => pct(x, 1)} />
              <Bar dataKey="within_target" fill="var(--ok)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Response time (seconds)</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={pol}><CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--muted)" fontSize={11} interval={0} angle={-15} textAnchor="end" height={50} />
              <YAxis stroke="var(--muted)" fontSize={11} /><Tooltip {...tip} formatter={(x: any) => fmtS(x)} /><Legend />
              <Bar dataKey="mean_rt" name="mean" fill="var(--accent-2)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="p90_rt" name="90th pct" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card" style={{ gridColumn: "1 / -1" }}>
          <h3>All metrics (mean per day)</h3>
          <table><thead><tr><th>Policy</th><th className="num">within 8 min</th><th className="num">mean</th><th className="num">p90</th>
            <th className="num">zone Gini</th><th className="num">moves/day</th><th className="num">km/day</th><th className="num">Critic rejections</th></tr></thead>
            <tbody>{pol.map((p: any) => <tr key={p.key} style={p.key === "lifeline" ? { fontWeight: 600 } : undefined}><td>{p.name}</td>
              <td className="num">{pct(p.within_target, 1)}</td><td className="num">{fmtS(p.mean_rt)}</td><td className="num">{fmtS(p.p90_rt)}</td>
              <td className="num">{p.gini_zone_p90.toFixed(3)}</td><td className="num">{p.moves.toFixed(1)}</td><td className="num">{p.km_moved.toFixed(1)}</td>
              <td className="num">{p.critic_rejections.toFixed(1)}</td></tr>)}</tbody></table>
        </div>
        <div className="card" style={{ gridColumn: "1 / -1" }}>
          <h3>LIFELINE vs each alternative: paired over days (95% bootstrap CI, Wilcoxon signed-rank)</h3>
          <table><thead><tr><th>Comparison</th><th>Metric</th><th className="num">LIFELINE − other</th><th className="num">95% CI</th><th className="num">p</th><th className="num">days LIFELINE better</th></tr></thead>
            <tbody>{Object.entries(b.comparisons).flatMap(([k, ms]: any) => Object.entries(ms).filter(([m]) => m !== "p90_rt").map(([m, r]: any) => (
              <tr key={k + m}><td>{NAMES[k.replace("lifeline_vs_", "")]}</td><td>{m}</td>
                <td className="num">{m === "within_target" ? pct(r.mean_diff, 2) : r.mean_diff?.toFixed(m === "gini_zone_p90" ? 4 : 1)}</td>
                <td className="num mono">[{r.ci95?.map((x: number) => (m === "within_target" ? (x * 100).toFixed(2) : x.toFixed(m === "gini_zone_p90" ? 4 : 1))).join(", ")}]</td>
                <td className="num" style={{ color: r.wilcoxon_p < 0.05 ? "var(--ok)" : undefined }}>{r.wilcoxon_p?.toFixed(3)}</td>
                <td className="num">{r.days_better}/{r.n_days}</td></tr>)))}</tbody></table>
        </div>
      </> : <div className="card empty" style={{ gridColumn: "1 / -1" }}>Run <span className="mono">python -m lifeline.eval.benchmark</span> to produce the benchmark.</div>}

      {f?.cvae && <div className="card">
        <h3>Generative model: probabilistic forecast quality ({f.meta.n_eval_hours} held-out hours)</h3>
        <table><thead><tr><th>Model</th><th className="num">CRPS / cell</th><th className="num">energy score</th><th className="num">CRPS of total</th><th className="num">90% interval hit</th></tr></thead>
          <tbody>{[["cvae", "Conditional VAE"], ["cvae_no_latent", "CVAE, latent off"], ["poisson", "Poisson GLM"], ["naive", "Seasonal naive"]].filter(([k]) => f[k]).map(([k, n]) => (
            <tr key={k} style={k === "cvae" ? { fontWeight: 600 } : undefined}><td>{n}</td><td className="num">{f[k].crps_cell.toFixed(4)}</td>
              <td className="num">{f[k].energy_score.toFixed(3)}</td><td className="num">{f[k].crps_total.toFixed(3)}</td><td className="num">{pct(f[k].coverage90_total)}</td></tr>))}</tbody></table>
        <div className="faint" style={{ fontSize: 12, marginTop: 6 }}>Lower is better for CRPS/energy; 90% is ideal for interval hit. Totals test the joint (cross-zone) structure.</div>
      </div>}
      {v && <div className="card">
        <h3>Simulator validation (status quo vs recorded, 12 days of 2025)</h3>
        <div className="row" style={{ marginBottom: 6 }}>
          <span className="pill">p50 {fmtS(v.sim_p50)} vs {fmtS(v.recorded_p50)} ({pct(v.p50_error, 1)})</span>
          <span className="pill">p90 {fmtS(v.sim_p90)} vs {fmtS(v.recorded_p90)} ({pct(v.p90_error, 1)})</span>
          <span className="pill">KS {v.ks_stat.toFixed(3)}</span></div>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={hist}><CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="s" stroke="var(--muted)" fontSize={10} tickFormatter={(i) => `${+i}m`} />
            <YAxis stroke="var(--muted)" fontSize={10} tickFormatter={(x) => pct(x)} /><Tooltip {...tip} formatter={(x: any) => pct(x, 1)} /><Legend />
            <Bar dataKey="recorded" fill="var(--faint)" /><Bar dataKey="simulated" fill="var(--accent-2)" />
          </BarChart>
        </ResponsiveContainer>
        <div className="faint" style={{ fontSize: 12 }}>Response-time distribution, 1-minute bins.</div>
      </div>}
      {ae?.tests && <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h3>Agent-quality tests</h3>
        <table><thead><tr><th>Component</th><th>Test</th><th className="num">result</th></tr></thead><tbody>
          {ae.tests.map((t: any) => <tr key={t.name}><td>{t.component}</td><td>{t.name}</td><td className="num">{t.result}</td></tr>)}</tbody></table>
      </div>}
    </div>
  );
}
