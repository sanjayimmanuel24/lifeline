import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { pct } from "../api";
import { project, seq } from "../geo";
import { useLive } from "../store";

export function ZoneMap({ values, max, width = 120, title, stroke = 0.3 }: { values: number[]; max: number; width?: number; title?: string; stroke?: number }) {
  const { meta } = useLive();
  const proj = useMemo(() => meta && project(meta.geojson, meta.zones, width), [meta, width]);
  if (!proj) return null;
  return (
    <svg width={proj.width} height={proj.height} viewBox={`0 0 ${proj.width} ${proj.height}`} role="img" aria-label={title}>
      {title && <title>{title}</title>}
      {proj.paths.map((d, i) => <path key={i} d={d} fill={seq((values[i] ?? 0) / Math.max(max, 1e-6))} stroke="var(--bg)" strokeWidth={stroke} />)}
    </svg>
  );
}

export default function Futures() {
  const { live, meta } = useLive();
  const f = live.agent?.forecast;
  if (!f || !meta) return <div className="card empty">The Demand Forecaster runs at the first agent review. Start a shift and wait for the first review.</div>;
  const S: number[][] = f.scenario_zone_totals;
  const max = Math.max(...S.flat());
  const totals = S.map((s) => s.reduce((a, b) => a + b, 0));
  const lo = Math.min(...totals), hi = Math.max(...totals);
  const bins = 12, w = Math.max(1, (hi - lo) / bins);
  const hist = Array.from({ length: bins }, (_, i) => ({ x: Math.round(lo + i * w), n: totals.filter((t) => t >= lo + i * w && (t < lo + (i + 1) * w || i === bins - 1)).length }));
  const zmax = Math.max(...f.zone_mean);
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)" }}>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <div className="row">
          <div><h2>50 possible futures for the next 2 hours</h2>
            <div className="muted">Each mini-map is one joint sample from the Conditional VAE ({f.model}). Brighter = more calls. Zones rise and fall together, which is the correlated-surge structure the planner protects against.</div></div>
          <div className="spacer" />
          <span className="pill">{f.total_mean.toFixed(0)} calls expected</span>
          <span className="pill">90%: {f.total_p05.toFixed(0)}–{f.total_p95.toFixed(0)}</span>
          <span className={`pill ${f.surge_prob > 0.5 ? "bad" : f.surge_prob > 0.2 ? "warn" : "ok"}`}>surge {pct(f.surge_prob)}</span>
          <span className={`pill ${f.ood.score > 0.6 ? "bad" : f.ood.score > 0.3 ? "warn" : "ok"}`}>novelty {f.ood.score.toFixed(2)}</span>
        </div>
      </div>
      <div className="card">
        <h3>Generated futures</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(92px, 1fr))", gap: 6 }}>
          {S.map((s, i) => (
            <div key={i} style={{ background: "var(--panel-2)", borderRadius: 6, padding: 4, textAlign: "center" }}>
              <ZoneMap values={s} max={max} width={84} title={`future ${i + 1}: ${totals[i]} calls`} />
              <div className="faint mono" style={{ fontSize: 10.5 }}>{totals[i]}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="grid" style={{ alignContent: "start" }}>
        <div className="card">
          <h3>Mean demand · uncertainty</h3>
          <div className="row" style={{ justifyContent: "space-around" }}>
            <div style={{ textAlign: "center" }}><ZoneMap values={f.zone_mean} max={zmax} width={150} /><div className="faint">mean</div></div>
            <div style={{ textAlign: "center" }}><ZoneMap values={f.zone_std} max={Math.max(...f.zone_std)} width={150} /><div className="faint">std. dev.</div></div>
          </div>
          <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>Most uncertain: {f.uncertain_zones.join(", ")}</div>
        </div>
        <div className="card">
          <h3>Total calls across futures</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={hist}><CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="x" stroke="var(--muted)" fontSize={11} /><YAxis stroke="var(--muted)" fontSize={11} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
              <ReferenceLine x={hist.reduce((a, b) => (Math.abs(b.x - f.historical_mean) < Math.abs(a.x - f.historical_mean) ? b : a)).x} stroke="var(--warn)" label={{ value: "hist. avg", fill: "var(--warn)", fontSize: 10 }} />
              <Bar dataKey="n" fill="var(--accent-2)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Latent space (2 of 8 dims)</h3>
          <ResponsiveContainer width="100%" height={170}>
            <ScatterChart><CartesianGrid stroke="var(--line)" />
              <XAxis type="number" dataKey="x" stroke="var(--muted)" fontSize={11} domain={[-3, 3]} />
              <YAxis type="number" dataKey="y" stroke="var(--muted)" fontSize={11} domain={[-3, 3]} />
              <Scatter data={f.latent_2d.map(([x, y]: number[], i: number) => ({ x, y, t: totals[i] }))} fill="var(--accent)" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="faint" style={{ fontSize: 12 }}>Novelty check: latent distance {f.ood.latent_distance.toFixed(2)} (99th pct of training {f.ood.latent_p99.toFixed(2)}){f.ood.outside_training_range.length ? ` · outside training range: ${f.ood.outside_training_range.join(", ")}` : ""}</div>
        </div>
      </div>
    </div>
  );
}
