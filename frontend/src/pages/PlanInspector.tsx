import { fmtS, pct } from "../api";
import { useLive } from "../store";

type TNode = { label: string; visits: number; q: number; children?: TNode[] };

function Tree({ root }: { root: TNode }) {
  // simple tidy layout: leaves get consecutive x slots, parents centred over children
  const nodes: { n: TNode; x: number; y: number; px?: number; py?: number }[] = [];
  let slot = 0;
  const place = (n: TNode, d: number, parent?: { x: number; y: number }): number => {
    const kids = n.children || [];
    const xs = kids.length ? kids.map((k) => place(k, d + 1, undefined)) : [slot++];
    const x = xs.reduce((a, b) => a + b, 0) / xs.length;
    nodes.push({ n, x, y: d });
    return x;
  };
  place(root, 0);
  const W = 120, H = 92, maxV = root.visits;
  const qs = nodes.map((p) => p.n.q), qmin = Math.min(...qs), qmax = Math.max(...qs);
  const pos = new Map(nodes.map((p) => [p.n, p]));
  const edges: [typeof nodes[0], typeof nodes[0]][] = [];
  nodes.forEach((p) => p.n.children?.forEach((c) => edges.push([p, pos.get(c)!])));
  const width = Math.max(1, slot) * W, height = (Math.max(...nodes.map((p) => p.y)) + 1) * H;
  const col = (q: number) => `hsl(${((q - qmin) / Math.max(1e-6, qmax - qmin)) * 130}, 70%, 52%)`;
  return (
    <div style={{ overflow: "auto" }}>
      <svg width={width} height={height + 10}>
        {edges.map(([a, b], i) => <line key={i} x1={a.x * W + W / 2} y1={a.y * H + 24} x2={b.x * W + W / 2} y2={b.y * H + 24}
          stroke="var(--line)" strokeWidth={1 + 6 * (b.n.visits / maxV)} />)}
        {nodes.map((p, i) => (
          <g key={i} transform={`translate(${p.x * W + W / 2},${p.y * H + 24})`}>
            <circle r={6 + 16 * Math.sqrt(p.n.visits / maxV)} fill={col(p.n.q)} stroke="var(--bg)" strokeWidth={2} />
            <text y={38} textAnchor="middle" fontSize={11} fill="var(--text)" className="mono">{p.n.label}</text>
            <text y={51} textAnchor="middle" fontSize={10} fill="var(--muted)">n={p.n.visits} q={p.n.q}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function PlanInspector() {
  const { live, meta } = useLive();
  const { plan, directive, critic = [] } = live.agent || {};
  if (!plan || !meta) return <div className="card empty">No plan yet. The Planner runs at each agent review.</div>;
  const b = plan.root_eval, a = plan.plan_eval;
  const ver = plan.verified || {};
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
      <div className="card">
        <h3>Planner directive (plan-and-execute)</h3>
        <ol style={{ margin: 0, paddingLeft: 18 }}>{directive?.tasks?.map((t: string, i: number) => <li key={i}>{t}</li>)}</ol>
        <div className="row" style={{ marginTop: 10 }}>
          <span className="pill">equity boost {directive?.equity_boost}</span>
          <span className="pill">max moves {directive?.max_moves}</span>
          {directive?.protect_zones?.length > 0 && <span className="pill">protect {directive.protect_zones.join(", ")}</span>}
          {Object.keys(directive?.demand_multipliers || {}).length > 0 && <span className="pill warn">event demand ×{JSON.stringify(directive.demand_multipliers)}</span>}
        </div>
        <p className="muted" style={{ fontSize: 13 }}>{directive?.reasoning}</p>
      </div>
      <div className="card">
        <h3>Chosen plan</h3>
        {plan.moves.length === 0 ? <div className="muted">Hold all positions.</div> :
          <table><thead><tr><th>Unit</th><th>Move</th><th>Route (A*)</th><th className="num">km</th></tr></thead><tbody>
            {plan.moves.map((m: any) => <tr key={m.unit}><td>{m.unit}</td><td className="mono">{m.from_zone} → {m.to_zone}</td>
              <td className="mono faint" style={{ fontSize: 11 }}>{m.route.join(" › ")}</td><td className="num">{m.km}</td></tr>)}
          </tbody></table>}
        <table style={{ marginTop: 10 }}><thead><tr><th>Coverage model (avg over futures)</th><th className="num">now</th><th className="num">plan</th></tr></thead><tbody>
          <tr><td>within 8 min</td><td className="num">{pct(b.within, 1)}</td><td className="num">{pct(a.within, 1)}</td></tr>
          <tr><td>mean response</td><td className="num">{fmtS(b.mean_rt)}</td><td className="num">{fmtS(a.mean_rt)}</td></tr>
          <tr><td>inequality (Gini)</td><td className="num">{b.gini.toFixed(3)}</td><td className="num">{a.gini.toFixed(3)}</td></tr>
          <tr><td>worst zone coverage</td><td className="num">{pct(b.min_zone_cov)}</td><td className="num">{pct(a.min_zone_cov)}</td></tr>
        </tbody></table>
        <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>{plan.support} of 50 futures favour this plan · {plan.iterations} MCTS rollouts</div>
      </div>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h3>Monte Carlo search tree (node size = visits, colour = value)</h3>
        <Tree root={plan.tree} />
      </div>
      <div className="card">
        <h3>Stage 2: verified in the full simulator (same 16 futures for every plan)</h3>
        <table><thead><tr><th>Candidate</th><th className="num">within 8 min</th><th className="num">mean</th><th className="num">p90</th><th className="num">reward</th></tr></thead><tbody>
          {Object.entries(ver).map(([k, v]: any) => <tr key={k} style={k === plan.chosen ? { color: "var(--ok)" } : undefined}>
            <td>{k}{k === plan.chosen ? " ✓" : ""}</td><td className="num">{pct(v.within, 1)}</td><td className="num">{fmtS(v.mean_rt)}</td>
            <td className="num">{fmtS(v.p90_rt)}</td><td className="num">{v.reward?.toFixed(1)}</td></tr>)}
        </tbody></table>
      </div>
      <div className="card">
        <h3>Safety &amp; Equity Critic</h3>
        {critic.map((c: any, i: number) => (
          <div key={i} className={`ev ${c.ok ? "human" : "blocked"}`} style={{ marginBottom: 6 }}>
            <div className="who">Round {c.round}: {c.ok ? "accepted" : "rejected"}</div>
            {c.violations.map((v: any, j: number) => <div key={j} style={{ fontSize: 13 }}><b>{v.rule}</b>: {v.evidence}
              {Object.keys(v.fix || {}).length > 0 && <span className="faint"> → fix {JSON.stringify(v.fix)}</span>}</div>)}
          </div>
        ))}
      </div>
    </div>
  );
}
