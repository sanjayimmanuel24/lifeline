import { useEffect, useMemo, useState } from "react";
import { Background, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { api } from "../api";
import { Feed } from "../components/Feed";
import { useLive } from "../store";

// hand-placed layout of the LangGraph so parallel branches and loops are readable
const POS: Record<string, [number, number]> = {
  START: [0, 160], guard: [130, 160], situation_analyst: [280, 160],
  demand_forecaster: [470, 60], memory_retriever: [470, 160], sop_retriever: [470, 260],
  planner: [660, 160], critic: [820, 160], meta_reasoner: [980, 160], explainer: [1140, 160],
  claims_audit: [1300, 160], autonomy_gate: [1460, 160], human_gate: [1620, 60], executor: [1620, 260], END: [1780, 260],
};

export default function AgentTrace() {
  const { meta, live } = useLive();
  const [filter, setFilter] = useState("");
  const [ckpt, setCkpt] = useState<any>();
  const visited = useMemo(() => new Set(live.trace.slice(-40).map((t) => t.node).filter(Boolean)), [live.trace]);

  useEffect(() => {
    if (!live.sessionId || !live.reviews.length) return;
    api(`/session/checkpoints`).then(setCkpt).catch(() => {});
  }, [live.reviews.length, live.sessionId]);

  const { nodes, edges } = useMemo(() => {
    if (!meta) return { nodes: [], edges: [] };
    const nodes: Node[] = meta.graph.nodes.map((n) => {
      const active = live.activeNode === n;
      return { id: n, position: { x: POS[n][0], y: POS[n][1] }, data: { label: n.replace("_", " ") },
        style: { background: active ? "var(--accent)" : visited.has(n) ? "var(--panel-2)" : "var(--panel)",
          color: active ? "#fff" : "var(--text)", border: `1px solid ${active ? "var(--accent)" : "var(--line)"}`,
          borderRadius: 8, fontSize: 12, width: 130, boxShadow: active ? "0 0 0 4px rgba(255,77,94,.25)" : undefined } };
    });
    const edges: Edge[] = meta.graph.edges.map((e, i) => ({ id: `e${i}`, source: e.source, target: e.target, label: e.label,
      animated: e.label === "parallel" || e.label === "reject", type: e.source === e.target ? "default" : "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: e.label === "reject" || e.label === "unsupported" ? "var(--warn)" : "var(--muted)" },
      labelStyle: { fill: "var(--muted)", fontSize: 10 }, labelBgStyle: { fill: "var(--panel)" } }));
    return { nodes, edges };
  }, [meta, live.activeNode, visited]);

  const items = live.trace.filter((t) => !filter || `${t.agent} ${t.summary} ${t.tool ?? ""}`.toLowerCase().includes(filter.toLowerCase()));
  return (
    <div className="grid">
      <div className="card" style={{ height: 380, padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "12px 14px 0" }}><h3>LangGraph workflow · active node highlighted</h3></div>
        <div style={{ height: 330 }}>
          <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }} nodesDraggable={false}>
            <Background color="var(--line)" />
          </ReactFlow>
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)" }}>
        <div className="card">
          <div className="row"><h3 style={{ margin: 0 }}>Trace ({live.trace.length} steps)</h3><div className="spacer" />
            <input placeholder="filter: critic, mcts_plan, blocked…" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>
          <div style={{ maxHeight: 560, overflow: "auto", marginTop: 10 }}><Feed items={[...items].reverse()} /></div>
        </div>
        <div className="card">
          <h3>Checkpoints (time travel)</h3>
          {!ckpt?.steps?.length ? <div className="faint">LangGraph saves a checkpoint after every step (SqliteSaver). They appear after the first review.</div> : <>
            <div className="row" style={{ marginBottom: 8 }}>
              <span className="muted">review</span>
              <select value={ckpt.epoch} onChange={(e) => api(`/session/checkpoints?epoch=${e.target.value}`).then(setCkpt)}>
                {ckpt.epochs.map((e: number) => <option key={e}>{e}</option>)}</select>
            </div>
            <table><thead><tr><th>step</th><th>next node</th><th>state keys</th></tr></thead><tbody>
              {ckpt.steps.map((s: any) => <tr key={s.checkpoint_id}><td className="num">{s.step}</td>
                <td className="mono">{s.next.join(", ") || "—"}</td><td className="faint" style={{ fontSize: 11 }}>{s.keys.length}</td></tr>)}
            </tbody></table></>}
        </div>
      </div>
    </div>
  );
}
