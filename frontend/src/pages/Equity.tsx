import { useEffect, useState } from "react";
import { api, fmtS } from "../api";
import { useLive } from "../store";
import { ZoneMap } from "./Futures";

export default function Equity() {
  const { meta, live } = useLive();
  const [st, setSt] = useState<any>();
  useEffect(() => {
    if (!live.sessionId) return;
    const load = () => api("/session/state").then(setSt).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [live.sessionId]);
  const zp: Record<string, number> = st?.metrics?.zone_p90 || {};
  if (!meta || !Object.keys(zp).length) return <div className="card empty">Per-zone response times appear once calls have been served.</div>;
  const vals = meta.zones.map((_, i) => zp[i] ?? 0);
  const rows = meta.zones.map((z, i) => ({ z, v: zp[i], pop: meta.geojson.features.find((f: any) => f.properties.zone === z)?.properties.pop }))
    .filter((r) => r.v != null).sort((a, b) => b.v - a.v);
  const max = Math.max(...rows.map((r) => r.v));
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
      <div className="card">
        <h3>90th-percentile response time by zone (this shift)</h3>
        <div style={{ textAlign: "center" }}><ZoneMap values={vals} max={max} width={420} stroke={0.8} /></div>
        <div className="row" style={{ marginTop: 10 }}>
          <span className="pill">Gini {st.metrics.gini_zone_p90?.toFixed(3)}</span>
          <span className="faint" style={{ fontSize: 12 }}>0 = every zone waits the same; the planner penalises rising inequality (SOP-3) and never uses personal attributes.</span>
        </div>
      </div>
      <div className="card" style={{ maxHeight: 640, overflow: "auto" }}>
        <h3>Zones, slowest first</h3>
        <table><thead><tr><th>Zone</th><th>p90</th><th></th><th className="num">population</th></tr></thead><tbody>
          {rows.map((r) => <tr key={r.z}><td className="mono">{r.z}</td><td className="num">{fmtS(r.v)}</td>
            <td style={{ width: "40%" }}><div className="bar"><i style={{ width: `${(r.v / max) * 100}%`, background: r.v > 900 ? "var(--bad)" : r.v > 600 ? "var(--warn)" : "var(--ok)" }} /></div></td>
            <td className="num faint">{r.pop?.toLocaleString()}</td></tr>)}
        </tbody></table>
      </div>
    </div>
  );
}
