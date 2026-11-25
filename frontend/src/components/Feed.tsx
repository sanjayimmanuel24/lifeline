import type { TraceEv } from "../api";

export function Feed({ items }: { items: (TraceEv & { time?: string })[] }) {
  if (!items.length) return <div className="faint">Agent steps appear here as they happen.</div>;
  return (
    <div className="feed">
      {items.map((t, i) => (
        <div key={i} className={`ev ${t.kind === "meta" ? "meta-k" : t.kind}`}>
          <div className="row" style={{ gap: 6 }}>
            <span className="who">{t.agent}</span>
            {t.tool && <span className="tag">{t.tool}</span>}
            {t.provider && <span className="tag">{t.provider}</span>}
            <span className="spacer" />
            <span className="meta">{t.time}{t.latency_s ? ` · ${t.latency_s}s` : ""}</span>
          </div>
          <div style={{ fontSize: 13 }}>{t.summary}</div>
        </div>
      ))}
    </div>
  );
}
