export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`/api${path}`, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

export type Unit = { id: number; status: "idle" | "busy" | "moving"; post: number; pos: number; loc: number; dest: number;
  depart_t: number; arrive_t: number };
export type TraceEv = { agent: string; kind: string; summary: string; provider?: string; latency_s?: number; tool?: string;
  [k: string]: any };
export type Meta = {
  zones: string[]; geojson: any; centroids: [number, number][]; hospitals: { name: string; lat: number; lon: number }[];
  fleet: number; test_days: string[]; zone_demand_per_hour: number[]; calibration: any; llm_providers: string[];
  graph: { nodes: string[]; edges: { source: string; target: string; label: string }[] }; embedder: string;
  adjacency: Record<string, number[]>;
};

export const fmtS = (s?: number) => (s == null || isNaN(s) ? "–" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
export const pct = (x?: number, d = 0) => (x == null || isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`);
