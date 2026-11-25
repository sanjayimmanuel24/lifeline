import { createContext, useContext, useEffect, useReducer, useRef, useState, type ReactNode } from "react";
import { api, type Meta, type TraceEv, type Unit } from "./api";

export type Call = { id: string; zone: number; sev: number; at: number };
export type Move = { unit: number; from: number; to: number; from_zone: string; to_zone: string; km: number; route: string[]; at: number };

export type Live = {
  sessionId?: string;
  status: string;
  time: string;
  simT: number;
  units: Unit[];
  queue: number;
  metricsHour: any;
  calls: Call[];
  trace: (TraceEv & { node?: string; time: string; i: number })[];
  activeNode?: string;
  agent: any;                 // merged latest agent state (situation, forecast, plan, meta, ...)
  approval?: any;
  moves: Move[];
  reviews: any[];
  injections: any[];
  lastError?: string;
};

const initial: Live = { status: "idle", time: "", simT: 0, units: [], queue: 0, metricsHour: {}, calls: [], trace: [],
  agent: {}, moves: [], reviews: [], injections: [] };

function reduce(s: Live, e: any): Live {
  switch (e.kind) {
    case "reset":
      return { ...initial, sessionId: e.id, status: "starting" };
    case "session":
      return { ...s, status: e.status, time: e.time };
    case "frame": {
      const calls = [...s.calls, ...e.calls.map((c: any) => ({ ...c, at: e.sim_t }))].filter((c) => e.sim_t - c.at < 900);
      return { ...s, units: e.units, queue: e.queue, metricsHour: e.metrics, calls, time: e.time, simT: e.sim_t,
        moves: s.moves.filter((m) => e.sim_t - m.at < 1800) };
    }
    case "review_start":
      return { ...s, status: "thinking", activeNode: "guard",
        trace: [...s.trace, { agent: "Scheduler", kind: "route", summary: `review #${e.epoch}: ${e.triggers.join(", ")}`, time: e.time, i: e.i }] };
    case "node": {
      const tr = (e.trace || []).map((t: TraceEv) => ({ ...t, node: e.node, time: e.time, i: e.i }));
      return { ...s, activeNode: e.node, agent: { ...s.agent, ...e.update }, trace: [...s.trace, ...tr].slice(-600) };
    }
    case "approval":
      return { ...s, status: "awaiting_approval", approval: e, activeNode: "human_gate" };
    case "review_end":
      return { ...s, status: "running", approval: undefined, activeNode: undefined, agent: { ...s.agent, ...e.state },
        reviews: [...s.reviews, e.review] };
    case "moves":
      return { ...s, moves: [...s.moves, ...e.moves.map((m: any) => ({ ...m, at: e.sim_t }))] };
    case "injection":
      return { ...s, injections: [...s.injections, e.injection],
        trace: [...s.trace, { agent: "Control room", kind: "human", summary: `injected: ${e.injection.name}` +
          (e.extra_calls ? ` (+${e.extra_calls} extra calls over 2h)` : ""), time: e.time, i: e.i }] };
    case "error":
      return { ...s, status: "error", lastError: e.message };
    default:
      return s;
  }
}

type Ctx = {
  meta?: Meta; live: Live; start: (o: any) => Promise<void>; control: (a: string, v?: number) => Promise<void>;
  inject: (k: string, extra?: any) => Promise<void>; decide: (d: any) => Promise<void>; paused: boolean; speed: number;
};
const C = createContext<Ctx>(null as any);
export const useLive = () => useContext(C);

export function LiveProvider({ children }: { children: ReactNode }) {
  const [meta, setMeta] = useState<Meta>();
  const [live, dispatch] = useReducer(reduce, initial);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(120);
  const es = useRef<EventSource | null>(null);

  useEffect(() => { api<Meta>("/meta").then(setMeta).catch(() => {}); }, []);

  const connect = (id: string) => {
    es.current?.close();
    const src = new EventSource(`/api/session/events?since=0`);
    src.onmessage = (m) => dispatch(JSON.parse(m.data));
    src.addEventListener("end", () => src.close());
    es.current = src;
    dispatch({ kind: "reset", id });
  };

  // reattach to a running session after a page reload
  useEffect(() => {
    api("/session/state").then((st: any) => { if (st?.id) { connect(st.id); setPaused(st.paused); setSpeed(st.speed); } })
      .catch(() => {});
    return () => es.current?.close();
  }, []);

  const value: Ctx = {
    meta, live, paused, speed,
    start: async (o) => { const st = await api<any>("/session/start", o); setPaused(false); setSpeed(st.speed); connect(st.id); },
    control: async (a, v) => {
      const r = await api<any>("/session/control", { action: a, value: v });
      setPaused(r.paused); setSpeed(r.speed);
    },
    inject: async (k, extra) => { await api("/session/inject", { kind: k, ...extra }); },
    decide: async (d) => { await api("/session/decision", d); },
  };
  return <C.Provider value={value}>{children}</C.Provider>;
}
