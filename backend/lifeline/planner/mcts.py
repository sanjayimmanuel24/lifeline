"""Monte Carlo Tree Search (UCT + progressive widening) over ambulance repositioning moves.

State  : positions of the available units.
Action : "move unit u to post p" or STOP. A plan is a path of at most MAX_MOVES actions.
Rollout: draw ONE generated future (CVAE scenario) at random and score the configuration with
         the analytic coverage model. Averaging over many rollouts = sample-average
         approximation over futures, so the chosen plan is robust to the spread of surges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lifeline.config import CANDIDATE_POSTS, MAX_MOVES, MCTS_C, MCTS_ITERS
from lifeline.planner.coverage import Evaluation, busy_prob, evaluate

STOP = ("stop",)
REWARD_SCALE = 20.0       # seconds of response time ~ one unit of normalized reward
FLOOR_PENALTY = 400.0


@dataclass
class Constraints:
    frozen_units: set[int] = field(default_factory=set)        # e.g. crews near shift end
    protected_zones: dict[int, float] = field(default_factory=dict)  # zone -> min P(covered)
    max_moves: int = MAX_MOVES
    equity_boost: float = 1.0
    max_km_per_move: float = 12.0


@dataclass
class Node:
    positions: np.ndarray
    moves: tuple
    parent: "Node | None" = None
    action: tuple | None = None
    children: dict = field(default_factory=dict)
    untried: list = field(default_factory=list)
    N: int = 0
    W: float = 0.0
    terminal: bool = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


@dataclass
class PlanResult:
    moves: list[tuple[int, int]]                 # (unit_id, post)
    alternatives: list[dict]
    tree: dict
    root_eval: dict
    plan_eval: dict
    iterations: int
    candidate_posts: list[int]


class MCTSPlanner:
    def __init__(self, T: np.ndarray, km: np.ndarray, service_h: float, fleet: int, rng: np.random.Generator,
                 proc_s: float = 75.0):
        self.T, self.km, self.service_h, self.fleet, self.rng, self.proc_s = T, km, service_h, fleet, rng, proc_s

    # ---- scoring -----------------------------------------------------------
    def _all(self, positions: np.ndarray) -> np.ndarray:
        """Movable units + busy units at the post they will return to (MEXCLP view of the fleet)."""
        return np.concatenate([positions, self.fixed]) if len(self.fixed) else positions

    def _score(self, positions: np.ndarray, s: int, km: float, cons: Constraints) -> tuple[float, Evaluation]:
        ev = evaluate(self._all(positions), self.demand[s], self.T, self.q[s], km, self.proc_s)
        r = ev.reward - (cons.equity_boost - 1) * 400.0 * ev.gini
        for z, floor in cons.protected_zones.items():
            if ev.zone_cov[z] < floor:
                r -= FLOOR_PENALTY * (floor - ev.zone_cov[z])
        return r, ev

    def _km(self, moves: tuple) -> float:
        return float(sum(self.km[self.origin[u], p] for u, p in moves))

    # ---- main ----------------------------------------------------------------
    def plan(self, avail: list[tuple[int, int]], scenarios: np.ndarray, busy_now: float,
             cons: Constraints | None = None, iters: int = MCTS_ITERS, fixed: list[int] | None = None) -> PlanResult:
        cons = cons or Constraints()
        self.fixed = np.array(fixed or [], dtype=int)
        self.unit_ids = [u for u, _ in avail]
        self.origin = {u: z for u, z in avail}
        root_pos = np.array([z for _, z in avail], dtype=int)
        S, H, Z = scenarios.shape
        self.demand = scenarios.sum(1)                                       # (S, Z) calls over horizon
        self.q = np.array([busy_prob(d.sum() / H, self.fleet, self.service_h, busy_now) for d in self.demand])
        self.base = np.array([self._score(root_pos, s, 0.0, cons)[0] for s in range(S)])
        mean_d = self.demand.mean(0)

        self.posts = self._candidate_posts(root_pos, mean_d)
        self.actions = self._ranked_actions(root_pos, mean_d, cons) if cons.max_moves > 0 else []
        root = Node(root_pos, (), untried=[STOP] + self.actions)

        for _ in range(iters):
            node = self._select(root, cons)
            s = int(self.rng.integers(S))
            r, _ = self._score(node.positions, s, self._km(node.moves), cons)
            r = (r - self.base[s]) / REWARD_SCALE
            while node is not None:
                node.N += 1
                node.W += r
                node = node.parent

        best = self._extract(root)
        alts = self._alternatives(root)
        root_ev = self._mean_eval(root_pos, (), cons)
        plan_ev = self._mean_eval(self._apply(root_pos, best), tuple(best), cons)
        return PlanResult(best, alts, self._tree_json(root), root_ev, plan_ev, iters, self.posts.tolist())

    # ---- tree policy ------------------------------------------------------------
    def _select(self, node: Node, cons: Constraints) -> Node:
        while True:
            if node.terminal:
                return node
            allowed = max(1, math.ceil(2.0 * (node.N + 1) ** 0.5))         # progressive widening
            if node.untried and len(node.children) < allowed:
                return self._expand(node, cons)
            if not node.children:
                return node
            logn = math.log(node.N + 1)
            node = max(node.children.values(),
                       key=lambda c: c.Q + MCTS_C * math.sqrt(logn / (c.N + 1e-9)) if c.N else math.inf)

    def _expand(self, node: Node, cons: Constraints) -> Node:
        a = node.untried.pop(0)
        if a == STOP:
            child = Node(node.positions, node.moves, node, a, terminal=True)
        else:
            u, p = a
            moves = node.moves + (a,)
            pos = self._apply(node.positions, moves[-1:])
            moved = {m[0] for m in moves}
            done = len(moves) >= cons.max_moves
            untried = [] if done else [STOP] + [b for b in self.actions if b[0] not in moved and b[1] != p][:30]
            child = Node(pos, moves, node, a, untried=untried, terminal=done)
        node.children[a] = child
        return child

    def _apply(self, positions: np.ndarray, moves) -> np.ndarray:
        pos = positions.copy()
        for u, p in moves:
            pos[self.unit_ids.index(u)] = p
        return pos

    # ---- action generation -------------------------------------------------------
    def _candidate_posts(self, positions: np.ndarray, mean_d: np.ndarray) -> np.ndarray:
        """Posts ranked by marginal coverage gain of one extra unit (pruning for widening)."""
        q = float(self.q.mean())
        base = evaluate(self._all(positions), mean_d, self.T, q, 0, self.proc_s).reward
        gains = np.array([evaluate(self._all(np.append(positions, z)), mean_d, self.T, q, 0, self.proc_s).reward - base
                          for z in range(self.T.shape[0])])
        return np.argsort(-gains)[:CANDIDATE_POSTS]

    def _ranked_actions(self, positions: np.ndarray, mean_d: np.ndarray, cons: Constraints) -> list[tuple]:
        q = float(self.q.mean())
        base = evaluate(self._all(positions), mean_d, self.T, q, 0, self.proc_s).reward
        scored = []
        for i, u in enumerate(self.unit_ids):
            if u in cons.frozen_units or int(positions[i]) in cons.protected_zones:
                continue                    # hard constraints: frozen crews stay, protected zones keep their units
            for p in self.posts:
                if p == positions[i] or self.km[positions[i], p] > cons.max_km_per_move:
                    continue
                pos = positions.copy()
                pos[i] = p
                gain = evaluate(self._all(pos), mean_d, self.T, q, self.km[positions[i], p], self.proc_s).reward - base
                scored.append((gain, (u, int(p))))
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:60]]

    # ---- results --------------------------------------------------------------------
    def _extract(self, root: Node) -> list[tuple[int, int]]:
        node, moves = root, []
        while node.children:
            child = max(node.children.values(), key=lambda c: c.N)
            if child.action == STOP or child.N == 0:
                break
            moves.append(child.action)
            node = child
        return [(int(u), int(p)) for u, p in moves]

    def _alternatives(self, root: Node, k: int = 3) -> list[dict]:
        leaves = []

        def walk(n: Node):
            if n.action is not None and n.N >= 5:
                leaves.append(n)
            for c in n.children.values():
                walk(c)
        walk(root)
        seen, out = set(), []
        for n in sorted(leaves, key=lambda n: -n.Q):
            key = tuple(sorted(n.moves))
            if key in seen:
                continue
            seen.add(key)
            out.append({"moves": [list(map(int, m)) for m in n.moves], "q": round(n.Q, 3), "visits": n.N})
            if len(out) >= k:
                break
        return out

    def _mean_eval(self, positions: np.ndarray, moves: tuple, cons: Constraints) -> dict:
        evs = [self._score(positions, s, self._km(moves), cons)[1] for s in range(len(self.demand))]
        zone_cov = np.mean([e.zone_cov for e in evs], 0)
        return {"mean_rt": float(np.mean([e.mean_rt for e in evs])),
                "within": float(np.mean([e.within for e in evs])),
                "within_p10": float(np.percentile([e.within for e in evs], 10)),
                "gini": float(np.mean([e.gini for e in evs])),
                "km": self._km(moves),
                "zone_cov": zone_cov.tolist(),
                "zone_rt": np.mean([e.zone_rt for e in evs], 0).tolist(),
                "min_zone_cov": float(zone_cov.min()),
                "worst_zone": int(zone_cov.argmin())}

    def _tree_json(self, root: Node, max_children: int = 6, depth: int = 3) -> dict:
        def enc(n: Node, d: int) -> dict:
            label = "root" if n.action is None else ("STOP" if n.action == STOP else f"U{n.action[0]}→{n.action[1]}")
            out = {"label": label, "visits": n.N, "q": round(n.Q, 3)}
            if d < depth and n.children:
                kids = sorted(n.children.values(), key=lambda c: -c.N)[:max_children]
                out["children"] = [enc(c, d + 1) for c in kids if c.N > 0]
            return out
        return enc(root, 0)
