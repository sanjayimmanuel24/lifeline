"""Zones, zone adjacency graph, A* routing and the travel-time model.

Zones are NYC Modified ZCTAs (the EMS data only records ZIP codes, not coordinates).
The road network is approximated by the zone adjacency graph: two zones are linked
when their polygons touch, the edge length is the centroid-to-centroid distance times
a detour factor. A* over this graph gives routes (for the map) and path lengths;
a per-hour speed curve calibrated on recorded travel seconds turns km into seconds.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from shapely.geometry import shape

from lifeline.config import DATA_DIR

DETOUR = 1.3          # street network vs straight line
EARTH_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def astar(graph: dict[int, dict[int, float]], start: int, goal: int,
          heuristic) -> tuple[list[int], float]:
    """Textbook A*: f = g + h with an admissible straight-line heuristic."""
    frontier = [(heuristic(start, goal), 0.0, start)]
    came_from: dict[int, int] = {}
    g = {start: 0.0}
    while frontier:
        _, cost, node = heapq.heappop(frontier)
        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            return path[::-1], cost
        if cost > g[node]:
            continue
        for nxt, w in graph[node].items():
            new = cost + w
            if new < g.get(nxt, math.inf):
                g[nxt] = new
                came_from[nxt] = node
                heapq.heappush(frontier, (new + heuristic(nxt, goal), new, nxt))
    raise ValueError(f"no path {start}->{goal}")


@dataclass
class Geography:
    zones: list[str]
    centroids: np.ndarray            # (Z, 2) lat, lon
    radius_km: np.ndarray            # (Z,) effective intra-zone travel distance
    graph: dict[int, dict[int, float]]
    geojson: dict
    hospitals: list[dict] = field(default_factory=list)

    @property
    def Z(self) -> int:
        return len(self.zones)

    def index(self, zone: str) -> int:
        return self.zones.index(zone)

    def heuristic(self, a: int, b: int) -> float:
        return haversine_km(tuple(self.centroids[a]), tuple(self.centroids[b]))

    def route(self, a: int, b: int) -> tuple[list[int], float]:
        if a == b:
            return [a], float(self.radius_km[a])
        return astar(self.graph, a, b, self.heuristic)

    @cached_property
    def km(self) -> np.ndarray:
        """All-pairs path length (km) by A*, intra-zone distance on the diagonal."""
        Z = self.Z
        m = np.zeros((Z, Z))
        for a in range(Z):
            for b in range(Z):
                m[a, b] = self.route(a, b)[1] if b >= a else m[b, a]
        return m

    def straight_km(self, a: int, b: int) -> float:
        return haversine_km(tuple(self.centroids[a]), tuple(self.centroids[b]))


def load_geography() -> Geography:
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    zones = splits["zones"]
    gj = json.loads((DATA_DIR / "zones.geojson").read_text())
    feats = {f["properties"]["zone"]: f for f in gj["features"]}
    zones = [z for z in zones if z in feats]
    polys = [shape(feats[z]["geometry"]) for z in zones]
    cent = np.array([[p.centroid.y, p.centroid.x] for p in polys])
    # area in km^2 (approximate at NYC latitude); mean distance to a random point in a disc = 2/3 r
    kx, ky = 111.32 * math.cos(math.radians(40.65)), 110.57
    radius = np.array([(2 / 3) * math.sqrt(p.area * kx * ky / math.pi) * DETOUR for p in polys])

    graph: dict[int, dict[int, float]] = {i: {} for i in range(len(zones))}
    buffered = [p.buffer(0.0008) for p in polys]
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            if buffered[i].intersects(buffered[j]):
                d = haversine_km(tuple(cent[i]), tuple(cent[j])) * DETOUR
                graph[i][j] = graph[j][i] = d
    _connect_components(graph, cent)

    for f in gj["features"]:
        f["properties"]["idx"] = zones.index(f["properties"]["zone"]) if f["properties"]["zone"] in zones else -1
    hospitals_path = DATA_DIR / "hospitals.json"
    hospitals = json.loads(hospitals_path.read_text()) if hospitals_path.exists() else []
    return Geography(zones, cent, radius, graph, gj, hospitals)


def _connect_components(graph: dict[int, dict[int, float]], cent: np.ndarray) -> None:
    """Islands (e.g. zones separated by water) get linked to their nearest zone."""
    seen: set[int] = set()
    comps = []
    for s in graph:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(graph[n])
        seen |= comp
        comps.append(comp)
    main = max(comps, key=len)
    for comp in comps:
        if comp is main:
            continue
        best = min(((a, b) for a in comp for b in main),
                   key=lambda ab: haversine_km(tuple(cent[ab[0]]), tuple(cent[ab[1]])))
        d = haversine_km(tuple(cent[best[0]]), tuple(cent[best[1]])) * DETOUR * 1.5
        graph[best[0]][best[1]] = graph[best[1]][best[0]] = d
        main |= comp


class TravelModel:
    """seconds(a, b, hour) = base_s + km(a, b) / speed_kmh(hour) * 3600.

    speed_kmh(hour) is calibrated from the recorded travel seconds (see calibrate.py)."""

    def __init__(self, geo: Geography, speed_by_hour: np.ndarray, base_s: float):
        self.geo = geo
        self.speed = np.asarray(speed_by_hour, dtype=float)   # (24,)
        self.base_s = float(base_s)
        self._cache: dict[int, np.ndarray] = {}

    def matrix(self, hour: int) -> np.ndarray:
        hour = int(hour) % 24
        if hour not in self._cache:
            self._cache[hour] = self.base_s + self.geo.km / self.speed[hour] * 3600.0
        return self._cache[hour]

    def seconds(self, a: int, b: int, hour: int) -> float:
        return float(self.matrix(hour)[a, b])

    def to_json(self) -> dict:
        return {"speed_by_hour": self.speed.tolist(), "base_s": self.base_s}
