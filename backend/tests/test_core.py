"""Unit tests for the deterministic building blocks (no data download or LLM needed)."""
import numpy as np
import pandas as pd
import pytest

from lifeline.forecast.cvae import CVAE
from lifeline.forecast.metrics import crps, energy_score
from lifeline.geo import Geography, TravelModel, astar
from lifeline.planner.coverage import evaluate
from lifeline.planner.mcts import Constraints, MCTSPlanner
from lifeline.planner.mexclp import mexclp, proportional
from lifeline.sim.engine import Incident, Samplers, Simulator, gini


def toy_geo(n=6) -> Geography:
    """A line of n zones 1 km apart."""
    cent = np.array([[40.6 + 0.009 * i, -73.95] for i in range(n)])
    graph = {i: {} for i in range(n)}
    for i in range(n - 1):
        graph[i][i + 1] = graph[i + 1][i] = 1.0
    return Geography([f"z{i}" for i in range(n)], cent, np.full(n, 0.4), graph, {"features": []})


def toy_travel(geo):
    return TravelModel(geo, np.full(24, 30.0), 30.0)


def toy_sim(geo, posts, seed=0):
    return Simulator(toy_travel(geo), posts, pd.Timestamp("2026-01-01"),
                     Samplers(np.array([20.0]), np.array([1800.0]), np.random.default_rng(seed)))


# ---- geography -----------------------------------------------------------------------------
def test_astar_finds_shortest_path():
    g = {0: {1: 1.0, 2: 5.0}, 1: {0: 1.0, 2: 1.0}, 2: {0: 5.0, 1: 1.0}}
    path, cost = astar(g, 0, 2, lambda a, b: 0.0)
    assert path == [0, 1, 2] and cost == 2.0


def test_km_matrix_is_symmetric_with_intrazone_diagonal():
    geo = toy_geo()
    assert np.allclose(geo.km, geo.km.T)
    assert np.allclose(np.diag(geo.km), geo.radius_km)
    assert geo.km[0, 5] == pytest.approx(5.0)


# ---- simulator -------------------------------------------------------------------------------
def test_every_call_is_served_and_response_includes_travel():
    geo = toy_geo()
    sim = toy_sim(geo, [0, 3, 5])
    sim.add_incidents([Incident(str(i), 60.0 * i, i % 6) for i in range(12)])
    sim.run_until(86400)
    df = sim.records_frame()
    assert len(df) == 12
    assert (df.response >= df.travel + df.proc - 1e-9).all()


def test_calls_queue_when_all_units_busy():
    geo = toy_geo()
    sim = toy_sim(geo, [0])
    sim.add_incidents([Incident("a", 0, 0), Incident("b", 10, 1)])
    sim.run_until(86400)
    df = sim.records_frame().set_index("id")
    assert df.loc["a", "wait"] == 0 and df.loc["b", "wait"] > 1000


def test_fork_never_sees_future_real_calls():
    geo = toy_geo()
    sim = toy_sim(geo, [0, 5])
    sim.add_incidents([Incident("future", 5000, 2)])
    f = sim.fork([], np.random.default_rng(1))
    f.run_until(86400)
    assert f.records_frame().empty


def test_reposition_moves_idle_unit():
    geo = toy_geo()
    sim = toy_sim(geo, [0])
    assert sim.reposition(0, 4)
    sim.run_until(3600)
    assert sim.units[0].loc == 4 and sim.units[0].status == "idle"


def test_gini_bounds():
    assert gini(np.ones(5)) == pytest.approx(0.0)
    assert 0.7 < gini(np.array([0, 0, 0, 0, 10.0])) <= 1.0


# ---- planner -----------------------------------------------------------------------------------
def test_adding_a_unit_never_reduces_coverage():
    geo = toy_geo()
    T = toy_travel(geo).matrix(12)
    d = np.ones(6)
    a = evaluate(np.array([0, 5]), d, T, 0.3)
    b = evaluate(np.array([0, 5, 2]), d, T, 0.3)
    assert b.within >= a.within and b.mean_rt <= a.mean_rt


def _planner(geo, seed=0):
    return MCTSPlanner(toy_travel(geo).matrix(12), geo.km, 1.0, 4, np.random.default_rng(seed), 20.0)


def test_mcts_moves_a_unit_toward_uncovered_demand():
    geo = toy_geo()
    scen = np.zeros((20, 2, 6))
    scen[:, :, 5] = 6                                    # all demand at the far end
    res = _planner(geo).plan([(0, 0), (1, 0), (2, 0)], scen, 0.1, iters=300)
    assert res.moves and all(geo.km[p, 5] < geo.km[0, 5] for _, p in res.moves)
    assert res.plan_eval["within"] >= res.root_eval["within"]


def test_mcts_is_deterministic_with_a_seed():
    geo = toy_geo()
    scen = np.random.default_rng(0).poisson(1.0, (20, 2, 6)).astype(float)
    a = _planner(geo, 3).plan([(0, 0), (1, 1), (2, 2)], scen, 0.2, iters=200).moves
    b = _planner(geo, 3).plan([(0, 0), (1, 1), (2, 2)], scen, 0.2, iters=200).moves
    assert a == b


def test_hard_constraints_are_respected():
    geo = toy_geo()
    scen = np.zeros((20, 2, 6))
    scen[:, :, 5] = 6
    avail = [(0, 0), (1, 0), (2, 1)]
    assert _planner(geo).plan(avail, scen, 0.1, Constraints(max_moves=0), iters=100).moves == []
    res = _planner(geo).plan(avail, scen, 0.1, Constraints(frozen_units={0, 1}), iters=200)
    assert all(u not in (0, 1) for u, _ in res.moves)
    res = _planner(geo).plan(avail, scen, 0.1, Constraints(protected_zones={0: 0.5}), iters=200)
    assert all(u == 2 for u, _ in res.moves)              # units in protected zone 0 stay put


def test_mexclp_and_proportional_place_every_unit():
    geo = toy_geo()
    d = np.array([1, 1, 1, 1, 1, 5.0])
    assert len(mexclp(d, toy_travel(geo).matrix(12), 5, 0.3, 20.0, time_limit_s=5)) == 5
    p = proportional(d, 10)
    assert len(p) == 10 and p.count(5) == 5


# ---- forecasting ----------------------------------------------------------------------------------
def test_scores_are_zero_for_a_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    s = np.tile(y, (40, 1))
    assert crps(s, y) == pytest.approx(0.0)
    assert energy_score(s, y) == pytest.approx(0.0)
    assert crps(s + 1, y) > crps(s, y)


def test_cvae_samples_have_the_right_shape_and_are_counts():
    m = CVAE(x_dim=12, c_dim=5)
    counts, z = m.sample(np.zeros(5, np.float32), 7, np.random.default_rng(0))
    assert counts.shape == (7, 12) and z.shape == (7, 8)
    assert (counts >= 0).all() and np.allclose(counts, np.round(counts))
