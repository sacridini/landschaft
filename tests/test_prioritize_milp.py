# SPDX-License-Identifier: GPL-3.0-or-later
"""MILP prioritization against prioritizr 8.0.6 / prioriactions 0.5.0 (tests/optimize_cases.py).

- the compiled model is prioritizr's ``compile()``, bit for bit;
- the optimum (gap 0) is prioritizr's on Gurobi;
- HiGHS built into landschaft solves every stored reference model to the same optimum.
"""

from __future__ import annotations

import math

import numpy as np
import optimize_cases
import pytest
from optimize_reference import case_ids, model, result

import landschaft as ls
from landschaft import _core

PY_CASES = [c for c in case_ids("core") if optimize_cases.CASES[c].py is not None]
ALL = [(g, c) for g in optimize_cases.GROUPS for c in case_ids(g)]


def _solve_stored(m, gap=0.0, time_limit=math.inf, presolve="choose", start=None, relaxation=False):
    a = m.csr()
    sense = np.asarray(m.sense)
    lo = np.where(np.isin(sense, (">=", "=", "==")), m.rhs, -np.inf)
    hi = np.where(np.isin(sense, ("<=", "=", "==")), m.rhs, np.inf)
    return _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=m.obj, lb=m.lb,
        ub=m.ub, row_lo=lo, row_hi=hi, integer=np.isin(np.asarray(m.vtype), ("B", "I")).astype(np.uint8),
        maximize=m.modelsense == "max", offset=0.0, gap=gap, time_limit=time_limit, threads=1, presolve=presolve,
        verbose=False, relaxation=relaxation, initial=start)


@pytest.mark.parametrize(("group", "name"), ALL, ids=[c for _, c in ALL])
def test_builtin_highs_same_optimum(group, name):
    m = model(group, name)
    if m.genconpow is not None:
        pytest.skip("general power constraints (Gurobi only)")
    r = _solve_stored(m)
    assert r["status"] == "optimal" and r["has_solution"]
    assert r["objective"] == pytest.approx(result(group, name)["objective"][0], rel=1e-9)
    assert r["x"].shape == (m.obj.size,)


@pytest.mark.parametrize("name", PY_CASES)
def test_compile_is_prioritizr(name):
    ours = optimize_cases.CASES[name].py().compile()
    ref = model("core", name)
    a = ours.A.tocoo()
    np.testing.assert_array_equal(a.row, ref.A_i, err_msg="A_i")
    np.testing.assert_array_equal(a.col, ref.A_j, err_msg="A_j")
    np.testing.assert_array_equal(a.data, ref.A_x, err_msg="A_x")
    for key in ("obj", "lb", "ub", "rhs"):
        np.testing.assert_array_equal(getattr(ours, key), getattr(ref, key), err_msg=key)
    assert list(ours.sense) == ref.sense
    assert list(ours.vtype) == ref.vtype
    assert ours.modelsense == ref.modelsense
    assert list(ours.row_ids) == ref.row_ids and list(ours.col_ids) == ref.col_ids


@pytest.mark.parametrize("name", PY_CASES)
def test_solve_reaches_prioritizr_optimum(name):
    sol = optimize_cases.CASES[name].py().add_highs_solver(gap=0).solve()
    ref = result("core", name)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(ref["objective"][0], rel=1e-9)
    assert sol.cost == pytest.approx(ref["objective"][0], rel=1e-9)
    assert sol.coverage["met"].all()


def test_solution_maps_back_to_the_raster():
    p = optimize_cases.CASES["minset_raster_relative"].py()
    sol = p.solve()
    cost = optimize_cases.pu_cost()
    assert sol.selection.shape == cost.shape
    assert np.array_equal(np.isnan(sol.selection.values), np.isnan(cost.values))
    assert set(np.unique(sol.values)) <= {0.0, 1.0}
    assert sol.selection.rio.crs == cost.rio.crs


def test_problem_is_immutable_and_order_free():
    base = ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features())
    a = base.add_relative_targets(0.3).add_min_set_objective().add_binary_decisions()
    b = base.add_binary_decisions().add_min_set_objective().add_relative_targets(0.3)
    assert base.objective is None and base.targets is None
    ma, mb = a.compile(), b.compile()
    assert np.array_equal(ma.rhs, mb.rhs) and (ma.A != mb.A).nnz == 0


def test_errors():
    base = ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features())
    with pytest.raises(ValueError, match="no objective"):
        base.add_relative_targets(0.3).compile()
    with pytest.raises(ValueError, match="needs targets"):
        base.add_min_set_objective().compile()
    with pytest.raises(ValueError, match="one per feature"):
        base.add_relative_targets([0.1, 0.2])
    li = optimize_cases.locked_in()
    with pytest.raises(ValueError, match="both locked in and locked out"):
        base.add_min_set_objective().add_relative_targets(0.3).add_locked_in_constraints(li) \
            .add_locked_out_constraints(li).compile()


def test_infeasible_is_reported():
    p = ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features()).add_min_set_objective() \
        .add_relative_targets(1.0)  # cells without cost hold part of the features: 100% cannot be met
    with pytest.raises(RuntimeError, match="infeasible"):
        p.solve()


def test_time_limit_is_kept_by_our_clock():
    m = model("core", "minset_raster_relative")
    r = _solve_stored(m, gap=0.0, time_limit=0.0)
    assert r["status"] == "time_limit"
    assert r["seconds"] < 5


def test_start_solution_is_used():
    m = model("core", "minset_raster_relative")
    x = result("core", "minset_raster_relative")["solution"]
    r = _solve_stored(m, gap=0.5, start=x)
    assert r["has_solution"] and r["objective"] <= m.obj @ x + 1e-9


def test_relaxation():
    m = model("core", "minset_raster_proportion")
    r = _solve_stored(m, relaxation=True)
    assert r["status"] == "optimal"
    assert r["objective"] == pytest.approx(result("core", "minset_raster_proportion")["objective"][0], rel=1e-9)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("gurobipy"), reason="gurobipy not installed")
def test_gurobi_backend():
    import gurobipy as gp

    try:
        with gp.Env(params={"OutputFlag": 0}):
            pass
    except gp.GurobiError:
        pytest.skip("no Gurobi licence")
    sol = optimize_cases.CASES["minset_raster_locked"].py().add_gurobi_solver(gap=0, n_threads=1).solve()
    assert sol.solver == "gurobi" and sol.status == "optimal"
    assert sol.objective == pytest.approx(result("core", "minset_raster_locked")["objective"][0], rel=1e-9)


# ----------------------------------------------------------------------------- the covering engine

def _wide_cover(n=20000, m=10, seed=3):
    """A random minimum-set model, wide enough for the bundle method and the sifting (n > 4 working sets)."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=0.3, random_state=rng, format="csr") * 5
    a.sort_indices()
    rhs = 0.3 * np.asarray(a.sum(axis=1)).ravel()
    cost = 1 + 9 * rng.random(n)
    return a, cost, rhs


def _cover(a, cost, rhs, method, gap=0.0, threads=1, time_limit=math.inf, integer=True):
    n = cost.size
    return _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=cost,
        lb=np.zeros(n), ub=np.ones(n), row_lo=rhs, row_hi=np.full(rhs.size, np.inf),
        integer=np.full(n, int(integer), dtype=np.uint8), maximize=False, offset=0.0, gap=gap,
        time_limit=time_limit, threads=threads, presolve="choose", verbose=False, relaxation=False, initial=None,
        method=method)


def test_covering_same_bits_on_any_threads():
    a, cost, rhs = _wide_cover()
    r1 = _cover(a, cost, rhs, "covering", gap=0.01, threads=1)
    r4 = _cover(a, cost, rhs, "covering", gap=0.01, threads=4)
    assert r1["status"] == r4["status"] == "optimal"
    assert np.array_equal(r1["x"], r4["x"])
    assert r1["objective"] == r4["objective"] and r1["bound"] == r4["bound"]


def test_covering_bounds_agree_with_highs():
    """Each solver's bound is below the other's solution, and ours is within the gap asked for."""
    a, cost, rhs = _wide_cover(n=12000, m=6)
    ours = _cover(a, cost, rhs, "covering", gap=0.01, threads=2, time_limit=60)
    highs = _cover(a, cost, rhs, "highs", gap=0.01, time_limit=60)
    assert ours["status"] == "optimal"
    tol = 1e-9 * abs(ours["objective"])
    assert ours["bound"] <= highs["objective"] + tol
    assert highs["bound"] <= ours["objective"] + tol
    assert ours["objective"] <= highs["objective"] * 1.01
    x = ours["x"]
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert np.all(a @ x >= rhs - 1e-9 * np.maximum(1, rhs))


def test_covering_lp_and_gap():
    a, cost, rhs = _wide_cover()
    lp = _cover(a, cost, rhs, "covering", integer=False)
    ref = _cover(a, cost, rhs, "highs", integer=False)
    assert lp["status"] == "optimal" and lp["gap"] == 0.0
    assert lp["objective"] == pytest.approx(ref["objective"], rel=1e-9)
    mip = _cover(a, cost, rhs, "covering", gap=0.01, threads=2)
    # the LP bound is valid and the reported gap within what was asked for
    assert mip["bound"] <= mip["objective"]
    assert mip["bound"] == pytest.approx(lp["objective"], rel=1e-9)
    assert (mip["objective"] - mip["bound"]) / mip["objective"] <= 0.01


def test_covering_infeasible_and_time_limit():
    a, cost, rhs = _wide_cover()
    total = np.asarray(a.sum(axis=1)).ravel()
    bad = rhs.copy()
    bad[2] = total[2] * 1.01
    assert _cover(a, cost, bad, "covering")["status"] == "infeasible"
    r = _cover(a, cost, rhs, "covering", time_limit=0.0)
    assert r["status"] == "time_limit" and r["seconds"] < 5
