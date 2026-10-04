# SPDX-License-Identifier: GPL-3.0-or-later
"""The anneal solver (Marxan's simulated annealing on a compiled minimum-set
model): the conversion is exact, every target is met, the solutions are near
prioritizr's optima (tests/optimize_cases.py) and the bound is below them."""

from __future__ import annotations

import numpy as np
import optimize_cases
import pandas as pd
import pytest
from optimize_reference import result

import landschaft as ls
from landschaft.prioritize._milp_anneal import marxan_model
from landschaft.prioritize._milp_spatial import boundary_matrix

ANNEALED = ("boundary_raster", "boundary_polygons", "boundary_matrix", "connectivity", "asym_connectivity",
            "linear_penalties")


def _full(m, n, x, mm):
    full = np.zeros(m.ncol)
    full[:n] = x
    full[n:] = x[mm["pi"]] * x[mm["pj"]]
    return full


@pytest.mark.parametrize("name", ANNEALED)
def test_conversion_is_exact(name):
    """Marxan's score (cost + boundary, BLM 1) of any selection is the compiled model's objective."""
    p = optimize_cases.CASES[name].py()
    m = p.compile()
    n = p.units.n
    mm = marxan_model(m, n)
    rng = np.random.default_rng(5)
    for _ in range(5):
        x = (rng.uniform(size=n) < 0.4).astype(np.float64)
        cut = x[mm["pi"]] + x[mm["pj"]] - 2 * x[mm["pi"]] * x[mm["pj"]]
        marxan = mm["cost"] @ x + mm["bound"] @ cut
        assert marxan == pytest.approx(m.obj @ _full(m, n, x, mm), rel=1e-12, abs=1e-9)
    # the features are the target rows
    targets = m.rhs[np.asarray(m.row_ids) == "spp_target"]
    np.testing.assert_array_equal(mm["target"], targets[targets > 0])


def test_boundary_penalty_is_marxan_blm():
    """prioritizr's boundary penalty is Marxan's: BLM = penalty, fixed boundary = exposed edge x edge
    factor, scored by ls.prioritize.marxan on the optimum (every unit locked to it)."""
    p = optimize_cases.CASES["boundary_polygons"].py()
    penalty, edge_factor = 0.2, 0.75
    sol = p.add_highs_solver(gap=0).solve()
    bm = boundary_matrix(p.units).tocoo()
    n = p.units.n
    total = np.asarray(boundary_matrix(p.units).diagonal())
    shared = np.asarray(boundary_matrix(p.units).sum(axis=1)).ravel() - total
    upper = bm.row < bm.col
    bound = pd.DataFrame({"id1": np.concatenate([bm.row[upper], np.arange(n)]),
                          "id2": np.concatenate([bm.col[upper], np.arange(n)]),
                          "boundary": np.concatenate([bm.data[upper], (total - shared) * edge_factor])})
    pu = pd.DataFrame({"id": np.arange(n), "cost": p.units.cost,
                       "status": np.where(sol.values > 0.5, 2, 3)})
    spec = pd.DataFrame({"id": [1], "target": [0.0]})
    puvspr = pd.DataFrame({"species": [1], "pu": [0], "amount": [1.0]})
    r = ls.prioritize.marxan(pu, spec, puvspr, bound, blm=penalty, num_reps=1, num_iters=0, start_temp=1)
    assert r.summary["Score"].iloc[0] == pytest.approx(sol.objective, rel=1e-12)


@pytest.mark.parametrize("name", ANNEALED)
def test_anneal_near_prioritizr_optimum(name):
    p = optimize_cases.CASES[name].py()
    sol = p.add_anneal_solver(seed=1, n_threads=2).solve()
    opt = result("spatial", name)["objective"][0]
    assert sol.coverage["met"].all()
    assert sol.solver == "anneal" and sol.status in ("optimal", "heuristic")
    assert opt - 1e-6 * abs(opt) <= sol.objective <= opt * 1.03
    assert sol.bound <= opt + 1e-6 * abs(opt)
    m = p.compile()
    mm = marxan_model(m, p.units.n)
    assert sol.objective == pytest.approx(m.obj @ _full(m, p.units.n, sol.values, mm), rel=1e-12)


def test_anneal_same_on_any_threads():
    p = optimize_cases.CASES["boundary_matrix"].py()
    a = p.add_anneal_solver(seed=7, num_reps=6, n_threads=1).solve()
    b = p.add_anneal_solver(seed=7, num_reps=6, n_threads=4).solve()
    np.testing.assert_array_equal(a.values, b.values)
    assert a.objective == b.objective


def test_anneal_locks_start_and_rows():
    p = optimize_cases.CASES["boundary_matrix"].py()
    n = p.units.n
    locked_in, locked_out = np.arange(0, 10), np.arange(n - 10, n)
    q = p.add_locked_in_constraints(locked_in).add_locked_out_constraints(locked_out)
    q = q.add_linear_constraints(30, ">=", data="f4")  # a '>=' row on the units: one more feature
    sol = q.add_anneal_solver(seed=2).solve(start=np.ones(n))
    assert (sol.values[locked_in] == 1).all() and (sol.values[locked_out] == 0).all()
    assert sol.coverage["met"].all()
    f4 = p.units.rij[p.units.feature_names.index("f4")]
    assert f4 @ sol.values >= 30 - 1e-9


def test_anneal_time_limit():
    p = optimize_cases.CASES["boundary_raster"].py()
    import time

    t = time.perf_counter()
    sol = p.add_anneal_solver(time_limit=1.0, num_reps=4, n_threads=2).solve()
    assert time.perf_counter() - t < 5
    assert sol.coverage["met"].all()


def test_anneal_unsupported_and_infeasible():
    with pytest.raises(ValueError, match="maximises"):
        optimize_cases.CASES["boundary_raster_edge_max"].py().add_anneal_solver().solve()
    with pytest.raises(ValueError, match="'<=' or '=' rows"):
        optimize_cases.CASES["linear_constraints"].py().add_anneal_solver().solve()
    with pytest.raises(ValueError, match="variables other than|rows on other variables"):
        optimize_cases.PY_SPATIAL["contiguity_matrix"]().add_anneal_solver().solve()
    with pytest.raises(ValueError, match="proportion"):
        optimize_cases.CASES["boundary_matrix"].py().add_proportion_decisions().add_anneal_solver().solve()
    with pytest.raises(ValueError, match="portfolio"):
        optimize_cases.CASES["boundary_matrix"].py().add_cuts_portfolio(2).add_anneal_solver().solve()
    p = optimize_cases.CASES["boundary_matrix"].py()
    with pytest.raises(RuntimeError, match="infeasible"):
        p.add_locked_out_constraints(np.arange(p.units.n)).add_anneal_solver().solve()
