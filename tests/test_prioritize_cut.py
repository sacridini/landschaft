# SPDX-License-Identifier: GPL-3.0-or-later
"""The cut engine (opt/cut.cpp): Boykov-Kolmogorov minimum cuts against scipy's
maximum flow, the Lagrangian bound against the linear relaxation, and minimum
sets with boundary penalties solved through the HiGHS solver."""

from __future__ import annotations

import time

import numpy as np
import optimize_cases
import pytest
import scipy.sparse as sp
import xarray as xr
from optimize_reference import result
from scipy.sparse.csgraph import maximum_flow

import landschaft as ls
from landschaft import _core
from landschaft.prioritize._milp_anneal import cut_applies, solve_cut


def _cut_value(i, j, cij, cji, s, t, side):
    side = side.astype(bool)
    return s[~side].sum() + t[side].sum() + cij[side[i] & ~side[j]].sum() + cji[side[j] & ~side[i]].sum()


def test_maxflow_is_scipy():
    rng = np.random.default_rng(1)
    for trial in range(150):
        n = int(rng.integers(2, 50))
        m = int(rng.integers(1, 4 * n))
        i, j = rng.integers(0, n, m), rng.integers(0, n, m)
        i, j = i[i != j], j[i != j]
        cij = rng.integers(0, 20, i.size).astype(float)
        cji = rng.integers(0, 20, i.size).astype(float) if trial % 2 else cij.copy()
        s = np.where(rng.uniform(size=n) < 0.4, rng.integers(0, 30, n), 0).astype(float)
        t = np.where(rng.uniform(size=n) < 0.4, rng.integers(0, 30, n), 0).astype(float)
        f, side = _core.opt_maxflow(n, i.astype(np.int32), j.astype(np.int32), cij, cji, s, t)
        g = sp.csr_matrix((np.concatenate([cij, cji, s, t]).astype(np.int64),
                           (np.concatenate([i, j, np.full(n, n), np.arange(n)]),
                            np.concatenate([j, i, np.arange(n), np.full(n, n + 1)]))), shape=(n + 2, n + 2))
        g.sum_duplicates()
        ref = maximum_flow(g, n, n + 1).flow_value
        assert f == pytest.approx(ref, abs=1e-9)
        assert _cut_value(i, j, cij, cji, s, t, side) == pytest.approx(ref, abs=1e-9)  # the minimum cut


@pytest.mark.parametrize("name", ["boundary_raster", "boundary_matrix", "connectivity", "asym_connectivity"])
def test_lagrangian_bound_is_the_relaxation(name):
    """The pair terms are submodular, so the subproblem is integral and max L is the LP's optimum."""
    p = optimize_cases.CASES[name].py()
    m = p.compile()
    r = solve_cut(m, p.units.n, cut_applies(m, p.units.n), 0.0, 60, False)
    lp = p.add_proportion_decisions().add_highs_solver(gap=0).solve()
    assert r["bound"] == pytest.approx(lp.objective, rel=1e-6)
    opt = result("spatial", name)["objective"][0]
    assert r["bound"] <= opt * (1 + 1e-9) and r["objective"] >= opt * (1 - 1e-9)


def test_cut_applies_only_to_submodular_min_sets():
    """Boundaries, symmetric connectivity and, in a minimum set, asymmetric connectivity too (its pair terms
    reward the pairs selected together: boundaries >= 0 in Marxan's form)."""
    for name in ("boundary_raster", "connectivity", "asym_connectivity"):
        p = optimize_cases.CASES[name].py()
        assert cut_applies(p.compile(), p.units.n) is not None, name
    for name in ("linear_penalties", "boundary_raster_edge_max", "linear_constraints"):
        p = optimize_cases.CASES[name].py()
        assert cut_applies(p.compile(), p.units.n) is None, name


def _synthetic(side: int, penalty: float = 0.5):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:side, 0:side] / side
    f = np.stack([np.exp(-((xx - rng.uniform()) ** 2 + (yy - rng.uniform()) ** 2) / rng.uniform(0.005, 0.05))
                  for _ in range(12)])
    f[f < 0.05] = 0
    coords = {"y": np.arange(side)[::-1] + 0.5, "x": np.arange(side) + 0.5}
    feats = xr.DataArray(f, dims=("band", "y", "x"), coords={**coords, "band": [f"f{k}" for k in range(12)]})
    cost = xr.DataArray(rng.uniform(1, 10, (side, side)), dims=("y", "x"), coords=coords)
    return (ls.prioritize.problem(cost, feats).add_min_set_objective().add_relative_targets(0.3)
            .add_boundary_penalties(penalty).add_binary_decisions())


def test_boundary_min_set_through_highs():
    """A boundary problem beyond HiGHS's MIP (phase 0: 10^5 units without a solution in 744 s), at a
    proven 1% gap in seconds."""
    p = _synthetic(200)  # 4 x 10^4 units, 1.2 x 10^5 columns
    t = time.perf_counter()
    sol = p.add_highs_solver(gap=0.01, time_limit=60).solve()
    assert time.perf_counter() - t < 30
    assert sol.status == "optimal" and sol.gap <= 0.01 and sol.bound <= sol.objective
    assert sol.coverage["met"].all()
    m = p.compile()
    mm = cut_applies(m, p.units.n)
    x = sol.values
    full = np.concatenate([x, x[mm["pi"]] * x[mm["pj"]]])
    assert sol.objective == pytest.approx(m.obj @ full, rel=1e-12)


def test_parallel_repairs_give_the_same_result():
    """The repairs run on other threads, but the bundle sees them a fixed number of evaluations late: the
    same selection, objective and bound on any number of threads."""
    p = _synthetic(120)
    m = p.compile()
    mm = cut_applies(m, p.units.n)
    runs = [solve_cut(m, p.units.n, mm, 0.002, float("inf"), False, None, threads) for threads in (1, 3, 8)]
    for r in runs[1:]:
        assert np.array_equal(r["x"], runs[0]["x"])
        assert r["objective"] == runs[0]["objective"] and r["bound"] == runs[0]["bound"]
        assert r["evaluations"] == runs[0]["evaluations"]


def test_small_boundary_goes_on_to_highs():
    """Below the gap asked for, small models go on to HiGHS's MIP from the cut engine's solution."""
    p = optimize_cases.CASES["boundary_polygons"].py()
    sol = p.add_highs_solver(gap=0).solve()
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(result("spatial", "boundary_polygons")["objective"][0], rel=1e-9)
