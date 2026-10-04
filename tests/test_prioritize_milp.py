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

PY_CASES = [(g, c) for g in ("core", "objectives") for c in case_ids(g) if optimize_cases.CASES[c].py is not None]
ALL = [(g, c) for g in ("core", "objectives", "actions") for c in case_ids(g)]


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


def _assert_reference_optimum(ours: float, ref: float, sense: str) -> None:
    """Our optimum is the reference's (Gurobi, gap 0). In the max objectives the cost only breaks ties
    (0.01 x cost / total cost): Gurobi stops within its tolerances there, at different optima from
    prioritizr and from gurobipy (max_features: 8.998077 and 8.998085), where HiGHS finds a feasible
    8.998091; so ours may be better than the reference by up to 1e-5, never worse."""
    if ours == pytest.approx(ref, rel=1e-9, abs=1e-9):
        return
    better = ours > ref if sense == "max" else ours < ref
    assert better and abs(ours - ref) <= 1e-5 * max(1.0, abs(ref)), (ours, ref)


@pytest.mark.parametrize(("group", "name"), ALL, ids=[c for _, c in ALL])
def test_builtin_highs_same_optimum(group, name):
    m = model(group, name)
    if m.genconpow is not None:
        pytest.skip("general power constraints (Gurobi only)")
    r = _solve_stored(m)
    assert r["status"] == "optimal" and r["has_solution"]
    _assert_reference_optimum(r["objective"], result(group, name)["objective"][0], m.modelsense)
    assert r["x"].shape == (m.obj.size,)


@pytest.mark.parametrize(("group", "name"), PY_CASES, ids=[c for _, c in PY_CASES])
def test_compile_is_prioritizr(group, name):
    """The same model, bit for bit: the matrix in canonical order (by row, then column; prioritizr also
    stores the zero costs and coefficients it pushes, which are no entries here)."""
    ours = optimize_cases.CASES[name].py().compile()
    ref = model(group, name)
    keep = ref.A_x != 0
    order = np.lexsort((ref.A_j[keep], ref.A_i[keep]))
    a = ours.A.tocoo()
    np.testing.assert_array_equal(a.row, ref.A_i[keep][order], err_msg="A_i")
    np.testing.assert_array_equal(a.col, ref.A_j[keep][order], err_msg="A_j")
    np.testing.assert_array_equal(a.data, ref.A_x[keep][order], err_msg="A_x")
    assert ours.A.shape == (len(ref.rhs), len(ref.obj))
    for key in ("obj", "lb", "ub", "rhs"):
        np.testing.assert_array_equal(getattr(ours, key), getattr(ref, key), err_msg=key)
    assert list(ours.sense) == ref.sense
    assert list(ours.vtype) == ref.vtype
    assert ours.modelsense == ref.modelsense
    assert list(ours.row_ids) == ref.row_ids and list(ours.col_ids) == ref.col_ids


@pytest.mark.parametrize(("group", "name"), PY_CASES, ids=[c for _, c in PY_CASES])
def test_solve_reaches_prioritizr_optimum(group, name):
    p = optimize_cases.CASES[name].py()
    sol = p.add_highs_solver(gap=0).solve()
    ref = result(group, name)
    assert sol.status == "optimal"
    _assert_reference_optimum(sol.objective, ref["objective"][0], "max" if p.objective[0].startswith("max") else "min")
    if p.objective[0] == "min_set":
        assert sol.cost == pytest.approx(ref["objective"][0], rel=1e-9)
    if p.objective[0] in ("min_set", "min_penalties"):
        assert sol.coverage["met"].all()
    if p.objective[0] != "min_set":
        assert sol.cost <= p.objective[1] * (1 + 1e-9)


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


def _cover(a, cost, rhs, method, gap=0.0, threads=1, time_limit=math.inf, integer=True, initial=None):
    n = cost.size
    return _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=cost,
        lb=np.zeros(n), ub=np.ones(n), row_lo=rhs, row_hi=np.full(rhs.size, np.inf),
        integer=np.full(n, int(integer), dtype=np.uint8), maximize=False, offset=0.0, gap=gap,
        time_limit=time_limit, threads=threads, presolve="choose", verbose=False, relaxation=False,
        initial=initial, method=method)


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


# ----------------------------------------------------------------------------- exact cell coverage (exactextract)

def _ragged(geoms):
    import shapely

    mp = [g if g.geom_type == "MultiPolygon" else shapely.MultiPolygon([g]) for g in geoms]
    _, xy, (ring, poly, feat) = shapely.to_ragged_array(mp)
    return xy, ring.astype(np.int64), poly.astype(np.int64), feat.astype(np.int64)


def test_exact_coverage_is_exactextract():
    """Covered fractions and coverage-weighted sums as exactextractr 0.10.0, bit for bit."""
    ref = result("exact", "exact_shapes")
    f = optimize_cases.features()
    ny, nx = f.shape[1:]
    grid = (0.0, 0.0, nx * optimize_cases.RES, ny * optimize_cases.RES, optimize_cases.RES, optimize_cases.RES)
    shapes = optimize_cases.shapes()
    for k, g in enumerate(shapes.geometry.values, start=1):
        xy, ring, poly, _ = _ragged([g])
        row0, col0, frac = _core.io_exact_coverage(grid, ny, nx, xy, ring, poly)
        full = np.zeros((ny, nx))
        full[row0:row0 + frac.shape[0], col0:col0 + frac.shape[1]] = frac
        want = np.nan_to_num(ref[f"coverage_{k}"].reshape(ny, nx), nan=0.0)
        np.testing.assert_array_equal(full, want, err_msg=f"shape {k}")
    xy, ring, poly, feat = _ragged(shapes.geometry.values)
    sums = _core.io_exact_sums(np.ascontiguousarray(f.values, dtype=np.float64), grid, xy, ring, poly, feat)
    np.testing.assert_array_equal(sums, ref["sum"])
    one = _core.io_exact_sums(np.ascontiguousarray(f.values, dtype=np.float64), grid, xy, ring, poly, feat,
                              n_threads=1)
    assert np.array_equal(one, sums)


def test_polygon_units():
    """Polygon planning units: the solution is a copy of their table; other CRSs are reprojected."""
    gdf = optimize_cases.pu_polygons_partial()
    sol = optimize_cases.CASES["minset_polygons_partial"].py().solve()
    assert list(sol.selection.columns) == [*gdf.columns, "solution"]
    assert sol.selection.crs == gdf.crs and set(np.unique(sol.values)) <= {0.0, 1.0}
    feats = optimize_cases.features()
    a = ls.prioritize.problem(gdf, feats, cost_column="cost").units.rij
    b = ls.prioritize.problem(gdf.to_crs("EPSG:4326"), feats, cost_column="cost").units.rij
    np.testing.assert_allclose(b.toarray(), a.toarray(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="no cost column"):
        ls.prioritize.problem(gdf, feats, cost_column="price")


def test_h3_units():
    """H3 hexagons over the cost raster: they tile it, so their costs add up to the raster's."""
    cost = optimize_cases.pu_cost()
    hexes = ls.prioritize.h3_units(cost, 9)
    assert hexes.crs == cost.rio.crs and list(hexes.columns) == ["h3", "cost", "geometry"]
    assert hexes["h3"].is_unique and (hexes["cost"] > 0).all()
    assert hexes["cost"].sum() == pytest.approx(float(np.nansum(cost.values)), rel=1e-5)
    sol = ls.prioritize.problem(hexes, optimize_cases.features(), cost_column="cost").add_min_set_objective() \
        .add_relative_targets(0.2).add_binary_decisions().add_highs_solver(gap=0).solve()
    assert sol.status == "optimal" and sol.coverage["met"].all()
    assert list(sol.selection["h3"]) == list(hexes["h3"])
    with pytest.raises(ValueError, match="from 0 to 15"):
        ls.prioritize.h3_units(cost, 16)


def test_raster_features_by_blocks(monkeypatch):
    """The features are read in blocks of rows: block size, dask chunks and threads give the same bits."""
    from landschaft.prioritize import _milp

    cost, feats = optimize_cases.pu_cost(), optimize_cases.features()
    ref = _milp._raster_units(cost, feats, n_threads=1)
    monkeypatch.setattr(_milp, "BLOCK_BYTES", 8 * feats.shape[0] * feats.shape[2] * 2 + 7)  # 2 rows
    for f, threads in ((feats, 4), (feats.chunk({"y": 5, "x": 4}), 3), ([feats[k] for k in range(feats.shape[0])], 2)):
        u = _milp._raster_units(cost, f, n_threads=threads)
        assert u.feature_names == ref.feature_names
        assert np.array_equal(u.totals, ref.totals)
        assert np.array_equal(u.rij.indptr, ref.rij.indptr) and np.array_equal(u.rij.indices, ref.rij.indices)
        assert np.array_equal(u.rij.data, ref.rij.data)


# ----------------------------------------------------------------------------- start solutions

def test_covering_takes_the_start_solution():
    """A feasible start is trimmed and kept when cheaper (here: no time for anything else); an
    infeasible one is ignored."""
    a, cost, rhs = _wide_cover()
    every = np.ones(cost.size)
    r = _cover(a, cost, rhs, "covering", time_limit=0.0, initial=every)
    assert r["status"] == "time_limit" and r["has_solution"]
    x = r["x"]
    assert set(np.unique(x)) <= {0.0, 1.0} and (a @ x >= rhs * (1 - 1e-9)).all()
    assert r["objective"] == pytest.approx(cost @ x) and r["objective"] < cost.sum()
    r = _cover(a, cost, rhs, "covering", time_limit=0.0, initial=np.zeros(cost.size))  # infeasible start
    assert not r["has_solution"]
    with pytest.raises(ValueError, match="one value per column"):
        _cover(a, cost, rhs, "covering", initial=np.ones(3))


def _minset(p, target=0.3):
    return p.add_min_set_objective().add_relative_targets(target).add_binary_decisions()


def test_zonation_start():
    """Starts from Zonation's ranking: feasible, and the same optimum."""
    from landschaft.prioritize import _milp

    cost, feats = optimize_cases.pu_cost(), optimize_cases.features()
    p = _minset(ls.prioritize.problem(cost, feats)).add_highs_solver(gap=0)
    best = p.solve()
    res = ls.prioritize.zonation([feats[k] for k in range(feats.shape[0])], cost=cost)
    m = p.compile()
    x = _milp._start_from_rank(m, _milp._unit_ranks(p.units, res.rank))
    assert set(np.unique(x)) <= {0.0, 1.0} and (m.A @ x >= m.rhs).all()
    # from the top of the ranking down: every unit taken outranks every unit left (ties aside)
    r = res.rank.values.ravel()[p.units.index]
    assert r[x == 1].min() >= r[x == 0].max()
    for start in ("zonation", res, res.rank):
        sol = p.solve(start=start)
        assert sol.status == "optimal" and sol.objective == pytest.approx(best.objective, rel=1e-9)
    # locked units are honoured
    li, lo = optimize_cases.locked_in(), optimize_cases.locked_out()
    q = p.add_locked_in_constraints(li).add_locked_out_constraints(lo)
    mq = q.compile()
    xq = _milp._start_from_rank(mq, _milp._unit_ranks(q.units, res.rank))
    assert (xq[mq.lb > 0] == 1).all() and (xq[mq.ub == 0] == 0).all() and (mq.A @ xq >= mq.rhs).all()
    assert q.solve(start="zonation").objective == pytest.approx(q.solve().objective, rel=1e-9)


def test_zonation_start_polygons_and_errors():
    hexes = ls.prioritize.h3_units(optimize_cases.pu_cost(), 9)
    p = _minset(ls.prioritize.problem(hexes, optimize_cases.features(), cost_column="cost"), 0.2)
    p = p.add_highs_solver(gap=0)
    assert p.solve(start="zonation").objective == pytest.approx(p.solve().objective, rel=1e-9)
    t = _minset(optimize_cases.CASES["minset_table"].py().add_relative_targets(0.3))
    with pytest.raises(ValueError, match="needs feature rasters"):
        t.solve(start="zonation")
    with pytest.raises(ValueError, match="unknown start"):
        p.solve(start="gurobi")
    with pytest.raises(ValueError, match="one value per planning unit"):
        p.solve(start=[1, 0])


def test_covering_keeps_the_time_limit_on_wide_models():
    """Gap 0 on 3e5 units: HiGHS's MIP root would ignore a 1 s limit for seconds; it is not called."""
    a, cost, rhs = _wide_cover(n=300_000, m=30, seed=5)
    r = _cover(a, cost, rhs, "covering", gap=0.0, threads=4, time_limit=1.0)
    assert r["has_solution"] and r["seconds"] < 1.5
    assert r["status"] in ("optimal", "time_limit") and r["gap"] < 1e-3


# ----------------------------------------------------------------------------- objectives, targets, weights

def test_ignored_targets_and_weights_warn():
    p = ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features())
    with pytest.warns(UserWarning, match="targets are ignored"):
        p.add_max_utility_objective(2500).add_relative_targets(0.3).compile()
    with pytest.warns(UserWarning, match="weights are ignored"):
        p.add_min_set_objective().add_relative_targets(0.3).add_feature_weights([1] * 5).compile()


def test_objective_and_target_errors():
    p = ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features())
    for bad in (0, -1, np.inf, [1, 2]):
        with pytest.raises(ValueError, match="budget"):
            p.add_min_shortfall_objective(bad)
    with pytest.raises(ValueError, match="needs targets"):
        p.add_max_features_objective(100).compile()
    with pytest.raises(ValueError, match="unknown features"):
        p.add_min_set_objective().add_manual_targets(
            {"feature": ["f9"], "type": ["absolute"], "target": [1.0]}).compile()
    with pytest.raises(ValueError, match="'type' column"):
        p.add_manual_targets({"feature": ["f1"], "target": [1.0]})
    with pytest.raises(ValueError, match="senses"):
        p.add_min_set_objective().add_manual_targets(
            {"feature": ["f1"], "type": ["absolute"], "target": [1.0], "sense": [">"]}).compile()
    with pytest.raises(ValueError, match="bound targets"):
        p.add_loglinear_targets(3, 1.5, 300, 0.2)
    with pytest.raises(ValueError, match="cap"):
        p.add_loglinear_targets(3, 0.6, 300, 0.2, cap_amount=100)
    with pytest.raises(ValueError, match="one value per target"):
        p.add_max_features_objective(2500).add_relative_targets(0.3).add_feature_weights([1, 2]).compile()
    with pytest.raises(ValueError, match=">= 0"):
        p.add_feature_weights([1, -1, 1, 1, 1])


def test_coverage_by_target():
    """One row per target, with its sense; without targets, every feature's amount."""
    sol = optimize_cases.CASES["targets_manual"].py().add_highs_solver(gap=0).solve()
    c = sol.coverage
    assert list(c["feature"]) == ["f1", "f2", "f4", "f4", "f5"]
    assert list(c["sense"]) == [">=", "<=", ">=", "<=", ">="] and c["met"].all()
    le = c["sense"] == "<="
    assert (c.loc[le, "absolute_held"] <= c.loc[le, "absolute_target"] + 1e-9).all()
    sol = optimize_cases.CASES["max_utility"].py().solve()
    assert list(sol.coverage.columns) == ["feature", "total_amount", "absolute_held", "relative_held"]
    assert sol.cost <= 2500


# ----------------------------------------------------------------------------- wide models with a budget

def _wide_problem(n=200, nf=12, seed=4):
    """A raster of n x n units (wider than the sifting's working set) with gaussian features."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    f = np.stack([np.exp(-((xx - rng.uniform()) ** 2 + (yy - rng.uniform()) ** 2) / rng.uniform(0.005, 0.05))
                  for _ in range(nf)])
    f[f < 0.05] = 0
    cost = optimize_cases.grid(rng.uniform(1, 10, (n, n)))
    feats = optimize_cases.grid(f, names=[f"s{k}" for k in range(nf)])
    return ls.prioritize.problem(cost, feats), float(np.sum(cost.values))


@pytest.mark.parametrize("objective", ["min_shortfall", "min_largest_shortfall", "max_utility", "max_features",
                                       "max_cover"])
def test_wide_engine_same_bits_and_within_budget(objective):
    p, total = _wide_problem()
    budget = 0.1 * total if objective != "max_cover" else 0.002 * total
    p = getattr(p, f"add_{objective}_objective")(budget)
    if objective not in ("max_utility", "max_cover"):
        p = p.add_relative_targets(0.3)
    a = p.add_highs_solver(gap=0.01, n_threads=1).solve()
    b = p.add_highs_solver(gap=0.01, n_threads=4).solve()
    assert np.array_equal(a.values, b.values) and a.objective == b.objective and a.bound == b.bound
    assert a.status == "optimal" and a.gap <= 0.01 and a.cost <= budget * (1 + 1e-9)
    assert set(np.unique(a.values)) <= {0.0, 1.0}


def test_wide_engine_lp_is_highs_lp():
    """Proportion decisions: the sifted LP is HiGHS's LP on the whole model."""
    p, total = _wide_problem(n=120)
    p = p.add_min_shortfall_objective(0.1 * total).add_relative_targets(0.3).add_proportion_decisions()
    sol = p.add_highs_solver(gap=0).solve()
    m = p.compile()
    lo, hi = m.row_bounds()
    a = m.A
    r = _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=m.obj, lb=m.lb,
        ub=m.ub, row_lo=lo, row_hi=hi, integer=np.zeros(m.ncol, np.uint8), maximize=False, offset=0.0, gap=0.0,
        time_limit=math.inf, threads=1, presolve="choose", verbose=False, relaxation=False, initial=None)
    assert sol.objective == pytest.approx(r["objective"], rel=1e-7)


def test_wide_engine_keeps_the_time_limit():
    p, total = _wide_problem(n=500, nf=30)
    sol = p.add_max_features_objective(0.1 * total).add_relative_targets(0.3).add_highs_solver(
        gap=0.0, time_limit=2.0).solve()
    assert sol.seconds < 3.0 and sol.status in ("optimal", "time_limit")
