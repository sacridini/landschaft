# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-action planning against prioriactions 0.5.0 (tests/optimize_cases.py, group
"actions"): the compiled model bit for bit, the optimum on Gurobi, and the
solution's figures (getCost, getSolutionBenefit, getPotentialBenefit,
getConnectivityPenalty, getActions) on prioriactions's own solution."""

from __future__ import annotations

import warnings

import numpy as np
import optimize_cases
import pandas as pd
import pytest
import scipy.sparse as sp
from optimize_reference import case_ids, model, result

import landschaft as ls

CASES = case_ids("actions")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _problem(name):
    return optimize_cases.PY_ACTIONS[name]()


@pytest.mark.parametrize("name", CASES)
def test_compile_is_prioriactions(name):
    ref = model("actions", name)
    r = result("actions", name)
    ours = _problem(name).compile()
    shape = (int(r["model/n_row"][0]), int(r["model/n_col"][0]))
    a = sp.coo_matrix((ref.A_x, (ref.A_i, ref.A_j)), shape=shape).tocsr()
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    o = ours.A.copy()
    o.eliminate_zeros()
    o.sort_indices()
    assert o.shape == a.shape
    np.testing.assert_array_equal(o.indptr, a.indptr)
    np.testing.assert_array_equal(o.indices, a.indices)
    np.testing.assert_array_equal(o.data, a.data)
    for key in ("obj", "lb", "ub", "rhs"):
        np.testing.assert_array_equal(getattr(ours, key), getattr(ref, key), err_msg=key)
    assert list(ours.sense) == [s if s != "==" else "=" for s in ref.sense]
    assert list(ours.vtype) == list(ref.vtype)
    assert ours.modelsense == ref.modelsense


@pytest.mark.parametrize("name", CASES)
def test_optimum_is_prioriactions(name):
    """The optimum on Gurobi (gap 0); with curve 3, Gurobi's piecewise-linear power function as ours."""
    sol = _problem(name).add_highs_solver(gap=0).solve()
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(result("actions", name)["objective"][0], rel=1e-9, abs=1e-9)


def _floats(v):
    return np.asarray([float(x) for x in v]) if isinstance(v, list) else np.asarray(v, dtype=np.float64)


@pytest.mark.parametrize("name", CASES)
def test_figures_are_prioriactions(name):
    """getCost, getSolutionBenefit (total and local), getPotentialBenefit, getConnectivityPenalty and
    getActions on prioriactions's solution."""
    r = result("actions", name)
    p = _problem(name)
    s = p._solution(np.asarray(r["solution"], dtype=np.float64))
    cost = s.cost()
    for col in cost.columns:
        assert cost[col].iloc[0] == _floats(r[f"cost/{col}"])[0], col
    b = s.benefit()
    for col in ("feature", "benefit.conservation", "benefit.recovery", "benefit.total"):
        np.testing.assert_array_equal(b[col].to_numpy(), r[f"benefit/{col}"], err_msg=col)
    loc = s.local_benefit
    for col in ("pu", "feature", "benefit.conservation", "benefit.recovery", "benefit.total"):
        np.testing.assert_array_equal(loc[col].to_numpy(), r[f"local/{col}"], err_msg=col)
    pot = p.data.potential_benefit()
    for col in pot.columns:
        np.testing.assert_array_equal(pot[col].to_numpy(), r[f"potential/{col}"], err_msg=col)
    conn = s.connectivity_penalty()
    for col in conn.columns:
        assert conn[col].iloc[0] == _floats(r[f"connectivity/{col}"])[0], col
    wide = s.action_summary()
    ref_cols = [k.split("/", 1)[1] for k in r if k.startswith("actions/") and k != "actions/solution_name"]
    assert sorted(wide.columns) == sorted(ref_cols)
    for col in wide.columns:
        np.testing.assert_array_equal(wide[col].to_numpy(), r[f"actions/{col}"], err_msg=col)
    long = s.action_summary("long")
    np.testing.assert_array_equal(long["pu"].to_numpy(), r["actions_long/pu"])
    assert list(long["action"]) == r["actions_long/action"]
    np.testing.assert_array_equal(long["solution"].to_numpy(), r["actions_long/solution"])


def test_solution_is_consistent():
    p = _problem("actions_mincost")
    s = p.add_highs_solver(gap=0).solve()
    cost = s.cost()
    assert cost.sum(axis=1).iloc[0] == pytest.approx(s.objective, rel=1e-9)  # no boundary: the objective
    b = s.benefit()
    f = p.data.features
    assert (b["benefit.recovery"].to_numpy() >= f["target_recovery"].to_numpy() - 1e-6).all()
    at = s.action_table()
    taken = at.loc[at["solution"] > 0, "pu"].unique()
    mon = s.monitoring().set_index("id")["solution"]
    assert (mon.loc[taken] == 1).all()  # an action needs its unit monitored


def test_inputs_checked_and_completed():
    d = optimize_cases.pa_python_data(sensitivity=None)
    # every feature harmed by every threat, deltas by default: 0, the threat's highest amount, 0, 1
    assert len(d.sensitivity) == len(d.features) * len(d.threats)
    top = d.dist_threats.groupby("threat")["amount"].max()
    np.testing.assert_array_equal(d.sensitivity["delta2"].to_numpy(), d.sensitivity["threat"].map(top).to_numpy())
    pu = optimize_cases.pa_pu()
    feats, thr, dt = optimize_cases.pa_features(), optimize_cases.pa_threats(), optimize_cases.pa_dist_threats()
    df = optimize_cases.pa_dist_features()
    with pytest.raises(ValueError, match="unique"):
        ls.prioritize.actions_data(pd.concat([pu, pu.head(1)]), feats, df, thr, dt)
    with pytest.raises(ValueError, match="duplicate"):
        ls.prioritize.actions_data(pu, feats, pd.concat([df, df.head(1)]), thr, dt)
    neg = pu.copy()
    neg.loc[0, "monitoring_cost"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        ls.prioritize.actions_data(neg, feats, df, thr, dt)
    with pytest.raises(ValueError, match="minimizeCosts"):
        d.problem("cheapest")


def test_no_action_status_keeps_unit_status():
    """prioriactions resets the units' status when dist_threats has none (a bug); here the actions are free
    and the locked-out units' actions locked out."""
    dt = optimize_cases.pa_dist_threats().drop(columns="status")
    d = optimize_cases.pa_python_data(dist_threats=dt)
    np.testing.assert_array_equal(d.pu["status"].to_numpy(), optimize_cases.pa_pu()["status"].to_numpy())
    out = d.pu.loc[d.pu["status"] == 3, "id"]
    assert (d.dist_threats.loc[d.dist_threats["pu"].isin(out), "status"] == 3).all()
    assert (d.dist_threats.loc[~d.dist_threats["pu"].isin(out), "status"] == 0).all()


def test_targets_above_the_maximum_are_capped():
    f = optimize_cases.pa_features()
    f.loc[0, "target_recovery"] = 1e6
    d = optimize_cases.pa_python_data(features=f)
    with pytest.warns(UserWarning, match="Infeasible model"):
        p = d.problem("minimizeCosts")
    top = d.potential_benefit()["maximum.recovery.benefit"].iloc[0]
    assert p.data.features["target_recovery"].iloc[0] == top - 0.1


# ----------------------------------------------------------------------------- the closure engine

def _grid_data(side=12, seed=3, boundary=False, blm_actions=0.0, locks=False):
    """A random multi-action problem on a side x side grid: 5 features, 3 threats."""
    rng = np.random.default_rng(seed)
    n = side * side
    ids = np.arange(1.0, n + 1)
    feat = rng.random((5, n)) < 0.35
    thr = rng.random((3, n)) < 0.45
    fp, fk = np.nonzero(feat.T)
    tp, tk = np.nonzero(thr.T)
    status = np.zeros(n)
    tstatus = np.zeros(tp.size)
    if locks:
        status[rng.choice(n, 4, replace=False)] = 3
        tstatus[rng.choice(tp.size, 3, replace=False)] = 2
    recoverable = (feat & thr.any(axis=0)).sum(axis=1)
    bnd = None
    if boundary:
        g = ids.reshape(side, side)
        a = np.concatenate([g[:, :-1].ravel(), g[:-1, :].ravel()])
        b = np.concatenate([g[:, 1:].ravel(), g[1:, :].ravel()])
        bnd = pd.DataFrame({"id1": a, "id2": b, "boundary": np.ones(a.size)})
    return ls.prioritize.actions_data(
        pd.DataFrame({"id": ids, "monitoring_cost": rng.integers(1, 4, n).astype(float), "status": status}),
        pd.DataFrame({"id": np.arange(1.0, 6.0), "target_recovery": np.floor(0.3 * recoverable),
                      "target_conservation": np.floor(0.2 * (feat & ~thr.any(axis=0)).sum(axis=1))}),
        pd.DataFrame({"pu": fp + 1.0, "feature": fk + 1.0, "amount": rng.integers(1, 3, fp.size).astype(float)}),
        pd.DataFrame({"id": [1.0, 2.0, 3.0], "blm_actions": [blm_actions] * 3}),
        pd.DataFrame({"pu": tp + 1.0, "threat": tk + 1.0, "amount": rng.random(tp.size).round(2) + 0.01,
                      "action_cost": rng.integers(1, 6, tp.size).astype(float), "status": tstatus}),
        None, bnd)


ENGINE_CASES = {
    "mincost": lambda: _grid_data().problem("minimizeCosts"),
    "mincost_locks": lambda: _grid_data(locks=True).problem("minimizeCosts"),
    "mincost_blm": lambda: _grid_data(boundary=True, blm_actions=0.5).problem("minimizeCosts", blm=1.0),
    "maxbenefit": lambda: _grid_data().problem("maximizeBenefits", budget=80),
    "maxbenefit_blm": lambda: _grid_data(boundary=True, blm_actions=0.25).problem("maximizeBenefits", budget=80,
                                                                                    blm=0.5),
}


@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_engine_finds_the_optimum(name):
    """The closure engine (then HiGHS from its solution) reaches the optimum of HiGHS alone at gap 0."""
    p = ENGINE_CASES[name]()
    ours = p.add_highs_solver(gap=0).solve()
    alone = p.add_highs_solver(gap=0, engine=False).solve()
    assert ours.status == alone.status == "optimal"
    assert ours.objective == pytest.approx(alone.objective, rel=1e-9, abs=1e-9)
    assert ours.bound == pytest.approx(ours.objective, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_engine_solutions_are_feasible(name):
    """At a loose gap the engine's own solution: targets or budget met, actions in monitored units, locks
    kept, objective as the figures add up and within the gap of its bound."""
    p = ENGINE_CASES[name]()
    s = p.add_highs_solver(gap=0.05).solve()
    assert s.status == "optimal" and s.gap <= 0.05
    d = p.data
    at = s.action_table()
    taken = at.loc[at["solution"] > 0, "pu"].unique()
    assert (s.monitoring().set_index("id")["solution"].loc[taken] == 1).all()
    assert (s.units[d.pu["status"].to_numpy() == 3] == 0).all()
    assert (s.actions[d.dist_threats["status"].to_numpy() == 2] == 1).all()
    cu, ca = s.cost().iloc[0, 0], s.cost().iloc[0, 1:].sum()
    if p.model == "minimizeCosts":
        b = s.benefit()
        f = d.features
        assert (b["benefit.recovery"].to_numpy() >= f["target_recovery"].to_numpy() - 1e-3).all()
        assert (b["benefit.conservation"].to_numpy() >= f["target_conservation"].to_numpy() - 1e-3).all()
        if p.blm == 0 and (d.threats["blm_actions"] == 0).all():
            assert cu + ca == pytest.approx(s.objective, rel=1e-9)
    else:
        assert cu + ca <= p.budget + 1e-6
    lo, hi = sorted([s.objective, s.bound])
    assert (hi - lo) / max(abs(s.objective), 1e-10) <= 0.05 + 1e-12


def test_engine_starts_from_a_solution():
    p = ENGINE_CASES["mincost"]().add_highs_solver(gap=0)
    first = p.solve()
    again = p.solve(start=first)
    assert again.objective == pytest.approx(first.objective)
    with pytest.raises(ValueError, match="start needs"):
        p.solve(start=np.zeros(3))


def test_engine_keeps_a_time_limit():
    import time

    p = _grid_data(side=40, seed=7, boundary=True, blm_actions=0.5).problem("minimizeCosts", blm=1.0)
    t0 = time.perf_counter()
    s = p.add_highs_solver(gap=0, time_limit=2).solve()
    assert time.perf_counter() - t0 < 10
    assert s.status in ("optimal", "time_limit")
    assert np.isfinite(s.objective) and s.bound <= s.objective + 1e-9


# ----------------------------------------------------------------------------- raster inputs

def _rasters(seed=11, ny=9, nx=8):
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 4, (ny, nx)).astype(float)
    cost[0, :3] = np.nan  # not planning units
    feats = np.where(rng.random((3, ny, nx)) < 0.4, rng.integers(1, 4, (3, ny, nx)), 0).astype(float)
    feats[1, 4, 4] = np.nan  # NaN = 0
    thr = np.where(rng.random((2, ny, nx)) < 0.5, rng.random((2, ny, nx)).round(2) + 0.01, 0.0)
    acost = rng.integers(1, 6, (ny, nx)).astype(float)
    return (optimize_cases.grid(cost), optimize_cases.grid(feats, names=["sp1", "sp2", "sp3"]),
            optimize_cases.grid(thr, names=["fire", "cattle"]), optimize_cases.grid(acost))


def _tables_of(cost, feats, thr, acost, fire_cost):
    """The same problem as tables, by hand: units = cells with a cost, in cell order."""
    c = cost.values.ravel()
    cells = np.flatnonzero(np.isfinite(c))
    pu = pd.DataFrame({"id": np.arange(1.0, cells.size + 1), "monitoring_cost": c[cells]})
    f = np.nan_to_num(feats.values.reshape(3, -1)[:, cells])
    fk, fp = np.nonzero(f)
    df = pd.DataFrame({"pu": fp + 1.0, "feature": fk + 1.0, "amount": f[fk, fp]})
    t = thr.values.reshape(2, -1)[:, cells]
    tk, tp = np.nonzero(t)
    cost_t = np.where(tk == 0, fire_cost, acost.values.ravel()[cells][tp])
    dt = pd.DataFrame({"pu": tp + 1.0, "threat": tk + 1.0, "amount": t[tk, tp], "action_cost": cost_t,
                       "status": 0.0}).sort_values(["pu", "threat"]).reset_index(drop=True)
    return pu, df, dt


def test_raster_inputs_are_the_tables():
    cost, feats, thr, acost = _rasters()
    d = ls.prioritize.actions_raster_data(cost, feats, thr, {"fire": 2.0, "cattle": acost},
                                          target_recovery=[3, 2, 4], target_conservation=1, relative=False)
    pu, df, dt = _tables_of(cost, feats, thr, acost, 2.0)
    fe = pd.DataFrame({"id": [1.0, 2.0, 3.0], "name": ["sp1", "sp2", "sp3"], "target_recovery": [3.0, 2.0, 4.0],
                       "target_conservation": 1.0})
    th = pd.DataFrame({"id": [1.0, 2.0], "name": ["fire", "cattle"], "blm_actions": 0.0})
    ref = ls.prioritize.actions_data(pu, fe, df, th, dt)
    a, b = d.problem("minimizeCosts").compile(), ref.problem("minimizeCosts").compile()
    assert (a.A != b.A).nnz == 0
    for key in ("obj", "rhs", "lb", "ub"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    assert list(d.features["name"]) == ["sp1", "sp2", "sp3"] and list(d.threats["name"]) == ["fire", "cattle"]


def test_raster_relative_targets():
    cost, feats, thr, acost = _rasters()
    d = ls.prioritize.actions_raster_data(cost, feats, thr, 1.0, target_recovery=0.5, target_conservation=0.25)
    pu, df, dt = _tables_of(cost, feats, thr, acost, 1.0)
    dt["action_cost"] = 1.0
    threatened = df["pu"].isin(dt["pu"])  # every threat harms every feature
    rec = df[threatened].groupby("feature")["amount"].sum().reindex([1.0, 2.0, 3.0], fill_value=0).to_numpy()
    con = df[~threatened].groupby("feature")["amount"].sum().reindex([1.0, 2.0, 3.0], fill_value=0).to_numpy()
    np.testing.assert_allclose(d.features["target_recovery"].to_numpy(), 0.5 * rec)
    np.testing.assert_allclose(d.features["target_conservation"].to_numpy(), 0.25 * con)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ls.prioritize.actions_raster_data(cost, feats, thr, 1.0, target_recovery=30)


def test_raster_solution_maps():
    cost, feats, thr, acost = _rasters()
    d = ls.prioritize.actions_raster_data(cost, feats, thr, acost, target_recovery=0.4, boundary=True,
                                          blm_actions=[0.5, 0.0])
    # the boundary: every pair of edge-sharing units once, and the units' perimeters
    b = d.boundary
    off = b[b["id1"] != b["id2"]]
    assert len(off) == 9 * 7 + 8 * 8 - 6  # 4-neighbour edges, less the 6 touching the 3 cells without a cost
    s = d.problem("minimizeCosts", blm=0.1).add_highs_solver(gap=0).solve()
    m = s.maps()
    assert list(m["band"].values) == ["monitoring", "fire", "cattle"]
    mon = m.sel(band="monitoring").values
    assert np.isnan(mon[0, :3]).all() and np.isfinite(mon.ravel()[np.isfinite(cost.values.ravel())]).all()
    acted = (np.nan_to_num(m.sel(band="fire").values) + np.nan_to_num(m.sel(band="cattle").values)) > 0
    assert (mon[acted] == 1).all()  # an action needs its cell monitored
    assert np.nansum(m.sel(band=["fire", "cattle"]).values) == s.actions.sum()
    assert np.nansum(mon) == s.units.sum()


def test_raster_inputs_checked():
    cost, feats, thr, acost = _rasters()
    holes = acost.copy(data=np.where(thr.values[1] > 0, np.nan, acost.values))
    with pytest.raises(ValueError, match="no value in cells where the threat occurs"):
        ls.prioritize.actions_raster_data(cost, feats, thr, [1.0, holes])
    with pytest.raises(ValueError, match="no value for"):
        ls.prioritize.actions_raster_data(cost, feats, thr, {"fire": 1.0})
    with pytest.raises(ValueError, match="maps need"):
        _problem("actions_mincost").add_highs_solver(gap=0).solve().maps()
    locked = cost.copy(data=np.zeros(cost.shape))
    locked.values[5, 5] = 1
    d = ls.prioritize.actions_raster_data(cost, feats, thr, 1.0, target_recovery=0.2, locked_out=locked)
    assert d.pu.loc[d.pu["status"] == 3].shape[0] == 1


def test_raster_solution_saves(tmp_path):
    cost, feats, thr, acost = _rasters()
    d = ls.prioritize.actions_raster_data(cost, feats, thr, 1.0, target_recovery=0.3)
    s = d.problem("minimizeCosts").add_highs_solver(gap=0).solve()
    ls.save(s, tmp_path / "sol")
    names = sorted(p.name for p in (tmp_path / "sol").iterdir())
    assert "maps.tif" in names and "monitoring.csv" in names
    back = ls.load_raster(tmp_path / "sol" / "maps.tif")
    np.testing.assert_array_equal(np.nan_to_num(back.values, nan=-1), np.nan_to_num(s.maps().values, nan=-1))
