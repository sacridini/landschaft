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
