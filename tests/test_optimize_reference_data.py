# SPDX-License-Identifier: GPL-3.0-or-later
"""The stored prioritizr / prioriactions references are complete and consistent.

Each compiled model is checked against its own optimal solution, and solved
again from the stored matrices on HiGHS and Gurobi (when highspy / gurobipy
are installed): the same model gives the same optimum on every solver.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import optimize_cases
import pytest
from optimize_reference import case_ids, model, result, solve_gurobi, solve_highs

CASES = [(g, c) for g in ("core", "objectives", "actions") for c in case_ids(g)]
IDS = [c for _, c in CASES]


def _gurobi_ok() -> bool:
    if importlib.util.find_spec("gurobipy") is None:
        return False
    import gurobipy as gp

    try:
        with gp.Env(params={"OutputFlag": 0}):
            return True
    except gp.GurobiError:  # no licence
        return False


@pytest.mark.parametrize(("group", "name"), CASES, ids=IDS)
def test_reference_case(group, name):
    m = model(group, name)
    r = result(group, name)
    nrow, ncol = m.shape
    assert m.lb.size == m.ub.size == len(m.vtype) == ncol
    assert len(m.sense) == nrow
    assert m.A_i.size == m.A_j.size == m.A_x.size
    assert m.A_i.min() >= 0 and m.A_i.max() < nrow and m.A_j.min() >= 0 and m.A_j.max() < ncol
    assert m.modelsense in ("min", "max")
    x = r["solution"]
    if group in ("core", "objectives"):
        # prioritizr reports the planning-unit columns, which come first
        assert set(m.col_ids) >= {"pu"} and len(m.row_ids) == nrow
        npu = m.col_ids.count("pu")
        assert x.size == npu
        assert r["status"] == ["OPTIMAL"]
        if ncol > npu:
            return  # the other columns are not reported
    else:
        assert x.size == ncol
        assert r["gap"][0] == 0
    # the optimum satisfies the model and gives the objective
    assert np.all(x >= m.lb - 1e-9) and np.all(x <= m.ub + 1e-9)
    integer = np.isin(np.asarray(m.vtype), ("B", "I"))
    assert np.allclose(x[integer], np.round(x[integer]), atol=1e-6)
    ax = m.csr() @ x
    sense = np.asarray(m.sense)
    tol = 1e-6 * np.maximum(1, np.abs(m.rhs))
    assert np.all(ax[sense == ">="] >= m.rhs[sense == ">="] - tol[sense == ">="])
    assert np.all(ax[sense == "<="] <= m.rhs[sense == "<="] + tol[sense == "<="])
    eq = np.isin(sense, ("=", "=="))
    assert np.allclose(ax[eq], m.rhs[eq], atol=1e-6)
    assert m.obj @ x == pytest.approx(r["objective"][0], rel=1e-9, abs=1e-9)


@pytest.mark.skipif(importlib.util.find_spec("highspy") is None, reason="highspy not installed")
@pytest.mark.parametrize(("group", "name"), CASES, ids=IDS)
def test_same_optimum_highs(group, name):
    m = model(group, name)
    if m.genconpow is not None:
        pytest.skip("general power constraints (Gurobi only)")
    obj, _ = solve_highs(m)
    assert obj == pytest.approx(result(group, name)["objective"][0], rel=1e-5)  # the max objectives' cost tie-break: see test_prioritize_milp


@pytest.mark.skipif(not _gurobi_ok(), reason="gurobipy or its licence missing")
@pytest.mark.parametrize(("group", "name"), CASES, ids=IDS)
def test_same_optimum_gurobi(group, name):
    m = model(group, name)
    obj, _ = solve_gurobi(m)
    assert obj == pytest.approx(result(group, name)["objective"][0], rel=1e-5)  # the max objectives' cost tie-break: see test_prioritize_milp


def test_raster_inputs_reach_prioritizr_unchanged():
    """Planning units are the cells with a cost, in cell order; the matrix holds the features' values (NaN = 0)
    bit for bit; relative targets are shares of the whole raster's totals, summed in cell order in double
    (terra), not of the planning units' totals."""
    m = model("core", "minset_raster_relative")
    cells = optimize_cases._units()
    f = optimize_cases._features().reshape(5, -1)
    assert np.array_equal(m.obj, optimize_cases._cost().ravel()[cells])
    assert np.array_equal(m.csr().toarray(), np.nan_to_num(f[:, cells]))
    totals = []
    for v in f:
        s = 0.0
        for a in v[np.isfinite(v)]:
            s += float(a)
        totals.append(s)
    assert np.array_equal(m.rhs, 0.3 * np.array(totals))


def test_table_targets_are_r_sums():
    """With tables, relative targets are shares of R's sum() (80-bit accumulation) of the amounts."""
    from landschaft import _core

    m = model("core", "minset_table")
    t = optimize_cases.pu_table()
    totals = [_core.nlm_r_sum(np.ascontiguousarray(t[f"f{k}"].to_numpy())) for k in range(1, 6)]
    assert np.array_equal(m.rhs, 0.3 * np.array(totals))
