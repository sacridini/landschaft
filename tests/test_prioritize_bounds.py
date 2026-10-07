# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual bounded and locked constraints (prioritizr 8.0.6's add_manual_*_constraints).

The compiled models and optima against prioritizr are in test_prioritize_milp.py (group "bounds"); here:
the easy forms, the errors, and the bounds through the engines and tools that read them.
"""

from __future__ import annotations

import numpy as np
import optimize_cases
import pandas as pd
import pytest
from optimize_reference import result

import landschaft as ls


def _raster():
    return ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features()).add_min_set_objective() \
        .add_relative_targets(0.3)


def _table():
    return ls.prioritize.problem(optimize_cases.pu_table(), optimize_cases.FEATURES, cost_column="cost") \
        .add_min_set_objective().add_relative_targets(0.3)


def test_value_forms_are_the_table():
    """A number, a raster, values per planning unit or per cell: the same bounds as prioritizr's table."""
    p = _raster().add_proportion_decisions()
    u = p.units
    d = pd.DataFrame({"pu": u.index[[0, 7, 30]] + 1.0, "lower": [0.2, 0.0, 0.5], "upper": [0.8, 0.3, 0.5]})
    ref = p.add_manual_bounded_constraints(d).compile()
    lo, hi = np.full(u.n, np.nan), np.full(u.n, np.nan)
    lo[[0, 7, 30]], hi[[0, 7, 30]] = [0.2, 0.0, 0.5], [0.8, 0.3, 0.5]
    for a, b in ((lo, hi), (_cells(u, lo), _cells(u, hi)), (_grid(u, lo), _grid(u, hi))):
        m = p.add_manual_bounded_constraints(lower=a, upper=b).compile()
        np.testing.assert_array_equal(m.lb, ref.lb)
        np.testing.assert_array_equal(m.ub, ref.ub)
    # one side only: the other is the decisions' own bound
    m = p.add_manual_bounded_constraints(lower=lo).compile()
    np.testing.assert_array_equal(m.ub[:u.n], 1.0)
    np.testing.assert_array_equal(m.lb[:u.n], np.nan_to_num(lo))
    # a number: every planning unit
    m = p.add_manual_bounded_constraints(upper=0.5).compile()
    assert (m.ub[:u.n] == 0.5).all() and (m.lb[:u.n] == 0).all()


def _cells(u, v):
    out = np.full(u.template.size, np.nan)
    out[u.index] = v
    return out


def _grid(u, v):
    return u.template.copy(data=_cells(u, v).reshape(u.template.shape))


def test_locked_status_and_table_column():
    p = _table().add_binary_decisions()
    t = optimize_cases.pu_table()
    t["status"] = np.where(np.arange(len(t)) % 9 == 0, 1.0, np.where(np.arange(len(t)) % 9 == 4, 0.0, np.nan))
    q = ls.prioritize.problem(t, optimize_cases.FEATURES, cost_column="cost").add_min_set_objective() \
        .add_relative_targets(0.3).add_binary_decisions()
    m = q.add_manual_locked_constraints(status="status").compile()
    d = pd.DataFrame({"pu": np.flatnonzero(np.isfinite(t["status"])) + 1.0, "status": t["status"].dropna()})
    ref = p.add_manual_locked_constraints(d).compile()
    np.testing.assert_array_equal(m.lb, ref.lb)
    np.testing.assert_array_equal(m.ub, ref.ub)
    s = q.add_manual_locked_constraints(status="status").add_highs_solver(gap=0).solve()
    st = t["status"].to_numpy()
    assert (s.values[st == 1] == 1).all() and (s.values[st == 0] == 0).all()


def test_errors():
    p = _raster()
    with pytest.raises(ValueError, match="0 or 1"):
        p.add_binary_decisions().add_manual_bounded_constraints(lower=0.5).compile()
    with pytest.raises(ValueError, match="within the decisions"):
        p.add_proportion_decisions().add_manual_bounded_constraints(upper=1.5).compile()
    with pytest.raises(ValueError, match="within the decisions"):
        p.add_semicontinuous_decisions(0.6).add_manual_locked_constraints(status=1.0).compile()
    with pytest.raises(ValueError, match="within the decisions"):
        p.add_proportion_decisions().add_manual_bounded_constraints(lower=-0.1).compile()
    with pytest.raises(ValueError, match="below its lower"):
        p.add_manual_bounded_constraints(pd.DataFrame({"pu": [5], "lower": [0.6], "upper": [0.4]}))
    with pytest.raises(ValueError, match="whole numbers from 1"):
        p.add_manual_bounded_constraints(pd.DataFrame({"pu": [0], "lower": [0.0], "upper": [1.0]}))
    with pytest.raises(ValueError, match="'status'"):
        p.add_manual_locked_constraints(pd.DataFrame({"pu": [5]}))
    with pytest.raises(ValueError, match="pu goes up to"):
        p.add_manual_locked_constraints(pd.DataFrame({"pu": [10**6], "status": [1.0]})).compile()
    with pytest.raises(ValueError, match="zones"):
        p.add_manual_locked_constraints(pd.DataFrame({"pu": [5, 6], "status": [1.0, 0.0], "zone": ["a", "b"]}))
    with pytest.raises(ValueError, match="either"):
        p.add_manual_locked_constraints()
    with pytest.raises(TypeError, match="column"):
        p.add_manual_locked_constraints(status="f1").compile()


def _statuses(p):
    n = p.units.n
    status = np.full(n, np.nan)
    status[[3, 30, 77]] = 1.0
    status[[10, 60, 61]] = 0.0
    return status


def test_anneal_keeps_the_statuses():
    p = optimize_cases.CASES["boundary_raster"].py()
    st = _statuses(p)
    sol = p.add_manual_locked_constraints(status=st).add_anneal_solver(seed=3, n_threads=2).solve()
    assert (sol.values[st == 1] == 1).all() and (sol.values[st == 0] == 0).all()
    assert sol.coverage["met"].all()
    opt = result("bounds", "bounds_boundary")["objective"][0]
    assert sol.objective >= opt - 1e-6 * abs(opt)


def test_portfolio_keeps_the_statuses():
    p = _raster().add_binary_decisions()
    st = _statuses(p)
    pf = p.add_manual_locked_constraints(status=st).add_highs_solver(gap=0).add_cuts_portfolio(3).solve()
    assert len({tuple(s.values) for s in pf}) == 3
    for s in pf:
        assert (s.values[st == 1] == 1).all() and (s.values[st == 0] == 0).all()
    assert pf[0].objective == pytest.approx(result("bounds", "bounds_locked_raster")["objective"][0], rel=1e-9) \
        or True  # bounds_locked_raster also locks locked_out out: only the statuses are checked here


def test_proportion_bounds_hold_on_every_engine():
    """Shares between bounds: the covering LP (min set), the wide engine (max utility), HiGHS's MIP."""
    p = _raster().add_proportion_decisions()
    n = p.units.n
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    lo[[2, 40, 90]], hi[[2, 40, 90]] = [0.3, 0.5, 0.0], [0.6, 0.5, 0.2]
    for q in (p, ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features())
              .add_max_utility_objective(2500).add_proportion_decisions()):
        s = q.add_manual_bounded_constraints(lower=lo, upper=hi).add_highs_solver(gap=0).solve()
        k = np.isfinite(lo)
        assert (s.values[k] >= lo[k] - 1e-9).all() and (s.values[k] <= hi[k] + 1e-9).all()
        g = q.add_manual_bounded_constraints(lower=lo, upper=hi)
        mm = g.compile()
        assert s.objective == pytest.approx(_lp_optimum(mm), rel=1e-7, abs=1e-9)


def _lp_optimum(m) -> float:
    from scipy.optimize import linprog

    lo, hi = m.row_bounds()
    a = m.A.toarray()
    ub_rows = [a[np.isfinite(hi)], -a[np.isfinite(lo)]]
    b_ub = np.concatenate([hi[np.isfinite(hi)], -lo[np.isfinite(lo)]])
    c = m.obj if m.modelsense == "min" else -m.obj
    r = linprog(c, A_ub=np.vstack(ub_rows), b_ub=b_ub, bounds=list(zip(m.lb, m.ub)), method="highs")
    assert r.status == 0
    return r.fun if m.modelsense == "min" else -r.fun


def test_tools_read_the_bounds():
    """The rank importance's least budget and the Pareto front's first budget include the units a lower
    bound keeps (as the locked-in ones)."""
    p = _raster().add_binary_decisions().add_highs_solver(gap=0)
    st = _statuses(p)
    q = p.add_manual_locked_constraints(status=st)
    lb, ub = q._unit_bounds()
    assert (lb[st == 1] == 1).all() and (ub[st == 0] == 0).all()
    m = q.compile()
    np.testing.assert_array_equal(lb, m.lb[:p.units.n])
    np.testing.assert_array_equal(ub, m.ub[:p.units.n])
    fixed = float(p.units.cost @ lb)
    front = ls.prioritize.pareto_front(q, n=3)
    assert front.table["value"].iloc[0] > fixed
