# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibrating trade-offs (tests/optimize_cases.py, group "calibrate"): the hierarchical and blended
approaches and the cost-shortfall front reach prioritizr's optima on Gurobi; TOPSIS, the knee, Cohon's
penalty and the iterative solve on their definitions."""

from __future__ import annotations

import warnings

import numpy as np
import optimize_cases as oc
import pytest
from optimize_reference import result

import landschaft as ls

P = ls.prioritize


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _base():
    return oc._raster().add_relative_targets(0.3).add_binary_decisions().add_highs_solver(gap=0)


def test_hierarchical_is_prioritizr():
    r = result("calibrate", "calibrate_hierarchical")
    cal = P.calibrate_budget(_base(), "boundary")
    np.testing.assert_array_equal(cal.table["value"].to_numpy(), r["threshold"])
    np.testing.assert_allclose(cal.table["objective"].to_numpy(), r["objective"], rtol=1e-9)
    # the criterion is the boundary the objective minimises (penalty 1, edge factor 0.5)
    np.testing.assert_allclose(cal.table["boundary"].to_numpy(), r["objective"], rtol=1e-9)
    assert (cal.table["cost"] <= cal.table["value"] + 1e-6).all()
    assert cal.table["topsis"].sum() == 1 and cal.table["knee"].sum() == 1


def test_blended_is_prioritizr():
    r = result("calibrate", "calibrate_blended")
    cal = P.calibrate_penalty(_base(), "boundary", values=r["penalty"])
    np.testing.assert_allclose(cal.table["objective"].to_numpy(), r["objective"], rtol=1e-9)
    np.testing.assert_allclose(cal.table["cost"] + cal.table["value"] * cal.table["boundary"], r["objective"],
                               rtol=1e-9)
    assert cal.best("topsis") is cal.solutions[int(np.flatnonzero(cal.table["topsis"])[0])]


def test_pareto_front_is_prioritizr():
    r = result("calibrate", "calibrate_pareto")
    front = P.pareto_front(oc._raster().add_relative_targets(0.3).add_binary_decisions().add_highs_solver(gap=0),
                           budgets=r["budget"])
    np.testing.assert_allclose(front.table["objective"].to_numpy(), r["objective"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(front.table["shortfall"].to_numpy(), r["objective"], rtol=1e-7, atol=1e-9)
    assert front.table["efficient"].all()  # more budget, less shortfall
    assert front.table["met"].iloc[-1] == 5


def test_cohon_penalty():
    p = _base()
    value, ideals = P.cohon_penalty(p, "boundary")
    least_cost = p.solve()
    assert ideals["cost"].iloc[0] == pytest.approx(least_cost.cost)
    assert ideals["boundary"].iloc[1] == pytest.approx(result("calibrate", "calibrate_hierarchical")["objective"][-1])
    want = abs((ideals["cost"].iloc[0] - ideals["cost"].iloc[1]) / (ideals["boundary"].iloc[0] - ideals["boundary"].iloc[1]))
    assert value == pytest.approx(round(want, 5), abs=1e-12)
    cal = P.calibrate_penalty(p, "boundary", n=3)  # the default range: around Cohon's penalty
    np.testing.assert_allclose(cal.table["value"], value * np.array([0.01, 1, 100]))


def test_topsis_and_knee():
    # Hwang & Yoon by hand: two criteria to minimise, equal weights
    d = np.array([[1.0, 9.0], [4.0, 4.0], [9.0, 1.0], [5.0, 6.0]])
    t = P.topsis(d, impacts=["-", "-"])
    norm = np.sqrt((d ** 2).sum(axis=0))
    v = d / norm * 0.5
    best, worst = v.min(axis=0), v.max(axis=0)
    du, dl = np.sqrt(((v - best) ** 2).sum(1)), np.sqrt(((v - worst) ** 2).sum(1))
    np.testing.assert_allclose(t["score"], dl / (dl + du))
    assert t.loc[t["rank"] == 1, "alt.row"].item() == 2
    with pytest.raises(ValueError, match="one weight"):
        P.topsis(d, impacts=["-"])
    x = np.array([0, 1, 2, 3, 4, 5.0])
    y = np.array([10, 4, 2, 1.5, 1.2, 1.0])
    assert P.knee_point(x, y) == 2  # farthest from the chord of the scaled curve
    assert P.knee_point(x[::-1], y[::-1]) == 3  # the same point, in another order


def test_iterative_solve():
    p = _base()
    seen = []

    def update(step, sol, q):
        seen.append(step)
        # the cheapest selected cells become locked out (lost), the targets rise
        lost = sol.values > 0.5
        lost[np.argsort(np.where(lost, p.units.cost, np.inf))[3:]] = False
        return q.add_locked_out_constraints(q._mask(q.locked_out) | lost)

    sols, table = P.iterative_solve(p, update, 3, lock_previous=False)
    assert seen == [0, 1] and len(sols) == 3 and list(table["step"]) == [0, 1, 2]
    assert table["cost"].is_monotonic_increasing
    sols2, _ = P.iterative_solve(p, lambda k, s, q: q.add_relative_targets(0.35), 2, lock_previous=True)
    assert (sols2[1].values[sols2[0].values > 0.5] == 1).all()  # the first solution kept
    assert P.iterative_solve(p, lambda k, s, q: None, 5)[1].shape[0] == 1
    mask = P.protected_cells(sols[0])
    assert mask.dtype == bool and int(mask.sum()) == int(sols[0].values.sum())
    with pytest.raises(ZeroDivisionError):
        P.iterative_solve(p, lambda k, s, q: 1 / 0, 2)  # a failing step raises


def test_calibrate_weights():
    """A feature's weight in max utility against the share of another feature held (the generic form)."""
    base = oc._rp().add_max_utility_objective(1500).add_binary_decisions().add_highs_solver(gap=0)

    def make(w):
        return base.add_feature_weights([w, 1, 1, 0.01, 1])

    def lost(q, s):  # how little of f4 the solution holds: a criterion to minimise
        rep = P.eval_feature_representation_summary(q, s)
        return 1 - rep.loc[rep["feature"] == "f4", "relative_held"].item()

    cal = P.calibrate(make, [0.1, 1, 10, 100], lost, name="f4_lost")
    t = cal.table
    held_f1 = [P.eval_feature_representation_summary(make(w), s)["relative_held"].iloc[0]
               for w, s in zip(t["value"], cal.solutions)]
    assert np.all(np.diff(held_f1) >= -1e-9)  # a larger weight on f1 holds more of it
    assert t["topsis"].sum() == 1 and "f4_lost" in t
    with pytest.raises(ValueError, match="no values"):
        P.calibrate(make, [], lost)


def test_pareto_front_budgets_start_above_the_locked_in_cost():
    p = oc._raster().add_relative_targets(0.3).add_binary_decisions().add_highs_solver(gap=0)
    locked = np.zeros(p.units.n, dtype=bool)
    locked[np.argsort(p.units.cost)[-5:]] = True  # the dearest units: more than a tenth of the plan's cost
    q = p.add_locked_in_constraints(locked)
    fixed = float(p.units.cost[locked].sum())
    top = q.add_min_set_objective().solve().cost
    front = P.pareto_front(q, n=4)
    np.testing.assert_allclose(front.table["value"], fixed + (top - fixed) * np.arange(1, 5) / 4)
    assert (front.table["cost"] >= fixed - 1e-9).all()
    assert front.table["shortfall"].iloc[-1] == pytest.approx(0, abs=1e-9)
