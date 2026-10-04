# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation (ls.simulate phase 8a): lulcc's three-map comparison, MOLUSCE's kappa and error budget."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate


@pytest.mark.parametrize("case", simulate_reference.case_ids("validation"))
def test_same_as_reference(case):
    ref = simulate_reference.result("validation", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref)
    for k in sorted(ref):
        a = np.asarray(ref[k], float)
        b = np.asarray(ours[k], float).reshape(a.shape)
        np.testing.assert_array_equal(b, a, err_msg=k)


def _maps():
    w = simulate_cases.val_synth()
    return w.sel(band="t0"), w.sel(band="t1"), w.sel(band="sim")


def test_perfect_simulation():
    t0, t1, _ = _maps()
    tmc = S.three_map_comparison(t0, t1, t1, factors=[1, 4])
    fom = tmc.figure_of_merit()
    np.testing.assert_allclose(fom["overall"].to_numpy(), 1.0)
    b = tmc.agreement_budget()
    o = b[(b.level == "overall") & (b.factor == 1)].iloc[0]
    valid = ~np.isnan(t0.values)
    share = np.sum((t0.values != t1.values) & valid) / valid.sum()
    assert o.misses == pytest.approx(0, abs=1e-15) and o.false_alarms == pytest.approx(0, abs=1e-15)
    assert o.hits == pytest.approx(share, rel=1e-12) and share > 0
    assert o.hits + o.correct_rejections == pytest.approx(1.0)


def test_tables_and_budget_add_up():
    t0, t1, sim = _maps()
    tmc = S.three_map_comparison(t0, t1, sim, factors=[1, 2, 3])
    for f in (1, 2, 3):
        t = tmc.table(f)
        assert t.loc[("total", "total"), "total"] == pytest.approx(1.0)
        inner = t.drop(index="total", level=0).drop(index="total", level=1).drop(columns="total")
        assert inner.to_numpy().sum() == pytest.approx(1.0)
    b = tmc.agreement_budget()
    o = b[b.level == "overall"]
    np.testing.assert_allclose(o[["misses", "hits", "wrong_hits", "false_alarms", "correct_rejections"]]
                               .sum(axis=1), 1.0)
    # at the cells, the figure of merit is the run's agreement()
    a = S.agreement(t1, sim, t0)
    assert tmc.figure_of_merit()["overall"].loc[1] == pytest.approx(a["figure_of_merit"], rel=1e-12)
    assert repr(tmc) == "ThreeMapComparison(4 classes, factors [1, 2, 3])"


def test_missing_cells_in_any_map_are_left_out():
    t0, t1, sim = _maps()
    t1v = t1.values.copy()
    t1v[20, 20] = np.nan
    a = S.three_map_comparison(t0, t1v, sim, factors=[1, 2])
    t0v, simv = t0.values.copy(), sim.values.copy()
    t0v[20, 20] = simv[20, 20] = np.nan
    b = S.three_map_comparison(t0v, t1v, simv, factors=[1, 2])
    for f in (1, 2):
        np.testing.assert_array_equal(a.tables[f], b.tables[f])


def test_compare_maps_and_error_budget():
    _, t1, sim = _maps()
    same = S.compare_maps(t1, t1)
    assert same["kappa"] == 1 and same["correctness"] == 100 and same["correlation"] == pytest.approx(1)
    c = S.compare_maps(t1, sim)
    assert 0 < c["kappa"] < 1 and c["kappa_histogram"] >= c["kappa"] - 1e-12
    eb = S.error_budget(t1, t1, iterations=3)
    np.testing.assert_allclose(eb[["medium_medium", "medium_perfect", "perfect_perfect"]].to_numpy(), 1.0)
    assert eb["cells"].tolist() == [1, 2, 4]
    eb = S.error_budget(t1, sim, iterations=3)
    assert (eb["medium_medium"] <= eb["medium_perfect"] + 1e-12).all()
    with pytest.raises(ValueError, match="lacks"):
        S.error_budget(t1, sim.where(sim != 4, 9))


def test_errors():
    t0, t1, sim = _maps()
    with pytest.raises(ValueError, match="two classes"):
        S.three_map_comparison(np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="factors"):
        S.three_map_comparison(t0, t1, sim, factors=[0])
    with pytest.raises(ValueError, match="shape"):
        S.three_map_comparison(t0, t1, np.ones((2, 2)))


@pytest.mark.parametrize("case", simulate_reference.case_ids("futures_validation"))
def test_same_as_r_futures_validation(case):
    ref = simulate_reference.result("futures_validation", case)
    ours = simulate_cases.run_python(case)
    assert list(ref["results.csv/key"]) == list(ours["results.csv/key"])
    np.testing.assert_array_equal(ours["results.csv/value"], np.asarray(ref["results.csv/value"], float))


def test_futures_validation_metrics():
    orig = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
    ref = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    sim = np.array([[1, 1, 1, 0], [0, 0, 1, 0]])
    r = S.futures_validation(ref, sim, orig)
    assert (r["hits"], r["misses"], r["false_alarms"], r["null_successes"]) == (1 / 8, 1 / 8, 1 / 8, 3 / 8)
    assert r["initially_developed"] == 2 / 8 and r["figure_of_merit"] == pytest.approx(1 / 3)
    assert r["producer"] == 0.5 and r["user"] == 0.5
    multi = S.futures_validation(ref + 2 * orig, sim, orig)
    assert multi["hits"] is None and multi["kappasimulation"] is not None
    assert S.futures_validation(ref, sim)["kappa"] == pytest.approx(S.compare_maps(ref, sim)["kappa"])
