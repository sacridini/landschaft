# SPDX-License-Identifier: GPL-3.0-or-later
"""Transition potential (ls.simulate phase 2): weights of evidence, logistic regression, estimators."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases

import landschaft as ls


def _woe_data():
    d = simulate_cases.woe_synth()
    drivers = {k: d.sel(band=k) for k in ("dist", "elev", "slope", "soil")}
    return d.sel(band="t0"), d.sel(band="t1"), drivers


def test_woe_default_transitions_and_api():
    t0, t1, drivers = _woe_data()
    w = ls.simulate.weights_of_evidence(t0, t1, drivers, {"dist": 25, "elev": 10, "slope": 0.5, "soil": "categorical"})
    assert set(w.transitions) >= {(1, 2), (1, 3), (3, 2)}
    tab = w.table
    assert set(tab["Variable"]) == {"dist", "elev", "slope", "soil"}
    assert (tab["Possible_Transitions"] >= tab["Executed_Transitions"]).all()
    assert w.ranges.shape[1] == 5
    corr = w.correlation()
    assert {"Chi_2", "Cramer", "Joint_Uncertainty"} <= set(corr.columns)
    assert ((corr["Cramer"] >= 0) & (corr["Cramer"] <= 1)).all()
    assert "transitions" in repr(w)


def test_woe_probability_dtypes_and_multinomial():
    t0, t1, drivers = _woe_data()
    w = ls.simulate.WeightsOfEvidence({"dist": 25, "elev": 10}, transitions=[(1, 2), (1, 3)]).fit(t0, t1, drivers)
    p = w.probability(t0, drivers)
    assert list(p.transition.values) == ["probability_1_to_2", "probability_1_to_3"]
    total = p.sum("transition", skipna=False)
    assert float(total.max()) < 1  # the class may also stay
    u8 = w.probability(t0, drivers, dtype="uint8")
    np.testing.assert_array_equal(u8.values, np.where(np.isnan(p.values), 0, np.trunc(p.values * 255)).astype("uint8"))
    single = w.probability(t0, drivers, transitions=[(1, 2)])
    s = np.log(p.values / (1 - p.values.sum(axis=0)))[0]  # the multinomial log-odds of 1 -> 2
    np.testing.assert_allclose(single.values[0], 1 / (1 + np.exp(-s)), rtol=1e-12)


def test_woe_negative_values_shift_invariant():
    # the consistent form: a variable shifted by a multiple of the increment gets the same ranges and weights
    # (Dinamica EGO drops, shifts and merges ranges below zero)
    t0, t1, drivers = _woe_data()
    elev = drivers["elev"]
    a = ls.simulate.WeightsOfEvidence({"elev": 10}, transitions=[(1, 3)]).fit(t0, t1, {"elev": elev}).table
    b = ls.simulate.WeightsOfEvidence({"elev": 10}, transitions=[(1, 3)]).fit(t0, t1, {"elev": elev - 1000}).table
    # positive variables are bucketed from 0 (the empty buckets below the minimum join the first range)
    np.testing.assert_allclose(a["Range_Lower_Limit"][1:], b["Range_Lower_Limit"][1:] + 1000)
    np.testing.assert_allclose(a["Range_Upper_Limit"], b["Range_Upper_Limit"] + 1000)
    np.testing.assert_array_equal(a["Possible_Transitions"], b["Possible_Transitions"])
    np.testing.assert_allclose(a["Weight_Coefficient"], b["Weight_Coefficient"], rtol=1e-14)


def test_woe_explicit_breaks_and_auto_increment():
    t0, t1, drivers = _woe_data()
    w = ls.simulate.WeightsOfEvidence({"dist": [0, 300, 900, 5000]}, transitions=[(1, 2)]).fit(
        t0, t1, {"dist": drivers["dist"]})
    tab = w.table
    assert tab["Range_Lower_Limit"].tolist() == [0, 300, 900]
    assert tab["Range_Upper_Limit"].tolist() == [300, 900, 5000]
    auto = ls.simulate.WeightsOfEvidence(transitions=[(1, 2)]).fit(t0, t1, {"dist": drivers["dist"]})
    assert len(auto.table) > 1


def test_woe_probability_beyond_the_calibrated_extent():
    t0, t1, drivers = _woe_data()
    w = ls.simulate.WeightsOfEvidence({"dist": 25}, transitions=[(1, 2)]).fit(t0, t1, {"dist": drivers["dist"]})
    far = drivers["dist"] + 1e6  # beyond every calibrated range: the last range's weight
    p = w.probability(t0, {"dist": far})
    last = w.table["Weight_Coefficient"].iloc[-1]
    v = p.values[0][np.isfinite(p.values[0])]
    np.testing.assert_allclose(np.unique(np.round(v, 12)), [np.round(1 / (1 + np.exp(-last)), 12)])


def test_woe_errors():
    t0, t1, drivers = _woe_data()
    with pytest.raises(ValueError, match="no driver"):
        ls.simulate.WeightsOfEvidence({"nope": 10}).fit(t0, t1, drivers)
    with pytest.raises(RuntimeError, match="fit"):
        _ = ls.simulate.WeightsOfEvidence().table
    with pytest.raises(ValueError, match="possible"):
        ls.simulate.WeightsOfEvidence(possible="never")


def test_logistic_regression_subset_and_alias():
    d = simulate_cases.glm_synth()
    lu, drivers = d[0], d[1:5]
    lr = ls.simulate.LogisticRegression({1: ["a", "b", "c"]}).fit(lu, drivers)
    coef = lr.coefficients
    assert np.isnan(coef.loc[1, "c"])  # c = 2a - 3b is aliased
    assert lr.summary.loc[1, "converged"]
    half = np.zeros(lu.shape, dtype=bool)
    half[:, :30] = True
    lr2 = ls.simulate.LogisticRegression({1: ["a", "b"]}).fit(lu, drivers, cells=half)
    assert lr2.summary.loc[1, "cells"] < lr.summary.loc[1, "cells"]
    p = lr2.predict(drivers)
    assert p.dims == ("class", "y", "x") and np.nanmax(p.values) < 1


def test_estimator_matches_glm_and_expansion():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression as SkLR

    d = simulate_cases.glm_synth()
    lu, drivers = d[0], d[1:5]
    sub = {k: drivers.sel(band=k) for k in ("a", "b", "d")}
    est = ls.simulate.Estimator(SkLR(C=1e10, max_iter=10000, tol=1e-12), classes=[2]).fit(lu, sub)
    glm = ls.simulate.LogisticRegression({2: ["a", "b", "d"]}).fit(lu, sub)
    np.testing.assert_allclose(est.predict(sub).values, glm.predict(sub).values, atol=2e-4)
    assert sklearn.__version__
    with pytest.raises(ValueError, match="t1"):
        ls.simulate.Estimator(SkLR(), target="expansion").fit(lu, sub)
    t1 = lu.copy(data=np.where(np.isfinite(lu.values) & (d.sel(band="a").values > 0.8), 2.0, lu.values))
    exp = ls.simulate.Estimator(SkLR(), target="expansion", classes=[2], sample=0.5, balance=True, seed=3)
    p = exp.fit(lu, sub, t1=t1).predict(sub)
    assert np.nanmax(p.values) <= 1


def test_raster_potential():
    m = np.random.default_rng(0).random((5, 6))
    pot = ls.simulate.RasterPotential({1: m, 2: 1 - m})
    out = pot.fit().predict(classes=[2])
    np.testing.assert_allclose(out.values[0], 1 - m)
    with pytest.raises(TypeError):
        ls.simulate.RasterPotential(m)
