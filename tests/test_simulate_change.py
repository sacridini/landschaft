# SPDX-License-Identifier: GPL-3.0-or-later
"""Change analysis (ls.simulate phase 1) against lulcc, OpenLand, diffeR and Dinamica EGO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

BITWISE = ["crosstab", "openland", "differ", "glm", "mce"]


def _cases(groups):
    return [(g, c) for g in groups for c in simulate_reference.case_ids(g)]


def _check(group, case, rtol=0.0):
    ref = simulate_reference.result(group, case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), (sorted(set(ours) ^ set(ref)))
    for key, value in ours.items():
        simulate_reference.assert_same(value, ref[key], f"{case}/{key}", rtol=rtol)


@pytest.mark.parametrize(("group", "case"), _cases(BITWISE))
@pytest.mark.external_reference
def test_same_as_r(group, case):
    _check(group, case)


@pytest.mark.parametrize("case", simulate_reference.case_ids("markov"))
def test_markov_as_dinamica(case):
    # Dinamica writes 15 significant digits and its root loses up to ~5e-12 on small rates
    _check("markov", case, rtol=2e-11)


# ----------------------------------------------------------------------------- properties

def test_markov_new_class_stays():
    # a class found only at the end keeps its cells (Dinamica gives it a row of zeros)
    s = simulate_cases.series()
    mk = ls.simulate.Markov.from_maps(s[0], s[3], steps=15)
    multi = mk.multi.to_numpy()
    k = mk.classes.index(12)
    assert multi[k, k] == pytest.approx(1.0) and np.allclose(np.delete(multi[k], k), 0)
    np.testing.assert_allclose(np.linalg.matrix_power(multi, 15), mk.single.to_numpy(), atol=1e-13)
    np.testing.assert_allclose(multi.sum(axis=1), 1, atol=1e-13)


@pytest.mark.parametrize("counts", [[[60, 40, 0], [0, 60, 40], [0, 0, 100]],     # repeated eigenvalue
                                    [[50, 50, 0], [0, 50, 50], [50, 0, 50]]])    # complex eigenvalues
def test_markov_root_where_dinamica_gives_up(counts):
    c = np.array(counts, float)
    mk = ls.simulate.Markov(c / c.sum(axis=1, keepdims=True), [1, 2, 3], steps=3)
    np.testing.assert_allclose(np.linalg.matrix_power(mk.multi.to_numpy(), 3), mk.single.to_numpy(), atol=1e-13)


def test_markov_negative_eigenvalue():
    mk = ls.simulate.Markov([[0.3, 0.7], [0.65, 0.35]], [1, 2], steps=3)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        _ = mk.multi


def test_markov_project():
    mk = ls.simulate.Markov([[0.9, 0.1], [0.0, 1.0]], [1, 2], steps=2)
    p = mk.project({1: 100.0, 2: 0.0})
    assert p.shape == (3, 2)
    np.testing.assert_allclose(p.iloc[-1].to_numpy(), [90.0, 10.0], atol=1e-12)  # two steps = the period
    np.testing.assert_allclose(p.sum(axis=1), 100.0)


@pytest.mark.parametrize("dtype", ["uint8", "int16", "int32", "int64", "float32", "float64"])
def test_crosstab_types_and_nodata(dtype):
    a, b = simulate_cases.small_change()
    a = a.astype(dtype)
    b = b.astype(dtype)
    a[0, :5] = 0
    ref = simulate_cases.crosstab(np.where(a == 0, np.nan, a), np.where(b == 0, np.nan, b))
    ours = ls.simulate.crosstab(a, b, nodata=0)
    np.testing.assert_array_equal(ours.to_numpy(), ref)


def test_crosstab_fractional_classes():
    rng = np.random.default_rng(4)
    a = rng.choice([0.5, 1.25, 7.0], (30, 40))
    b = rng.choice([0.5, 1.25, 3.5], (30, 40))
    a[2, 3] = np.nan
    ours = ls.simulate.crosstab(a, b)
    assert list(ours.index) == [0.5, 1.25, 3.5, 7.0]
    np.testing.assert_array_equal(ours.to_numpy(), simulate_cases.crosstab(a, b))


def test_threads_do_not_change_results():
    s = simulate_cases.series()
    one = ls.simulate.changes(s, n_threads=1)
    many = ls.simulate.changes(s, n_threads=8)
    pd.testing.assert_frame_equal(one.table, many.table)
    np.testing.assert_array_equal(one.change_count()[0], many.change_count()[0])


def test_changes_api():
    lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    ch = ls.simulate.changes(lu, labels={1: "forest", 2: "built", 3: "other"})
    assert ch.times == [1985, 1991, 1999]
    m = ch.matrix(1985, 1999)
    assert list(m.index) == ["forest", "built", "other"]
    assert int(m.to_numpy().sum()) == ch.total_area["cells"]
    areas = ch.areas
    assert list(areas.index) == [1985, 1991, 1999]
    np.testing.assert_array_equal(areas.iloc[0].to_numpy(), m.sum(axis=1).to_numpy())
    np.testing.assert_array_equal(areas.iloc[-1].to_numpy(), m.sum(axis=0).to_numpy())
    comp = ch.components()
    assert comp.columns.tolist()[:4] == ["Category", "Gain", "Persistence", "Loss"]
    counts, table = ch.change_count()
    assert counts.rio.crs == lu.rio.crs and int(table["Qt"].sum()) == ch.total_area["cells"]
    mk = ch.markov(steps=14)
    assert mk.steps == 14
    with pytest.raises(ValueError, match="no map at time"):
        ch.matrix(1990)


def test_series_validation():
    a, b = simulate_cases.small_change()
    with pytest.raises(ValueError, match="at least two"):
        ls.simulate.changes([a])
    with pytest.raises(ValueError, match="increase"):
        ls.simulate.changes([a, b], times=[5, 1])
    with pytest.raises(ValueError, match="shape"):
        ls.simulate.changes([a, b[:, :-1]])


def test_overall_identity():
    lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    d = ls.simulate.overall_difference(ls.simulate.crosstab(lu[0], lu[1]))
    assert d["overall"] == pytest.approx(d["quantity"] + d["exchange"] + d["shift"])
    assert d["allocation"] == pytest.approx(d["exchange"] + d["shift"])


# ----------------------------------------------------------------------------- weights of evidence (phase 2)

_WOE_TOL = {"Weight_Coefficient": 1e-13, "Contrast": 1e-13, "Chi_2": 1e-13, "Cramer": 1e-13,
            "Contingency": 1e-13, "Joint_Entropy": 1e-13, "Joint_Uncertainty": 1e-13, "prob.tif": 1e-13,
            "Range_Lower_Limit*": 1e-14, "Range_Upper_Limit*": 1e-14}


@pytest.mark.parametrize("case", simulate_reference.case_ids("woe"))
def test_woe_as_dinamica(case):
    # Dinamica writes 15 significant digits: weights compared to 1e-13, counts and ranges exactly
    ref = simulate_reference.result("woe", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))
    for key, value in ours.items():
        col = key.split("/")[-1]
        tol = _WOE_TOL.get(col, 0.0)
        simulate_reference.assert_same(value, ref[key], f"{case}/{key}", rtol=tol, atol=tol)
