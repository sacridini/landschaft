# SPDX-License-Identifier: GPL-3.0-or-later
"""Fuzzy similarity (Dinamica EGO) and validation in one call (ls.simulate phase 8b)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate


def test_same_as_dinamica():
    ref = simulate_reference.result("similarity", "similarity_dinamica")
    ours = simulate_cases.run_python("similarity_dinamica")
    for k in sorted(ref):
        a = np.asarray(ref[k], float)
        b = np.asarray(ours[k], float).reshape(a.shape)
        if k == "means.csv/Value":  # Dinamica writes 15 significant digits
            np.testing.assert_allclose(b, a, rtol=1e-14, atol=0, err_msg=k)
        else:
            np.testing.assert_array_equal(b, a, err_msg=k)


def test_decay_and_window():
    a = np.full((15, 15), np.nan)
    b = np.full((15, 15), np.nan)
    a[7, 7] = 1
    for d, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (3, 3)]):
        b[:] = np.nan
        b[7 + dy, 7 + dx] = 1
        e = S.fuzzy_similarity(a, b, window=7).first_mean
        assert e == pytest.approx(float(np.float32(2 ** (-np.hypot(dy, dx) / 2))), rel=0, abs=0)
        c = S.fuzzy_similarity(a, b, window=5, decay="constant").first_mean
        assert c == (1.0 if max(dy, dx) <= 2 else 0.0)
    with pytest.raises(ValueError, match="odd"):
        S.fuzzy_similarity(a, b, window=4)


def test_two_way_where_both_have_a_class():
    a = np.full((9, 9), np.nan)
    b = np.full((9, 9), np.nan)
    a[4, 4] = 1
    b[4, 4] = 2
    b[4, 6] = 1  # class 1 is near, but b's class 2 is nowhere in a: the minimum is 0
    s = S.fuzzy_similarity(a, b, window=5)
    assert s.first_mean == 0.0 and s.second_mean == pytest.approx((0.0 + 0.5) / 2)
    assert s.minimum == 0.0


def test_similarity_of_differences_grows_with_the_window():
    w = simulate_cases.sim_world()
    t = S.similarity_of_differences(w.sel(band="t0"), w.sel(band="observed"), w.sel(band="simulated"))
    assert list(t["window"]) == [1, 3, 5, 7, 9, 11]
    assert (np.diff(t["minimum"]) >= 0).all() and (t["minimum"] == np.fmin(t["simulated"], t["observed"])).all()
    same = S.similarity_of_differences(w.sel(band="t0"), w.sel(band="observed"), w.sel(band="observed"))
    np.testing.assert_allclose(same["minimum"], 1.0)


def test_validate_and_projection():
    w = simulate_cases.sim_world()
    t0, obs, sim = (w.sel(band=k) for k in ("t0", "observed", "simulated"))
    v = S.validate(t0, obs, sim, factors=[1, 4], windows=[1, 5], metrics=["pland"])
    s = v.summary()
    assert s["figure_of_merit"] == pytest.approx(S.three_map_comparison(t0, obs, sim).figure_of_merit()["overall"]
                                                 .iloc[0])
    assert s["kappa"] == S.compare_maps(obs, sim)["kappa"]
    assert s["similarity_5"] >= s["similarity_1"]
    assert set(v.metrics.columns) >= {"metric", "class", "observed", "simulated", "difference"}
    assert "Validation(" in repr(v)
    lu = ls.load_raster(S.example_data("sibuyan_landuse"))
    dr = ls.load_raster(S.example_data("sibuyan_drivers")).sel(band=["dist_sea", "dist_roads"])
    proj = S.run(lu[0], lu[1], dr, seed=1, woe=250)
    pv = proj.validate(factors=[1, 2], windows=[1, 3])
    assert pv.summary()["figure_of_merit"] == pytest.approx(proj.validation["figure_of_merit"], rel=1e-12)
    with pytest.raises(ValueError, match="validate=False"):
        S.run(lu[0], lu[1], dr, seed=1, woe=250, validate=False).validate()


def test_futures_grid_validation():
    w = simulate_cases.futval_world()
    ref, sim, orig = (w.sel(band=k) for k in ("reference", "simulated", "original"))
    g = S.futures_validation_grid(ref, sim, orig, size=20)
    assert len(g) == 3 * 3 and {"row", "col", "x", "y", "cells", "kappa", "figure_of_merit"} <= set(g.columns)
    one = S.futures_validation(np.asarray(ref)[:20, 20:40], np.asarray(sim)[:20, 20:40], np.asarray(orig)[:20, 20:40])
    row = g[(g.row == 0) & (g.col == 1)].iloc[0]
    assert row["kappa"] == one["kappa"] and row["figure_of_merit"] == one["figure_of_merit"]
    regions = np.zeros(ref.shape)
    regions[:, 30:] = 1
    r = S.futures_validation_grid(ref, sim, orig, regions=regions)
    assert list(r["region"]) == [0, 1]
    with pytest.raises(ValueError, match="tile size or the regions"):
        S.futures_validation_grid(ref, sim)


def test_validation_plot():
    w = simulate_cases.sim_world()
    v = S.validate(*(w.sel(band=k) for k in ("t0", "observed", "simulated")), factors=[1, 2], windows=[1, 3])
    assert [p.title for p in ls.plot(v).panels] == ["budget", "figure of merit", "similarity", None]
