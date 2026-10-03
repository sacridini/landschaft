# SPDX-License-Identifier: GPL-3.0-or-later
"""FUTURES (ls.simulate phase 5a): patch growing, development pressure and demand against GRASS GIS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate


def _curves(lines):
    out = []
    for line in lines[1:]:
        sid, method, coef, rmse = line.split(",")
        out.append((sid, method, [float(c) for c in coef.split(";")], float(rmse)))
    return out


@pytest.mark.parametrize("case", simulate_reference.case_ids("futures"))
def test_same_as_grass(case):
    ref = simulate_reference.result("futures", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))
    for key, value in ours.items():
        if key == "curves.txt":  # numpy and scipy versions differ between the two sides: the last digits
            for a, b in zip(_curves(value), _curves(ref[key]), strict=True):
                assert a[:2] == b[:2]
                np.testing.assert_allclose(a[2], b[2], rtol=1e-12)
                np.testing.assert_allclose(a[3], b[3], rtol=1e-12)
            continue
        simulate_reference.assert_same(value, ref[key], f"{case}/{key}")


# ----------------------------------------------------------------------------- pieces

def test_development_pressure_methods():
    dev = np.zeros((9, 9))
    dev[4, 4] = 1
    occ = S.development_pressure(dev, 2, method="occurrence").values
    assert occ[4, 4] == 0 and occ[4, 6] == 1 and occ[4, 7] == 0 and occ[6, 6] == 0  # d = 2.83 > 2
    grav = S.development_pressure(dev, 3, gamma=2.0, scaling_factor=4.0).values
    assert grav[4, 6] == pytest.approx(1.0) and grav[4, 5] == pytest.approx(4.0)
    ker = S.development_pressure(dev, 3, method="kernel", gamma=2.0, scaling_factor=1.0).values
    assert ker[4, 5] == pytest.approx(np.exp(-1.0), rel=1e-6)
    dev[0, 0] = np.nan
    edges = S.development_pressure(dev, 2, method="occurrence", pad=False).values
    assert np.isnan(edges[2, 2]) and edges[0, 5] == 0 and np.isnan(edges[0, 0])
    padded = S.development_pressure(dev, 2, method="occurrence").values
    assert np.isnan(padded[0, 0]) and padded[2, 2] == 0
    with pytest.raises(ValueError, match="gamma"):
        S.development_pressure(dev, 2, gamma=None)


def test_demand_and_potential_tables():
    d = S.DevelopmentDemand(pd.DataFrame({"1": [3.0, 4.0]}, index=[2021, 2022]))
    assert d.subregions == [1] and d.steps == 2 and "DevelopmentDemand" in repr(d)
    with pytest.raises(ValueError, match="negative"):
        S.DevelopmentDemand({1: [3, -1]})
    with pytest.raises(ValueError, match="subregion of each"):
        S.DevelopmentDemand(np.ones((2, 2)))
    p = S.DevelopmentPotential(pd.DataFrame({"Intercept": [-1.0], "development_pressure": [0.2], "x": [0.1]},
                                            index=[5]))
    assert p.predictors == ["x"] and p.row(5)["devpressure"] == 0.2
    one = S.DevelopmentPotential({"intercept": -1.0, "devpressure": 0.2})
    assert one.row(99)["intercept"] == -1.0
    with pytest.raises(ValueError, match="intercept and devpressure"):
        S.DevelopmentPotential({"a": 1.0})


def test_patch_library_and_settings():
    pg = S.PatchGrowing([10, 3, 1, 0], discount_factor=0.5)
    sizes, single = pg._patches([1, 2])
    assert single and sizes == [[5, 1]]
    per = S.PatchGrowing(pd.DataFrame({1: [4.0, 2.0], 2: [np.nan, np.nan]}))._patches([1, 2])
    assert per == ([[4, 2], [1]], False)  # a subregion without patches gets size 1, as the module
    with pytest.raises(ValueError, match="no patch sizes"):
        S.PatchGrowing({1: [2]})._patches([1, 2])
    for kw in ({"num_neighbors": 6}, {"seed_search": "best"}, {"approach": "linear"}, {"incentive_power": 11}):
        with pytest.raises(ValueError):
            S.PatchGrowing([1], **kw)


# ----------------------------------------------------------------------------- simulation

def _toy(**kw):
    w = simulate_cases.futures_world()
    layers = {b: w.sel(band=b).drop_vars("band") for b in w.band.values}
    pot = S.DevelopmentPotential(pd.DataFrame(simulate_cases._FUT_POT, columns=[
        "ID", "intercept", "devpressure", "slope", "road", "forest"]).set_index("ID"))
    demand = S.DevelopmentDemand({r: simulate_cases._FUT_DEMAND[r] for r in (30, 10, 20)})
    alloc = S.PatchGrowing(simulate_cases._FUT_PATCHES, subregions=layers["sub"], neighborhood=8, **kw)
    drivers = {k: layers[k] for k in ("slope", "road", "forest")}
    return S.Model(pot, demand, alloc), layers, drivers


def test_growth_is_monotone_and_reproducible():
    model, layers, drivers = _toy()
    sim = model.run(layers["developed"], drivers, seed=5)
    m = sim.maps.values
    assert m.shape == (6, 60, 80)
    assert (np.nan_to_num(np.diff(m, axis=0)) >= 0).all()  # developed cells stay developed
    lake = np.isnan(layers["developed"].values)
    assert np.isnan(m[:, lake]).all()
    code = sim.development.values
    for k in range(1, 6):
        assert ((code[k] == k) == ((m[k] == 1) & (m[k - 1] == 0))).all()
    np.testing.assert_array_equal(model.run(layers["developed"], drivers, seed=5).maps.values, m)
    steps = sim.steps
    assert {"developed_30", "overflow_10"} <= set(steps.columns)
    total = sum(simulate_cases._FUT_DEMAND[r][0] for r in (30, 10, 20))
    assert abs(int(np.nansum(m[1] - m[0])) - total) <= 25  # patches may overshoot into the next step


def test_restrictions_and_errors():
    model, layers, drivers = _toy()
    blocked = np.zeros(layers["developed"].shape, dtype=bool)
    blocked[:, 60:] = True
    model.rules.restrictions = blocked
    sim = model.run(layers["developed"], drivers, seed=2)
    assert np.isnan(sim.maps.values[:, blocked]).all()
    with pytest.raises(ValueError, match="drivers lack"):
        model.run(layers["developed"], {"slope": layers["slope"]}, seed=1)
    model.demand = S.DevelopmentDemand({30: [5], 10: [5]})
    with pytest.raises(ValueError, match="no demand for subregions"):
        model.run(layers["developed"], drivers, seed=1)
    model.demand = S.AreaDemand(np.ones((2, 1)), classes=[1])
    with pytest.raises(TypeError, match="DevelopmentDemand"):
        model.run(layers["developed"], drivers, seed=1)


def test_zoning_checks():
    model, layers, drivers = _toy(zoning=simulate_cases.futures_world().sel(band="zone").drop_vars("band"))
    sim = model.run(layers["developed"], drivers, seed=1)
    assert sim.maps.shape[0] == 6
    bad = pd.DataFrame({"stringency": [2.0, 1.0, 1.0], 130: [0.0] * 3}, index=[30, 10, 20])
    model.allocation.zoning_effects = bad
    with pytest.raises(ValueError, match="stringency"):
        model.run(layers["developed"], drivers, seed=1)
    model.allocation.zoning_effects = pd.DataFrame({"stringency": [1.0] * 3, 130: [0.0] * 3}, index=[30, 10, 20])
    with pytest.raises(RuntimeError, match="no zoning effect"):
        model.run(layers["developed"], drivers, seed=1)


def test_not_enough_cells_warns():
    model, layers, drivers = _toy()
    model.demand = S.DevelopmentDemand({30: [5000], 10: [5], 20: [5]})
    with pytest.warns(UserWarning, match="not enough undeveloped cells"):
        sim = model.run(layers["developed"], drivers, seed=1)
    sub = layers["sub"].values
    assert np.nansum(sim.maps.values[1][sub == 30] == 0) < np.nansum(layers["developed"].values[sub == 30] == 0)
