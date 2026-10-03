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


MIXED = ("potential_intercept", "potential_slope", "potential_dredge")


@pytest.mark.parametrize("case", simulate_reference.case_ids("futures"))
def test_same_as_reference(case):
    ref = simulate_reference.result("futures", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))
    for key, value in ours.items():
        if key == "curves.txt":  # numpy and scipy versions differ between the two sides: the last digits
            for a, b in zip(_curves(value), _curves(ref[key]), strict=True):
                assert a[:2] == b[:2]
                np.testing.assert_allclose(a[2], b[2], rtol=1e-12)
                np.testing.assert_allclose(a[3], b[3], rtol=1e-12)
        elif case == "futures_potsurface":  # r.mapcalc's exp (glibc) and the MSVC one differ by up to 2 ulps
            simulate_reference.assert_same(value, ref[key], f"{case}/{key}", rtol=1e-15)
        elif case in MIXED and key != "terms":
            # lme4 stops at an absolute deviance tolerance of 1e-5: the optimum here is lower or equal
            if key == "m2ll":
                assert value[0] <= ref[key][0] + 1e-9 and ref[key][0] - value[0] < 1e-3
            else:
                np.testing.assert_allclose(value, ref[key], rtol=1e-3, atol=1e-4, err_msg=f"{case}/{key}")
        else:
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


# ----------------------------------------------------------------------------- floods, calibration, potential

def _flood_model(**kw):
    f = simulate_cases.futures_flood_world()
    layers = {str(b): f.sel(band=b).drop_vars("band") for b in f.band.values}
    damage = pd.DataFrame([simulate_cases._DDF["curves"][10]], columns=list(simulate_cases._DDF["levels"]))
    depths = {1: {p: layers[f"fd_1_{int(p * 1000):03d}"] for p in simulate_cases._FLOOD_PROBS}}
    flooding = S.Flooding(layers["huc"], layers["ac"], damage, (0.5, 0.35, 0.3, 0.5), depths=depths, **kw)
    model, wl, drivers = _toy()
    model.allocation.climate = flooding
    return model, wl, drivers


def test_flooding_abandons_and_adapts():
    model, layers, drivers = _flood_model()
    sim = model.run(layers["developed"], drivers, seed=2)
    code = sim.development.values
    assert (code[1:] <= -2).any()  # abandoned cells
    assert sim.floods.columns.tolist() == ["step", "catchment", "probability"] and len(sim.floods) > 0
    assert set(sim.floods.catchment) <= {101, 102, 103, 104}
    assert np.nanmax(sim.adaptation.values) in (0, 2, 5, 10, 20, 50, 100)
    abandoned = code[-1] <= -2
    assert (sim.maps.values[-1][abandoned] == 0).all()
    assert not hasattr(sim, "redistribution")


def test_flooding_checks():
    f = simulate_cases.futures_flood_world()
    huc = f.sel(band="huc").drop_vars("band")
    with pytest.raises(ValueError, match="either depths or probabilities"):
        S.Flooding(huc, huc, pd.DataFrame([[1.0]], columns=[0.5]), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="HAND"):
        S.Flooding(huc, huc, pd.DataFrame([[1.0]], columns=[0.5]), (0, 0, 0, 0), probabilities={1: huc})
    with pytest.raises(ValueError, match="response"):
        S.Flooding(huc, huc, pd.DataFrame([[1.0]], columns=[0.5]), (0, 0, 0), depths={1: {0.1: huc}})
    model, layers, drivers = _flood_model()
    model.allocation.climate.depths = {1: {0.5: f.sel(band="fd_1_500").drop_vars("band")},
                                       3: {0.2: f.sel(band="fd_3_200").drop_vars("band")}}
    with pytest.raises(ValueError, match="lacks depth maps"):
        model.run(layers["developed"], drivers, seed=1)
    model, layers, drivers = _flood_model(redistribution=pd.DataFrame({30: [100.0]}, index=[30]))
    with pytest.raises(ValueError, match="population"):
        model.run(layers["developed"], drivers, seed=1)


def test_patch_library_shapes():
    a = np.zeros((6, 8))
    b = a.copy()
    b[0, 0:3] = 1           # 3 cells, in subregion 1
    b[2:4, 2:4] = 1         # 4 cells, crossing into subregion 2 (2 + 2)
    b[5, 7] = 1             # 1 cell in subregion 2
    sub = np.ones((6, 8))
    sub[:, 3:] = 2
    assert S.patch_library(a, b) == [1, 3, 4]
    assert S.patch_library(a, b, subregions=sub, per_subregion=True) == {1: [2, 3], 2: [1, 2]}
    assert S.patch_library(a, b, threshold=3.5) == [4]


def test_calibrate_patches_table():
    w = simulate_cases.futures_world()
    maps = simulate_cases._fut_observed(w)
    model, _, drivers = _toy()
    t = S.calibrate_patches(model, maps[0], maps[3], drivers, compactness_means=(0.3, 0.6),
                            discount_factors=(1.0,), repeat=1, seed=2)
    assert list(t.columns) == ["discount_factor", "compactness_mean", "compactness_range", "area_error",
                               "compactness_error", "combined_error"]
    assert len(t) == 2 and t["combined_error"].is_monotonic_increasing and t["combined_error"].max() <= 1
    assert isinstance(t.attrs["library"], list) and len(t.attrs["library"]) > 0


def test_potential_fit_and_surface():
    samples = simulate_cases.futures_samples()
    pot = S.DevelopmentPotential.fit(samples, pressure="pressure", subregions="county")
    assert len(pot.table) == 10 and pot.attrs["aic"] > 0
    assert set(pot.predictors) == {"x1", "x2", "x3"}
    one = S.DevelopmentPotential.fit(samples, pressure="pressure", subregions=None,
                                     predictors=["x1", "x2", "x3", "pressure"])
    assert len(one.table) == 1
    rng = np.random.default_rng(0)
    drivers = {k: rng.normal(size=(4, 5)) for k in ("x1", "x2", "x3")}
    surf = one.surface(drivers, np.zeros((4, 5))).values
    row = one.table.iloc[0]
    expect = 1 / (1 + np.exp(-(row["intercept"] + sum(row[k] * drivers[k] for k in ("x1", "x2", "x3")))))
    np.testing.assert_allclose(surf, expect, rtol=1e-15)
