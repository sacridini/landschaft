# SPDX-License-Identifier: GPL-3.0-or-later
"""Scenarios and replicates in parallel, their summaries and plots (ls.simulate phase 7c)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import xarray as xr

import landschaft as ls

S = ls.simulate


def _model(n=40):
    land = np.ones((n, n))
    land[10:20, 10:20] = 2
    land[30:, :5] = 3
    pot = S.RasterPotential({1: np.full((n, n), 0.2), 2: np.random.default_rng(0).random((n, n)),
                             3: np.full((n, n), 0.1)})
    start = np.array([np.sum(land == c) for c in (1, 2, 3)], float)
    demand = S.AreaDemand(np.array([start + k * np.array([-50, 50, 0]) for k in range(4)]), classes=[1, 2, 3])
    fast = S.AreaDemand(np.array([start + k * np.array([-100, 100, 0]) for k in range(4)]), classes=[1, 2, 3])
    return S.Model(pot, demand, S.CARS(step_size=10)), simulate_cases.grid(land), fast


def test_replicates_are_the_single_runs_with_any_workers():
    model, land, _ = _model()
    a = S.scenarios(model, land, replicates=4, seed=10, n_workers=1)
    b = S.scenarios(model, land, replicates=4, seed=10, n_workers=4, n_threads=4)
    np.testing.assert_array_equal(a.codes, b.codes)
    assert a.seeds == [10, 11, 12, 13] and a.scenarios == ["baseline"]
    for r in (0, 3):
        np.testing.assert_array_equal(a.maps(replicate=r).values, model.run(land, seed=10 + r).maps.values)
    assert set(a.steps.columns) >= {"scenario", "replicate", "step", "converged"}
    assert len(a.steps) == 4 * 3


def test_scenarios_share_their_draws():
    model, land, fast = _model()
    ens = S.scenarios(model, land, replicates=3, seed=1, years=2020,
                      scenarios={"trend": {}, "same": lambda m: m, "fast": {"demand": fast}})
    np.testing.assert_array_equal(ens.codes[0], ens.codes[1])  # common random numbers
    cover = ens.cover().set_index(["scenario", "replicate", "year"])
    assert (cover.loc["fast"][2.0].xs(2023, level="year") == 100 + 300).all()
    assert (cover.loc["trend"][2.0].xs(2023, level="year") == 100 + 150).all()
    p = ens.probability(2, year=2023, scenario="fast")
    assert float(p.sum()) == pytest.approx(400)
    total = sum(ens.probability(c).values for c in ens.classes)
    np.testing.assert_allclose(total, 1.0)
    ch = ens.change_probability(scenario="trend")
    assert ch.sel(year=2020).values.max() == 0 and ch.sel(year=2023).values.max() <= 1
    assert ens.probability(2).rio.crs is not None


def test_years_keep_and_dataset():
    model, land, _ = _model()
    ens = S.scenarios(model, land, replicates=2, seed=1, years=[2000, 2010, 2020, 2030], keep=[2000, 2030])
    assert ens.years == [2000, 2030] and ens.codes.shape[:3] == (1, 2, 2)
    ds = ens.to_dataset()
    assert isinstance(ds, xr.Dataset) and ds.landuse.dims == ("scenario", "replicate", "year", "y", "x")
    with pytest.raises(ValueError, match="years for"):
        S.scenarios(model, land, replicates=1, years=[1, 2])


def test_from_a_projection():
    lu = ls.load_raster(S.example_data("sibuyan_landuse"))
    dr = ls.load_raster(S.example_data("sibuyan_drivers")).sel(band=["dist_sea", "dist_roads"])
    proj = S.run(lu[0], lu[1], dr, until=2014, seed=1, woe=250, validate=False)
    ens = S.scenarios(proj, replicates=2, seed=1)
    assert ens.years == [2011, 2012, 2013, 2014]
    # run() projects with seed + 1: replicate 0 with seed 2 is the projection itself
    again = S.scenarios(proj, replicates=1, seed=2)
    np.testing.assert_array_equal(again.maps().values, proj.maps.values)


def test_metrics():
    model, land, fast = _model()
    ens = S.scenarios(model, land, replicates=2, seed=1, years=2020, scenarios={"trend": {}, "fast": {"demand": fast}})
    m = ens.metrics(["pland", "np"], years=[2023])
    assert set(m.columns) == {"scenario", "replicate", "year", "metric", "class", "value"}
    pl = m[(m.metric == "pland") & (m["class"] == 2)].set_index(["scenario", "replicate"])["value"]
    assert pl.loc["fast"].iloc[0] == pytest.approx(100 * 400 / 1600)
    f = ens.metrics({"share2": lambda a: float((a == 2).mean()), "both": lambda a: {"a": 1.0, "b": 2.0}},
                    years=[2020])
    assert set(f.metric) == {"share2", "both_a", "both_b"} and len(f) == 2 * 2 * 3


def test_errors():
    model, land, _ = _model()
    with pytest.raises(ValueError, match="changes potential"):
        S.scenarios(model, land, scenarios={"x": {"colour": 1}})
    with pytest.raises(TypeError, match="Model or a Projection"):
        S.scenarios("model", land)
    with pytest.raises(ValueError, match="replicates"):
        S.scenarios(model, land, replicates=0)
    with pytest.raises(ValueError, match="name the scenario"):
        S.scenarios(model, land, replicates=1, scenarios={"a": {}, "b": {}}).maps()


def test_plots():
    model, land, fast = _model()
    sim = model.run(land, seed=1)
    kinds = [(p.kind, p.title) for p in ls.plot(sim).panels]
    assert kinds[0][0] == "map" and ("chart", "cover") in kinds and ("chart", "flows") in kinds
    ens = S.scenarios(model, land, replicates=2, seed=1, scenarios={"trend": {}, "fast": {"demand": fast}})
    kinds = [(p.kind, p.title) for p in ls.plot(ens).panels]
    assert kinds[0][0] == "map" and ("chart", "cover") in kinds
    lu = ls.load_raster(S.example_data("sibuyan_landuse"))
    dr = ls.load_raster(S.example_data("sibuyan_drivers")).sel(band=["dist_sea", "dist_roads"])
    proj = S.run(lu[0], lu[1], dr, until=2013, seed=1, woe=250)
    kinds = [p.kind for p in ls.plot(proj).panels]
    assert kinds.count("chart") >= 3 and "html" in kinds  # no georeference: the last map as a chart
