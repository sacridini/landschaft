# SPDX-License-Identifier: GPL-3.0-or-later
"""Presets and the one-call projection (ls.simulate phase 7b)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import xarray as xr

import landschaft as ls

S = ls.simulate


@pytest.fixture(scope="module")
def sibuyan():
    lu = ls.load_raster(S.example_data("sibuyan_landuse"))
    dr = ls.load_raster(S.example_data("sibuyan_drivers"))
    return lu, dr.sel(band=["dist_sea", "dist_roads"])


def test_observed_patches():
    t0 = np.ones((12, 12))
    t0[0:3, 0:3] = 2
    t1 = t0.copy()
    t1[3, 0:3] = 2          # an expansion of 3 cells
    t1[8:10, 8:10] = 2      # a new patch of 4
    t1[6, 11] = 2           # a new patch of 1
    tab = S.presets.observed_patches(t0, t1).set_index(["from", "to"]).loc[(1, 2)]
    assert tab["cells"] == 8 and tab["expansion"] == pytest.approx(3 / 8)
    assert tab["expansion_patches"] == 1 and tab["expansion_mean"] == 3 and tab["expansion_variance"] == 0
    assert tab["new_patches"] == 2 and tab["new_mean"] == 2.5 and tab["new_variance"] == pytest.approx(2.25)


def test_dinamica_preset(sibuyan):
    lu, dr = sibuyan
    model = S.presets.dinamica(lu[0], lu[1], dr, steps=14, woe=250)
    assert isinstance(model.potential, S.WeightsOfEvidence) and isinstance(model.allocation, S.ExpanderPatcher)
    obs = S.presets.observed_patches(lu[0], lu[1]).set_index(["from", "to"])
    for tr, share in model.allocation.expansion.items():
        assert share == obs.loc[tr, "expansion"]
        assert model.allocation.patcher[tr][:2] == (obs.loc[tr, "new_mean"], obs.loc[tr, "new_variance"])
    fixed = S.presets.dinamica(lu[0], lu[1], dr, steps=14, woe=250, patches=0.7, isometry=1.2)
    assert set(fixed.allocation.expansion.values()) == {0.7}
    assert {v[2] for v in fixed.allocation.expander.values()} == {1.2}


def test_plus_preset_models_the_classes_that_expanded(sibuyan):
    pytest.importorskip("sklearn")
    lu, dr = sibuyan
    model = S.presets.plus(lu[0], lu[1], dr, steps=14, seed=1, step_size=50)
    a, b = lu[0].values, lu[1].values
    grew = sorted({float(c) for c in np.unique(b[(a != b) & ~np.isnan(a) & ~np.isnan(b)])})
    assert sorted(model.potential.models) == grew
    assert isinstance(model.demand, S.Markov) and isinstance(model.allocation, S.CARS)
    assert model.allocation.step_size == 50


def test_stratified_sample_keeps_rare_changes():
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeClassifier

    t0 = np.ones((50, 50))
    t1 = t0.copy()
    t1[0, :3] = 2  # 3 cells of 2500
    drivers = {"x": np.tile(np.arange(50.0), (50, 1))}
    est = S.Estimator(DecisionTreeClassifier(random_state=0), target="expansion", classes=[2], sample=0.01,
                      stratify=True, seed=1).fit(t0, drivers, t1=t1)
    idx, y = est._train[2]
    assert y.sum() == 1 and idx.size == 1 + round(0.01 * 2497)
    with pytest.raises(ValueError, match="all absences"):
        S.Estimator(DecisionTreeClassifier(), target="expansion", classes=[2], sample=0.01, seed=1).fit(
            t0, drivers, t1=t1)


def test_molusce_preset_is_the_manual_model():
    w = simulate_cases.molusce_world()
    t0, t1 = w.sel(band="t0"), w.sel(band="t1")
    drivers = {k: w.sel(band=k) for k in ("dist", "soil", "slope")}
    model = S.presets.molusce(t0, t1, drivers, bins=simulate_cases._MOL_BINS, iterations=3)
    sim = model.run(t1, seed=7)
    ref = simulate_cases.molusce_py(w)
    for k in (1, 2, 3):
        np.testing.assert_array_equal(sim.maps.values[k], ref[f"map{k}"])


def test_piece_presets():
    pot = S.RasterPotential({1: np.zeros((3, 3)), 2: np.ones((3, 3))})
    demand = S.AreaDemand(np.array([[5, 4], [4, 5]]), classes=[1, 2])
    m = S.presets.clue_s(pot, demand, jitter=0.0, max_iter=10)
    assert isinstance(m.allocation, S.CLUE) and m.allocation.max_iter == 10
    m = S.presets.ordered(pot, demand, [2, 1], stochastic=False)
    assert isinstance(m.allocation, S.Ordered) and m.allocation.order == [2, 1]
    assert np.sum(m.run(np.array([[1, 1, 1], [1, 1, 2], [2, 2, 2]]), seed=1).maps.values[-1] == 2) == 5


def test_run_calibrates_validates_and_projects(sibuyan):
    lu, dr = sibuyan
    proj = S.run(lu[0], lu[1], dr, until=2016, seed=3, woe=250)
    assert proj.settings["years"] == [1997, 2011] and proj.settings["seed"] == 3
    assert proj.maps["year"].values.tolist() == list(range(2011, 2017))
    np.testing.assert_array_equal(proj.maps.values[0], lu[1].where(lu[1] != 255).values)  # starts at t1
    assert proj.calibration.maps.shape[0] == 15
    v = proj.validation
    assert set(v) >= {"overall", "figure_of_merit", "hits", "misses", "false_alarms", "wrong_hits", "quantity"}
    assert 0 < v["figure_of_merit"] < 1
    assert list(proj.report["stage"]) == ["fit", "validation", "projection"]
    ds = proj.to_dataset()
    assert isinstance(ds, xr.Dataset) and ds["calibration"]["calibration_year"].values[-1] == 2011
    again = proj.redo(seed=3)
    np.testing.assert_array_equal(again.maps.values, proj.maps.values)
    other = proj.redo(seed=4, until=2013, validate=False)
    assert other.calibration is None and other.maps["year"].values[-1] == 2013


def test_run_with_years_and_a_custom_preset(sibuyan):
    lu, dr = sibuyan

    def mine(t0, t1, drivers, *, steps, **kw):
        return S.presets.dinamica(t0, t1, drivers, steps=steps, woe=500, patches=0.5, **kw)

    a, b = (np.where(m.values == 255, np.nan, m.values) for m in (lu[0], lu[1]))
    proj = S.run(a, b, dr, years=(0, 2), until=4, preset=mine, seed=1)
    assert proj.settings["preset"] == "mine" and proj.maps["year"].values.tolist() == [2, 3, 4]
    only = S.run(lu[0], lu[1], dr, seed=1, woe=500)
    assert only.maps.shape[0] == 1 and only.simulation is None


def test_agreement():
    start = np.array([[1, 1, 1, 1]])
    obs = np.array([[2, 2, 1, 1]])
    sim = np.array([[2, 3, 2, 1]])
    a = S.agreement(obs, sim, start)
    assert (a["hits"], a["wrong_hits"], a["misses"], a["false_alarms"]) == (1, 1, 0, 1)
    assert a["figure_of_merit"] == pytest.approx(1 / 3) and a["overall"] == 0.5
    assert a["quantity"] == {1: -1, 2: 0, 3: 1}


def test_run_errors(sibuyan):
    lu, dr = sibuyan
    with pytest.raises(ValueError, match="after the earlier"):
        S.run(lu[0], lu[1], dr, years=(2011, 1997))
    with pytest.raises(ValueError, match="until"):
        S.run(lu[0], lu[1], dr, until=2000)
    with pytest.raises(ValueError, match="preset"):
        S.run(lu[0], lu[1], dr, preset="sleuth")
