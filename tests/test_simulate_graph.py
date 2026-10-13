# SPDX-License-Identifier: GPL-3.0-or-later
"""Models as Terraflux graphs: Model.save / Model.load give the same simulation, and Terraflux runs it bit for bit.

The Terraflux tests need its ``terraflux`` command (the TERRAFLUX variable or PATH); they are skipped without it.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings

import numpy as np
import pytest

import landschaft as ls

S = ls.simulate

PIE_FORMS = {2: ["elevation", "slope", "dist_built_1985"], 1: ["elevation", "slope"], 3: ["elevation", "slope"]}
VIGNETTE = {"jitter": 0.0002, "scale": 0.000001, "max_iter": 1000, "max_diff": 50, "mean_diff": 50}
HAS_TERRAFLUX = bool(os.environ.get("TERRAFLUX") or shutil.which("terraflux"))


@pytest.fixture(scope="module")
def pie():
    return ls.load_raster(S.example_data("pie_landuse")), ls.load_raster(S.example_data("pie_drivers"))


@pytest.fixture(scope="module")
def sibuyan():
    return ls.load_raster(S.example_data("sibuyan_landuse")), ls.load_raster(S.example_data("sibuyan_drivers"))


def _maps(model, landscape, drivers, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.run(landscape, drivers, seed=seed).maps.values


def _check(model, landscape, drivers, seed, tmp_path):
    path = model.save(tmp_path / "model.json", landscape, drivers, seed=seed)
    expected = _maps(model, landscape, drivers, seed)
    again, inputs = S.Model.load(path)
    assert inputs["seed"] == seed
    got = _maps(again, inputs["start"], inputs.get("drivers"), inputs["seed"])
    np.testing.assert_array_equal(got, expected)
    if HAS_TERRAFLUX:
        tf = S.terraflux(path).values
        np.testing.assert_array_equal(tf, expected[1:])
    return path


def _lulcc(pie, allocation, **rules):
    lu, dr = pie
    return S.Model(S.LogisticRegression(PIE_FORMS).fit(lu[0], dr), S.AreaDemand.from_maps(lu, times=[0, 6, 14]),
                   allocation, S.Rules(**rules))


def test_dinamica(sibuyan, tmp_path):
    lu, dr = sibuyan
    path = _check(S.presets.dinamica(lu[0], lu[1], dr, steps=14, woe=250), lu[0], dr, 1, tmp_path)
    g = json.loads(path.read_text())
    assert g["format"] == "terraflux/1"
    assert {n["op"] for n in g["nodes"]} >= {"woe.weights", "markov.matrix", "patches.settings", "control.repeat"}
    assert (tmp_path / "model_data" / "drivers.tif").exists()


def test_clue_with_rules(pie, tmp_path):
    lu, dr = pie
    nan = np.isnan(lu[0].where(lu[0] != 255).values)
    hist = np.where(nan, np.nan, np.random.default_rng(5).integers(0, 13, lu[0].shape).astype(float))
    restricted = np.zeros(lu[0].shape)
    restricted[100:200, 150:300] = 1
    model = _lulcc(pie, S.CLUE(**VIGNETTE), elasticity=[0.2, 0.6, 0.1],
                   allowed=np.array([[1, 1008, 105], [0, 1, 0], [-1, -2, 1]]),
                   history=lu[0].copy(data=hist), restrictions=lu[0].copy(data=restricted),
                   neighbourhood=S.Neighbourhood({1: 0.3, 2: 0.2, 3: 0.3}))
    _check(model, lu[0], dr, 11, tmp_path)


def test_ordered(pie, tmp_path):
    lu, dr = pie
    _check(_lulcc(pie, S.Ordered([2, 1, 3])), lu[0], dr, 3, tmp_path)


def test_dynamic_variables(pie, tmp_path):
    # the distance to built land moves with the simulated map; a share of forest is added
    lu, dr = pie
    lr = S.LogisticRegression({**PIE_FORMS, 1: ["elevation", "slope", "forest_share"]})
    start = lu[0]
    fitted = lr.fit(start, {**{str(n): dr[k] for k, n in enumerate(dr.band.values)},
                            "forest_share": S.share_of(1, 5)(start)})
    model = S.Model(fitted, S.AreaDemand.from_maps(lu, times=[0, 6, 14]), S.CLUE(**VIGNETTE),
                    S.Rules(elasticity=0.2), dynamic={"dist_built_1985": S.distance_to(2, unit="cells"),
                             "forest_share": S.share_of(1, 5)})
    _check(model, lu[0], dr, 2, tmp_path)


def test_cars(pie, tmp_path):
    lu, dr = pie
    lr = S.LogisticRegression(PIE_FORMS).fit(lu[0], dr)
    model = S.Model(lr, S.Markov.from_maps(lu[0], lu[2], steps=4), S.CARS(seeds={2: 0.3}, step_size=50, max_iter=300))
    _check(model, lu[0], dr, 4, tmp_path)


def test_raster_potential_and_paths(pie, tmp_path):
    lu, dr = pie
    p = S.LogisticRegression(PIE_FORMS).fit(lu[0], dr).predict(dr)
    model = S.Model(S.RasterPotential(p), S.AreaDemand.from_maps(lu, times=[0, 6, 14]), S.Ordered([2, 1, 3]))
    _check(model, lu[0], None, 9, tmp_path)


def test_estimator_as_its_predictions(pie, tmp_path):
    # a scikit-learn estimator goes as its predictions on the start (the drivers given as a path too)
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression as SkLogit

    lu, dr = pie
    est = S.Estimator(SkLogit(max_iter=500), sample=4000, seed=1).fit(lu[0], dr)
    model = S.Model(est, S.AreaDemand.from_maps(lu, times=[0, 6, 14]), S.CLUE(**VIGNETTE), S.Rules(elasticity=0.2))
    path = _check(model, lu[0], dr, 6, tmp_path)
    g = json.loads(path.read_text())
    pot = next(n for n in g["nodes"] if n["id"] == "potential")
    assert pot["op"] == "stack.classes" and "Estimator" in pot["note"]
    again, _ = S.Model.load(path)
    assert isinstance(again.potential, S.RasterPotential)
    ls.write_raster(dr, tmp_path / "drivers.tif")
    model.save(tmp_path / "paths.json", lu[0], "drivers.tif", seed=6)
    again, inputs = S.Model.load(tmp_path / "paths.json")
    np.testing.assert_array_equal(_maps(again, inputs["start"], None, 6), _maps(model, lu[0], dr, 6))


def test_estimator_for_the_expander_and_patcher(sibuyan, tmp_path):
    # transition probabilities of a piece with probability() (MOLUSCE's weights here)
    lu, dr = sibuyan
    woe = S.MolusceWoE().fit(lu[0], lu[1], {"slope": dr.sel(band="slope") // 5,
                                             "elevation": dr.sel(band="elevation") // 100})
    dyn = S.presets.dinamica(lu[0], lu[1], dr, steps=4, woe=250)
    model = S.Model(woe, dyn.demand, dyn.allocation)
    _check(model, lu[0], None, 3, tmp_path)


def test_what_terraflux_cannot_run(pie, tmp_path):
    lu, dr = pie
    regional = S.Regional(np.ones(lu[0].shape), {1: S.AreaDemand.from_maps(lu, times=[0, 6, 14])})
    model = S.Model(S.LogisticRegression(PIE_FORMS).fit(lu[0], dr), regional, S.CLUE())
    with pytest.raises(TypeError, match="regional"):
        model.save(tmp_path / "m.json", lu[0], dr, seed=1)
    with pytest.raises(ValueError, match="JSON"):
        _lulcc(pie, S.CLUE()).save(tmp_path / "m.toml", lu[0], dr, seed=1)
    bad = _lulcc(pie, S.CLUE())
    bad.dynamic = {"x": lambda m: m}
    with pytest.raises(TypeError, match="does not know"):
        bad.save(tmp_path / "m.json", lu[0], dr, seed=1)
    p = S.LogisticRegression(PIE_FORMS).fit(lu[0], dr).predict(dr)
    moving = S.Model(S.RasterPotential(p), S.AreaDemand.from_maps(lu, times=[0, 6, 14]), S.CLUE(),
                     dynamic={"dist_built_1985": S.distance_to(2)})
    moving.potential = type("Fitted", (), {"predict": lambda self, *a, **k: p})()
    with pytest.raises(ValueError, match="at every step"):
        moving.save(tmp_path / "m.json", lu[0], dr, seed=1)


@pytest.mark.skipif(HAS_TERRAFLUX, reason="terraflux is installed")
def test_terraflux_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TERRAFLUX", raising=False)
    with pytest.raises(FileNotFoundError, match="terraflux"):
        S.terraflux(tmp_path / "m.json")
