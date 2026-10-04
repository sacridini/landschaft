# SPDX-License-Identifier: GPL-3.0-or-later
"""PLUS (ls.simulate phase 6): LEAS through Estimator(target="expansion") and the CARS allocation.

PLUS is distributed without source code or licence, so CARS is written from the paper (Liang et al. 2021)
and checked by its properties: demand met, patch growth from neighbours, random seeds, the decreasing
threshold, the rules, reproducibility with any number of threads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import simulate_cases
from scipy import ndimage

import landschaft as ls
from landschaft import _core

S = ls.simulate


def _world(n=60, seed=3):
    """Class 1 everywhere, a block of 2 and a strip of 3; P(2) rises to the east, P(3) flat."""
    land = np.ones((n, n))
    land[20:30, 10:20] = 2
    land[:, -4:] = 3
    rng = np.random.default_rng(seed)
    p2 = np.clip(np.linspace(0.1, 0.9, n)[None, :] + 0.1 * rng.standard_normal((n, n)), 0, 1)
    p1 = np.full((n, n), 0.3)
    p3 = np.full((n, n), 0.2)
    return simulate_cases.grid(land), S.RasterPotential({1: p1, 2: p2, 3: p3})


def _demand(land, gains, steps=2):
    v = land.values
    start = np.array([np.sum(v == c) for c in (1, 2, 3)], dtype=float)
    rows = [start]
    for _ in range(steps):
        rows.append(rows[-1] + np.asarray(gains, dtype=float))
    return S.AreaDemand(np.array(rows), classes=[1, 2, 3])


def _near(mask, cls_map, window=3):
    """Cells of mask with a cell of cls_map in the window around them (the centre left out)."""
    k = np.ones((window, window), int)
    return (ndimage.convolve(cls_map.astype(int), k, mode="constant") - cls_map.astype(int))[mask] > 0


def test_demand_met_every_step():
    land, pot = _world()
    sim = S.Model(pot, _demand(land, [-150, 120, 30], steps=3), S.CARS(step_size=20)).run(land, seed=1)
    m = sim.maps.values
    assert m.shape == (4, 60, 60)
    for k in range(4):
        n0 = np.sum(land.values == 1)
        assert np.sum(m[k] == 1) == n0 - 150 * k
        assert np.sum(m[k] == 2) == 100 + 120 * k
        assert np.sum(m[k] == 3) == 240 + 30 * k
    assert sim.steps["converged"].all() and (sim.steps["gap"] == 0).all()
    assert list(sim.steps.columns[:4]) == ["iterations", "converged", "decays", "gap"]


def test_reproducible_with_any_threads():
    land, pot = _world()
    model = S.Model(pot, _demand(land, [-150, 120, 30]), S.CARS(step_size=20))
    a = model.run(land, seed=7, n_threads=1).maps.values
    b = model.run(land, seed=7, n_threads=4).maps.values
    c = model.run(land, seed=8, n_threads=4).maps.values
    np.testing.assert_array_equal(a, b)
    assert (a != c).any()


def test_patches_grow_from_neighbours_without_seeds():
    land, pot = _world()
    sim = S.Model(pot, _demand(land, [-200, 200, 0], steps=1), S.CARS(seeds=0.0, step_size=20)).run(land, seed=2)
    end = sim.maps.values[-1]
    new = (end == 2) & (land.values != 2)
    assert new.sum() == 200
    assert _near(new, end == 2).all()  # every new cell touches class 2
    _, n = ndimage.label(end == 2, np.ones((3, 3)))
    assert n == 1  # one patch, grown from the block


def test_random_seeds_start_new_patches():
    land, pot = _world()
    counts = []
    for mu in (0.0, 0.5, 1.0):
        sim = S.Model(pot, _demand(land, [-200, 200, 0], steps=1), S.CARS(seeds={2: mu}, step_size=20)).run(
            land, seed=4)
        counts.append(ndimage.label(sim.maps.values[-1] == 2, np.ones((3, 3)))[1])
    assert counts[0] == 1 and counts[1] > 1 and counts[2] >= counts[1]


def test_growth_follows_probability():
    land, pot = _world()
    sim = S.Model(pot, _demand(land, [-300, 300, 0], steps=1), S.CARS(seeds=0.5, step_size=20)).run(land, seed=5)
    new = (sim.maps.values[-1] == 2) & (land.values == 1)
    p2 = pot.predict().sel({"class": 2}).values
    assert p2[new].mean() > p2[land.values == 1].mean() + 0.1


def test_threshold_decays_when_progress_is_slow():
    land, pot = _world()
    fast = S.Model(pot, _demand(land, [-150, 150, 0], steps=1), S.CARS(step_size=0)).run(land, seed=1)
    slow = S.Model(pot, _demand(land, [-150, 150, 0], steps=1), S.CARS(step_size=1e9)).run(land, seed=1)
    assert fast.steps.loc[1, "decays"] < slow.steps.loc[1, "decays"]
    assert slow.steps.loc[1, "decays"] == slow.steps.loc[1, "iterations"]  # every iteration decays


def test_rules_allowed_and_restrictions():
    land, pot = _world()
    allowed = pd.DataFrame([[1, 1, 0], [1, 1, 1], [0, 0, 1]], index=[1, 2, 3], columns=[1, 2, 3])
    restricted = np.zeros(land.shape, bool)
    restricted[:, 40:50] = True
    rules = S.Rules(allowed=allowed, restrictions=restricted)
    sim = S.Model(pot, _demand(land, [-100, 100, 0], steps=2), S.CARS(step_size=20), rules).run(land, seed=3)
    m = sim.maps.values
    assert (m[:, restricted] == land.values[restricted]).all()
    for k in range(1, 3):
        a, b = m[k - 1], m[k]
        assert not ((a == 1) & (b == 3)).any() and not ((a == 3) & (b != 3)).any()


def test_classes_held_at_their_demand():
    # 3 stays the same: it neither gains nor loses, though P(3) > 0 everywhere
    land, pot = _world()
    sim = S.Model(pot, _demand(land, [-150, 150, 0], steps=2), S.CARS(seeds=1.0, step_size=20)).run(land, seed=6)
    for k in range(3):
        np.testing.assert_array_equal(sim.maps.values[k] == 3, land.values == 3)


def test_markov_demand():
    land, pot = _world()
    end = land.values.copy()
    end[30:40, 10:20] = 2
    markov = S.Markov.from_maps(land, simulate_cases.grid(end), steps=2)
    sim = S.Model(pot, markov, S.CARS(step_size=20)).run(land, seed=1)
    proj = markov.project({c: np.sum(land.values == c) for c in (1, 2, 3)})
    assert sim.maps.shape[0] == 3
    for k in (1, 2):
        cells = [np.sum(sim.maps.values[k] == c) for c in (1, 2, 3)]
        assert sum(cells) == land.size
        np.testing.assert_allclose(cells, proj.loc[k].to_numpy(), atol=1)


def test_nodata_and_new_class():
    land, pot = _world()
    v = land.values.copy()
    v[:3, :3] = np.nan
    land = simulate_cases.grid(v)
    n = int(np.sum(~np.isnan(v)))
    start = [np.sum(v == c) for c in (1, 2, 3)]
    p4 = np.zeros(v.shape)
    p4[50:, :30] = 0.9
    pot4 = S.RasterPotential({**{c: pot.predict().sel({"class": c}).values for c in (1, 2, 3)}, 4: p4})
    table = np.array([start + [0], [start[0] - 50, start[1], start[2], 50]])
    assert table[1].sum() == n
    sim = S.Model(pot4, S.AreaDemand(table, classes=[1, 2, 3, 4]), S.CARS(seeds=1.0, step_size=5)).run(land, seed=2)
    end = sim.maps.values[-1]
    assert np.isnan(end[:3, :3]).all()
    assert np.sum(end == 4) == 50 and (np.argwhere(end == 4)[:, 0] >= 50).all()


def test_adaptive_coefficient():
    # class 0 grows from 100 to 150 cells over class 1: the gaps only shrink, so D stays 1
    land = np.ones((20, 20), np.int32)
    land[:, :5] = 0
    prob = np.zeros((2, 400))
    prob[0] = 0.5
    prob[1] = 0.9
    allowed = np.array([[1, 1], [1, 1]], np.uint8)
    r = _core.sim_cars(land, None, None, prob, np.ones(2), np.full(2, 0.1), allowed, np.array([[150, 250]]), 3, 0.9,
                       5.0, 200, 1)
    assert r["converged"] and r["trace"].shape[1] == 2
    assert (r["trace"][0] == [100, 300]).all() and (r["trace"][-1] == [150, 250]).all()
    d = r["trace_d"]
    assert d.shape == (r["iterations"], 2) and (d > 0).all()
    np.testing.assert_array_equal(d, 1.0)


def test_not_converged_warns():
    land, pot = _world()
    allowed = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    with pytest.warns(UserWarning, match="CARS stopped"):
        sim = S.Model(pot, _demand(land, [-50, 50, 0], steps=1), S.CARS(max_iter=30),
                      S.Rules(allowed=allowed)).run(land, seed=1)
    assert not sim.steps.loc[1, "converged"] and sim.steps.loc[1, "gap"] == 100


def test_errors():
    land, pot = _world()
    d = _demand(land, [-10, 10, 0], steps=1)
    with pytest.raises(ValueError, match="decay"):
        S.CARS(decay=0)
    with pytest.raises(ValueError, match="window"):
        S.CARS(window=4)
    with pytest.raises(ValueError, match="only the rules"):
        S.Model(pot, d, S.CARS(), S.Rules(elasticity=0.5)).run(land, seed=1)
    with pytest.raises(ValueError, match="0/1"):
        S.Model(pot, d, S.CARS(), S.Rules(allowed=np.full((3, 3), 100))).run(land, seed=1)
    with pytest.raises(ValueError, match="adds up"):
        S.Model(pot, S.AreaDemand(np.array([[3256, 100, 240], [3000, 100, 240]]), classes=[1, 2, 3]),
                S.CARS()).run(land, seed=1)
    with pytest.raises(ValueError, match="without demand"):
        S.Model(pot, S.AreaDemand(np.array([[3256, 100], [3256, 100]]), classes=[1, 2]), S.CARS()).run(land, seed=1)
    with pytest.raises(ValueError, match="no growth probability"):
        S.Model(S.RasterPotential({1: np.zeros(land.shape)}), d, S.CARS()).run(land, seed=1)
    with pytest.raises(ValueError, match="seeds"):
        S.Model(pot, d, S.CARS(seeds={9: 0.1})).run(land, seed=1)
    with pytest.raises(TypeError, match="AreaDemand or a Markov"):
        S.Model(pot, {(1, 2): 0.1}, S.CARS()).run(land, seed=1)


def test_leas_and_cars_on_plum_island():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    lu = ls.load_raster(S.example_data("pie_landuse"))
    ef = ls.load_raster(S.example_data("pie_drivers"))
    leas = S.Estimator(RandomForestClassifier(n_estimators=20, random_state=0), target="expansion",
                       sample=0.2, seed=1).fit(lu[0], ef, t1=lu[1])
    imp = leas.importance()
    assert list(imp.columns) == list(leas.models) and imp.shape[0] == len(leas.drivers)
    np.testing.assert_allclose(imp.sum().to_numpy(), 1.0)
    perm = leas.importance("permutation", n_repeats=2, seed=0)
    assert perm.shape == imp.shape
    markov = S.Markov.from_maps(lu[0], lu[1], steps=6)
    sim = S.Model(leas, markov, S.CARS(step_size=50)).run(lu[0], ef, seed=1)
    assert sim.steps["converged"].all()
    proj = markov.project({c: int(np.sum(lu[0].values == c)) for c in markov.classes})
    end = sim.maps.values[-1]
    np.testing.assert_allclose([np.sum(end == c) for c in markov.classes], proj.iloc[-1].to_numpy(), atol=1)
