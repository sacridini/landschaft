# SPDX-License-Identifier: GPL-3.0-or-later
"""Dynamic variables and regional demands (ls.simulate phase 7a)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import simulate_cases
import xarray as xr
from scipy import ndimage

import landschaft as ls

S = ls.simulate


def _land(n=40, seed=1):
    rng = np.random.default_rng(seed)
    land = np.ones((n, n))
    land[5:12, 5:12] = 2
    land[25:35, 20:30] = 3
    land[rng.random((n, n)) < 0.02] = 3
    return land


CLUE = S.CLUE(jitter=1e-5, scale=1e-4, max_diff=1, mean_diff=1, max_iter=5000)  # meets the demand within a cell


class Recording:
    """A potential from fixed maps that records the drivers it was given (per class or per transition)."""

    def __init__(self, layers, names, dim="class", effect=True):
        self.layers, self.names, self.dim, self.effect, self.seen = layers, names, dim, effect, []

    def predict(self, drivers=None, *, landscape=None, **kw):
        self.seen.append({k: np.array(v.as_float() if hasattr(v, "as_float") else v) for k, v in
                          (drivers or {}).items()})
        d2 = self.seen[-1].get("d2")
        layers = [lay if d2 is None or not self.effect else lay * np.exp(-np.nan_to_num(d2) / 10)
                  for lay in self.layers]
        return xr.DataArray(np.stack(layers), dims=(self.dim, "y", "x"), coords={self.dim: self.names})

    def probability(self, landscape, drivers=None):
        return self.predict(drivers, landscape=landscape)


# ----------------------------------------------------------------------------- pieces

def test_distance_to_euclidean():
    land = _land()
    land[0, 0] = np.nan
    d = S.distance_to([2, 3])(simulate_cases.grid(land, res=30.0))
    ref = ndimage.distance_transform_edt(~np.isin(land, [2, 3]), sampling=30.0)
    ref[0, 0] = np.nan
    np.testing.assert_allclose(d.values, ref, rtol=1e-12)
    assert d.name == "distance_to_2_3" and d.rio.crs is not None
    cells = S.distance_to(2, unit="cells")(simulate_cases.grid(land))
    np.testing.assert_allclose(cells.values[~np.isnan(land)],
                               ndimage.distance_transform_edt(land != 2)[~np.isnan(land)], rtol=1e-12)
    none = S.distance_to(9)(np.ones((3, 4)))
    np.testing.assert_allclose(none, 5.0)  # the diagonal
    with pytest.raises(ValueError, match="square"):
        S.distance_to(2)(simulate_cases.grid(land).rio.write_transform(
            simulate_cases.grid(land).rio.transform() * __import__("affine").Affine.scale(1, 2)))


def test_distance_to_by_cost_follows_a_road():
    n = 30
    land = np.ones((n, n))
    land[15, 0] = 2
    cond = np.full((n, n), 0.1)
    cond[15, :] = 10.0  # a road along row 15
    tr = ls.cost.transition(cond, "mean", 8)
    d = S.distance_to(2, cost=tr)(land)
    np.testing.assert_allclose(d, np.asarray(ls.cost.acc_cost(tr, (land == 2).astype(np.uint8))))
    assert d[15, 29] < d[0, 15]  # far along the road is nearer than a little off it


def test_share_of():
    land = _land()
    s = S.share_of(2, window=5)(land)
    pres = (land == 2).astype(float)
    num = ndimage.convolve(pres, np.ones((5, 5)), mode="constant")
    den = ndimage.convolve(np.ones_like(pres), np.ones((5, 5)), mode="constant")
    np.testing.assert_allclose(s, num / den, rtol=1e-12)
    with pytest.raises(ValueError, match="odd"):
        S.share_of(2, window=np.ones((2, 3)))


# ----------------------------------------------------------------------------- dynamic variables in the models

def _check_dynamic(pot, sim):
    """The driver of each step is the distance to class 2 in the map at the start of that step."""
    seen = [s["d2"] for s in pot.seen]
    assert len(seen) == sim.maps.shape[0] - 1
    for k, d in enumerate(seen):
        ref = S.distance_to(2)(sim.maps.values[k])
        np.testing.assert_allclose(d, ref)


def test_dynamic_in_clue_and_ordered():
    land = _land()
    rng = np.random.default_rng(2)
    layers = [rng.random(land.shape) * 0.5 for _ in range(3)]
    start = np.array([np.sum(land == c) for c in (1, 2, 3)], float)
    demand = S.AreaDemand(np.array([start + k * np.array([-40, 40, 0]) for k in range(4)]), classes=[1, 2, 3])
    for alloc in (S.Ordered(order=[2, 3, 1]),
                  CLUE):
        pot = Recording(layers, [1, 2, 3])
        sim = S.Model(pot, demand, alloc, dynamic={"d2": S.distance_to(2)}).run(land, seed=1)
        _check_dynamic(pot, sim)
        tol = 1 if isinstance(alloc, S.CLUE) else 0
        for k in range(4):
            assert abs(np.sum(sim.maps.values[k] == 2) - (start[1] + 40 * k)) <= tol
    static = Recording(layers, [1, 2, 3])
    a = S.Model(static, demand, S.Ordered(order=[2, 3, 1])).run(land, {"d2": S.distance_to(2)(land)}, seed=1)
    assert len(static.seen) == 1  # without dynamic variables the potential is predicted once
    b = S.Model(Recording(layers, [1, 2, 3]), demand, S.Ordered(order=[2, 3, 1]),
                dynamic={"d2": S.distance_to(2)}).run(land, {"d2": S.distance_to(2)(land)}, seed=1)
    np.testing.assert_array_equal(a.maps.values[1], b.maps.values[1])  # step 1: the same driver
    assert (a.maps.values[3] != b.maps.values[3]).any()


def test_dynamic_in_cars():
    land = _land()
    rng = np.random.default_rng(3)
    pot = Recording([rng.random(land.shape) * 0.6 for _ in range(3)], [1, 2, 3])
    start = np.array([np.sum(land == c) for c in (1, 2, 3)], float)
    demand = S.AreaDemand(np.array([start + k * np.array([-30, 30, 0]) for k in range(3)]), classes=[1, 2, 3])
    sim = S.Model(pot, demand, S.CARS(step_size=10), dynamic={"d2": S.distance_to(2)}).run(land, seed=2)
    _check_dynamic(pot, sim)
    assert sim.steps["converged"].all()


def test_dynamic_in_expander_patcher():
    land = _land()
    rng = np.random.default_rng(4)
    pot = Recording([rng.random(land.shape)], ["probability_1_to_2"], dim="band")
    alloc = S.ExpanderPatcher(steps=3, expander=(4, 2, 1.5), patcher=(4, 2, 1.5))
    sim = S.Model(pot, {(1, 2): 0.02}, alloc, dynamic={"d2": S.distance_to(2)}).run(land, seed=3)
    _check_dynamic(pot, sim)
    with pytest.raises(ValueError, match="dynamic variables need"):
        S.Model({(1, 2): np.ones(land.shape)}, {(1, 2): 0.02}, alloc, dynamic={"d2": S.distance_to(2)}).run(land)


def test_dynamic_rejected_by_futures():
    with pytest.raises(ValueError, match="no dynamic variables"):
        S.Model(None, None, S.PatchGrowing.__new__(S.PatchGrowing),
                dynamic={"d": S.distance_to(1)}).run(np.zeros((3, 3)))


# ----------------------------------------------------------------------------- regional demands

def _regions(n=40):
    r = np.ones((n, n))
    r[:, n // 2:] = 2
    r[:2, :] = np.nan  # no region: the cells keep their class
    return r


def test_regional_from_maps_and_markov():
    t0 = _land()
    t1 = t0.copy()
    t1[12:16, 5:12] = 2  # region 1 only
    reg = _regions()
    d = S.Regional.from_maps(reg, [t0, t1])
    assert d.ids == [1, 2] and d.steps == 1 and d.classes == [1, 2, 3]
    for rid in (1, 2):
        inside = reg == rid
        np.testing.assert_array_equal(d.at(1, [1, 2, 3], region=rid), [np.sum(t1[inside] == c) for c in (1, 2, 3)])
    np.testing.assert_array_equal(d.at(1, [1, 2, 3]), d.at(1, [1, 2, 3], region=1) + d.at(1, [1, 2, 3], region=2))
    m = S.Regional.markov(reg, t0, t1, steps=2)
    assert 2 not in m.demands[2].single.columns and m.demands[1].single.loc[1, 2] > 0
    with pytest.raises(ValueError, match="different numbers of steps"):
        S.Regional(reg, {1: S.Markov.from_maps(t0, t1, steps=2), 2: S.Markov.from_maps(t0, t1, steps=3)})


def _regional_demand(land, reg, gains):
    demands = {}
    for rid, g in gains.items():
        inside = reg == rid
        start = np.array([np.sum(land[inside] == c) for c in (1, 2, 3)], float)
        demands[rid] = S.AreaDemand(np.array([start + k * np.asarray(g, float) for k in range(3)]), classes=[1, 2, 3])
    return S.Regional(reg, demands)


@pytest.mark.parametrize("alloc", [S.Ordered(order=[2, 3, 1]),
                                   CLUE,
                                   S.CARS(step_size=10)])
def test_regional_allocations_meet_each_region(alloc):
    land = _land()
    reg = _regions()
    rng = np.random.default_rng(5)
    pot = S.RasterPotential({c: rng.random(land.shape) * 0.5 for c in (1, 2, 3)})
    demand = _regional_demand(land, reg, {1: [-30, 30, 0], 2: [-10, 0, 10]})
    sim = S.Model(pot, demand, alloc).run(land, seed=1)
    m = sim.maps.values
    tol = 1 if isinstance(alloc, S.CLUE) else 0
    for k in range(3):
        for rid in (1, 2):
            inside = reg == rid
            np.testing.assert_allclose([np.sum(m[k][inside] == c) for c in (1, 2, 3)],
                                       demand.at(k, [1, 2, 3], region=rid), atol=tol)
    np.testing.assert_array_equal(m[:, np.isnan(reg)], np.broadcast_to(land[np.isnan(reg)], (3, 80)))


def test_one_region_is_the_plain_model():
    land = _land()
    rng = np.random.default_rng(6)
    pot = S.RasterPotential({c: rng.random(land.shape) * 0.5 for c in (1, 2, 3)})
    start = np.array([np.sum(land == c) for c in (1, 2, 3)], float)
    d = S.AreaDemand(np.array([start + k * np.array([-30, 30, 0]) for k in range(3)]), classes=[1, 2, 3])
    for alloc in (CLUE, S.CARS(step_size=10)):
        a = S.Model(pot, d, alloc).run(land, seed=9).maps.values
        b = S.Model(pot, S.Regional(np.ones(land.shape), {1: d}), alloc).run(land, seed=9).maps.values
        np.testing.assert_array_equal(a, b)


def test_regional_expander_patcher():
    land = _land()
    reg = _regions()
    rng = np.random.default_rng(7)
    prob = rng.random(land.shape)
    demand = S.Regional(reg, {1: {(1, 2): 0.05}, 2: {(1, 3): 0.05}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = S.Model({(1, 2): prob, (1, 3): prob}, demand,
                      S.ExpanderPatcher(steps=2, expander=(3, 1, 1.5), patcher=(3, 1, 1.5))).run(land, seed=4)
    m = sim.maps.values
    for k in (1, 2):
        new2 = (m[k] == 2) & (m[k - 1] == 1)
        new3 = (m[k] == 3) & (m[k - 1] == 1)
        assert new2.sum() > 0 and new3.sum() > 0
        assert (reg[new2] == 1).all() and (reg[new3] == 2).all()
        assert new2.sum() == np.floor(0.05 * np.sum((m[k - 1] == 1) & (reg == 1)) + 0.5)
        assert sim.steps.loc[k, "requested_1_2"] == new2.sum()
    assert (m[:, np.isnan(reg)] == land[np.isnan(reg)]).all()


def test_dynamic_in_clumondo():
    model, land, drivers = simulate_cases.clumondo_model(simulate_cases._clumondo_services(), {})
    p = model.potential.predict(drivers)
    pot = Recording([p.values[c] for c in range(5)], list(range(5)), effect=False)
    model.potential = pot
    model.dynamic = {"d2": S.distance_to(2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model.run(land, seed=3)
    _check_dynamic(pot, sim)
