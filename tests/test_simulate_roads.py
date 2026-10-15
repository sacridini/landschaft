# SPDX-License-Identifier: GPL-3.0-or-later
"""Endogenous road building (Dinamica EGO's CalcCostMap, CalcDestinationMap, CalcPathwayMap)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import simulate_reference
from scipy.spatial.distance import pdist

import landschaft as ls

S = ls.simulate
CASE = ("roads", "roads_dinamica")
CONFIGS = [(tag, fmap, diag) for tag, fmap in (("real", "fr"), ("int", "fri")) for diag in (True, False)]


def _key(tag, diag):
    return f"{tag}_{'diag' if diag else 'flat'}"


@pytest.mark.parametrize(("tag", "fmap", "diag"), CONFIGS)
def test_cost_paths_as_dinamica(tag, fmap, diag):
    ref = simulate_reference.result(*CASE)
    w = simulate_cases.build_input("roads_world")
    k = _key(tag, diag)
    cost, dirs = S.friction_cost(w["net"], w[fmap], diagonal=diag, directions=True)
    simulate_reference.assert_same(cost, ref[f"cost_{k}.tif"], f"cost_{k}")
    # directions: Dinamica's where there is one cheapest predecessor; else both are cheapest predecessors
    c, fr = np.asarray(cost), np.asarray(w[fmap])
    d = np.asarray(dirs).astype(float)
    d[d == 255] = np.nan
    ref_d = ref[f"dir_{k}.tif"]
    assert np.array_equal(np.isnan(d), np.isnan(ref_d))
    step = np.sqrt(2) if diag else 1.0
    for i, j in np.argwhere((d != ref_d) & ~np.isnan(d)):
        for code in (d[i, j], ref_d[i, j]):
            a, b = S._roads.DIRECTIONS[int(code)]
            assert c[i + a, j + b] + fr[i, j] * (step if a and b else 1) == c[i, j]
    if tag == "real":
        assert np.array_equal(d, ref_d, equal_nan=True)
    simulate_reference.assert_same(S.pathways(w["net"], w["src"], cost), ref[f"path_{k}.tif"], f"path_{k}")
    simulate_reference.assert_same(S.pathways(w["net"], w["src"], cost, merge=False), ref[f"new_{k}.tif"], f"new_{k}")
    # along Dinamica's own directions
    simulate_reference.assert_same(S.pathways(w["net"], w["src"], directions=ref_d, merge=False),
                                   ref[f"dirpath_{k}.tif"], f"dirpath_{k}")


def test_relative_friction_as_dinamica():
    ref = simulate_reference.result(*CASE)
    w = simulate_cases.build_input("roads_world")
    simulate_reference.assert_same(S.friction_cost(w["net"], w["fr"], relative=True), ref["cost_relative.tif"])


def test_destinations_as_dinamica():
    ref = simulate_reference.result(*CASE)
    w = simulate_cases.build_input("roads_world")
    att = np.asarray(w["att"])

    def stats(maps):
        big, mind, mean = [], [], []
        for d in maps:
            cells = np.argwhere(~np.isnan(d) & (d > 0))
            assert len(cells) == 12  # 3 in each of the 2 x 2 quadrants
            assert (att[tuple(cells.T)] > 0).all()
            big.append((7, 9) in {tuple(c) for c in cells})
            mind.append(pdist(cells * 30.0).min())
            mean.append(att[tuple(cells.T)][att[tuple(cells.T)] < 200].mean())
        return np.mean(big), np.min(mind), np.mean(mean), np.std(mean, ddof=1) / np.sqrt(len(mean))

    dinamica = stats([ref[f"dest_{k}.tif"] for k in range(simulate_cases._DEST_DRAWS)])
    ours = stats([np.asarray(S.destinations(w["att"], (2, 2), per_quadrant=3, min_distance=90, seed=k))
                  for k in range(300)])
    assert dinamica[1] >= 90 and ours[1] >= 90  # never closer than the minimum distance
    assert abs(dinamica[0] - ours[0]) < 0.15  # the attractive cell
    assert abs(dinamica[2] - ours[2]) < 4 * np.hypot(dinamica[3], ours[3])


def test_destinations_quadrants_and_draw():
    # the quadrants cover the map (Dinamica leaves out the last n % q rows and columns)
    a = np.zeros((20, 23))
    a[:, 22] = 1
    d = np.asarray(S.destinations(a, (3, 2), seed=1))
    assert np.argwhere(~np.isnan(d)).tolist() == [[r, 22] for r in sorted(np.argwhere(~np.isnan(d))[:, 0])]
    assert np.sum(~np.isnan(d)) == 2
    # in proportion to the attraction (Dinamica rounds its draw and can take a cell of none)
    a = np.ones((7, 10))
    a[4, 6] = 100
    a[6, 9] = 0
    hits = [np.asarray(S.destinations(a, seed=k)) for k in range(2000)]
    assert abs(np.mean([h[4, 6] == 100 for h in hits]) - 100 / 168) < 0.04
    assert not any(not np.isnan(h[6, 9]) for h in hits)


def test_pathway_ties_and_lottery():
    ny, nx = 7, 11
    rr, cc = np.mgrid[0:ny, 0:nx]
    cost = np.minimum(cc, 10 - cc) + np.abs(rr - 3) * 0.5  # west and east tie
    net = np.zeros((ny, nx))
    net[:, [0, 10]] = 1
    src = np.zeros((ny, nx))
    src[3, 5] = 1
    p = np.asarray(S.pathways(net, src, cost, merge=False))
    assert np.argwhere(~np.isnan(p)).tolist() == [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5]]  # west: first in row order
    sides = {bool(np.argwhere(~np.isnan(np.asarray(S.pathways(net, src, cost, merge=False, lottery=True,
                                                                  seed=k))))[:, 1].min() < 5) for k in range(40)}
    assert sides == {True, False}
    bad = cost.copy()
    bad[3, 4] = bad[3, 6] = 9
    bad[2:5, 4] = bad[2:5, 6] = 9
    bad[[2, 4], 5] = 9
    with pytest.raises(RuntimeError, match="local minimum"):
        S.pathways(net, src, bad)
    with pytest.raises(ValueError, match="cost or the directions"):
        S.pathways(net, src)


def test_road_builder_steps_and_dynamic():
    rng = np.random.default_rng(3)
    net = np.zeros((60, 60))
    net[0, :] = 1
    land = np.ones((60, 60))
    land[30:, :] = 2
    attraction = simulate_cases.grid(rng.random((60, 60)))
    rb = S.RoadBuilder(np.ones((60, 60)), attraction, quadrants=(3, 3), distance=(150, 600), spacing=150)
    da = simulate_cases.grid(net)
    one = rb.step(da, seed=1)
    new = ~np.isnan(np.asarray(one)) & (net == 0)
    assert new.any() and one.rio.crs == da.rio.crs
    # every new road reaches the old network (8-connected)
    from scipy import ndimage

    lab, n = ndimage.label(~np.isnan(np.asarray(one)), structure=np.ones((3, 3)))
    assert n == 1
    nets = rb.run(da, 3, seed=2)
    assert nets.shape == (4, 60, 60)
    counts = [int(np.sum(~np.isnan(nets[k].values))) for k in range(4)]
    assert counts[0] == 60 and counts == sorted(counts)
    # attraction and mask as functions of the map: roads end only in class 2
    rb2 = S.RoadBuilder(np.ones((60, 60)), lambda m: np.where(np.asarray(m) == 2, 1.0, 0.0), quadrants=(2, 2),
                        distance=(30 * 30, 30 * 40), spacing=0, mask=lambda m: np.asarray(m) == 2)
    two = np.asarray(rb2.step(da, landscape=simulate_cases.grid(land), seed=4))
    added = ~np.isnan(two) & (net == 0)
    assert added.any() and added[30:].any()
    # as a dynamic variable: a step of roads each time the map changes
    dyn = rb.dynamic(da, seed=5)
    l0 = simulate_cases.grid(land)
    d0 = np.asarray(dyn(l0))
    assert len(dyn.networks) == 1 and d0[0].max() == 0 and d0[10, 0] == 300
    np.testing.assert_array_equal(np.asarray(dyn(l0)), d0)  # the same map: no new roads
    l1 = l0.copy(data=np.where(np.arange(60)[:, None] < 31, 1.0, 2.0) * np.ones((1, 60)))
    d1 = np.asarray(dyn(l1))
    assert len(dyn.networks) == 2 and (d1 <= d0).all() and (d1 < d0).any()
    dyn.reset()
    assert len(dyn.networks) == 1


def test_road_distance_in_a_model():
    rng = np.random.default_rng(6)
    land = np.ones((40, 40))
    land[:5] = 2
    net = np.zeros((40, 40))
    net[0, :] = 1
    rb = S.RoadBuilder(np.ones((40, 40)), rng.random((40, 40)), quadrants=(2, 2), distance=(4, 15), spacing=4)
    roads = rb.dynamic(net, seed=1)

    class Near:
        def __init__(self):
            self.seen = []

        def probability(self, landscape, drivers=None):
            import xarray as xr

            d = np.asarray(drivers["roads"].as_float() if hasattr(drivers["roads"], "as_float") else drivers["roads"])
            self.seen.append(d)
            return xr.DataArray(np.exp(-d / 5)[None], dims=("band", "y", "x"),
                                coords={"band": ["probability_1_to_2"]})

    pot = Near()
    sim = S.Model(pot, {(1, 2): 0.02}, S.ExpanderPatcher(steps=3), dynamic={"roads": roads}).run(land, seed=2)
    assert len(pot.seen) == 3 and len(roads.networks) == 3
    assert (pot.seen[2] <= pot.seen[0]).all() and (pot.seen[2] < pot.seen[0]).any()
    assert sim.maps.shape[0] == 4


def test_friction_cost_edges():
    with pytest.raises(ValueError, match="positive"):
        S.friction_cost(np.ones((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="georeferenced"):
        S.friction_cost(np.ones((2, 2)), np.ones((2, 2)), relative=True)
    c = S.friction_cost(np.zeros((2, 3)), np.ones((2, 3)))
    assert np.isnan(c).all()
    with pytest.raises(ValueError, match="quadrants"):
        S.destinations(np.ones((3, 3)), (0, 1))
