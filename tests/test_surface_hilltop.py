# SPDX-License-Identifier: GPL-3.0-or-later
"""Hilltops as permanent protection areas (APP de topo de morro, Lei 12.651/2012)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

import landschaft as ls

EIGHT = np.ones((3, 3))
EXT = (0.0, 1500.0, 0.0, 1000.0)  # 10 m cells on 100 x 150


def _grid():
    yy, xx = np.mgrid[0:100, 0:150] * 10.0 + 5.0
    return yy, xx


def _cone(cy, cx, h, run, base=500.0):
    yy, xx = _grid()
    return base + np.clip(h - h / run * np.hypot(yy - cy, xx - cx), 0, None)


def _component(mask, cell):
    lab, _ = ndimage.label(mask, structure=EIGHT)
    return lab == lab[cell]


def test_cone_on_a_plain():
    dem = _cone(505, 705, 300, 300)  # 45 degrees
    h = ls.surface.hilltop(dem, extent=EXT)
    top = h.hills.iloc[0]
    assert (top.top, top.base, top.height, top.base_kind) == (800.0, 500.0, 300.0, "lowest")
    assert 43 < top.mean_slope < 46 and 43 < top.line_slope < 46
    np.testing.assert_array_equal(h.app.values, (dem >= 700).astype(float))
    assert h.hills["qualifies"].sum() == 1 and int(top.app_cells) == int(np.sum(dem >= 700))
    assert np.isnan(h.hill.values[dem < 700]).all() and (h.hill.values[dem >= 700] == 0).all()


def test_criteria():
    gentle = _cone(505, 705, 300, 900)  # 18 degrees
    assert ls.surface.hilltop(gentle, extent=EXT).hills["qualifies"].sum() == 0
    assert ls.surface.hilltop(gentle, extent=EXT, min_slope=15).hills["qualifies"].sum() == 1
    low = _cone(505, 705, 80, 80)
    assert ls.surface.hilltop(low, extent=EXT).hills["qualifies"].sum() == 0
    conama = ls.surface.hilltop(low, extent=EXT, min_height=50, min_slope=16.7, slope="line")
    assert conama.hills["qualifies"].sum() == 1
    upper = ls.surface.hilltop(low, extent=EXT, min_height=50, fraction=0.5)
    np.testing.assert_array_equal(upper.app.values, (low >= 540).astype(float))


def test_saddle_range_and_plain():
    # a massif rising 400 m from the plain with two peaks (900 and 880 m) and a saddle at about 850 m,
    # and an isolated hill whose base is a lake at 560 m
    yy, xx = _grid()
    massif = 500 + np.clip(400 - 1.2 * np.hypot(yy - 505, xx - 405), 0, None)
    second = 500 + np.clip(380 - 1.2 * np.hypot(yy - 505, xx - 475), 0, None)
    dem = np.maximum(massif, second)
    hill = _cone(505, 1205, 250, 200, base=560.0)
    dem = np.maximum(dem, np.where(np.hypot(yy - 505, xx - 1205) < 300, hill, 0))
    lake = (np.hypot(yy - 505, xx - 1205) >= 200) & (np.hypot(yy - 505, xx - 1205) < 300)
    dem[lake] = 560.0
    h = ls.surface.hilltop(dem, extent=EXT, plain=lake)
    t = h.hills
    big = t.iloc[0]
    assert big.top == 900 and big.base == 500 and big.qualifies
    peak2 = t[(t.x - 470).abs() < 30].iloc[0]
    assert peak2.base_kind == "saddle" and peak2.height < 100 and not peak2.qualifies
    iso = t[(t.x - 1200).abs() < 30].iloc[0]
    assert iso.base_kind == "plain" and iso.base == 560 and iso.qualifies
    # the range's APP holds both peaks: above 500 + 2/3 x 400 and joined to the highest
    thr = big.base + 2 / 3 * big.height
    yb, xb = int(big.y // 10), int(big.x // 10)
    expect = _component(dem >= thr, (99 - yb, xb))
    app = h.app.values == 1
    np.testing.assert_array_equal(app & (xx < 900), expect & (xx < 900))
    assert app[50, 47]  # the second peak
    np.testing.assert_array_equal(app & (xx > 900), (dem >= 560 + 2 / 3 * iso.height) & (xx > 900))
    assert set(np.unique(h.hill.values[app])) == {float(big.name), float(iso.name)}


def _naive(dem):
    """Bases from components above each level (no ties: random floats): the highest level at which a
    summit's component holds a higher cell; the area's lowest elevation for the highest summit."""
    ny, nx = dem.shape
    out = {}
    levels = np.sort(np.unique(dem))[::-1]
    for s in np.argsort(-dem, axis=None, kind="stable"):
        cell = divmod(int(s), nx)
        # summits: no higher neighbour already joined at its level means a local maximum
        nb = dem[max(0, cell[0] - 1):cell[0] + 2, max(0, cell[1] - 1):cell[1] + 2]
        if dem[cell] < nb.max():
            continue
        base = None
        for lv in levels[levels < dem[cell]]:
            comp = _component(dem >= lv, cell)
            if dem[comp].max() > dem[cell]:
                base = lv
                break
        if base is None:
            base = dem.min()
        out[cell] = base
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bases_and_app_against_components(seed):
    rng = np.random.default_rng(seed)
    dem = ndimage.gaussian_filter(rng.random((18, 22)), 1.5) * 1000 + rng.random((18, 22)) * 1e-3
    h = ls.surface.hilltop(dem, extent=(0, 220, 0, 180), min_height=5, min_slope=0)
    naive = _naive(dem)
    t = h.hills
    cells = {(int((180 - y) // 10), int(x // 10)): row for (y, x), row in
             zip(zip(t.y, t.x), t.itertuples())}
    assert set(cells) == set(naive)
    for cell, base in naive.items():
        assert cells[cell].base == base
    # the APP: each qualifying hill's component above its threshold, joined to its summit
    expect = np.zeros(dem.shape, bool)
    for cell, row in cells.items():
        if row.qualifies:
            expect |= _component(dem >= row.threshold, cell)
            # its mean slope: the cells above its base joined to it
            region = _component(dem > row.base, cell)
            sl = ls.surface.terrain(dem, extent=(0, 220, 0, 180)).values[region]
            assert row.mean_slope == pytest.approx(np.nanmean(sl), rel=1e-12)
    np.testing.assert_array_equal(h.app.values == 1, expect)


def test_threads_and_errors():
    rng = np.random.default_rng(9)
    dem = ndimage.gaussian_filter(rng.random((200, 260)), 6) * 3000
    a = ls.surface.hilltop(dem, extent=(0, 2600, 0, 2000), n_threads=1)
    b = ls.surface.hilltop(dem, extent=(0, 2600, 0, 2000), n_threads=4)
    np.testing.assert_array_equal(a.hill.values, b.hill.values)
    pd_equal = a.hills.equals(b.hills)
    assert pd_equal
    nan_dem = dem.copy()
    nan_dem[:20] = np.nan
    c = ls.surface.hilltop(nan_dem, extent=(0, 2600, 0, 2000))
    assert np.isnan(c.app.values[:20]).all()
    with pytest.raises(ValueError, match="slope"):
        ls.surface.hilltop(dem, slope="steepest")
    with pytest.raises(ValueError, match="fraction"):
        ls.surface.hilltop(dem, fraction=1.5)
    with pytest.raises(ValueError, match="plain"):
        ls.surface.hilltop(dem, plain=np.ones((3, 3)))


def test_against_dinamica_calc_hilltop():
    # Dinamica EGO's CalcHilltop (CONAMA 303/2002) on cones on a plain, the DEM in 10 m bands: the same heights
    # (its base is the mean elevation of the plain's band), but its dip slope measures from the summit to the
    # nearest cell of that band bordering other features or the map's edge, far below the cones' slopes
    import simulate_cases
    import simulate_reference

    ref = simulate_reference.result("hilltop", "hilltop_dinamica")
    world = simulate_cases.build_input("hilltop_world")
    ours = ls.surface.hilltop(world["dem"], min_height=0, min_slope=0).hills
    ours = ours[ours.height > 0].sort_values("top").reset_index(drop=True)
    true_slope = np.degrees(np.arctan([h / run for _, _, h, run in simulate_cases._HILL_CONES]))
    np.testing.assert_array_equal(ours.top, ref["maxima.csv/Value"])
    np.testing.assert_array_equal(ours.height, [h for _, _, h, _ in simulate_cases._HILL_CONES])
    np.testing.assert_allclose(ref["height.csv/Value"], ours.height, atol=0.1)
    np.testing.assert_allclose(ours.line_slope, true_slope, atol=0.5)
    assert (np.abs(ours.mean_slope - true_slope) < 2.5).all()  # Horn at the apex and the foot
    assert (ref["dip.csv/Value"] < true_slope - 10).all()
