# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain hydrology (ls.hydro) against WhiteboxTools 2.4 and its own properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls

H = ls.hydro
REF = Path(__file__).parent / "data" / "reference" / "hydro" / "whitebox.npz"
CASES = ["oregon", "holes", "walk", "terraces"]


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


def _kw(ref, name):
    dem = ref[f"{name}/dem"]
    dx, dy = ref[f"{name}/cell"]
    return dict(extent=(0.0, dem.shape[1] * dx, 0.0, dem.shape[0] * dy))


def _codes(a):
    """Our D8 codes as WhiteboxTools stores them (-1 where missing)."""
    return np.where(np.isnan(a), -1, a).astype(np.int16)


@pytest.mark.parametrize("name", CASES)
def test_directions_and_accumulations_match_whitebox(ref, name):
    kw = _kw(ref, name)
    dem, filled = ref[f"{name}/dem"], ref[f"{name}/filled"]
    valid = ~np.isnan(dem)
    np.testing.assert_array_equal(_codes(H.flow_direction(dem, flats=False, **kw).values), ref[f"{name}/d8_raw"])
    np.testing.assert_array_equal(_codes(H.flow_direction(filled, flats=False, **kw).values), ref[f"{name}/d8"])
    for units, key in (("cells", "d8acc"), ("sca", "d8sca")):    # stored as float32 by WhiteboxTools
        ours = H.flow_accumulation(filled, flats=False, units=units, **kw).values
        np.testing.assert_array_equal(ours.astype(np.float32), ref[f"{name}/{key}"].astype(np.float32))
    dinf = H.flow_direction(filled, method="dinf", flats=False, **kw).values
    np.testing.assert_array_equal(dinf[valid], ref[f"{name}/dinf"][valid])   # (WhiteboxTools: -1 where missing)
    # stored as float32 (WhiteboxTools sums in the order its threads hand over the rows: the
    # same float32 values, the last float64 bits may differ)
    for method, key in (("dinf", "dinfacc"), ("fd8", "fd8acc")):
        ours = H.flow_accumulation(filled, method=method, flats=False, **kw).values
        np.testing.assert_array_equal(ours.astype(np.float32), ref[f"{name}/{key}"].astype(np.float32))
    streams = (ref[f"{name}/streams"] == 1).astype(float)
    np.testing.assert_array_equal(H.hand(filled, streams, flats=False, **kw).values, ref[f"{name}/hand"])


@pytest.mark.parametrize("name", ["oregon", "walk", "terraces"])
def test_fill_is_whitebox_with_the_edge_as_outlet(ref, name):
    kw = _kw(ref, name)
    np.testing.assert_array_equal(H.fill(ref[f"{name}/dem"], **kw).values, ref[f"{name}/filled_edge"])


def test_whitebox_raises_depressions_at_the_edge(ref):
    # FillDepressions keeps water from leaving at the edge: edge cells raised
    dem, wbt = ref["walk/dem"], ref["walk/filled"]
    edge = np.zeros(dem.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    raised = edge & (wbt > dem)
    assert raised.sum() > 0
    ours = H.fill(dem, **_kw(ref, "walk")).values
    np.testing.assert_array_equal(ours[edge], dem[edge])


def test_missing_cells_are_outlets():
    dem = np.full((5, 5), 9.0)
    dem[1:4, 1:4] = 5.0
    dem[2, 2] = 1.0
    filled = H.fill(dem).values
    assert filled[2, 2] == 9.0 and filled[1, 1] == 9.0   # a basin raised to its rim
    dem[2, 3] = np.nan                            # a missing cell next to the pit: water leaves there
    filled = H.fill(dem).values
    assert filled[2, 2] == 1.0 and filled[1, 1] == 5.0 and np.isnan(filled[2, 3])


def _path_ends(codes, valid):
    """Each valid cell's walk downstream must end at a cell without direction, without cycles."""
    step = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    ny, nx = codes.shape
    for r in range(ny):
        for c in range(nx):
            if not valid[r, c]:
                continue
            seen = 0
            rr, cc = r, c
            while codes[rr, cc] > 0:
                dr, dc = step[int(codes[rr, cc])]
                rr, cc = rr + dr, cc + dc
                seen += 1
                assert seen <= ny * nx, "a cycle"
            yield rr, cc


def test_flats_drain_to_the_edge(ref):
    dem = ref["terraces/dem"]
    kw = _kw(ref, "terraces")
    filled = H.fill(dem, **kw).values
    codes = H.flow_direction(filled, **kw).values
    ny, nx = codes.shape
    for r, c in _path_ends(codes, ~np.isnan(filled)):
        assert r in (0, ny - 1) or c in (0, nx - 1)          # every walk leaves at the edge
    plain = H.flow_direction(filled, flats=False, **kw).values
    assert (plain[1:-1, 1:-1] == 0).sum() > 1000           # many flat cells without it


def test_drain_flats_fill_slopes_to_the_outlets(ref):
    dem = ref["walk/dem"]
    kw = _kw(ref, "walk")
    eps = H.fill(dem, drain_flats=True, **kw).values
    plain = H.flow_direction(eps, flats=False, **kw).values
    assert (plain[1:-1, 1:-1] == 0).sum() == 0               # no interior cell without direction
    assert np.all(eps >= dem)
    assert np.max(eps - H.fill(dem, **kw).values) < 1e-9


@pytest.mark.parametrize("method", ["d8", "dinf", "fd8", "quinn"])
def test_accumulation_conserves_the_area(ref, method):
    kw = _kw(ref, "holes")
    filled = H.fill(ref["holes/dem"], **kw).values
    acc = H.flow_accumulation(filled, method=method, **kw).values
    d8 = H.flow_direction(filled, **kw).values
    valid = ~np.isnan(filled)
    # what leaves the map: the accumulation of the cells that pass nothing on
    exits = valid & (d8 == 0)
    assert exits.sum() > 0
    if method == "d8":
        assert acc[exits].sum() == valid.sum()
    else:  # multiple-flow methods also pass flow off the edge from cells that have a direction
        assert acc[exits].sum() <= valid.sum() + 1e-6


def test_threads_do_not_change_the_results(ref):
    kw = _kw(ref, "oregon")
    dem = ref["oregon/dem"]
    for method in ("d8", "dinf"):
        a = H.flow_direction(dem, method=method, n_threads=1, **kw).values
        b = H.flow_direction(dem, method=method, n_threads=4, **kw).values
        np.testing.assert_array_equal(a, b)


def test_units_and_weights():
    dem = np.array([[4, 4, 4], [3, 3, 3], [2, 1, 2]], float)
    kw = dict(extent=(0.0, 30.0, 0.0, 60.0))                 # cells 10 wide, 20 high
    cells = H.flow_accumulation(dem, **kw).values
    np.testing.assert_array_equal(H.flow_accumulation(dem, units="area", **kw).values, cells * 200.0)
    np.testing.assert_array_equal(H.flow_accumulation(dem, units="sca", **kw).values, cells * 200.0 / 15.0)
    w = np.full(dem.shape, 2.0)
    np.testing.assert_array_equal(H.flow_accumulation(dem, weights=w, **kw).values, cells * 2.0)


def test_quinn_splits_by_slope_and_contour_length():
    dem = np.array([[9, 9, 9], [9, 4, 9], [2, 2, 2]], float)
    only_centre = np.zeros(dem.shape)
    only_centre[1, 1] = 1.0
    for p in (1.0, 2.0):
        acc = H.flow_accumulation(dem, method="quinn", exponent=p, weights=only_centre, flats=False).values
        straight, diagonal = 2.0 ** p * 0.5, (2.0 / np.sqrt(2.0)) ** p * 0.354
        total = straight + 2 * diagonal
        assert acc[2, 1] == pytest.approx(straight / total, rel=1e-15)
        assert acc[2, 0] == pytest.approx(diagonal / total, rel=1e-15)
    fd8 = H.flow_accumulation(dem, method="fd8", weights=only_centre, flats=False).values
    straight, diagonal = 2.0 ** 1.1, (2.0 / np.sqrt(2.0)) ** 1.1
    assert fd8[2, 1] == pytest.approx(straight / (straight + 2 * diagonal), rel=1e-15)


def test_hand_from_a_threshold_matches_given_streams(ref):
    kw = _kw(ref, "walk")
    filled = H.fill(ref["walk/dem"], **kw)
    acc = H.flow_accumulation(filled)
    rivers = H.streams(acc, 40)
    np.testing.assert_array_equal(H.hand(filled, rivers).values, H.hand(filled, threshold=40).values)
    h = H.hand(filled, rivers).values
    assert np.nanmin(h) == 0.0 and np.all(h[rivers.values == 1] == 0)


def test_longitude_latitude_cells_in_metres():
    import xarray as xr

    dem = np.array([[3, 3, 3], [3, 2, 3], [3, 1, 3]], float)
    da = xr.DataArray(dem, dims=("y", "x"), coords={"y": [0.002, 0.001, 0.0], "x": [0.0, 0.001, 0.002]})
    da = da.rio.write_crs("EPSG:4326")
    area = H.flow_accumulation(da, units="area").values
    cell = 0.001 * 111320.0 * 0.001 * 110574.0
    assert area[0, 0] == pytest.approx(cell, rel=1e-3)


@pytest.mark.parametrize("name", CASES)
def test_indices_match_whitebox_given_the_same_area_and_slope(ref, name):
    kw = _kw(ref, name)
    filled, sca, slope = ref[f"{name}/filled"], ref[f"{name}/fd8sca"], ref[f"{name}/slope"]
    ours = H.flow_accumulation(filled, method="fd8", units="sca", flats=False, **kw).values
    np.testing.assert_array_equal(ours.astype(np.float32), sca.astype(np.float32))
    for fn, key in ((H.twi, "twi"), (H.spi, "spi"), (H.sti, "sti")):     # stored as float32
        out = fn(filled, sca=sca, slope=slope, **kw).values
        np.testing.assert_array_equal(out.astype(np.float32), ref[f"{name}/{key}"].astype(np.float32))


def test_horn_gradient_of_a_plane():
    from landschaft.hydro._indices import _tan_slope
    from landschaft.surface._grid import read_surface

    y, x = np.mgrid[0:6, 0:7]
    dem = 0.3 * x * 10.0 - 0.2 * (5 - y) * 20.0                  # cells 10 wide, 20 high
    t = _tan_slope(read_surface(dem, extent=(0.0, 70.0, 0.0, 120.0)))
    np.testing.assert_allclose(t[1:-1, 1:-1], np.hypot(0.3, 0.2), rtol=1e-14)


def test_wetness_on_flats_and_the_slope_floor():
    dem = np.zeros((5, 5))
    dem[0] = 1.0
    w = H.twi(dem).values
    assert np.isnan(w[3, 2])                                     # flat: tan(slope) = 0
    w = H.twi(dem, min_slope=0.001).values
    assert np.isfinite(w[3, 2])


@pytest.mark.parametrize("name", ["oregon", "walk", "terraces"])
@pytest.mark.parametrize("min_dist", [True, False])
def test_breach_is_whitebox_least_cost(ref, name, min_dist):
    # WhiteboxTools on one thread (on several it takes equal pits in the order its threads finish),
    # on the DEM in a frame of low cells: channels that reach the edge are cut
    kw = _kw(ref, name)
    out = H.breach(ref[f"{name}/dem"], max_length=20, min_dist=min_dist, fill_rest=False, **kw).values
    np.testing.assert_array_equal(out, ref[f"{name}/{'breach_dist' if min_dist else 'breach_cost'}"])


def test_breach_cuts_a_channel_to_the_edge():
    dem = np.full((5, 6), 9.0)
    dem[2, 1:5] = [5.0, 1.0, 6.0, 6.0]
    dem[2, 5] = 7.0                                   # the edge cell is higher than the pit
    out = H.breach(dem, flat_increment=0.01, fill_rest=False).values
    walk = H.flow_direction(out, flats=False).values
    ends = list(_path_ends(walk, np.ones(dem.shape, bool)))
    assert all(r in (0, 4) or c in (0, 5) for r, c in ends)   # every cell drains off the map


def test_breach_fills_what_it_cannot_cut(ref):
    kw = _kw(ref, "walk")
    dem = ref["walk/dem"]
    left = H.breach(dem, max_length=2, max_cost=0.5, fill_rest=False, **kw).values
    done = H.breach(dem, max_length=2, max_cost=0.5, **kw).values
    np.testing.assert_array_equal(done, H.fill(left, **kw).values)
    plain = H.flow_direction(done, flats=True, **kw).values
    assert (plain[1:-1, 1:-1] == 0).sum() == 0
