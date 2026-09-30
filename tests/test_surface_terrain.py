# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface terrain indices against spatialEco / terra (corrected where they err) and unit checks."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("terrain"))
def test_terrain_matches_r(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "terrain", c.tolerance) > 0


def _plane(a=0.3, b=-0.2, shape=(20, 25), res=10.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x, y = (xx + 0.5) * res, (ny - yy - 0.5) * res
    return 100 + a * x + b * y, (0.0, nx * res, 0.0, ny * res)


def test_plane():
    a, b = 0.3, -0.2
    z, e = _plane(a, b)
    inner = (slice(3, -3), slice(3, -3))
    slope = ls.surface.terrain(z, "slope", "radians", extent=e).values
    np.testing.assert_allclose(slope[1:-1, 1:-1], np.arctan(np.hypot(a, b)), rtol=1e-12)
    # the downslope direction: z rises to the east (a > 0) and the south (b < 0) -> faces north-west
    aspect = ls.surface.terrain(z, "aspect", extent=e).values
    np.testing.assert_allclose(aspect[1:-1, 1:-1], np.degrees(np.arctan2(-a, -b)) % 360, rtol=1e-12)
    assert np.isnan(slope[0]).all() and np.isnan(slope[:, -1]).all()
    np.testing.assert_allclose(ls.surface.tpi(z, 5, extent=e).values[inner], 0, atol=1e-10)
    np.testing.assert_allclose(ls.surface.vrm(z, 3, extent=e).values[inner], 0, atol=1e-12)
    for t in ("planform", "profile", "total"):
        np.testing.assert_allclose(ls.surface.curvature(z, t, extent=e).values[inner], 0, atol=1e-12)


def test_hill_and_valley():
    yy, xx = np.mgrid[-10:11, -10:11].astype(float)
    hill = 50 - 0.1 * (xx ** 2 + yy ** 2)
    assert ls.surface.tpi(hill, 5).values[10, 10] > 0
    assert ls.surface.tpi(-hill, 5).values[10, 10] < 0
    assert ls.surface.dissection(hill).values[10, 10] == 1.0
    tri = ls.surface.tri(hill).values
    assert tri[10, 10] == pytest.approx(np.sqrt(4 * 0.1**2 + 4 * 0.2**2))
    assert not np.isnan(tri[0, 0])  # the edges have neighbours too


def test_hli_folds():
    aspect = np.arange(0.0, 360.0, 5.0).reshape(1, -1)
    slope = np.full_like(aspect, 30.0)
    north = ls.surface.hli(aspect, slope, 40).values.ravel()
    # the southern hemisphere folds about NW-SE (315) where the northern folds about SW (225)
    south = ls.surface.hli((aspect + 90) % 360, slope, -40, hemisphere="southern").values.ravel()
    np.testing.assert_allclose(south, north, rtol=1e-12)
    assert 180 < aspect.ravel()[north.argmax()] < 270  # warm south-west slopes
    np.testing.assert_allclose(ls.surface.hli(np.radians(aspect), np.radians(slope), 40, units="radians").values.ravel(),
                               north, rtol=1e-14)


def test_hsp_and_windows():
    z, e = _plane()
    z = z + np.random.default_rng(3).normal(0, 1, z.shape)
    h = ls.surface.hsp(z, 3, 7, 2, normalize=True, extent=e).values
    assert np.nanmin(h) == 0 and np.nanmax(h) == 1
    t3 = ls.surface.tpi(z, (3, 5), extent=e)
    assert t3.shape == z.shape
    with pytest.raises(ValueError, match="odd"):
        ls.surface.tpi(z, 4)
    with pytest.raises(ValueError, match="two cells"):
        ls.surface.tpi(z, 15.0, "circle", extent=e)
    with pytest.raises(ValueError, match="type"):
        ls.surface.curvature(z, "gaussian")
    c = ls.surface.tpi(z, 25.0, "circle", extent=e)
    assert np.isnan(c.values[:2]).all() and not np.isnan(c.values[5:-5, 5:-5]).any()
