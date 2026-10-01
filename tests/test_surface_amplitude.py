# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface amplitude parameters and trend surfaces against geodiv (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("amplitude"))
def test_amplitude_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "amplitude", c.tolerance) > 0


@pytest.mark.parametrize("case", surface_reference.case_ids("plane"))
def test_plane_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "plane", c.tolerance) > 0


def test_amplitude_by_definition():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(40, 50))
    z[rng.random(z.shape) < 0.1] = np.nan
    v = z[~np.isnan(z)]
    n, m, s = v.size, v.mean(), v.std()  # ISO's Sq: over N
    assert ls.surface.sa(z) == pytest.approx(np.abs(v - m).mean(), rel=1e-13)
    assert ls.surface.sq(z) == pytest.approx(s, rel=1e-13)
    g1 = ((v - m) ** 3).mean() / s**3
    assert ls.surface.ssk(z, adj=False) == pytest.approx(g1, rel=1e-12)
    assert ls.surface.ssk(z) == pytest.approx(np.sqrt(n * (n - 1)) / (n - 2) * g1, rel=1e-12)
    k = ((v - m) ** 4).mean() / s**4
    assert ls.surface.sku(z, excess=False) == pytest.approx(k, rel=1e-12)
    assert ls.surface.sku(z) == pytest.approx(k - 3, rel=1e-12)
    assert ls.surface.sv(z) == abs(v.min())
    assert ls.surface.sph(z) == abs(v.max())
    assert ls.surface.smean(z) == pytest.approx(v[v > 0].mean(), rel=1e-13)


def test_metrics_table():
    z = surface_cases.RASTERS["rand_na"].values
    table = ls.surface.metrics(z)
    assert list(table.index) == list(ls.surface.list_metrics().metric)
    assert table["sa"] == ls.surface.sa(z)
    assert list(ls.surface.metrics(z, "sq").index) == ["sq"]
    with pytest.raises(ValueError, match="unknown surface metric"):
        ls.surface.metrics(z, ["sa", "nope"])


def test_planes_recover_polynomials():
    ny, nx = 30, 40
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x, y = xx + 0.5, ny - (yy + 0.5)  # cell centres of unit cells
    for order, z in [(0, np.full((ny, nx), 3.0)), (1, 1 + 0.5 * x - 0.25 * y),
                     (2, 1 + 0.1 * x * y - 0.02 * x**2), (3, 0.001 * x**3 - 0.002 * y**2 * x + y)]:
        fit = ls.surface.fit_plane(z, order)
        np.testing.assert_allclose(fit.values, z, rtol=1e-10, atol=1e-10)
        assert fit.attrs["order"] == order
    # remove_plane subtracts the best fit, whose choice rests on rounding (as geodiv)
    z = 1 + 0.5 * x - 0.25 * y + np.random.default_rng(1).normal(0, 0.1, x.shape)
    z[3, 4] = np.nan
    best = ls.surface.best_fit_plane(z)
    removed = ls.surface.remove_plane(z)
    assert removed.attrs["order"] == best.attrs["order"] in (0, 1, 2, 3)
    assert np.isnan(removed.values[3, 4])
    np.testing.assert_array_equal(removed.values, z - best.values)
    np.testing.assert_array_equal(best.values, ls.surface.fit_plane(z, best.attrs["order"]).values)


def test_plane_keeps_the_grid():
    xr = pytest.importorskip("xarray")
    path = ls.surface.example_data("orforest")
    da = ls.load_raster(path)
    fit = ls.surface.fit_plane(da, 1)
    assert isinstance(fit, xr.DataArray)
    np.testing.assert_array_equal(fit.x.values, da.x.values)
    np.testing.assert_array_equal(fit.y.values, da.y.values)
    assert fit.rio.crs == da.rio.crs
    # arrays: cell centres of the extent
    arr = ls.surface.fit_plane(np.ones((2, 3)), 0, extent=(10, 16, 0, 4))
    np.testing.assert_array_equal(arr.x.values, [11, 13, 15])
    np.testing.assert_array_equal(arr.y.values, [3, 1])


def test_plane_errors():
    with pytest.raises(ValueError, match="order"):
        ls.surface.fit_plane(np.ones((5, 5)), 7)
    with pytest.raises(ValueError, match="order"):
        ls.surface.fit_plane(np.ones((5, 5)), 1.5)
    with pytest.raises(ValueError, match="at least 10 cells"):
        ls.surface.best_fit_plane(np.ones((3, 3)))
