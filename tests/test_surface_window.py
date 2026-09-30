# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface.window_metrics against each window's metrics in R (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("window"))
def test_window_matches_r(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "window", c.tolerance) > 0


def _window(z, r, c, size, extent=None):
    """The window around (r, c) with NA beyond the edges, and its extent."""
    h = size // 2
    ny, nx = z.shape
    w = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            rr, cc = r - h + i, c - h + j
            if 0 <= rr < ny and 0 <= cc < nx:
                w[i, j] = z[rr, cc]
    xmin, xmax, ymin, ymax = extent or (0.0, nx, 0.0, ny)
    xr, yr = (xmax - xmin) / nx, (ymax - ymin) / ny
    wx = xmin + (c - h) * xr
    wy = ymax - (r - h) * yr
    return w, (wx, wx + size * xr, wy - size * yr, wy)


def test_windows_are_whole_raster_metrics():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(14, 17))
    names = ["sa", "sq", "sv", "sph", "sds", "sdq", "ssc", "s10z", "sk"]
    t = ls.surface.window_metrics(z, names, 5)
    for r, c in [(0, 0), (3, 7), (13, 16), (6, 0)]:
        w, e = _window(z, r, c, 5)
        ref = ls.surface.metrics(w, names, extent=e)
        for n in names:
            np.testing.assert_equal(t[n].values[r, c], ref[n], err_msg=f"{n} at {(r, c)}")


def test_circle_masks_the_corners():
    z = np.arange(49, dtype=float).reshape(7, 7)
    t = ls.surface.window_metrics(z, ["sph", "sv"], 5, "circle")
    # the centre's window: radius 2, the corners (and cells at distance > 2) masked
    assert t["sph"].values[3, 3] == z[5, 3] and t["sv"].values[3, 3] == z[1, 3]
    sq = ls.surface.window_metrics(z, ["sph"], 5)
    assert sq["sph"].values[3, 3] == z[5, 5]


def test_window_errors():
    z = np.zeros((6, 6))
    with pytest.raises(ValueError, match="odd"):
        ls.surface.window_metrics(z, "sa", 4)
    with pytest.raises(ValueError, match="unknown"):
        ls.surface.window_metrics(z, "nope")
    with pytest.raises(ValueError, match="shape"):
        ls.surface.window_metrics(z, "sa", 3, "hexagon")
    t = ls.surface.window_metrics(z, "sa", 3)
    assert t.attrs["window"] == 3 and t["sa"].shape == z.shape
