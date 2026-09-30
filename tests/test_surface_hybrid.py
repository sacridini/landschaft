# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface hybrid parameters and local features against geodiv (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("hybrid"))
def test_hybrid_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "hybrid", c.tolerance) > 0


@pytest.mark.parametrize("case", surface_reference.case_ids("features"))
def test_features_match_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "features", c.tolerance) > 0


def _plane(ny=24, nx=30, gx=0.3, gy=-0.2, res=(2.0, 3.0)):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x, y = (xx + 0.5) * res[0], (ny - yy - 0.5) * res[1]
    return 5 + gx * x + gy * y, (0.0, nx * res[0], 0.0, ny * res[1])


def test_iso_slopes_and_areas_of_a_plane():
    gx, gy, res = 0.3, -0.2, (2.0, 3.0)
    z, extent = _plane(gx=gx, gy=gy, res=res)
    # per cell: a step of gx * res_x along x, gy * res_y along y (rows go down)
    step = np.hypot(gx * res[0], gy * res[1])
    assert ls.surface.sdq6(z, "iso", extent=extent) == pytest.approx(step, rel=1e-12)
    assert ls.surface.sdq6(z, extent=extent) > 10 * step  # geodiv's stencil keeps the height
    ny, nx = z.shape
    flat = (ny - 1) * (nx - 1) * res[0] * res[1]
    assert ls.surface.flat_area(z, "iso", extent=extent) == pytest.approx(flat, rel=1e-12)
    # ISO 25178's area of a cell: the product of the lengths of its two edges
    # (exact when they are orthogonal: a plane tilted along one axis)
    area = flat * np.sqrt(1 + gx**2) * np.sqrt(1 + gy**2)
    assert ls.surface.surface_area(z, "iso", extent=extent) == pytest.approx(area, rel=1e-12)
    assert ls.surface.sdr(z, "iso", extent=extent) == pytest.approx((np.sqrt(1 + gx**2) * np.sqrt(1 + gy**2) - 1)
                                                                     * 100, rel=1e-10)
    tilted, extent = _plane(gx=0.4, gy=0.0, res=res)
    assert ls.surface.sdr(tilted, "iso", extent=extent) == pytest.approx((np.hypot(1, 0.4) - 1) * 100, rel=1e-10)
    # sdq: differences per cell, no spacing
    assert ls.surface.sdq(z, extent=extent) == pytest.approx(step, rel=1e-12)


def test_geodiv_areas_are_rescaled():
    z, extent = _plane()
    # geodiv rescales heights to 0-1: the same Sdr for any scaling of z
    assert ls.surface.sdr(3 * z + 7, extent=extent) == pytest.approx(ls.surface.sdr(z, extent=extent), rel=1e-12)
    assert ls.surface.sdr(3 * z, "iso", extent=extent) != pytest.approx(ls.surface.sdr(z, "iso", extent=extent))


def test_peaks_and_valleys_by_definition():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(17, 23))
    z[rng.random(z.shape) < 0.05] = np.nan

    def extremes(v, sign):
        out = []
        for r in range(1, v.shape[0] - 1):
            for c in range(1, v.shape[1] - 1):
                w = v[r - 1:r + 2, c - 1:c + 2]
                others = np.delete(w.ravel(), 4)
                if sign * v[r, c] > 0 and np.all(sign * v[r, c] > sign * others):
                    out.append((r, c))
        return out

    peaks = ls.surface.find_peaks(z)
    assert list(zip(peaks.row, peaks.col)) == extremes(z, 1)
    np.testing.assert_array_equal(peaks.value, z[peaks.row, peaks.col])
    valleys = ls.surface.find_valleys(z, "iso")
    assert list(zip(valleys.row, valleys.col)) == extremes(z, -1)
    assert ls.surface.sds(z) == len(peaks) / ((z.shape[0] - 1) * (z.shape[1] - 1))
    top = np.sort(peaks.value)[::-1][:5].sum() + np.abs(np.sort(valleys.value)[:5]).sum()
    assert ls.surface.s10z(z, "iso") == pytest.approx(top / 5, rel=1e-14)
    # geodiv's valleys on a square raster: each cell against the transposed neighbourhood
    sq = z[:17, :17]
    t = ls.surface.find_valleys(sq)
    for r, c, v in zip(t.row, t.col, t.value):
        others = np.delete(sq.T[r - 1:r + 2, c - 1:c + 2].ravel(), 4)
        assert v < 0 and np.all(v < others)


def test_ssc_by_definition():
    rng = np.random.default_rng(8)
    z = rng.normal(size=(15, 20))
    zn = (z - z.min()) / (z.max() - z.min())
    p = ls.surface.find_peaks(z)
    r, c = p.row.to_numpy(), p.col.to_numpy()
    terms = ((zn[r, c] - zn[r, c + 1]) + (zn[r, c] - zn[r, c - 1])) / 2 + \
        ((zn[r, c] - zn[r + 1, c]) + (zn[r, c] - zn[r - 1, c])) / 2
    assert ls.surface.ssc(z) == pytest.approx(-terms.sum() / (2 * len(p)), rel=1e-12)
    assert np.isnan(ls.surface.ssc(np.zeros((5, 5))))


def test_metrics_conventions():
    z = surface_cases.RASTERS["rect"].values
    geodiv = ls.surface.metrics(z)
    iso = ls.surface.metrics(z, convention="iso")
    for name in ("sa", "sq", "sdq", "sds", "ssc"):
        assert geodiv[name] == iso[name]
    assert geodiv["sdq6"] == ls.surface.sdq6(z)
    assert iso["sdq6"] == ls.surface.sdq6(z, "iso")
    assert iso["s10z"] == ls.surface.s10z(z, "iso")
    with pytest.raises(ValueError, match="convention"):
        ls.surface.metrics(z, convention="fragstats")
