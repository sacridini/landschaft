# SPDX-License-Identifier: GPL-3.0-or-later
"""Gravitational interaction potential (Dinamica EGO's CalcInteractionPotentialMap)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls


def _naive(mass, beta, dx=1.0, dy=1.0):
    """The sum in the same order (sources in row order, float64), term by term."""
    ny, nx = mass.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    out = np.zeros((ny, nx))
    for c in np.flatnonzero(~np.isnan(mass) & (mass != 0)):
        i, j = divmod(int(c), nx)
        d2 = ((yy - i) * dy) ** 2 + ((xx - j) * dx) ** 2
        d2[i, j] = dx * dy
        w = {0.0: np.ones_like(d2), 1.0: 1.0 / np.sqrt(d2), 2.0: 1.0 / d2, 3.0: 1.0 / (d2 * np.sqrt(d2)),
             4.0: 1.0 / (d2 * d2)}.get(beta)
        out += mass[i, j] * (d2 ** (-0.5 * beta) if w is None else w)
    return out


def test_same_as_dinamica():
    # Dinamica adds float32 terms in float32: about 1e-6 apart with 300 sources
    ref = simulate_reference.result("gravity", "gravity_dinamica")
    ours = simulate_cases.run_python("gravity_dinamica")
    assert len(ref) == 10
    for k in sorted(ref):
        a = np.asarray(ref[k], float)
        b = np.asarray(ours[k], float).reshape(a.shape)
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        np.testing.assert_allclose(b, a, rtol=3e-6, atol=0, err_msg=k)


@pytest.mark.parametrize("beta", [2.0, 1.0, 0.7, 0.0, 3.0])
def test_direct_is_the_sum_in_row_order(beta):
    rng = np.random.default_rng(5)
    mass = np.where(rng.random((23, 31)) < 0.1, rng.random((23, 31)) * 100, np.nan)
    mass[0, 0] = 0.0  # a zero mass adds nothing
    ours = np.asarray(ls.surface.gravity(mass, beta, method="direct"))
    np.testing.assert_array_equal(ours, _naive(mass, beta))


def test_fft_and_threads():
    rng = np.random.default_rng(6)
    mass = np.where(rng.random((70, 90)) < 0.2, rng.random((70, 90)) * 50, np.nan)
    one = np.asarray(ls.surface.gravity(mass, 1.5, method="direct", n_threads=1))
    four = np.asarray(ls.surface.gravity(mass, 1.5, method="direct", n_threads=4))
    np.testing.assert_array_equal(one, four)
    fft = np.asarray(ls.surface.gravity(mass, 1.5, method="fft"))
    np.testing.assert_allclose(fft, one, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ls.surface.gravity(mass, 1.5)), one, rtol=1e-12)


def test_map_units_and_mask():
    da = simulate_cases.grid(np.where(np.arange(48).reshape(6, 8) % 7 == 0, 10.0, np.nan), res=30.0)
    cells = np.asarray(ls.surface.gravity(da, 2.0))
    meters = ls.surface.gravity(da, 2.0, unit="map")
    np.testing.assert_allclose(np.asarray(meters), cells / 900.0, rtol=1e-15)
    assert meters.rio.crs == da.rio.crs and meters.attrs["unit"] == "map"
    mask = np.ones((6, 8))
    mask[:2, :3] = np.nan
    masked = np.asarray(ls.surface.gravity(da, 2.0, mask=mask))
    assert np.isnan(masked[:2, :3]).all()
    # sources under the mask still count
    np.testing.assert_array_equal(masked[2:], cells[2:])


def test_edges():
    none = np.full((4, 5), np.nan)
    np.testing.assert_array_equal(np.asarray(ls.surface.gravity(none)), np.zeros((4, 5)))
    one = none.copy()
    one[1, 2] = 3.0
    flat = np.asarray(ls.surface.gravity(one, 0.0))
    np.testing.assert_array_equal(flat, np.full((4, 5), 3.0))
    p = np.asarray(ls.surface.gravity(one, 2.0))
    assert p[1, 2] == 3.0 and p[1, 3] == 3.0 and p[0, 3] == 1.5
    with pytest.raises(ValueError, match="unit"):
        ls.surface.gravity(one, unit="m")
    with pytest.raises(ValueError, match="method"):
        ls.surface.gravity(one, method="x")
    with pytest.raises(ValueError, match="mask"):
        ls.surface.gravity(one, mask=np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        ls.surface.gravity(one, np.inf)
