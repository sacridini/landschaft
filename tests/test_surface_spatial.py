# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface spatial parameters and fractal dimension against geodiv (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("spatial"))
@pytest.mark.external_reference
def test_spatial_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "spatial", c.tolerance) > 0


@pytest.mark.parametrize("case", surface_reference.case_ids("fractal"))
def test_fractal_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "fractal", c.tolerance) > 0


def test_aacf_and_lengths():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(31, 40))
    a = ls.surface.aacf(z)
    assert a.shape == z.shape
    assert a.values[31 // 2, 40 // 2] == 1.0 == np.nanmax(a.values)
    lengths = ls.surface.scl(z)
    assert lengths.shape == (4,)
    assert np.all(lengths[:2] <= lengths[2:])
    np.testing.assert_array_equal(ls.surface.stxr(z), lengths[:2] / lengths[2:])
    one = ls.surface.scl(z, 0.5)
    assert one.shape == (2,)
    with pytest.raises(ValueError, match="threshold"):
        ls.surface.scl(z, -1)


def test_aacf_crops_missing_cells():
    z = surface_cases.RASTERS["rand_na"].values
    a = ls.surface.aacf(z, extent=(0, 240, 0, 240))
    assert a.shape[0] == a.shape[1] < z.shape[0]
    assert not np.isnan(a.values).any()
    # the lengths use the cropped image (geodiv fails here)
    assert np.isfinite(ls.surface.scl(z, extent=(0, 240, 0, 240))).all()


def test_fractal_dimension():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    rough = np.random.default_rng(2).normal(size=(n, n))
    smooth = np.sin(xx / 20) + np.cos(yy / 25)
    assert ls.surface.sfd(rough) > ls.surface.sfd(smooth) >= 2.0
    assert np.isfinite(ls.surface.sfd(rough[:9, :12]))  # geodiv reads past this one
    z = rough.copy()
    z[3, 3] = np.nan
    assert np.isnan(ls.surface.sfd(z))


def test_spatial_metrics_table():
    z = surface_cases.RASTERS["square"].values
    t = ls.surface.metrics(z, ["sal", "str", "std", "stdi", "srw", "srwi", "shw", "sfd"])
    lengths = ls.surface.scl(z)
    assert t["sal"] == lengths[0] and t["str"] == lengths[0] / lengths[2]
    assert [t["std"], t["stdi"]] == ls.surface.std(z).tolist()
    assert [t["srw"], t["srwi"], t["shw"]] == ls.surface.srw(z).tolist()
    assert t["sfd"] == ls.surface.sfd(z)


def test_texture_direction_of_stripes():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # stripes varying along x: frequency vector along x (0 degrees); along y: 90
    assert ls.surface.std(np.sin(2 * np.pi * xx / 8))[0] == 0.0
    assert ls.surface.std(np.sin(2 * np.pi * yy / 8))[0] == 90.0
    assert ls.surface.std(np.sin(2 * np.pi * (xx - yy) / 8))[0] == 45.0
    # a wavelength of 8 cells: 64 / 8 = 8 frequency steps
    assert ls.surface.srw(np.sin(2 * np.pi * yy / 8))[0] == pytest.approx(8, abs=0.6)
