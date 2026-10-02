# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.thin against spThin::thin.algorithm (same records for the same seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import sdm_cases
import sdm_reference
from conftest import BITWISE


@pytest.mark.parametrize("case", sdm_reference.case_ids("thin"))
def test_thin_matches_spthin(case):
    if case == "thin_threshold" and not BITWISE:
        pytest.skip("distances within an ulp of the threshold: the UCRT's acos")
    ref = sdm_reference.result("thin", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        sdm_reference.assert_same(ours[key], ref[key], key)


def test_kept_records_are_far_enough():
    rng = np.random.default_rng(3)
    occ = pd.DataFrame({"lon": rng.uniform(-60, -40, 3000), "lat": rng.uniform(-20, 0, 3000)})
    kept = ls.sdm.thin(occ, 100_000, reps=3, seed=1)
    from pyproj import Geod

    lon, lat = kept.lon.to_numpy(), kept.lat.to_numpy()
    i, j = np.triu_indices(len(kept), 1)
    # spThin's sphere (6378.388 km) is wider than the ellipsoid's meridian radius near the equator (~0.7 %)
    d = Geod(ellps="WGS84").inv(lon[i], lat[i], lon[j], lat[j])[2]
    assert d.min() > 100_000 * 0.99
    assert kept.attrs["thin"]["seed"] == 1 and len(kept.attrs["thin"]["kept"]) == 3


def test_neighbours_match_brute_force():
    rng = np.random.default_rng(5)
    lon, lat = rng.uniform(-180, 180, 400), rng.uniform(-90, 90, 400)
    from landschaft import _core

    start, index = _core.sdm_earth_neighbours(lon, lat, 2000.0)
    a = np.deg2rad(np.column_stack([lat, lon]))
    v = np.column_stack([np.cos(a[:, 0]) * np.cos(a[:, 1]), np.cos(a[:, 0]) * np.sin(a[:, 1]), np.sin(a[:, 0])])
    d = 6378.388 * np.arccos(np.clip(v @ v.T, -1, 1))
    near = (d < 2000.0) & ~np.eye(400, dtype=bool)
    margin = np.abs(d - 2000.0) > 1e-6  # leave out pairs within rounding of the threshold
    for i in range(400):
        ours = set(index[start[i]:start[i + 1]].tolist())
        theirs = set(np.flatnonzero(near[i]).tolist())
        assert {k for k in ours ^ theirs if margin[i, k]} == set()
