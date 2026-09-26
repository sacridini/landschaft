# SPDX-License-Identifier: GPL-3.0-or-later
"""Omniscape moving-window connectivity (ls.omniscape)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft import _core

REF = Path(__file__).parent / "data" / "reference" / "circuit"
OUTPUTS = ("cum_currmap", "flow_potential", "normalized_cum_currmap")
OPTION_NAMES = {"connect_four_neighbors_only": "four_neighbors", "calc_flow_potential": "flow_potential",
                "calc_normalized_current": "normalized_current"}


def _map(seed=1, shape=(40, 50)):
    rng = np.random.default_rng(seed)
    r = np.exp(rng.normal(0.5, 1.0, shape))
    r[rng.random(shape) < 0.05] = np.nan
    return r


def test_single_target_is_one_advanced_solve():
    # block_size 1 and one source cell pair: the cumulative map is the sum of
    # the windows' advanced-mode currents.
    r = _map(2, (15, 15))
    r[np.isnan(r)] = 1.0
    s = np.zeros(r.shape)
    s[7, 7] = 1.0
    s[5, 9] = 2.0
    ds = ls.omniscape(r, 4, source=s)
    assert ds.attrs["n_targets"] == 2
    total = np.zeros(r.shape)
    for (ty, tx), amps, other in [((7, 7), 1.0, (5, 9)), ((5, 9), 2.0, (7, 7))]:
        yy, xx = np.mgrid[:15, :15]
        g = np.where((yy - ty) ** 2 + (xx - tx) ** 2 <= 16, 1 / r, np.nan)
        src = np.zeros(r.shape)
        src[other] = amps
        gr = np.zeros(r.shape)
        gr[ty, tx] = np.inf
        total += ls.circuit.advanced(g, src, gr).current
    np.testing.assert_allclose(ds.cum_currmap.values, total, rtol=1e-12, atol=1e-15)


def test_threads_do_not_change_the_result():
    r = _map(3)
    s = np.where(np.isnan(r), 0, 1 / np.nan_to_num(r, nan=1))
    kw = dict(source=s, block_size=3, buffer=2, flow_potential=True, normalized_current=True)
    a = ls.omniscape(r, 6, n_threads=1, **kw)
    b = ls.omniscape(r, 6, **kw)
    for k in OUTPUTS:
        np.testing.assert_array_equal(a[k].values, b[k].values)


def test_options_and_masks():
    r = _map(4)
    with pytest.warns(UserWarning, match="block_size is even"):
        ds = ls.omniscape(r, 5, block_size=2, flow_potential=True, normalized_current=True)
    assert ds.attrs["block_size"] == 3
    nodata = np.isnan(r)
    for k in OUTPUTS:
        assert np.isnan(ds[k].values[nodata]).all() and not np.isnan(ds[k].values[~nodata]).any()
    ds2 = ls.omniscape(r, 5, block_size=3, flow_potential=True, normalized_current=True, mask_nodata=False)
    assert not np.isnan(ds2.cum_currmap.values).any()
    np.testing.assert_array_equal(ds2.cum_currmap.values[~nodata], ds.cum_currmap.values[~nodata])
    # Normalised current: current / flow potential, 0 where both are 0.
    fp = ds2.flow_potential.values
    with np.errstate(invalid="ignore", divide="ignore"):
        expect = np.where((ds2.cum_currmap.values == 0) & (fp == 0), 0, ds2.cum_currmap.values / fp)
    np.testing.assert_array_equal(ds2.normalized_cum_currmap.values, expect)
    # Only the requested outputs.
    assert list(ls.omniscape(r, 5).data_vars) == ["cum_currmap"]
    assert list(ls.omniscape(r, 5, normalized_current=True).data_vars) == ["cum_currmap", "normalized_cum_currmap"]


def test_source_from_resistance():
    r = _map(5)
    s = ls.circuit.source_from_resistance(r, r_cutoff=2.0)
    expect = np.where(np.isnan(r) | (1 / r < 0.5), 0, 1 / r)
    np.testing.assert_array_equal(s, expect)
    a = ls.omniscape(r, 4, r_cutoff=2.0)
    b = ls.omniscape(r, 4, source=expect)
    np.testing.assert_array_equal(a.cum_currmap.values, b.cum_currmap.values)
    # Conductance input: the same circuit.
    c = ls.omniscape(1 / r, 4, source=expect, resistance_is_conductance=True)
    np.testing.assert_allclose(c.cum_currmap.values, a.cum_currmap.values, rtol=1e-12)


def test_source_threshold_and_nan_sources():
    r = _map(6)
    s = np.random.default_rng(6).uniform(0, 1, r.shape)
    s[::7, ::5] = np.nan
    a = ls.omniscape(r, 4, source=s, source_threshold=0.4)
    b = ls.omniscape(r, 4, source=np.where(np.isnan(s) | (s < 0.4), 0, s))
    np.testing.assert_array_equal(a.cum_currmap.values, b.cum_currmap.values)


def test_correction_factors():
    # No correction with blocks of one cell: every factor 1.
    np.testing.assert_array_equal(_core.circuit_omniscape_correction(4, 1, 1, False, 50, 50), 1.0)
    c = _core.circuit_omniscape_correction(6, 2, 5, False, 50, 50)
    assert c.shape == (17, 17)
    np.testing.assert_allclose(c, c.T, rtol=1e-9)
    np.testing.assert_allclose(c, c[::-1, :], rtol=1e-9)
    assert np.all(c > 0)


def test_dataarray_input():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    r = _map(7, (20, 30))
    da = xr.DataArray(r, dims=("y", "x"), coords={"y": 100 - np.arange(20) * 10.0, "x": np.arange(30) * 10.0})
    ds = ls.omniscape(da, 3, block_size=3)
    np.testing.assert_array_equal(ds.x, da.x)
    np.testing.assert_array_equal(ds.cum_currmap.values, ls.omniscape(r, 3, block_size=3).cum_currmap.values)


def test_bad_inputs():
    r = _map(8)
    with pytest.raises(ValueError, match="> 0"):
        ls.omniscape(np.where(np.isnan(r), 0, r), 3)
    with pytest.raises(ValueError, match="shape"):
        ls.omniscape(r, 3, source=np.ones((3, 3)))


def _cases():
    z = np.load(REF / "omniscape.npz")
    return sorted({k.split("/")[0] for k in z.files})


@pytest.mark.reference
@pytest.mark.parametrize("name", _cases())
def test_matches_omniscape(name):
    z = np.load(REF / "omniscape.npz")
    options = json.loads(str(z[f"{name}/options"]))
    kw = {OPTION_NAMES.get(k, k): v for k, v in options.items() if k != "source_from_resistance"}
    source = None if options.get("source_from_resistance") else z[f"{name}/source"]
    with pytest.warns(UserWarning) if options.get("block_size", 1) % 2 == 0 else _nothing():
        ds = ls.omniscape(z[f"{name}/resistance"], source=source, **kw)
    checked = 0
    for out in OUTPUTS:
        key = f"{name}/cholmod/{out}"
        if key not in z.files:
            assert out not in ds
            continue
        ours, chol, cg = ds[out].values, z[key], z[f"{name}/cgamg/{out}"]
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(chol))
        np.testing.assert_array_equal(ours == 0, chol == 0)
        top = np.nanmax(chol)
        # Omniscape with CHOLMOD: the same up to rounding (the artifact
        # correction divides currents, so a few more digits go).
        np.testing.assert_allclose(ours, chol, rtol=1e-9, atol=1e-13 * top)
        # With its default cg+amg (relative residual 1e-6, amplified by the correction).
        np.testing.assert_allclose(ours, cg, rtol=0, atol=1e-3 * top)
        checked += 1
    assert checked


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
