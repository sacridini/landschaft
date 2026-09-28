# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral landscape models (landschaft.nlm) against NLMR and on their own."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

import landschaft as ls
import nlm_cases
from landschaft.nlm._common import as_values

REF = Path(__file__).parent / "data" / "reference" / "nlm" / "models.npz"


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("no NLMR reference (tools/make_reference_nlm.py models)")
    return np.load(REF)


@pytest.mark.parametrize("key", sorted(nlm_cases.MODELS))
def test_matches_nlmr(ref, key):
    if key not in ref:
        pytest.skip(f"{key} not in the reference file")
    got = as_values(nlm_cases.MODELS[key][1](ls))
    np.testing.assert_array_equal(got, ref[key])


# -- common behaviour -----------------------------------------------------------------


def test_grid_and_attrs():
    r = ls.nlm.random(30, 20, resolution=2.5, seed=11)
    assert r.dims == ("y", "x") and r.shape == (20, 30)
    assert r.attrs["seed"] == 11
    assert tuple(r.rio.transform())[:6] == (2.5, 0.0, 0.0, 0.0, -2.5, 50.0)
    np.testing.assert_allclose(r.x.values[[0, -1]], [1.25, 73.75])
    np.testing.assert_allclose(r.y.values[[0, -1]], [48.75, 1.25])


def test_seed_none_is_recorded_and_reproducible():
    a = ls.nlm.random(10, 10)
    b = ls.nlm.random(10, 10, seed=a.attrs["seed"])
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("seed", [1.5, True, 2**31, "a"])
def test_bad_seed(seed):
    with pytest.raises((ValueError, TypeError)):
        ls.nlm.random(5, 5, seed=seed)


@pytest.mark.parametrize("ncol,nrow", [(0, 5), (5, -1), (2.5, 3)])
def test_bad_size(ncol, nrow):
    with pytest.raises(ValueError):
        ls.nlm.random(ncol, nrow, seed=1)


def test_threads_do_not_change_results():
    for f in (lambda: ls.nlm.random(400, 300, seed=3),
              lambda: ls.nlm.edge_gradient(500, 400, direction=33),
              lambda: ls.nlm.distance_gradient(300, 80, origin=(10, 20, 30, 40)),
              lambda: ls.nlm.distance_gradient(500, 400, origin=(10, 20, 30, 40), nlmr_compat=False)):
        with ls._threads.limit(1):
            one = f().values
        with ls._threads.limit(4):
            four = f().values
        np.testing.assert_array_equal(one, four)


# -- models --------------------------------------------------------------------------


def test_percolation_values():
    p = ls.nlm.percolation(200, 150, prob=0.25, seed=5)
    assert p.dtype == np.uint8 and set(np.unique(p.values)) <= {0, 1}
    assert abs(float(p.mean()) - 0.25) < 0.01
    assert int(ls.nlm.percolation(20, 20, prob=0, seed=1).sum()) == 0
    assert int(ls.nlm.percolation(20, 20, prob=1, seed=1).sum()) == 400
    with pytest.raises(ValueError):
        ls.nlm.percolation(5, 5, prob=1.5)


def test_gradient_direction():
    east = ls.nlm.planar_gradient(10, 6, direction=90).values
    assert np.all(np.diff(east, axis=1) > 0) and np.allclose(np.diff(east, axis=0), 0)
    north = ls.nlm.planar_gradient(10, 6, direction=0).values
    assert np.all(np.diff(north, axis=0) < 0)
    g = ls.nlm.planar_gradient(10, 6, seed=4)
    assert 0 <= g.attrs["direction"] < 360
    e = ls.nlm.edge_gradient(11, 5, direction=90).values[0]
    np.testing.assert_allclose(e, [0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0], atol=1e-12)


def test_distance_gradient_exact():
    d = ls.nlm.distance_gradient(70, 50, origin=(5, 9, 30, 33), resolution=2, nlmr_compat=False, rescale=False)
    mask = np.ones((50, 70), bool)
    mask[29:33, 4:9] = False
    np.testing.assert_allclose(d.values, ndimage.distance_transform_edt(mask) * 2, rtol=1e-12)


def test_distance_gradient_origin_checks():
    with pytest.raises(ValueError):
        ls.nlm.distance_gradient(10, 5, origin=(1, 2, 3, 6))  # NLMR: origin[4] <= nrow
    with pytest.raises(ValueError):
        ls.nlm.distance_gradient(10, 5, origin=(4, 2, 1, 1), nlmr_compat=False)


def test_rescale():
    x = np.array([[3.0, 5.0], [np.nan, 7.0]])
    np.testing.assert_array_equal(ls.nlm.rescale(x), [[0, 0.5], [np.nan, 1]])
    np.testing.assert_array_equal(ls.nlm.rescale(np.full((2, 2), 4.0)), np.zeros((2, 2)))
    np.testing.assert_array_equal(ls.nlm.rescale(np.array([1.0, 1.0 + 1e-9])), [0.0, 0.0])  # all.equal
    da = xr.DataArray(np.array([[1.0, -9.0], [3.0, 5.0]]), dims=("y", "x")).rio.write_nodata(-9.0)
    out = ls.nlm.rescale(da)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_array_equal(out.values, [[0, np.nan], [0.5, 1]])


def test_classify():
    r = ls.nlm.random(40, 25, seed=12)
    c = ls.nlm.classify(r, weighting=[0.1, 0.6, 0.3], level_names=["a", "b", "c"])
    assert c.dtype == np.int32 and c.rio.nodata == 0
    assert c.attrs["levels"] == {1: "a", 2: "b", 3: "c"}
    np.testing.assert_array_equal(np.bincount(c.values.ravel()), [0, 100, 600, 300])
    # plain arrays in, arrays out; missing cells stay missing
    v = r.values.copy()
    v[0, :5] = np.nan
    out = ls.nlm.classify(v, n=3)
    assert isinstance(out, np.ndarray) and np.all(out[0, :5] == 0) and set(np.unique(out)) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        ls.nlm.classify(r)
    with pytest.raises(ValueError):
        ls.nlm.classify(r, weighting=[1, -1])
    with pytest.raises(ValueError):
        ls.nlm.classify(r, weighting=[1, 1], level_names=["only one"])


def test_jenks_breaks_small():
    # classes of clearly separated groups
    x = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 9.0, 9.2, 9.1])
    np.testing.assert_array_equal(ls._core.nlm_jenks_breaks(np.sort(x), 3), [1.0, 1.2, 5.1, 9.2])
    out = ls.nlm.classify(x.reshape(2, 4), n=3)
    np.testing.assert_array_equal(out.ravel(), [1, 1, 1, 2, 2, 3, 3, 3])
