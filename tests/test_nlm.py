# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral landscape models (landschaft.nlm) against NLMR and on their own."""

from __future__ import annotations

import warnings
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # NLMR's mpd sizes
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


def test_mpd_sizes():
    with pytest.warns(UserWarning, match="nlmr_compat=False"):
        m = ls.nlm.mpd(100, 60, seed=1)
    assert m.shape == (99, 99)
    with pytest.warns(UserWarning):
        assert ls.nlm.mpd(10, 10, seed=1).shape == (9, 9)
    assert ls.nlm.mpd(33, 33, seed=1).shape == (33, 33)
    for ncol, nrow in ((100, 60), (1, 1), (129, 3), (130, 7)):
        m = ls.nlm.mpd(ncol, nrow, seed=2, nlmr_compat=False)
        assert m.shape == (nrow, ncol)
    with pytest.raises(ValueError):
        ls.nlm.mpd(10, 10, roughness=1.5)


def test_mpd_roughness_and_threads():
    smooth = ls.nlm.mpd(257, 257, roughness=0.2, seed=4, nlmr_compat=False, rescale=False).values
    rough = ls.nlm.mpd(257, 257, roughness=0.9, seed=4, nlmr_compat=False, rescale=False).values
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(rough, axis=1)).mean()
    for torus in (False, True):
        with ls._threads.limit(1):
            one = ls.nlm.mpd(513, 513, seed=5, torus=torus).values
        with ls._threads.limit(4):
            four = ls.nlm.mpd(513, 513, seed=5, torus=torus).values
        np.testing.assert_array_equal(one, four)


def test_neigh_counts():
    n = ls.nlm.neigh(50, 40, p_neigh=0.7, p_empty=0.1, categories=4, seed=3, rescale=False)
    assert n.dtype == np.int32
    counts = np.bincount(n.values.ravel(), minlength=4)
    # floor(2000 / 4) cells for categories 3..1; category 0 and the rest are 0
    assert list(counts[1:]) == [500, 500, 500] and counts[0] == 500
    p = ls.nlm.neigh(40, 25, p_neigh=0.6, p_empty=0.2, proportions=[0.2, 0.3, 0.5], seed=1, rescale=False)
    assert list(np.bincount(p.values.ravel())) == [200, 300, 500]
    # compact clusters: more same-category neighbours than a random arrangement
    v = ls.nlm.neigh(100, 100, p_neigh=0.9, p_empty=0.01, categories=2, seed=5, rescale=False).values
    assert (v[:, 1:] == v[:, :-1]).mean() > 0.7
    with pytest.raises(ValueError):
        ls.nlm.neigh(10, 10, p_neigh=0.5, p_empty=0.5, proportions=[0.5, 0.4, 0.2])
    with pytest.raises(ValueError):
        ls.nlm.neigh(10, 10, p_neigh=0, p_empty=0)
    with pytest.raises(ValueError):
        ls.nlm.neigh(10, 10, p_neigh=0.5, p_empty=0.5, neighbourhood=6)


def test_random_rectangular_cluster():
    r = ls.nlm.random_rectangular_cluster(80, 60, minl=4, maxl=9, seed=8, rescale=False).values
    assert np.all((r >= 0) & (r < 1))
    # patches of equal value are unions of rectangles: few distinct values per row run
    assert np.unique(r).size < r.size / 8
    with pytest.raises(ValueError):
        ls.nlm.random_rectangular_cluster(10, 5, minl=3, maxl=6)


def test_random_cluster():
    r = ls.nlm.random_cluster(80, 60, p=0.45, ai=[0.2, 0.3, 0.5], seed=6, rescale=False)
    assert r.dtype == np.int32 and set(np.unique(r.values)) <= {1, 2, 3}
    # clusters: most cells share their type with their right-hand neighbour
    v = r.values
    assert (v[:, 1:] == v[:, :-1]).mean() > 0.5
    # the weights set the shares of the types, roughly
    counts = np.bincount(ls.nlm.random_cluster(300, 300, p=0.5, ai=[0.2, 0.8], seed=1, rescale=False).values.ravel())
    assert counts[2] > 2 * counts[1]
    # p = 0: no clusters, every cell drawn with the weights
    z = ls.nlm.random_cluster(20, 20, p=0, ai=[1, 0], seed=2, rescale=False)
    assert np.all(z.values == 1)
    with ls._threads.limit(1):
        one = ls.nlm.random_cluster(400, 300, p=0.4, seed=9).values
    with ls._threads.limit(4):
        four = ls.nlm.random_cluster(400, 300, p=0.4, seed=9).values
    np.testing.assert_array_equal(one, four)
    with pytest.raises(ValueError):
        ls.nlm.random_cluster(10, 10, p=0.5, ai=[-1, 2])
    with pytest.raises(ValueError):
        ls.nlm.random_cluster(10, 10, p=0.5, neighbourhood=6)


def test_curds():
    c = ls.nlm.curds([0.5, 0.5], [4, 4], seed=1)
    assert c.shape == (16, 16) and c.dtype == np.uint8
    # habitat kept: half of the 16 blocks, then half of their cells
    assert int(c.sum()) == 8 * 8
    blocks = c.values.reshape(4, 4, 4, 4).sum(axis=(1, 3))
    assert (blocks == 0).sum() >= 8
    w = ls.nlm.curds([0.5, 0.5], [4, 4], wheyes=[0.25, 0.0], seed=1)
    assert int(w.sum()) > 64
    with pytest.raises(ValueError):
        ls.nlm.curds([0.5], [2, 2])
    with pytest.raises(ValueError):
        ls.nlm.curds([0.5], [0])


def test_gaussian_field():
    g = ls.nlm.gaussian_field(120, 100, autocorr_range=20, mag_var=1, nug=0, mean=3, seed=4, rescale=False).values
    assert abs(g.mean() - 3) < 1
    # autocorrelated: neighbours close, far cells independent
    assert np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1] > 0.9
    noisy = ls.nlm.gaussian_field(120, 100, autocorr_range=20, mag_var=1, nug=1, mean=3, seed=4, rescale=False).values
    assert np.corrcoef(noisy[:, :-1].ravel(), noisy[:, 1:].ravel())[0, 1] < 0.8
    with ls._threads.limit(1):
        one = ls.nlm.gaussian_field(150, 130, seed=2).values
    with ls._threads.limit(4):
        four = ls.nlm.gaussian_field(150, 130, seed=2).values
    np.testing.assert_array_equal(one, four)
    with pytest.raises(ValueError):
        ls.nlm.gaussian_field(10, 10, autocorr_range=0)


def test_fbm():
    rough = ls.nlm.fbm(128, 128, fract_dim=0.3, seed=1, rescale=False).values
    smooth = ls.nlm.fbm(128, 128, fract_dim=1.8, seed=1, rescale=False).values
    # unit standard deviation of the steps, zero mean
    for f in (rough, smooth):
        steps = np.concatenate([np.diff(f, axis=0).ravel(order="F"), np.diff(f, axis=1).ravel(order="F")])
        assert abs(np.std(steps, ddof=1) - 1) < 1e-12 and abs(f.mean()) < 1e-12
    assert np.ptp(smooth) > np.ptp(rough)
    plane = ls.nlm.fbm(20, 10, fract_dim=2, seed=3, rescale=False).values
    np.testing.assert_allclose(np.diff(plane, axis=1, n=2), 0, atol=1e-12)
    with ls._threads.limit(1):
        one = ls.nlm.fbm(300, 200, seed=2).values
    with ls._threads.limit(4):
        four = ls.nlm.fbm(300, 200, seed=2).values
    np.testing.assert_array_equal(one, four)
    with pytest.raises(ValueError):
        ls.nlm.fbm(10, 10, fract_dim=2.5)


def test_mosaic_field():
    one = ls.nlm.mosaic_field(80, 60, n=1, seed=3, nlmr_compat=False, rescale=False).values
    # one mosaic: piecewise constant tiles, few distinct values
    assert np.unique(one).size < 200
    many = ls.nlm.mosaic_field(80, 60, n=40, seed=3, rescale=False).values
    assert np.unique(many).size > np.unique(one).size
    # NLMR's n = 1 sums three fields
    compat = ls.nlm.mosaic_field(80, 60, n=1, seed=3, rescale=False).values
    assert not np.array_equal(compat, one)
    out = ls.nlm.mosaic_field(30, 20, n=3, collect=True, infinit=True, seed=4)
    assert set(out) == {"mosaic_field", "steps", "infinit"}
    assert out["steps"].dims == ("step", "y", "x") and list(out["steps"].step.values) == [1, 2, 3]
    only = ls.nlm.mosaic_field(30, 20, n=None, infinit=True, seed=4)
    assert only.shape == (20, 30)
    with ls._threads.limit(1):
        a = ls.nlm.mosaic_field(150, 120, n=5, infinit=True, seed=6)
    with ls._threads.limit(4):
        b = ls.nlm.mosaic_field(150, 120, n=5, infinit=True, seed=6)
    for k in a:
        np.testing.assert_array_equal(a[k].values, b[k].values)
    with pytest.raises(ValueError):
        ls.nlm.mosaic_field(10, 10, n=None)
    with pytest.raises(ValueError):
        ls.nlm.mosaic_field(10, 10, n=0)


def test_mosaic_tess():
    t = ls.nlm.mosaic_tess(120, 90, germs=60, seed=5, nlmr_compat=False, rescale=False)
    x, y, v = ls._core.nlm_tess_germs(5, 60, 120.0, 90.0)
    # every cell takes the value of its nearest germ
    yy, xx = np.mgrid[0:90, 0:120]
    cx, cy = xx + 0.5, 90 - (yy + 0.5)
    d = (cx[..., None] - x) ** 2 + (cy[..., None] - y) ** 2
    np.testing.assert_array_equal(t.values, v[np.argmin(d, axis=2)])
    # NLMR's order: the same tiles and values, another assignment, in float32
    c = ls.nlm.mosaic_tess(120, 90, germs=60, seed=5, rescale=False).values
    assert set(np.unique(c)) == set(v.astype(np.float32).astype(np.float64))
    with ls._threads.limit(1):
        one = ls.nlm.mosaic_tess(500, 400, germs=800, seed=7).values
    with ls._threads.limit(4):
        four = ls.nlm.mosaic_tess(500, 400, germs=800, seed=7).values
    np.testing.assert_array_equal(one, four)
    with pytest.raises(ValueError):
        ls.nlm.mosaic_tess(10, 10, germs=0)


def test_mosaic_gibbs():
    x, y, classes = ls._core.nlm_gibbs_germs(3, 80, 6.0, 4, 100.0, 80.0)
    assert len(x) == 80
    d = np.hypot(x[:, None] - x, y[:, None] - y) + np.eye(80) * 1e9
    assert d.min() > 6
    assert sorted(np.bincount(classes.astype(int))[1:]) == [20, 20, 20, 20]
    g = ls.nlm.mosaic_gibbs(100, 80, germs=80, R=6, patch_classes=4, seed=3, nlmr_compat=False, rescale=False)
    assert set(np.unique(g.values)) == {1.0, 2.0, 3.0, 4.0}
    with pytest.warns(UserWarning, match="only"):
        ls.nlm.mosaic_gibbs(20, 20, germs=50, R=10, patch_classes=2, seed=1, nlmr_compat=False)


def test_voronoi_without_shapely(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_shapely(name, *args, **kwargs):
        if name == "shapely" or name.startswith("shapely."):
            raise ImportError("no shapely")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_shapely)
    with pytest.warns(UserWarning, match="shapely"):
        t = ls.nlm.mosaic_tess(40, 30, germs=10, seed=1)
    assert t.shape == (30, 40)
