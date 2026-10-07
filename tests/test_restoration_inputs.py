# SPDX-License-Identifier: GPL-3.0-or-later
"""Original vegetation, area of habitat, ecological intactness and uncertainty propagation
(the parity with plangea is in test_plangea_reference.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
import plangea_cases as pc


def _cover():
    return pc.grid(pc.land_cover(), names=pc.CLASSES)


# ----------------------------------------------------------------------------- original vegetation

def test_original_vegetation_accounts_for_the_converted_land():
    cover = _cover()
    veg = ls.original_vegetation(cover, pc.NATURAL, pc.ANTHROPIC, past=pc.grid(pc.past(), names=pc.NATURAL),
                                 ecoregions=pc.grid(pc.ecoregions(), names=pc.NATURAL))
    v = np.nan_to_num(cover.values)
    nat = v[:3].sum(axis=0)
    land = ~np.isnan(veg.source.values)
    # the original natural cover is the current natural plus the anthropic share
    np.testing.assert_allclose(veg.original.values.sum(axis=0)[land], (nat + v[3] + v[4])[land], rtol=1e-12)
    np.testing.assert_allclose(veg.anthropic.values[land], (v[3] + v[4])[land], rtol=1e-15)
    assert veg.classes == pc.NATURAL
    assert list(veg.original["band"].values) == pc.NATURAL


def test_original_vegetation_without_maps_uses_equal_shares():
    cover = xr.DataArray(np.array([[[0.0]], [[0.0]], [[1.0]]]), dims=("band", "y", "x"),
                         coords={"band": ["a", "b", "c"]})
    veg = ls.original_vegetation(cover, ["a", "b"], ["c"])
    np.testing.assert_array_equal(veg.original.values.ravel(), [0.5, 0.5])
    assert veg.source.values.item() == 1


def test_original_vegetation_restore_to_current():
    cover = xr.DataArray(np.array([[[0.2]], [[0.0]], [[0.8]]]), dims=("band", "y", "x"),
                         coords={"band": ["a", "b", "c"]})
    past = xr.DataArray(np.array([[[0.0]], [[1.0]]]), dims=("band", "y", "x"), coords={"band": ["a", "b"]})
    orig = ls.original_vegetation(cover, ["a", "b"], ["c"], past=past)
    cur = ls.original_vegetation(cover, ["a", "b"], ["c"], past=past, restore_to="current")
    np.testing.assert_allclose(orig.restore.values.ravel(), [0.2, 0.8])
    np.testing.assert_allclose(cur.restore.values.ravel(), [1.0, 0.0])


def test_original_vegetation_checks_its_inputs():
    cover = _cover()
    with pytest.raises(ValueError, match="natural and anthropic"):
        ls.original_vegetation(cover, ["forest"], ["forest"])
    with pytest.raises(ValueError, match="no layer"):
        ls.original_vegetation(cover, ["forest", "tundra"], ["crops"])
    with pytest.raises(ValueError, match="failsafe"):
        ls.original_vegetation(cover, ["forest"], ["crops"], failsafe=["crops"])
    with pytest.raises(ValueError, match="restore_to"):
        ls.original_vegetation(cover, ["forest"], ["crops"], restore_to="past")


def test_original_vegetation_saves(tmp_path):
    veg = ls.original_vegetation(_cover(), pc.NATURAL, pc.ANTHROPIC)
    folder = veg.save(tmp_path / "veg")
    assert {p.name for p in folder.iterdir()} >= {"original.tif", "restore.tif", "source.tif", "anthropic.tif"}


# ----------------------------------------------------------------------------- area of habitat

def test_area_of_habitat_sums_the_suitable_shares():
    cover = _cover()
    names, r = pc.ranges()
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table())
    v = np.nan_to_num(cover.values)
    rr = np.nan_to_num(r) > 0
    t = aoh.table.set_index("species")
    np.testing.assert_allclose(t.loc["sp_d", "current"], v[0][rr[3]].sum(), rtol=1e-12)  # forest, cells of 1 km²
    np.testing.assert_allclose(t.loc["sp_b", "current"], (v[1] + v[2])[rr[1]].sum(), rtol=1e-12)
    # without the original vegetation the potential habitat is the current one, and there is no gain
    np.testing.assert_array_equal(t["current"], t["potential"])
    assert aoh.gain is None
    with pytest.raises(ValueError, match="no gain"):
        aoh.gain_map("sp_a")


def test_area_of_habitat_unlisted_and_elevation():
    cover = _cover()
    names, r = pc.ranges()
    elev = pc.elevation()
    elev[0, 0:4] = np.nan
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 include_unlisted=False, elevation=pc.grid(elev),
                                 elevation_limits=pc.limits_table())
    t = aoh.table.set_index("species")
    assert not t.loc["sp_e", "valid"] and t.loc["sp_e", "current"] == 0
    rr = np.nan_to_num(r[0]) > 0
    keep = rr & (np.isnan(elev) | (elev >= 500))   # sp_a: 500 m and up; cells without elevation stay
    assert t.loc["sp_a", "range_cells"] == keep.sum()


def test_cell_area_from_the_grid():
    proj = pc.grid(np.zeros((2, 3)))
    np.testing.assert_allclose(ls.sdm.cell_area_km2(proj), 1.0)
    geo = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), coords={"y": [0.5, -0.5], "x": [0.5, 1.5, 2.5]})
    geo = geo.rio.write_crs("EPSG:4326")
    a = ls.sdm.cell_area_km2(geo)
    assert a.shape == (2, 3)
    np.testing.assert_allclose(a, 12308.8, rtol=1e-3)  # one degree at the equator
    with pytest.raises(ValueError, match="cell_area"):
        ls.sdm.cell_area_km2(xr.DataArray(np.zeros((2, 2)), dims=("y", "x")))


# ----------------------------------------------------------------------------- intactness

def test_intactness_of_a_uniform_footprint_is_the_squared_quality():
    h = np.full((6, 7), 3.0)
    q = np.exp(-ls.INTACTNESS_GAMMA * 3.0)
    for r in (0, 1, 3):
        np.testing.assert_allclose(ls.intactness(h, r, cell_size=1.0).values, q ** (2 * 0.5), rtol=1e-14)


def test_intactness_restoration_and_nodata():
    h = pc.footprint((8, 9), 5)
    a, x = np.full(h.shape, 2.0), np.full(h.shape, 0.5)
    q0 = ls.intactness(h, 2, cell_size=1.0).values
    q1 = ls.intactness(h, 2, cell_size=1.0, reduction=a, restored=x).values
    np.testing.assert_array_equal(np.isnan(q0), np.isnan(h))
    assert np.all(q1[~np.isnan(h)] >= q0[~np.isnan(h)])
    np.testing.assert_allclose(q1, ls.intactness(np.maximum(h - 1.0, 0.0), 2, cell_size=1.0).values, rtol=1e-15)


def test_intactness_cell_size_from_the_grid_and_threads():
    h = pc.footprint((8, 9), 6)
    da = pc.grid(h, res=2000.0)
    np.testing.assert_array_equal(ls.intactness(da, 2).values, ls.intactness(h, 2, cell_size=2.0).values)
    np.testing.assert_array_equal(ls.intactness(da, 2, n_threads=1).values, ls.intactness(da, 2, n_threads=4).values)
    geo = xr.DataArray(h, dims=("y", "x"), coords={"y": -0.05 - 0.1 * np.arange(8), "x": 0.05 + 0.1 * np.arange(9)})
    with pytest.raises(ValueError, match="degrees"):
        ls.intactness(geo.rio.write_crs("EPSG:4326"))
    with pytest.raises(ValueError, match="coordinate system"):
        ls.intactness(geo)
    with pytest.raises(ValueError, match="radius"):
        ls.intactness(h, 1.5, cell_size=1.0)


def test_intactness_gradient_matches_finite_differences():
    h = pc.footprint((7, 8), 7, low=0.0)
    h[2, 3] = 0.0          # no footprint left: restoring it changes nothing
    a = 1.0 + pc.restored((7, 8), 8)
    g = ls.intactness_gradient(h, a, 2, cell_size=1.5).values
    eps = 1e-6
    total = lambda x: np.nansum(ls.intactness(h, 2, cell_size=1.5, reduction=a, restored=x).values)  # noqa: E731
    for k in [(0, 0), (2, 3), (3, 4), (6, 7), (4, 2)]:
        x = np.zeros(h.shape)
        x[k] = eps
        fd = (total(x) - total(np.zeros(h.shape))) / eps
        if np.isnan(h[k]):
            assert np.isnan(g[k])
        else:
            assert g[k] == pytest.approx(fd, rel=1e-4, abs=1e-9), k
    assert g[2, 3] == 0.0


# ----------------------------------------------------------------------------- uncertainty

def test_sum_sd_forms():
    s = np.array([1.0, -2.0, 2.0, np.nan])
    assert ls.uncertainty.sum_sd(s) == pytest.approx(3.0)
    assert ls.uncertainty.sum_sd(s, method="linear") == pytest.approx(5.0)
    assert ls.uncertainty.sum_sd(s, correlation=0.0) == pytest.approx(3.0)
    assert ls.uncertainty.sum_sd(s, correlation=1.0) == pytest.approx(1.0)
    assert ls.uncertainty.sum_sd(s, weights=2.0) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        ls.uncertainty.sum_sd(s, method="cubic")
    with pytest.raises(ValueError):
        ls.uncertainty.sum_sd(s, correlation=2.0)


def test_product_sd_is_goodmans_variance():
    rng = np.random.default_rng(3)
    x, v = 2.0 + rng.standard_normal(400000) * 0.3, 5.0 + rng.standard_normal(400000) * 0.7
    assert float(ls.uncertainty.product_sd(2.0, 0.3, 5.0, 0.7)) == pytest.approx(np.std(x * v), rel=5e-3)
    da = pc.grid(np.full((2, 2), 2.0))
    out = ls.uncertainty.product_sd(da, 0.3, 5.0, 0.7)
    assert isinstance(out, xr.DataArray) and out.shape == (2, 2)


def test_delta_method_and_monte_carlo_agree():
    f = lambda p: np.array([p[0] * p[1], p[0] + 2 * p[1], np.exp(0.1 * p[0])])  # noqa: E731
    p, sd = [2.0, 3.0], [0.05, 0.1]
    d = ls.uncertainty.delta_method(f, p, sd)
    m = ls.uncertainty.monte_carlo(f, p, sd, n=40000, seed=4)
    np.testing.assert_allclose(d.sd, m.sd, rtol=0.02)
    np.testing.assert_allclose(d.jacobian[1], [1.0, 2.0], rtol=1e-8)
    assert d.sd[1] == pytest.approx(np.hypot(0.05, 0.2), rel=1e-8)
    lin = ls.uncertainty.delta_method(f, p, sd, method="linear")
    assert lin.sd[1] == pytest.approx(0.25, rel=1e-8)
    cov = ls.uncertainty.delta_method(f, p, sd, covariance=np.diag(np.square(sd)))
    np.testing.assert_allclose(cov.sd, d.sd, rtol=1e-12)
    corr = np.array([[0.0025, 0.004], [0.004, 0.01]])  # correlation 0.8
    assert ls.uncertainty.delta_method(f, p, sd, covariance=corr).sd[1] == pytest.approx(
        np.sqrt(0.0025 + 4 * 0.01 + 4 * 0.004), rel=1e-8)
    mc = ls.uncertainty.monte_carlo(f, p, sd, n=200, seed=4, keep=True, quantiles=(0.5,))
    assert mc.samples.shape == (200, 3) and mc.quantiles.shape == (1, 3)


def test_interval_levels():
    low, high = ls.uncertainty.interval(np.array([1.0, 2.0]), np.array([0.1, 0.2]), level=0.9)
    np.testing.assert_allclose(high - low, 2 * 1.6448536269514722 * np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        ls.uncertainty.interval(1.0, 1.0, level=1.0)
