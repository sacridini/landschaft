# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.bias_surface against MASS::kde2d, and its options."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("bias"))
def test_matches_kde2d(case):
    ref = sdm_reference.result("bias", case)
    ours = sdm_cases.run_python(case)
    sdm_reference.assert_same(ours["bw"], ref["bw"], "bw", rtol=1e-14)
    # kde2d sums with BLAS (tcrossprod) and its own dnorm: equal up to rounding
    sdm_reference.assert_same(ours["z"], ref["z"], "z", rtol=1e-10, atol=1e-12 * np.max(ref["z"]))


@pytest.fixture(scope="module")
def grid():
    return sdm_cases.build_input("stack_80")


@pytest.fixture(scope="module")
def points():
    return sdm_cases.build_input("bias_points")


def test_nrd_is_kde2d_default(grid, points):
    b = ls.sdm.bias_surface(points, grid)
    bx = ls.sdm._bias._bw_nrd(points["x"].to_numpy())
    by = ls.sdm._bias._bw_nrd(points["y"].to_numpy())
    assert b.attrs["bandwidth"] == (bx, by)
    assert b.attrs["n"] == len(points)
    assert b.dims == ("y", "x") and b.shape == grid.shape[1:]
    assert b.rio.crs == grid.rio.crs


def test_isotropic_and_limits(grid, points):
    b = ls.sdm.bias_surface(points, grid, isotropic=True)
    bx, by = b.attrs["bandwidth"]
    assert bx == by == max(ls.sdm.bias_surface(points, grid).attrs["bandwidth"])
    b = ls.sdm.bias_surface(points, grid, bandwidth=(500, 50_000), limits=(2_000, 10_000))
    assert b.attrs["bandwidth"] == (2_000, 10_000)
    b = ls.sdm.bias_surface(points, grid, bandwidth=500, limits=(2_000, None))
    assert b.attrs["bandwidth"] == (2_000, 2_000)


def test_integer_weights_are_duplicates(grid, points):
    w = np.tile([1, 2, 3], len(points) // 3 + 1)[: len(points)]
    rep = points.loc[points.index.repeat(w)].reset_index(drop=True)
    a = ls.sdm.bias_surface(points, grid, bandwidth=3000, weights=w)
    b = ls.sdm.bias_surface(rep, grid, bandwidth=3000)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)


def test_density_integrates_to_records_inside(grid):
    # one record in the middle, a narrow kernel: the cells hold all of its mass (1 per km2 * 1000^2 m2)
    one = pd.DataFrame({"x": [340_000.0], "y": [7_460_000.0]})
    b = ls.sdm.bias_surface(one, grid, bandwidth=5000, mask=False)
    assert float(b.sum()) * 1000.0 * 1000.0 == pytest.approx(1.0, rel=1e-6)


def test_mask_and_rescale(points):
    env = sdm_cases.build_input("stack_200")  # 3000 cells missing in the first band
    b = ls.sdm.bias_surface(points, env.isel(y=slice(0, 80), x=slice(0, 80)), bandwidth=4000,
                            rescale=(0.05, 0.95))
    missing = np.isnan(env.isel(y=slice(0, 80), x=slice(0, 80)).values).any(axis=0)
    assert np.array_equal(np.isnan(b.values), missing)
    assert float(b.min()) == 0.05 and float(b.max()) == 0.95
    full = ls.sdm.bias_surface(points, env.isel(y=slice(0, 80), x=slice(0, 80)), bandwidth=4000, mask=False)
    assert not np.isnan(full.values).any()


def test_reprojects_records():
    import geopandas as gpd

    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    utm = ls.reproject(env.isel(y=slice(100, 140), x=slice(100, 140)), "EPSG:32722", res=50_000)
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(occ["lon"], occ["lat"]), crs="EPSG:4326")
    a = ls.sdm.bias_surface(occ, utm, bandwidth=100_000, mask=False)
    b = ls.sdm.bias_surface(gdf.to_crs("EPSG:32722"), utm, bandwidth=100_000, mask=False)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-9)


def test_feeds_background():
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    bias = ls.sdm.bias_surface(occ, env, bandwidth=1.0, rescale=(0.0, 1.0))
    bg = ls.sdm.background(env, 200, bias=bias, exclude_occurrences=False, seed=3)
    flat = ls.sdm.background(env, 200, exclude_occurrences=False, seed=3)
    # drawn near the records: the background's mean density is far above that of a uniform draw
    vals = bias.values.reshape(-1)
    assert vals[bg["cell"]].mean() > 3 * vals[flat["cell"]].mean()


def test_errors(grid, points):
    with pytest.raises(ValueError, match="unknown bandwidth"):
        ls.sdm.bias_surface(points, grid, bandwidth="scott")
    with pytest.raises(ValueError, match="positive"):
        ls.sdm.bias_surface(points.iloc[:1], grid)  # one record: no normal reference bandwidth
    with pytest.raises(ValueError, match="one value per record"):
        ls.sdm.bias_surface(points, grid, bandwidth=1000, weights=[1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        ls.sdm.bias_surface(points, grid, bandwidth=1000, weights=-np.ones(len(points)))
    with pytest.raises(ValueError, match="lo < hi"):
        ls.sdm.bias_surface(points, grid, bandwidth=1000, rescale=(1, 0))
    far = pd.DataFrame({"x": [9e6, 9e6 + 1], "y": [9e6, 9e6 + 1]})
    with pytest.raises(ValueError, match="same in every cell"):
        ls.sdm.bias_surface(far, grid, bandwidth=1000, rescale=(0, 1))
    with pytest.raises(TypeError):
        ls.sdm.bias_surface(points, grid.values)
