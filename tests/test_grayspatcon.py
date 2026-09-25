# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Continuous Grayscale Spatial Convolution (GraySpatCon)."""

import numpy as np
import pytest
import xarray as xr

import guidospy as gp
from guidospy.grayspatcon import gray_spatial_convolution


def test_grayspatcon_constant_raster():
    # 15x15 raster of constant value 42
    data = np.full((15, 15), 42, dtype=np.uint8)

    # Test all metrics on uniform surface
    mean_res = gray_spatial_convolution(data, metric="mean", window_size=5)
    assert np.allclose(mean_res, 42.0)

    med_res = gray_spatial_convolution(data, metric="median", window_size=5)
    assert np.allclose(med_res, 42.0)

    maj_res = gray_spatial_convolution(data, metric="majority", window_size=5)
    assert np.allclose(maj_res, 42.0)

    min_res = gray_spatial_convolution(data, metric="min", window_size=5)
    assert np.allclose(min_res, 42.0)

    max_res = gray_spatial_convolution(data, metric="max", window_size=5)
    assert np.allclose(max_res, 42.0)

    rng_res = gray_spatial_convolution(data, metric="range", window_size=5)
    assert np.allclose(rng_res, 0.0)

    std_res = gray_spatial_convolution(data, metric="stddev", window_size=5)
    assert np.allclose(std_res, 0.0)

    shan_res = gray_spatial_convolution(data, metric="shannon_diversity", window_size=5)
    assert np.allclose(shan_res, 0.0)

    even_res = gray_spatial_convolution(data, metric="shannon_evenness", window_size=5)
    assert np.allclose(even_res, 0.0)


def test_grayspatcon_checkerboard_majority_and_range():
    # 10x10 pattern with two values 10 and 90
    data = np.zeros((10, 10), dtype=np.uint8)
    data[:5, :] = 10
    data[5:, :] = 90

    # A 3x3 window strictly in the top half has min=10, max=10, range=0
    rng_res = gray_spatial_convolution(data, metric="range", window_size=3)
    assert rng_res[2, 2] == 0.0

    # A 3x3 window straddling row 4 and 5 has min=10, max=90, range=80
    assert rng_res[4, 4] == 80.0
    assert rng_res[5, 5] == 80.0


def test_grayspatcon_mask_missing():
    data = np.full((11, 11), 50, dtype=np.uint8)
    data[5, 5] = 255  # missing pixel

    # With mask_missing=True, center pixel is masked to -0.01 (GTB nodata value)
    res_masked = gray_spatial_convolution(data, metric="mean", window_size=3, mask_missing=True)
    assert np.isclose(res_masked[5, 5], -0.01)
    assert np.isclose(res_masked[0, 0], 50.0)

    # With mask_missing=False, center pixel computes the mean of its neighbors
    res_unmasked = gray_spatial_convolution(data, metric="mean", window_size=3, mask_missing=False)
    assert np.isclose(res_unmasked[5, 5], 50.0)


def test_grayspatcon_dataarray():
    da = xr.DataArray(
        np.full((11, 11), 30, dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": np.arange(11), "x": np.arange(11)},
        attrs={"unit": "density"},
    )
    res = gray_spatial_convolution(da, metric="mean", window_size=5)
    assert isinstance(res, xr.DataArray)
    assert res.dims == ("y", "x")
    assert res.attrs["guidospy_product"] == "GRAY_SPATIAL_CONVOLUTION"
    assert res.attrs["spatial_metric"] == "mean"
    assert res.attrs["window_size"] == 5


def test_grayspatcon_invalid_inputs():
    data = np.full((10, 10), 10, dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown metric"):
        gray_spatial_convolution(data, metric="invalid_metric")

    with pytest.raises(ValueError, match="window_size must be an odd integer >= 3"):
        gray_spatial_convolution(data, metric="mean", window_size=4)

    with pytest.raises(ValueError, match="window_size must be an odd integer >= 3"):
        gray_spatial_convolution(data, metric="mean", window_size=1)
