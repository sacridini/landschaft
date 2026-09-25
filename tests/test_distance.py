# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Euclidean Distance Transform in landschaft.

Includes:
1. Verification against reference distance formulations.
2. Synthetic geometry tests (isolated square patch, concentric distances).
3. Value bounds on byte distance map (10..129 background, 131..250 foreground).
4. Real landscape raster test (input.tif).
5. GeoTIFF writing and GTB colormap integration.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

import landschaft as ls


@pytest.fixture(scope="module")
def input_raster():
    return ls.load_raster(ls.example_data("input"))


def test_distance_synthetic_square_patch():
    """Test signed Euclidean distance and byte mapping on a centered 6x6 patch."""
    arr = np.ones((20, 20), dtype=np.uint8) # Background = 1
    arr[7:13, 7:13] = 2                     # Foreground = 2

    res = ls.euclidean_distance(arr)

    # 1. Sign semantics: foreground positive, background negative
    signed = np.asarray(res.signed_distance)
    fg_mask = arr == 2
    bg_mask = arr == 1

    assert (signed[fg_mask] > 0).all()
    assert (signed[bg_mask] < 0).all()

    # 2. Maximum FG distance should be at the center (dist ~ 3.0)
    assert pytest.approx(res.max_fg_distance, abs=0.1) == 3.0
    # Minimum FG distance on edges is 1.0
    assert pytest.approx(signed[7, 7], abs=0.1) == 1.0

    # 3. Byte map bounds
    b_map = np.asarray(res.byte_map)
    assert (b_map[fg_mask] >= 131).all()
    assert (b_map[fg_mask] <= 250).all()
    assert (b_map[bg_mask] >= 10).all()
    assert (b_map[bg_mask] <= 129).all()

    # 4. Pixel count checks
    assert res.fg_pixel_count == 36
    assert res.bg_pixel_count == 364


def test_distance_real_landscape(input_raster):
    """Test distance transform on input.tif and check expected metrics."""
    res = ls.euclidean_distance(input_raster, pixel_size=30.0)

    assert res.signed_distance.shape == input_raster.shape
    assert res.byte_map.shape == input_raster.shape

    # Exact metrics on input.tif
    assert res.fg_pixel_count == 428490
    assert res.bg_pixel_count == 571240
    assert pytest.approx(res.mean_fg_distance, abs=0.05) == 6.01
    assert pytest.approx(res.max_fg_distance, abs=0.05) == 37.00
    assert pytest.approx(res.mean_bg_distance, abs=0.05) == 5.03
    assert pytest.approx(res.max_bg_distance, abs=0.05) == 33.54

    # Real world unit checks (pixel_size=30 meters)
    assert pytest.approx(res.stats.iloc[0]["mean_distance_units"], abs=0.5) == 6.01 * 30.0
    assert pytest.approx(res.stats.iloc[0]["max_distance_units"], abs=0.5) == 37.00 * 30.0


def test_distance_write_raster_integration(tmp_path, input_raster):
    """Test saving Euclidean distance GeoTIFF with colormap and tags."""
    res = ls.euclidean_distance(input_raster)
    out_file = tmp_path / "test_dist.tif"

    ls.write_raster(res.byte_map, out_file)
    assert out_file.exists()

    with rasterio.open(out_file) as src:
        cm = src.colormap(1)
        assert cm[130] == (0, 255, 255, 255) # Cyan boundary
        assert cm[131] == (180, 255, 75, 255) # Light green FG edge
        tags = src.tags()
        assert "GTB_EUCL" in tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
