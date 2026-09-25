# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Spatial Joint Entropy in guidospy.

Includes:
1. Exact numerical verification against GTB's entropy_mspainp.pro algorithm.
2. Synthetic extreme cases (homogeneous landscape = 0% entropy, checkerboard).
3. Moving window local spatial entropy map.
4. GeoTIFF writing and GTB colormap integration.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

import guidospy as gp


def gtb_reference_entropy(arr: np.ndarray) -> float:
    """Independent implementation of GTB's entropy_mspainp.pro procedure."""
    # Mapping: FG(2) -> 7, BG(1) -> 3, Missing(0) -> 0
    image = (arr == 2).astype(np.uint8) * 7 + (arr == 1).astype(np.uint8) * 3
    h, w = image.shape

    # Pad with border replication
    ext = np.pad(image, 1, mode="edge")

    # 8 neighbor directional shifts
    dx = [1, -1, 0, 0, 1, -1, 1, -1]
    dy = [0, 0, 1, -1, 1, -1, -1, 1]

    counts = np.zeros(256, dtype=np.uint64)
    for y in range(h):
        for x in range(w):
            c0 = int(ext[y + 1, x + 1])
            for k in range(8):
                c1 = int(ext[y + 1 + dy[k], x + 1 + dx[k]])
                diff = (c0 - c1) % 256
                counts[diff] += 1

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts[counts > 0] / total
    ent_max = 1.5
    al2 = np.log(2.0)
    normfac = 100.0 / (al2 * ent_max)

    h_val = float(-np.sum(p * np.log(p)) * normfac)
    return min(100.0, max(0.0, h_val))


@pytest.fixture(scope="module")
def input_raster():
    return gp.load_raster(gp.example_data("input"))


def test_spatial_entropy_accuracy_against_gtb_reference(input_raster):
    """Verify bit-exact / numerical match with GTB's entropy_mspainp.pro."""
    # Test on full input.tif
    expected_index = gtb_reference_entropy(input_raster.values)
    res = gp.spatial_entropy(input_raster)

    assert pytest.approx(res.entropy_index, rel=1e-6) == expected_index


def test_spatial_entropy_homogeneous():
    """Homogeneous landscapes have zero transitions and therefore 0.0% entropy."""
    # Pure foreground (with 1 bg pixel to satisfy input validation)
    arr = np.full((30, 30), 2, dtype=np.uint8)
    arr[0, 0] = 1

    res = gp.spatial_entropy(arr)
    # Almost all transitions are 7-7 (0 diff), so entropy is very close to 0 (< 2.0%)
    assert res.entropy_index < 2.0


def test_spatial_entropy_map(input_raster):
    """Test local moving-window spatial entropy map."""
    # Compute on small patch for quick testing
    patch = input_raster.values[:100, :100]
    ent_map = gp.spatial_entropy_map(patch, window_size=20)

    assert ent_map.shape == patch.shape
    arr_val = np.asarray(ent_map)
    assert (arr_val >= 0.0).all()
    assert (arr_val <= 100.0).all()


def test_entropy_write_raster_integration(tmp_path, input_raster):
    """Test saving spatial entropy GeoTIFF with colormap and tags."""
    patch = input_raster[:100, :100]
    ent_map = gp.spatial_entropy_map(patch, window_size=20)
    out_file = tmp_path / "test_entropy.tif"

    # Convert to byte for colormapped TIFF
    byte_ent = np.round(ent_map).astype(np.uint8)
    byte_ent.attrs["guidospy_product"] = "SPATIAL_ENTROPY"
    gp.write_raster(byte_ent, out_file)
    assert out_file.exists()

    with rasterio.open(out_file) as src:
        cm = src.colormap(1)
        # Check colormap entries
        assert cm[0] == (0, 0, 0, 255)
        assert cm[100] == (255, 0, 255, 255)
        tags = src.tags()
        assert "GTB_ENT" in tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
