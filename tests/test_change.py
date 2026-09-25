# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Change Analysis and Morphological Change Detection (MCD)."""

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from landschaft.change import (
    FOSChangeResult,
    MorphologicalChangeResult,
    SimpleChangeResult,
    fos_change,
    morphological_change,
    simple_change,
)


def test_simple_change_basic():
    # 4x4 raster with known transitions:
    # (1,1)->11, (1,2)->12, (2,1)->21, (2,2)->22, (0,anything)->255
    t1 = np.array(
        [
            [1, 1, 2, 2],
            [1, 2, 1, 2],
            [0, 1, 2, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    t2 = np.array(
        [
            [1, 2, 1, 2],
            [1, 2, 2, 1],
            [2, 0, 0, 0],
            [1, 1, 1, 2],
        ],
        dtype=np.uint8,
    )

    res = simple_change(t1, t2)
    assert isinstance(res, SimpleChangeResult)
    cmap = res.change_map

    # Check known pixel coordinates
    assert cmap[0, 0] == 11  # 1->1 Stable BG
    assert cmap[0, 1] == 12  # 1->2 Gain
    assert cmap[0, 2] == 21  # 2->1 Loss
    assert cmap[0, 3] == 22  # 2->2 Stable FG
    assert cmap[1, 2] == 12  # 1->2 Gain
    assert cmap[1, 3] == 21  # 2->1 Loss

    # Missing pixels where either is 0
    assert cmap[2, 0] == 255  # 0 at t1
    assert cmap[2, 1] == 255  # 0 at t2
    assert cmap[2, 2] == 255  # 0 at t2
    assert cmap[2, 3] == 255  # 0 at both

    # Check counts match histogram
    assert res.stable_bg == int(np.sum(cmap == 11))
    assert res.gain == int(np.sum(cmap == 12))
    assert res.loss == int(np.sum(cmap == 21))
    assert res.stable_fg == int(np.sum(cmap == 22))
    assert res.missing == int(np.sum(cmap == 255))
    assert res.total_valid == res.stable_bg + res.gain + res.loss + res.stable_fg
    assert res.net_change == res.gain - res.loss

    summary = res.summary()
    assert "Simple Change Analysis Summary" in summary
    assert "Net Change" in summary


def test_morphological_change_core_and_spurious():
    # Construct a 20x20 scenario:
    # Background = 1
    # Time 1: A large 10x10 square of foreground (core pixels will be at center > 2 distance from BG)
    # Time 2: The 10x10 square is expanded by a 4x4 block connected to it (real MCD gain)
    # plus a single 1-pixel isolated speckle toggle at (1, 1) far away from core (spurious boundary chatter)
    t1 = np.ones((20, 20), dtype=np.uint8)
    t1[5:15, 5:15] = 2  # 10x10 FG (center [7:13, 7:13] is core > 2 pixels from edge)

    t2 = np.ones((20, 20), dtype=np.uint8)
    t2[5:15, 5:15] = 2
    t2[15:18, 5:9] = 2  # Core-connected expansion
    t2[1, 1] = 2  # 1-pixel chatter far away (no core > 2)

    res = morphological_change(t1, t2)
    assert isinstance(res, MorphologicalChangeResult)

    cmap = res.change_map
    assert cmap.shape == (20, 20)

    # (1, 1) had no core shift (>2) -> spurious boundary chatter (176)
    assert cmap[1, 1] == 176
    assert res.spurious >= 1

    # Center of stable forest is stable FG (22)
    assert cmap[10, 10] == 22

    # Background corners remain stable BG (11)
    assert cmap[0, 0] == 11
    assert cmap[19, 19] == 11

    # Check area metrics
    assert res.fg_area_t1 == int(np.sum(t1 == 2))
    assert res.fg_area_t2 == int(np.sum(t2 == 2))
    assert res.net_change == res.fg_area_t2 - res.fg_area_t1

    summary = res.summary()
    assert "Morphological Change Detection (MCD) Summary" in summary
    assert "MCD-Spurious" in summary


def test_fos_change_density_and_matrix():
    # 10x10 raster of FOS density values:
    # 0..100 = FG density
    # 101 = BG
    # 102 = Missing
    # 105 = Water
    fos1 = np.full((10, 10), 101, dtype=np.uint8)  # all BG
    fos2 = np.full((10, 10), 101, dtype=np.uint8)

    # Pixel (0, 0): BG -> BG => 252 (Stable BG)
    # Pixel (1, 1): BG -> FG (density 50) => 250 (FG Gain)
    fos2[1, 1] = 50
    # Pixel (2, 2): FG (density 80) -> BG => 251 (FG Loss)
    fos1[2, 2] = 80
    # Pixel (3, 3): Missing at t1 => 254 (Missing)
    fos1[3, 3] = 102
    # Pixel (4, 4): Water at t2 => 253 (Water)
    fos2[4, 4] = 105

    # Common FG pixels:
    # (5, 5): Rare (5%) -> Intact (100%) => shift to higher density = fragmentation decrease
    fos1[5, 5] = 5
    fos2[5, 5] = 100
    # Delta FOS = 5 + 100 - 100 = 5

    # (6, 6): Dominant (70%) -> Dominant (70%) => unchanged class
    fos1[6, 6] = 70
    fos2[6, 6] = 70
    # Delta FOS = 70 + 100 - 70 = 100 (exact no change)

    # (7, 7): Interior (95%) -> Patchy (20%) => shift to lower density = fragmentation increase
    fos1[7, 7] = 95
    fos2[7, 7] = 20
    # Delta FOS = 95 + 100 - 20 = 175

    res = fos_change(fos1, fos2)
    assert isinstance(res, FOSChangeResult)

    cmap = res.change_map
    assert cmap[0, 0] == 252  # Stable BG
    assert cmap[1, 1] == 250  # FG Gain
    assert cmap[2, 2] == 251  # FG Loss
    assert cmap[3, 3] == 254  # Missing
    assert cmap[4, 4] == 253  # Water
    assert cmap[5, 5] == 5    # Delta FOS
    assert cmap[6, 6] == 100  # Delta FOS no change
    assert cmap[7, 7] == 175  # Delta FOS

    assert res.fg_gain == 1
    assert res.fg_loss == 1
    assert res.forcom == 3
    assert res.diag_same_class == 1
    assert res.diff_class == 2
    assert res.frag_decrease_pct == 50.0
    assert res.frag_increase_pct == 50.0

    # 6x6 matrix checks
    mat = res.matrix_6x6
    assert mat.shape == (6, 6)
    # Rare (0-9%) -> Intact (100%)
    assert mat.loc["Rare (0-9%)", "Intact (100%)"] == 1
    # Dominant -> Dominant
    assert mat.loc["Dominant (60-89%)", "Dominant (60-89%)"] == 1
    # Interior -> Patchy
    assert mat.loc["Interior (90-99%)", "Patchy (10-39%)"] == 1

    summary = res.summary()
    assert "Fragmentation (FOS) Change Summary" in summary
    assert "FOS 6x6 Transition Matrix" in summary


def test_dataarray_and_metadata_preservation():
    da1 = xr.DataArray(
        np.array([[1, 2], [2, 1]], dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": [10.0, 20.0], "x": [100.0, 200.0]},
        attrs={"crs": "EPSG:4326"},
    )
    da2 = xr.DataArray(
        np.array([[2, 2], [1, 1]], dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": [10.0, 20.0], "x": [100.0, 200.0]},
        attrs={"crs": "EPSG:4326"},
    )

    res = simple_change(da1, da2)
    assert isinstance(res.change_map, xr.DataArray)
    assert res.change_map.dims == ("y", "x")
    assert np.all(res.change_map.coords["y"] == [10.0, 20.0])
    assert res.change_map.attrs["landschaft_product"] == "SIMPLE_CHANGE"


def test_dimension_mismatch():
    t1 = np.ones((10, 10), dtype=np.uint8)
    t2 = np.ones((10, 12), dtype=np.uint8)

    with pytest.raises(ValueError, match="Input rasters must have identical dimensions"):
        simple_change(t1, t2)
