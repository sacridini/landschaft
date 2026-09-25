# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Forest Area Density (FAD) & FAD-Multiscale in guidospy.

Includes:
1. Accuracy tests against the original GTB spatcon64.exe binary (rule 81).
2. Synthetic landscape edge cases (pure foreground, pure background, missing values).
3. Multiscale FAD consistency and statistics tests.
4. GeoTIFF writing and GTB colormap integration.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import guidospy as gp
from guidospy.validation import InvalidInputError

SPATCON_EXE = Path(r"C:\Users\eduar\Documents\dev\GTB\external_sources\spatcon64.exe")


def run_original_spatcon(arr: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Run GTB's original spatcon64.exe on a numpy array."""
    if not SPATCON_EXE.is_file():
        pytest.skip(f"spatcon executable not found at {SPATCON_EXE}")

    ny, nx = arr.shape
    with tempfile.TemporaryDirectory(prefix="gtb_spatcon_") as tmpdir:
        tmp = Path(tmpdir)
        with open(tmp / "scsize.txt", "w") as f:
            f.write(f"nrows {ny}\nncols {nx}\n")
        with open(tmp / "scpars.txt", "w") as f:
            f.write(f"w {window_size}\nr 81\na 2\nh 1\nb 0\nm 0\nf 0\n")
        arr.tofile(tmp / "scinput")

        proc = subprocess.run([str(SPATCON_EXE)], cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"spatcon failed: {proc.stderr}")

        out = np.fromfile(tmp / "scoutput", dtype=np.uint8).reshape((ny, nx))
        return out


def gtb_rescale_fad(spatcon_raw: np.ndarray) -> np.ndarray:
    """Apply the exact IDL rescaling from guidostoolbox.pro (lines 5250-5255)."""
    rescaled = np.zeros_like(spatcon_raw, dtype=np.uint8)
    q255 = (spatcon_raw == 255)
    val = (spatcon_raw.astype(np.float64) - 1.0) / 254.0
    val = np.clip(val, 0.0, 0.994999)
    rescaled = np.round(val * 100.0).astype(np.uint8)
    rescaled[q255] = 100
    return rescaled


@pytest.fixture(scope="module")
def input_raster():
    return gp.load_raster(gp.example_data("input"))


def test_fad_accuracy_against_gtb_spatcon(input_raster):
    """Verify bit-exact agreement between guidospy FAD and original GTB spatcon64.exe."""
    # Test on a representative 100x100 sub-patch containing fg, bg, and complex edges
    patch = input_raster.values[50:150, 50:150].copy()

    for w in [7, 13, 27]:
        spatcon_raw = run_original_spatcon(patch, window_size=w)
        expected_density = gtb_rescale_fad(spatcon_raw)

        res = gp.fad(patch, window_size=w)

        # Check raw density
        raw_val = np.asarray(res.raw_density)
        diff = np.abs(raw_val.astype(int) - expected_density.astype(int))

        # Must have zero mismatched pixels
        assert diff.max() == 0, f"Scale {w} had {np.count_nonzero(diff)} mismatched pixels"


def test_fad_synthetic_pure_foreground():
    """All foreground pixels (except 1 background pixel for compliance) should be Intact."""
    arr = np.full((30, 30), 2, dtype=np.uint8)
    arr[0, 0] = 1  # 1 background pixel to satisfy MSPA input validation
    res = gp.fad(arr, window_size=7)

    # Core interior pixels away from the 1 background corner must have 100% density
    density = np.asarray(res.density)
    classes = np.asarray(res.classes)
    assert density[15, 15] == 100
    assert classes[15, 15] == 6  # 6 = Intact
    assert res.stats.iloc[0]["intact_%"] > 95.0


def test_fad_synthetic_pure_background():
    """Background-only arrays must have 0% foreground and all pixels masked as 101."""
    arr = np.full((30, 30), 1, dtype=np.uint8)
    # validate_mspa_input requires at least one FG pixel:
    arr[15, 15] = 2

    res = gp.fad(arr, window_size=7)
    density = np.asarray(res.density)
    classes = np.asarray(res.classes)
    assert density[0, 0] == 101  # GTB background code
    assert classes[0, 0] == 101


def test_fad_synthetic_missing_values():
    """Missing values (0) must be masked as 102 and ignored in neighbor denominators."""
    arr = np.full((21, 21), 2, dtype=np.uint8)
    arr[0, :] = 0  # Missing row
    arr[10, 10] = 1 # Background dot

    res = gp.fad(arr, window_size=7)
    density = np.asarray(res.density)
    classes = np.asarray(res.classes)
    assert (density[0, :] == 102).all()
    assert (classes[0, :] == 102).all()


def test_fad_multiscale(input_raster):
    """Test multiscale FAD across 5 scales on full test raster."""
    scales = [7, 13, 27]
    res = gp.fad_multiscale(input_raster, window_sizes=scales)

    assert res.multiscale_map.shape == input_raster.shape
    assert len(res.single_scales) == 3
    assert len(res.stats) == 4  # 3 scales + 1 multiscale summary

    # Verify percentages in each row sum approximately to 100%
    for _, row in res.stats.iterrows():
        total_pct = (
            row["intact_%"]
            + row["interior_%"]
            + row["dominant_%"]
            + row["transitional_%"]
            + row["patchy_%"]
            + row["rare_%"]
        )
        assert pytest.approx(total_pct, abs=0.1) == 100.0

    # Summary string must contain GTB title
    assert "GuidosToolbox (GTB)" in res.summary()


def test_fad_invalid_inputs():
    """Test validation and error handling for bad parameters."""
    arr = np.ones((20, 20), dtype=np.uint8)
    arr[10, 10] = 2

    with pytest.raises(ValueError, match="odd integer"):
        gp.fad(arr, window_size=6)  # Even window size

    with pytest.raises(ValueError, match="odd integer"):
        gp.fad(arr, window_size=2)  # Window size < 3


def test_fad_write_raster_integration(tmp_path, input_raster):
    """Test saving FAD GeoTIFF with colormap and GDAL metadata."""
    res = gp.fad(input_raster, window_size=7)
    out_file = tmp_path / "test_fad.tif"

    gp.write_raster(res.density, out_file)
    assert out_file.exists()

    # Re-read and check colormap
    import rasterio

    with rasterio.open(out_file) as src:
        cm = src.colormap(1)
        assert cm[100] == (0, 120, 0, 255)    # Intact color (green)
        assert cm[101] == (175, 175, 175, 255) # Background (gray)
        assert cm[102] == (255, 255, 255, 255) # Missing (white)
        tags = src.tags()
        assert "GTB_FAD" in tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
