# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Landscape Mosaic (LM) Tri-Polar Analysis."""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import landschaft as ls

SPATCON_EXE = r"C:\Users\eduar\Documents\dev\GTB\external_sources\spatcon64.exe"
RECODE_EXE = r"C:\Users\eduar\Documents\dev\GTB\external_sources\recode64.exe"
RECODE_TXT = r"C:\Users\eduar\Documents\dev\GTB\guidos_progs\recodelm103.sav"
HAS_GTB_BINARIES = os.path.exists(SPATCON_EXE) and os.path.exists(RECODE_EXE) and os.path.exists(RECODE_TXT)


def run_gtb_lm(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Run spatcon rule 7 and recode to get both 103-class and 19-class results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "scsize.txt"), "w") as f:
            f.write(f"nrows {data.shape[0]}\n")
            f.write(f"ncols {data.shape[1]}\n")
        with open(os.path.join(tmpdir, "scpars.txt"), "w") as f:
            f.write(f"w {window}\n")
            f.write("r 7\n")
            f.write("a 2\n")
            f.write("h 1\n")
            f.write("b 0\n")
            f.write("m 0\n")
            f.write("f 0\n")
        data.astype(np.uint8).tofile(os.path.join(tmpdir, "scinput"))
        subprocess.run([SPATCON_EXE], cwd=tmpdir, check=True, capture_output=True)
        im103 = np.fromfile(os.path.join(tmpdir, "scoutput"), dtype=np.uint8).reshape(data.shape)

        # Missing pixels in input are set to 0 in 103-class map
        im103[data == 0] = 0

        # Recode to 19 classes
        with open(os.path.join(tmpdir, "recsize.txt"), "w") as f:
            f.write(f"nrows {data.shape[0]}\n")
            f.write(f"ncols {data.shape[1]}\n")
        with open(RECODE_TXT, "r") as f_in, open(os.path.join(tmpdir, "recode.txt"), "w") as f_out:
            f_out.write(f_in.read())
        im103.tofile(os.path.join(tmpdir, "recinput"))
        subprocess.run([RECODE_EXE], cwd=tmpdir, check=True, capture_output=True)
        im19 = np.fromfile(os.path.join(tmpdir, "recoutput"), dtype=np.uint8).reshape(data.shape)

    return im103, im19


def test_landscape_mosaic_pure_classes():
    # 20x20 raster with 4 quadrants: missing (0), agr (1), for (2), dev (3)
    data = np.zeros((20, 20), dtype=np.uint8)
    data[:10, :10] = 1   # agr
    data[:10, 10:] = 2   # for
    data[10:, :10] = 3   # dev
    data[10:, 10:] = 0   # missing

    lm19 = ls.landscape_mosaic(data, window_size=3, output_classes=19)
    assert lm19.shape == (20, 20)
    assert lm19.dtype == np.uint8

    # Missing pixels in input must remain 0
    assert (lm19[10:, 10:] == 0).all()

    # Deep inside pure quadrants (window size 3 centered at 5, 5):
    assert lm19[5, 5] == 18   # Pure agriculture (100%) -> 18
    assert lm19[5, 15] == 17  # Pure forest (100%) -> 17
    assert lm19[15, 5] == 19  # Pure developed (100%) -> 19

    lm103 = ls.landscape_mosaic(data, window_size=3, output_classes=103)
    assert lm103[5, 5] == 180   # 103-code for pure agr
    assert lm103[5, 15] == 170  # 103-code for pure for
    assert lm103[15, 5] == 190  # 103-code for pure dev
    assert (lm103[10:, 10:] == 0).all()


def test_landscape_mosaic_stats():
    data = np.zeros((30, 30), dtype=np.uint8)
    data[:15, :15] = 1
    data[:15, 15:] = 2
    data[15:, :] = 3

    df = ls.landscape_mosaic_stats(data, window_size=5)
    assert isinstance(df, pd.DataFrame)
    assert "class_code" in df.columns
    assert "pixel_count" in df.columns
    assert "percentage" in df.columns
    assert "recode_class_19" in df.columns
    assert "recode_name_19" in df.columns

    # Sum of percentages over all non-missing classes must be ~100%
    assert np.isclose(df["percentage"].sum(), 100.0)


def test_landscape_mosaic_xarray_metadata():
    da = xr.DataArray(
        np.array([[1, 2], [2, 3]], dtype=np.uint8),
        dims=("y", "x"),
        attrs={"crs": "EPSG:32632"},
    )
    res = ls.landscape_mosaic(da, window_size=3)
    assert isinstance(res, xr.DataArray)
    assert res.attrs["landschaft_product"] == "LANDSCAPE_MOSAIC"
    assert res.attrs["output_classes"] == 19
    assert res.attrs["window_size"] == 3
    assert res.attrs["colormap"] == "lm"


def test_invalid_parameters():
    data = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd integer"):
        ls.landscape_mosaic(data, window_size=4)
    with pytest.raises(ValueError, match="output_classes must be either 19 or 103"):
        ls.landscape_mosaic(data, window_size=3, output_classes=50)
    with pytest.raises(ValueError, match="Landscape Mosaic input must contain only classes"):
        ls.landscape_mosaic(np.array([[1, 4]], dtype=np.uint8), window_size=3)


@pytest.mark.skipif(not HAS_GTB_BINARIES, reason="GTB binaries or recodelm103.sav not found")
def test_accuracy_against_gtb_spatcon_and_recode():
    rng = np.random.default_rng(999)
    sample = rng.choice([0, 1, 2, 3], size=(40, 40), p=[0.05, 0.35, 0.35, 0.25]).astype(np.uint8)

    # 1. Compare 103-class output
    gtb_103, gtb_19 = run_gtb_lm(sample, window=7)
    py_103 = ls.landscape_mosaic(sample, window_size=7, output_classes=103)
    py_19 = ls.landscape_mosaic(sample, window_size=7, output_classes=19)

    np.testing.assert_array_equal(py_103, gtb_103)
    np.testing.assert_array_equal(py_19, gtb_19)

    # 2. Compare larger window 13
    gtb_103_w13, gtb_19_w13 = run_gtb_lm(sample, window=13)
    py_103_w13 = ls.landscape_mosaic(sample, window_size=13, output_classes=103)
    py_19_w13 = ls.landscape_mosaic(sample, window_size=13, output_classes=19)

    np.testing.assert_array_equal(py_103_w13, gtb_103_w13)
    np.testing.assert_array_equal(py_19_w13, gtb_19_w13)
