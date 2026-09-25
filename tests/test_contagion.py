# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Contagion (Pff) and Foreground Area Clustering (FAC)."""

import os
import subprocess
import tempfile
import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from landschaft.validation import InvalidInputError

SPATCON_EXE = r"C:\Users\eduar\Documents\dev\GTB\external_sources\spatcon64.exe"
HAS_SPATCON = os.path.exists(SPATCON_EXE)


def run_gtb_spatcon(data: np.ndarray, window: int, rule: int) -> np.ndarray:
    """Run original GTB spatcon64.exe binary and apply GTB byte rescaling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "scsize.txt"), "w") as f:
            f.write(f"nrows {data.shape[0]}\n")
            f.write(f"ncols {data.shape[1]}\n")
        with open(os.path.join(tmpdir, "scpars.txt"), "w") as f:
            f.write(f"w {window}\n")
            f.write(f"r {rule}\n")
            f.write("a 2\n")
            f.write("h 1\n")
            f.write("b 2\n" if rule == 76 else "b 0\n")
            f.write("m 0\n")
            f.write("f 0\n")
        data.astype(np.uint8).tofile(os.path.join(tmpdir, "scinput"))
        subprocess.run([SPATCON_EXE], cwd=tmpdir, check=True, capture_output=True)
        raw = np.fromfile(os.path.join(tmpdir, "scoutput"), dtype=np.uint8).reshape(data.shape)

    # Apply GTB rescaling matching batch_pff / batch_fac
    q255 = np.where(raw == 255)
    scaled = (raw.astype(float) - 1.0) / 254.0
    scaled = np.clip(scaled, 0.0, 0.994999)
    res = np.round(scaled * 100.0).astype(np.uint8)
    res[raw == 0] = 0
    res[q255] = 100
    res[data == 1] = 101
    res[data == 0] = 102
    return res


def test_contagion_pff_basic():
    # Simple 10x10 synthetic raster
    data = np.ones((10, 10), dtype=np.uint8)
    data[2:8, 2:8] = 2  # 6x6 foreground core
    data[0, 0] = 0      # 1 missing pixel

    res = ls.contagion_pff(data, window_size=3)
    assert isinstance(res, np.ndarray)
    assert res.shape == (10, 10)
    assert res.dtype == np.uint8

    # Background pixels should be 101, missing should be 102
    assert res[0, 0] == 102
    assert res[0, 1] == 101
    # Center of 6x6 foreground should have 100% self-joins
    assert res[4, 4] == 100


def test_clustering_fac_basic():
    data = np.ones((10, 10), dtype=np.uint8)
    data[2:8, 2:8] = 2
    data[0, 0] = 0

    res = ls.clustering_fac(data, window_size=3)
    assert isinstance(res, np.ndarray)
    assert res.shape == (10, 10)
    assert res.dtype == np.uint8

    assert res[0, 0] == 102
    assert res[0, 1] == 101
    # Center of 6x6 core in 3x3 window: all 12 edges are 2-2, so FAC is 100%
    assert res[4, 4] == 100


def test_contagion_float_output():
    data = np.ones((10, 10), dtype=np.uint8)
    data[1:8, 1:8] = 2  # 7x7 core
    data[0, 0] = 0

    res = ls.contagion_pff(data, window_size=5, as_byte=False)
    assert res.dtype == np.float32
    assert np.isnan(res[0, 0])
    assert np.isnan(res[0, 1])
    assert np.isclose(res[4, 4], 1.0)


def test_clustering_float_output():
    data = np.ones((10, 10), dtype=np.uint8)
    data[1:8, 1:8] = 2  # 7x7 core
    data[0, 0] = 0

    res = ls.clustering_fac(data, window_size=5, as_byte=False)
    assert res.dtype == np.float32
    assert np.isnan(res[0, 0])
    assert np.isnan(res[0, 1])
    assert np.isclose(res[4, 4], 1.0)


def test_contagion_xarray_metadata():
    da = xr.DataArray(
        np.array([[1, 2], [2, 2]], dtype=np.uint8),
        dims=("y", "x"),
        attrs={"crs": "EPSG:4326"},
    )
    res = ls.contagion_pff(da, window_size=3)
    assert isinstance(res, xr.DataArray)
    assert res.attrs["landschaft_product"] == "CONTAGION"
    assert res.attrs["metric"] == "Pff"
    assert res.attrs["window_size"] == 3


def test_invalid_parameters():
    data = np.array([[1, 2], [2, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="odd integer"):
        ls.contagion_pff(data, window_size=4)
    with pytest.raises(ValueError, match="odd integer"):
        ls.clustering_fac(data, window_size=2)
    with pytest.raises(InvalidInputError):
        ls.contagion_pff(np.array([[5, 6]]), window_size=3)


@pytest.mark.skipif(not HAS_SPATCON, reason="spatcon64.exe not found")
def test_accuracy_against_gtb_spatcon():
    # Test on a randomized 50x50 image with missing, background, and foreground
    rng = np.random.default_rng(12345)
    sample = rng.choice([0, 1, 2], size=(50, 50), p=[0.1, 0.4, 0.5]).astype(np.uint8)

    # 1. Pff (Contagion) window 7
    gtb_pff = run_gtb_spatcon(sample, 7, 77)
    py_pff = ls.contagion_pff(sample, window_size=7)
    np.testing.assert_array_equal(py_pff, gtb_pff)

    # 2. FAC (Clustering) window 7
    gtb_fac = run_gtb_spatcon(sample, 7, 76)
    py_fac = ls.clustering_fac(sample, window_size=7)
    np.testing.assert_array_equal(py_fac, gtb_fac)

    # 3. Larger window 13
    gtb_pff13 = run_gtb_spatcon(sample, 13, 77)
    py_pff13 = ls.contagion_pff(sample, window_size=13)
    np.testing.assert_array_equal(py_pff13, gtb_pff13)
