# SPDX-License-Identifier: GPL-3.0-or-later
"""R's arithmetic as the landscape metrics reproduce it, bit for bit.

``tests/data/reference/lsm_rmath`` holds R's ``sum``, ``mean``, ``sd``,
``median``, ``log`` and ``sqrt`` of random vectors (``make_reference_lsm.py
rmath``). R accumulates in 80-bit ``long double`` (emulated here where the
compiler lacks it) and takes ``log`` and the ``sqrt`` of a vector from the x87
unit; the ``sqrt`` inside ``sd`` is the correctly rounded one of R's
byte-code. ``lm.csv.gz`` holds the slopes of ``lm(y ~ x)`` that pafrac takes
(LINPACK's QR with R's reference BLAS; NA when R drops ``x`` as collinear).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import DATA
from landschaft import _core

RMATH = DATA / "reference" / "lsm_rmath" / "rmath.csv.gz"
LM = DATA / "reference" / "lsm_rmath" / "lm.csv.gz"
COLUMNS = ["sum", "mean", "sd", "median", "log", "sqrt"]


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(RMATH, dtype=str)


@pytest.mark.reference
def test_same_bits_as_r(reference):
    ours = np.array([_core.lsm_r_statistics([float(v) for v in x.split(",")]) for x in reference["x"]])
    ref = reference[COLUMNS].to_numpy(dtype=np.float64)  # float() of the 17 digits: exact
    for j, col in enumerate(COLUMNS):
        same = ours[:, j] == ref[:, j]
        assert same.all(), f"{col}: {(~same).sum()} values differ, e.g. row {np.flatnonzero(~same)[:5].tolist()}"


def test_extended_accumulation_differs_from_double():
    # the reference is only meaningful if 80-bit sums matter for these vectors
    x = [1.0, 2.0**-53, 2.0**-53]  # in double: 1 + 2^-53 rounds back to 1, twice
    double_sum = 0.0
    for v in x:
        double_sum += v
    assert double_sum == 1.0
    assert _core.lsm_r_statistics(x)[0] == 1.0 + 2.0**-52


def test_edge_cases():
    nan = float("nan")
    s = _core.lsm_r_statistics([2.0])
    assert s[:2] == [2.0, 2.0] and np.isnan(s[2]) and s[3] == 2.0
    assert np.isnan(_core.lsm_r_statistics([1.0, nan])[1])
    assert _core.lsm_r_statistics([0.0, 0.0])[4] == -np.inf
    assert _core.lsm_r_statistics([-0.0, 1.0])[0] == 1.0


@pytest.mark.reference
def test_lm_slope_same_bits_as_r():
    ref = pd.read_csv(LM, dtype=str, keep_default_na=False)
    ours = np.array([_core.lsm_r_lm_slope([float(v) for v in x.split(",")], [float(v) for v in y.split(",")])
                     for x, y in zip(ref["x"], ref["y"])])
    expected = np.array([np.nan if s == "NA" else float(s) for s in ref["slope"]])
    assert np.isnan(expected).sum() > 100  # the collinear cases are there
    same = (ours == expected) | (np.isnan(ours) & np.isnan(expected))
    assert same.all(), f"{(~same).sum()} slopes differ, e.g. row {np.flatnonzero(~same)[:5].tolist()}"


def test_lm_slope_edge_cases():
    assert _core.lsm_r_lm_slope([1.0, 2.0], [3.0, 5.0]) == 2.0
    assert np.isnan(_core.lsm_r_lm_slope([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
    assert np.isnan(_core.lsm_r_lm_slope([1.0], [1.0]))
    with pytest.raises(ValueError):
        _core.lsm_r_lm_slope([1.0, 2.0], [1.0])
