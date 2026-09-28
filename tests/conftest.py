# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from rasterio.errors import NotGeoreferencedWarning

from gtb_reference import find_executable
from landschaft import _core

DATA = Path(__file__).parent / "data"

# The references of R (landscapemetrics, NLMR) and Perl (Biodiverse) come from x86-64,
# where log, sqrt, exp, x^y, sin, cos and atan2 run on the x87 unit; x86 builds do the
# same (lsm/rmath.hpp) and must give their bits. Other builds (ARM64) take these
# functions from the system library, which can differ in the last bits: there the
# comparisons allow a few units in the last place.
BITWISE = bool(_core.x87_math)


def same_bits(a, b, *, scale=None, ulps: float = 4) -> np.ndarray:
    """Element by element: ``a`` has the bits of ``b`` (NaN matching NaN).

    Without the x87 functions (see ``BITWISE``): within ``ulps`` units in the last
    place of ``b``, or of ``scale`` for results that add up many terms (the
    magnitude of the result, e.g. the largest value of a transform).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if BITWISE:
        return same
    size = np.abs(b) if scale is None else np.maximum(np.abs(b), scale)
    with np.errstate(invalid="ignore"):
        return same | (np.abs(a - b) <= ulps * np.spacing(size))


def assert_same_bits(a, b, label: str = "", *, scale=None, ulps: float = 4) -> None:
    """Assert :func:`same_bits` for every element, naming the first differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{label}: shapes {a.shape} and {b.shape}"
    ok = same_bits(a, b, scale=scale, ulps=ulps)
    if not ok.all():
        bad = np.argwhere(~ok)[:5].tolist()
        raise AssertionError(f"{label}: {int((~ok).sum())} of {ok.size} values differ "
                             f"({'bit for bit' if BITWISE else f'beyond {ulps} ulp'}), first at {bad}: "
                             f"{a[~ok][:3].tolist()} != {b[~ok][:3].tolist()}")


@pytest.fixture(autouse=True)
def _quiet_georef_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        yield


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    return DATA / "reference"


@pytest.fixture(scope="session")
def mspa_exe() -> Path:
    exe = find_executable()
    if exe is None:
        pytest.skip("set LANDSCHAFT_MSPA_EXE to the original GTB mspa executable to run live comparisons")
    return exe


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def mspa_input(shape, foreground=(), missing=()) -> np.ndarray:
    """Background image with the given foreground/missing slices set."""
    a = np.ones(shape, dtype=np.uint8)
    for sl in foreground:
        a[sl] = 2
    for sl in missing:
        a[sl] = 0
    return a
