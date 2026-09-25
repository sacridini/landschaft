# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from rasterio.errors import NotGeoreferencedWarning

from gtb_reference import find_executable

DATA = Path(__file__).parent / "data"


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
        pytest.skip("set GUIDOSPY_MSPA_EXE to the original GTB mspa executable to run live comparisons")
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
