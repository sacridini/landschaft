# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from landschaft import InvalidInputError, validate_mspa_input


def valid() -> np.ndarray:
    a = np.ones((10, 10), np.uint8)
    a[3:7, 3:7] = 2
    return a


def test_valid_input_passes():
    validate_mspa_input(valid())
    validate_mspa_input(xr.DataArray(valid(), dims=("y", "x")))


@pytest.mark.parametrize(
    "data, message",
    [
        (valid().astype(np.float32), "not uint8"),
        (valid().astype(np.int16), "not uint8"),
        (np.where(valid() == 2, 3, 1).astype(np.uint8), "larger than 2"),
        (np.ones((10, 10), np.uint8), "no foreground"),
        (np.full((10, 10), 2, np.uint8), "no background"),
        (np.stack([valid(), valid()]), "2-D"),
        (np.ones((0, 5), np.uint8), "empty"),
    ],
)
def test_invalid_inputs_are_rejected(data, message):
    with pytest.raises(InvalidInputError, match=message):
        validate_mspa_input(data)


def test_missing_and_foreground_only_is_accepted_like_gtb():
    # GTB only requires min <= 1: missing (0) without background passes.
    a = np.zeros((10, 10), np.uint8)
    a[2:8, 2:8] = 2
    validate_mspa_input(a)


def test_mspa_validates_by_default():
    with pytest.raises(InvalidInputError):
        ls.mspa(np.ones((5, 5), np.uint8))


def test_core_rejects_bad_values_even_without_validation():
    a = valid()
    a[0, 0] = 7
    with pytest.raises(InvalidInputError):
        ls.mspa(a, validate=False)


def test_dask_input_is_validated_lazily():
    da = pytest.importorskip("dask.array")
    validate_mspa_input(da.from_array(valid(), chunks=5))
    with pytest.raises(InvalidInputError):
        validate_mspa_input(da.from_array(np.ones((10, 10), np.uint8), chunks=5))
