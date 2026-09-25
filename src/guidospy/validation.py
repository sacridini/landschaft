# SPDX-License-Identifier: GPL-3.0-or-later
"""Input checks, mirroring GuidosToolbox's ``MSPA_Compliance`` procedure."""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["InvalidInputError", "validate_mspa_input"]

MISSING, BACKGROUND, FOREGROUND = 0, 1, 2


class InvalidInputError(ValueError):
    """The raster cannot be processed by the requested GTB tool."""


def _minmax(values) -> tuple[int, int]:
    """Min and max of a numpy or dask array (dask: one pass, computed together)."""
    if hasattr(values, "dask"):
        import dask

        lo, hi = dask.compute(values.min(), values.max())
        return int(lo), int(hi)
    arr = np.asarray(values)
    return int(arr.min()), int(arr.max())


def validate_mspa_input(data, *, spatial_only: bool = True) -> None:
    """Raise :class:`InvalidInputError` unless ``data`` is a valid MSPA input.

    MSPA expects a single-band ``uint8`` map coded as
    ``0`` = missing, ``1`` = background, ``2`` = foreground, with at least
    one foreground pixel and at least one pixel that is not foreground.

    ``data`` may be a numpy array, a dask array or an ``xarray.DataArray``.
    With ``spatial_only`` the array must be 2-D; otherwise extra leading
    dimensions (e.g. time) are allowed and every slice is checked jointly.
    """
    if isinstance(data, xr.DataArray):
        values = data.data  # keeps dask arrays lazy
    elif hasattr(data, "dask"):
        values = data
    else:
        values = np.asarray(data)
    ndim = values.ndim
    if spatial_only and ndim != 2:
        raise InvalidInputError(
            f"MSPA-compliant map required: expected a single-band 2-D map, got {ndim} dimensions "
            f"with shape {tuple(values.shape)}."
        )
    if ndim < 2:
        raise InvalidInputError(f"MSPA-compliant map required: expected at least 2 dimensions, got {ndim}.")
    if values.shape[-1] == 0 or values.shape[-2] == 0:
        raise InvalidInputError("MSPA-compliant map required: the map is empty.")
    if values.dtype != np.uint8:
        raise InvalidInputError(
            f"MSPA-compliant map required: map is of type {values.dtype}, not uint8 (BYTE). "
            "Convert it with guidospy.reclassify() or astype('uint8')."
        )
    lo, hi = _minmax(values)
    if hi > FOREGROUND:
        raise InvalidInputError(
            f"MSPA-compliant map required: map maximum is {hi}, larger than 2. Use 0 = missing, "
            "1 = background, 2 = foreground (see guidospy.reclassify())."
        )
    if hi < FOREGROUND:
        raise InvalidInputError("MSPA-compliant map required: map has no foreground (value 2).")
    if lo > BACKGROUND:
        raise InvalidInputError("MSPA-compliant map required: map has no background (value 1).")
