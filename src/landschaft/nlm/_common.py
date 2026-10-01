# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers shared by the neutral landscape models."""

from __future__ import annotations

import numpy as np
import xarray as xr
from rasterio.transform import Affine

from .. import _threads

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def resolve_seed(seed) -> int:
    """The seed of R's ``set.seed()`` (a 32-bit integer); ``None`` draws one."""
    if seed is None:
        return int(np.random.SeedSequence().entropy % _INT32_MAX) + 1
    if isinstance(seed, (bool, np.bool_)) or not float(seed).is_integer():
        raise ValueError(f"seed must be an integer, got {seed!r}")
    s = int(seed)
    if not _INT32_MIN < s <= _INT32_MAX:
        raise ValueError(f"seed must be a 32-bit integer, got {seed!r}")
    return s


def check_size(ncol, nrow) -> tuple[int, int]:
    out = []
    for name, v in (("ncol", ncol), ("nrow", nrow)):
        if isinstance(v, (bool, np.bool_)) or not float(v).is_integer() or int(v) < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
        out.append(int(v))
    return out[0], out[1]


def check_resolution(resolution) -> float:
    r = float(resolution)
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"resolution must be positive, got {resolution!r}")
    return r


def grid(values: np.ndarray, resolution: float, name: str, attrs: dict | None = None, nodata=None) -> xr.DataArray:
    """A DataArray on NLMR's grid: extent 0..ncol*res x 0..nrow*res, first row
    on top, cell-centre coordinates, no CRS."""
    ny, nx = values.shape
    transform = Affine(resolution, 0.0, 0.0, 0.0, -resolution, ny * resolution)
    x = (np.arange(nx) + 0.5) * resolution
    y = ny * resolution - (np.arange(ny) + 0.5) * resolution
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name=name, attrs=dict(attrs or {}))
    from ..io import _KEEP, _with_georef

    return _with_georef(da, transform=transform, nodata=_KEEP if nodata is None else nodata)


def as_values(x) -> np.ndarray:
    """The values of a DataArray / array as float64 (masked cells as NaN)."""
    if isinstance(x, xr.DataArray):
        values = np.asarray(x.values, dtype=np.float64)
        nodata = x.rio.nodata if hasattr(x, "rio") else None
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            values = np.where(x.values == nodata, np.nan, values)
        return values
    if np.ma.isMaskedArray(x):
        return np.ma.filled(np.ma.asarray(x, dtype=np.float64), np.nan)
    return np.asarray(x, dtype=np.float64)


def threads(n_threads) -> int:
    return _threads.resolve(n_threads)
