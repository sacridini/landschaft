# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading categorical maps: one map, a list of maps or a stack with a band per date."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import xarray as xr


@dataclass
class Map:
    """A map's values (any numeric numpy type, C-contiguous) and its missing value."""

    values: np.ndarray
    nodata: float | None = None
    template: xr.DataArray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def missing(self) -> np.ndarray:
        v = self.values
        m = np.isnan(v) if np.issubdtype(v.dtype, np.floating) else np.zeros(v.shape, dtype=bool)
        if self.nodata is not None:
            m |= v == self.nodata
        return m

    def as_float(self) -> np.ndarray:
        """float64 values, NaN where missing."""
        out = self.values.astype(np.float64)
        out[self.missing()] = np.nan
        return out


def _nodata_of(da: xr.DataArray, nodata):
    if nodata is not None:
        return float(nodata)
    nd = None
    try:
        nd = da.rio.nodata
    except Exception:
        nd = None
    if nd is None:
        nd = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
    if nd is None or (isinstance(nd, float) and np.isnan(nd)):
        return None
    return float(nd)


def _one(x, nodata) -> Map:
    if isinstance(x, Map):
        return x
    if isinstance(x, xr.DataArray):
        if x.ndim != 2:
            raise ValueError(f"a map must be 2-D, got dimensions {x.dims}")
        return Map(np.ascontiguousarray(x.values), _nodata_of(x, nodata), x)
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"a map must be 2-D, got shape {a.shape}")
    if a.dtype == bool:
        a = a.astype(np.uint8)
    return Map(np.ascontiguousarray(a), None if nodata is None else float(nodata))


def read_map(x, nodata=None) -> Map:
    """One categorical map from a 2-D DataArray or array (NaN and ``nodata`` are missing)."""
    if isinstance(x, xr.DataArray) and x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return _one(x, nodata)


def read_series(maps, times=None, nodata=None) -> tuple[list[Map], list]:
    """Maps of a series and their times.

    ``maps``: a 3-D DataArray (a band per date; the band names give the
    times when they are numbers), or a list of 2-D maps. ``times`` default to
    the band names, else 0, 1, 2...
    """
    names = None
    if isinstance(maps, xr.DataArray) and maps.ndim == 3:
        dim = maps.dims[0]
        names = [str(v) for v in maps[dim].values] if dim in maps.coords else None
        out = [_one(maps[k], nodata) for k in range(maps.shape[0])]
    elif isinstance(maps, (list, tuple)):
        out = [read_map(m, nodata) for m in maps]
    else:
        raise TypeError("maps must be a 3-D DataArray (a band per date) or a list of 2-D maps")
    if len(out) < 2:
        raise ValueError("at least two maps are needed")
    shape = out[0].shape
    if any(m.shape != shape for m in out):
        raise ValueError("the maps differ in shape")
    if times is None:
        times = list(range(len(out)))
        if names is not None:
            try:
                times = [int(n) if float(n).is_integer() else float(n) for n in names]
            except ValueError:
                pass
    times = list(times)
    if len(times) != len(out):
        raise ValueError(f"{len(times)} times for {len(out)} maps")
    if any(b <= a for a, b in pairwise(times)):
        raise ValueError("times must increase")
    return out, times


def cell_area_of(m: Map, cell_area) -> float:
    """The area of a cell: given, else the product of the template's resolutions (1 without one)."""
    if cell_area is not None:
        return float(cell_area)
    t = m.template
    if t is not None:
        try:
            rx, ry = t.rio.resolution()
            return abs(float(rx) * float(ry))
        except Exception:
            pass
    return 1.0
