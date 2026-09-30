# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading a surface (path, DataArray or array) and writing rasters on its grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from affine import Affine

from ..cost._transition import read_grid_values


@dataclass(frozen=True)
class Surface:
    """The values (float64, NaN = NA) of a raster and its grid."""

    values: np.ndarray
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    crs: object = None
    template: xr.DataArray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def res(self) -> tuple[float, float]:
        ny, nx = self.shape
        xmin, xmax, ymin, ymax = self.extent
        return (xmax - xmin) / nx, (ymax - ymin) / ny

    def raster(self, values: np.ndarray, name: str, attrs: dict | None = None) -> xr.DataArray:
        """``values`` on this grid: the input's coordinates, or cell centres of the extent."""
        if self.template is not None:
            t = self.template
            da = xr.DataArray(values, coords={k: t.coords[k] for k in t.dims}, dims=t.dims, name=name,
                              attrs=dict(attrs or {}))
            try:
                da = da.rio.write_transform(t.rio.transform())
                if t.rio.crs is not None:
                    da = da.rio.write_crs(t.rio.crs)
            except Exception:
                pass
            return da
        ny, nx = self.shape
        xmin, xmax, ymin, ymax = self.extent
        xr_, yr_ = (xmax - xmin) / nx, (ymax - ymin) / ny
        x = xmin + (np.arange(nx) + 0.5) * xr_
        y = ymax - (np.arange(ny) + 0.5) * yr_
        da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name=name, attrs=dict(attrs or {}))
        da = da.rio.write_transform(Affine(xr_, 0.0, xmin, 0.0, -yr_, ymax))
        if self.crs is not None:
            try:
                da = da.rio.write_crs(self.crs)
            except Exception:
                pass
        return da


def read_surface(x, *, extent=None, crs=None, band=None, nodata=None) -> Surface:
    """A surface from a path, ``xarray.DataArray`` or 2-D array (NaN or ``nodata`` = NA).

    Arrays without ``extent`` get unit cells with the origin at 0 and the first row
    on top (as ``terra::rast(matrix)``).
    """
    if isinstance(x, Surface):
        return x
    values, grid, template = read_grid_values(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return Surface(np.ascontiguousarray(values), grid.extent, grid.crs, template)
