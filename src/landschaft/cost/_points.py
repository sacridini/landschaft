# SPDX-License-Identifier: GPL-3.0-or-later
"""Locations given to the grid-distance functions: coordinates, points or masks."""

from __future__ import annotations

import warnings

import numpy as np


def as_xy(points) -> np.ndarray:
    """(n, 2) coordinates of points: an (x, y) pair, an (n, 2) array, shapely points or a GeoDataFrame."""
    geoms = None
    if hasattr(points, "geometry") and hasattr(points, "crs"):  # GeoDataFrame / GeoSeries
        geoms = np.asarray(points.geometry.values if hasattr(points.geometry, "values") else points.geometry)
    elif hasattr(points, "geom_type"):  # a shapely geometry
        geoms = np.array([points], dtype=object)
    elif isinstance(points, (list, tuple, np.ndarray)) and len(points) and hasattr(np.asarray(points, dtype=object)
                                                                                    .ravel()[0], "geom_type"):
        geoms = np.asarray(points, dtype=object).ravel()
    if geoms is not None:
        import shapely

        xy = shapely.get_coordinates(geoms)
        return np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim == 1:
        if xy.size != 2:
            raise ValueError("coordinates given as a vector must have two values (x, y)")
        xy = xy.reshape(1, 2)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must be an (n, 2) array of x, y")
    return xy


def cells_of_xy(xy: np.ndarray, grid) -> np.ndarray:
    """raster's cellFromXY (0-based, -1 outside the grid)."""
    ny, nx = grid.shape
    xmin, xmax, ymin, ymax = grid.extent
    yres_inv = ny / (ymax - ymin)
    xres_inv = nx / (xmax - xmin)
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        row = np.floor((ymax - y) * yres_inv)
        row = np.where(y == ymin, ny - 1, row)
        col = np.floor((x - xmin) * xres_inv)
        col = np.where(x == xmax, nx - 1, col)
        outside = (row < 0) | (row >= ny) | (col < 0) | (col >= nx) | np.isnan(row) | np.isnan(col)
    return np.where(outside, -1, row * nx + col).astype(np.int64)


def as_cells(where, grid, what: str = "locations") -> np.ndarray:
    """Cells of points (coordinates, shapely, GeoDataFrame) or of the true/non-zero cells of a mask raster.

    Cells outside the grid are -1 (with a warning).
    """
    mask = _as_mask(where, grid)
    if mask is not None:
        return np.flatnonzero(mask.ravel()).astype(np.int64)
    cells = cells_of_xy(as_xy(where), grid)
    if (cells < 0).any():
        warnings.warn(f"{int((cells < 0).sum())} {what} outside the grid are left out", RuntimeWarning, stacklevel=3)
    return cells


def _as_mask(where, grid):
    """The mask of a raster of locations, or None for points. An array is a mask when it is
    boolean or has the grid's shape (an (n, 2) array is coordinates on a two-column grid)."""
    import xarray as xr

    if isinstance(where, xr.DataArray):
        values = np.asarray(where.squeeze().values)
    elif isinstance(where, np.ndarray) and where.ndim == 2 and where.shape == grid.shape and (
            where.dtype == bool or grid.shape[1] != 2):
        values = where
    else:
        return None
    if values.shape != grid.shape:
        raise ValueError("the mask must have the transition's shape")
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0) != 0
