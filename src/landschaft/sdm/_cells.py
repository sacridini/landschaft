# SPDX-License-Identifier: GPL-3.0-or-later
"""One occurrence record per raster cell."""

from __future__ import annotations

import numpy as np

from . import _points

__all__ = ["one_per_cell"]


def one_per_cell(occ, raster, *, x: str | None = None, y: str | None = None, drop_missing: bool = False):
    """Keep one occurrence record per cell of the predictor grid.

    Several records in the same cell carry the same environmental values:
    kept, they weigh that cell more than the others (pseudo-replication) and
    put the same values on both sides of a validation split. The first record
    of each cell is kept, in the order of the table, as
    ``occ[!duplicated(terra::cellFromXY(r, xy)), ]`` in R. Records without
    coordinates or outside the grid are dropped.

    Parameters
    ----------
    occ
        Occurrence records: a GeoDataFrame of points, or a table with
        coordinate columns (``lon`` / ``lat``, ``decimalLongitude`` /
        ``decimalLatitude``, ``x`` / ``y``...).
    raster
        The predictors (or any raster on their grid): a DataArray, one band or
        a stack. The records are reprojected to its CRS to find their cells.
    x, y
        Names of the coordinate columns of a table (default: found by name).
    drop_missing
        Also drop the records whose cell has no value (NoData) in some band.

    Returns
    -------
    GeoDataFrame or DataFrame
        The records kept, with their original index.

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> len(occ), len(ls.sdm.one_per_cell(occ, env))
    (116, 94)
    """
    xs, ys, crs = _points.coordinates(occ, x=x, y=y)
    xs, ys = _points.to_crs(xs, ys, crs, raster.rio.crs)
    cells = _points.grid_cells(xs, ys, raster)
    keep = cells >= 0
    if drop_missing and keep.any():
        values = np.asarray(raster.values)
        values = values.reshape(-1, values.shape[-2] * values.shape[-1]) if values.ndim == 3 else values.reshape(1, -1)
        at = values[:, cells[keep]]
        nodata = raster.rio.nodata
        missing = np.isnan(at) if np.issubdtype(at.dtype, np.floating) else np.zeros(at.shape, bool)
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            missing |= at == nodata
        keep[np.flatnonzero(keep)[missing.any(axis=0)]] = False
    first = np.zeros(len(cells), dtype=bool)
    idx = np.flatnonzero(keep)
    _, pos = np.unique(cells[idx], return_index=True)
    first[idx[pos]] = True
    return _points.subset(occ, first)
