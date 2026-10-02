# SPDX-License-Identifier: GPL-3.0-or-later
"""Occurrence records as coordinates: GeoDataFrames or tables of longitude / latitude."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._vector import _LONLAT, _X_NAMES, _Y_NAMES, _coordinate_column


def coordinates(occ, *, x: str | None = None, y: str | None = None, crs=None):
    """The x and y of every record (NaN where missing) and their CRS.

    ``occ`` is a GeoDataFrame of points (empty or missing geometries give
    NaN) or a table whose coordinate columns are found by name (``lon`` /
    ``lat``, ``decimalLongitude`` / ``decimalLatitude``, ``x`` / ``y``...;
    longitude / latitude columns are taken as EPSG:4326 unless ``crs`` says
    otherwise).
    """
    import geopandas as gpd

    if isinstance(occ, gpd.GeoDataFrame) and x is None and y is None:
        geom = occ.geometry
        if len(geom) and not geom.isna().all():
            types = set(geom.dropna()[~geom.dropna().is_empty].geom_type)
            if types - {"Point"}:
                raise ValueError(f"occurrence records must be points, not {', '.join(sorted(types - {'Point'}))}")
        xs = np.full(len(geom), np.nan)
        ys = np.full(len(geom), np.nan)
        ok = (~geom.isna() & ~geom.is_empty).to_numpy()
        if ok.any():
            xs[ok] = geom[ok].x.to_numpy(dtype=float)
            ys[ok] = geom[ok].y.to_numpy(dtype=float)
        return xs, ys, occ.crs if crs is None else crs
    if not isinstance(occ, pd.DataFrame):
        raise TypeError(f"occurrences must be a GeoDataFrame or a DataFrame, not {type(occ).__name__}")
    xc = _coordinate_column(occ.columns, x, _X_NAMES, "x")
    yc = _coordinate_column(occ.columns, y, _Y_NAMES, "y")
    if crs is None and str(xc).lower() in _LONLAT and str(yc).lower() in _LONLAT:
        crs = "EPSG:4326"
    xs = pd.to_numeric(occ[xc], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(occ[yc], errors="coerce").to_numpy(dtype=float)
    return xs, ys, crs


def lonlat(occ, *, x: str | None = None, y: str | None = None, crs=None):
    """Longitude and latitude (EPSG:4326) of every record, NaN where missing."""
    xs, ys, src = coordinates(occ, x=x, y=y, crs=crs)
    return to_crs(xs, ys, src, "EPSG:4326")


def to_crs(xs: np.ndarray, ys: np.ndarray, src, dst):
    """Coordinates in another CRS (unchanged when the CRS are the same or unknown)."""
    if src is None or dst is None:
        return xs, ys
    from pyproj import CRS, Transformer

    s, d = CRS.from_user_input(src), CRS.from_user_input(dst)
    if s == d:
        return xs, ys
    ok = np.isfinite(xs) & np.isfinite(ys)
    ox, oy = np.full_like(xs, np.nan), np.full_like(ys, np.nan)
    if ok.any():
        ox[ok], oy[ok] = Transformer.from_crs(s, d, always_xy=True).transform(xs[ok], ys[ok])
    return ox, oy


def subset(occ, keep: np.ndarray):
    """The records where ``keep`` is true, the type and index kept."""
    return occ.loc[np.asarray(keep, dtype=bool)]


def grid_cells(xs: np.ndarray, ys: np.ndarray, raster) -> np.ndarray:
    """Cell number (row-major, 0-based) of each point in a raster's grid; -1 outside.

    The rule of ``terra::cellFromXY``: a point on the edge between two cells
    falls in the cell to its right / below; on the right and bottom edges of
    the raster, in the last column / row.
    """
    from ..io import _transform

    tr = _transform(raster)
    if tr.b != 0 or tr.d != 0:
        raise ValueError("rotated rasters are not supported")
    ny, nx = raster.shape[-2:]
    # terra's extent (GDAL's: origin + n * cell size) and its resolution (extent / n)
    xmin, ymax = tr.c, tr.f
    xmax, ymin = xmin + nx * tr.a, ymax + ny * tr.e
    xres, yres = (xmax - xmin) / nx, (ymax - ymin) / ny
    with np.errstate(invalid="ignore"):
        col = np.where((xs >= xmin) & (xs < xmax), np.trunc((xs - xmin) / xres), np.where(xs == xmax, nx - 1, -1))
        row = np.where((ys > ymin) & (ys <= ymax), np.trunc((ymax - ys) / yres), np.where(ys == ymin, ny - 1, -1))
    ok = (col >= 0) & (row >= 0) & (col < nx) & (row < ny)
    return np.where(ok, row.astype(np.int64) * nx + col.astype(np.int64), -1)
