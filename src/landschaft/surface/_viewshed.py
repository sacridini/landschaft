# SPDX-License-Identifier: GPL-3.0-or-later
"""What observers see over an elevation map: Franklin and Ray's R2 lines of sight."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .. import _core
from .._threads import resolve
from ._grid import read_surface
from ._hilltop import _metres

__all__ = ["viewshed"]


def _observers(points, crs) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of the observers, in ``crs`` when the points and ``crs`` have one."""
    if hasattr(points, "geometry"):
        g = points.geometry
        if getattr(points, "crs", None) is not None and crs is not None:
            g = g.to_crs(crs)
        if set(g.geom_type.dropna()) - {"Point"}:
            raise ValueError("observers: points")
        return g.x.to_numpy(np.float64), g.y.to_numpy(np.float64)
    a = np.asarray(points, dtype=np.float64)
    if a.ndim == 1 and a.size == 2:
        a = a.reshape(1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise TypeError("observers: a GeoDataFrame of points, an (n, 2) array or an (x, y) pair")
    return a[:, 0], a[:, 1]


def viewshed(dem, observers, *, observer_height: float = 2.0, target_height: float = 0.0,
             radius: float | None = None, curvature: bool = False, extent=None, crs=None, band=None, nodata=None,
             n_threads: int | None = None) -> xr.DataArray:
    """How many observers see every cell of an elevation map (gdal_viewshed, GRASS's r.viewshed, for many observers).

    A cell is seen when the line of sight from an observer's eye (the ground
    plus ``observer_height``) to the cell (plus ``target_height``) passes
    above every cell between, traced along rays from the observer to every
    cell of the edge of its reach (Franklin and Ray's R2), and for the cells no
    ray passes through, along their own line of sight. With ``curvature``, the
    earth's curvature and refraction lower the far cells (GDAL's coefficient,
    0.85714). As Terraflux's ``raster.viewshed``; the observers in parallel
    (C++ with OpenMP).

    Parameters
    ----------
    dem
        The elevation (metres): a DataArray, a path or a 2-D array; NaN and
        ``nodata`` are missing (neither seen nor blocking).
    observers
        A GeoDataFrame of points (reprojected to the DEM's CRS), an ``(n, 2)``
        array or an ``(x, y)`` pair, in the DEM's coordinates. Observers off
        the map or on a missing cell see nothing.
    observer_height
        The eye above the ground (default 2, as GDAL's).
    target_height
        What is seen above the ground (default 0).
    radius
        How far an observer sees, in metres (default: the whole map).
    curvature
        Lower the far cells by the earth's curvature and refraction.
    extent, crs, band
        For an array: its extent ``(xmin, xmax, ymin, ymax)``, CRS and band.
    nodata
        Value marking missing cells.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        The observers seeing every cell (one observer: 1 seen, 0 not); NaN
        where the DEM has no value. Distances are in metres also for a DEM in
        longitude / latitude (the cells' size at the map's centre latitude).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.zeros((5, 9)); dem[:, 4] = 10                     # a wall down the middle
    >>> v = ls.surface.viewshed(dem, (1.5, 2.5), extent=(0, 9, 0, 5))
    >>> v.values[2].tolist()
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    """
    s = read_surface(dem, extent=extent, crs=crs, band=band, nodata=nodata)
    z = np.ascontiguousarray(s.values, dtype=np.float64)
    xmin, xmax, ymin, ymax = (float(v) for v in s.extent)
    ny, nx = z.shape
    ox, oy = _observers(observers, s.crs)
    cols = np.floor((ox - xmin) / ((xmax - xmin) / nx))
    rows = np.floor((ymax - oy) / ((ymax - ymin) / ny))
    ok = np.isfinite(cols) & np.isfinite(rows)
    rows = np.where(ok, rows, -1).astype(np.int64)
    cols = np.where(ok, cols, -1).astype(np.int64)
    rw, rh = _metres(s)
    reach = math.inf if radius is None else float(radius)
    if not reach > 0:
        raise ValueError("radius: above 0")
    out = _core.stats_viewshed(z, float(rw), float(rh), rows, cols, float(observer_height), float(target_height),
                               reach, bool(curvature), resolve(n_threads))
    return s.raster(out, "viewshed")
