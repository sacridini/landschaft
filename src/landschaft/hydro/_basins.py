# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream links, outlets and watersheds along D8 directions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import threaded
from ..surface._grid import read_surface
from ._flow import _cells, _d8, _read

__all__ = ["snap_outlets", "stream_links", "watersheds"]

# neighbour offsets of the internal D8 codes (0 north-east, clockwise ... 7 north)
_DROW = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
_DCOL = np.array([1, 1, 1, 0, -1, -1, -1, 0])


def _steps(dx: float, dy: float) -> np.ndarray:
    diag = math.hypot(dx, dy)
    return np.array([diag, dx, diag, dy, diag, dx, diag, dy])


def _down(d: np.ndarray) -> np.ndarray:
    """The flat index of the cell each cell drains to (-1: none, off the map or missing)."""
    ny, nx = d.shape
    r, c = np.indices(d.shape)
    k = np.clip(d, 0, 7)
    rn, cn = r + _DROW[k], c + _DCOL[k]
    ok = (d >= 0) & (rn >= 0) & (cn >= 0) & (rn < ny) & (cn < nx)
    j = np.where(ok, rn * nx + cn, -1).ravel()
    flat = d.ravel()
    j[(j >= 0) & (flat[np.maximum(j, 0)] == -2)] = -1
    return j


def _points(points, s) -> np.ndarray:
    """(n, 2) map coordinates of points: an (n, 2) array, (x, y) pairs or a GeoDataFrame."""
    if hasattr(points, "geometry"):
        g = points.geometry
        if getattr(points, "crs", None) is not None and s.crs is not None:
            g = g.to_crs(s.crs)
        return np.column_stack([g.x.to_numpy(float), g.y.to_numpy(float)])
    a = np.asarray(points, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("outlets are (x, y) pairs, an (n, 2) array or a GeoDataFrame of points")
    return a


def snap_outlets(points, streams, *, distance: float | None = None, accumulation=None) -> np.ndarray:
    """The stream cell each outlet point stands for.

    Each point moves to the stream cell nearest to it (cell centres, in map units),
    within ``distance``; of equally near cells, the one with the largest
    ``accumulation`` (then the first in raster order).

    Parameters
    ----------
    points
        (x, y) pairs, an (n, 2) array or a GeoDataFrame of points (map coordinates
        of the stream raster).
    streams
        1 on the streams (:func:`streams`).
    distance
        The largest move, in map units (default: 300 m, QSWAT+'s snap threshold,
        or 3 cells for unprojected grids).
    accumulation
        A flow accumulation to break ties (optional).

    Returns
    -------
    numpy.ndarray
        (n, 2) row and column of the cell of each point.



    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)

    a point half a cell west of the stream, in the raster's x, y (cell centres 0.5, 1.5...):

    >>> ls.hydro.snap_outlets([(0.9, 1.5)], st, distance=1.0)
    array([[2, 1]])
    """
    s = read_surface(streams)
    pts = _points(points, s)
    ny, nx = s.shape
    xmin, xmax, ymin, ymax = s.extent
    rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny
    on = np.nan_to_num(s.values) == 1
    rows, cols = np.nonzero(on)
    if rows.size == 0:
        raise ValueError("there are no stream cells to snap the outlets to")
    xs = xmin + (cols + 0.5) * rx
    ys = ymax - (rows + 0.5) * ry
    acc = None if accumulation is None else read_surface(accumulation).values[rows, cols]
    if distance is None:
        distance = 300.0 if s.crs is not None and not _cells_lonlat(s) else 3 * max(rx, ry)
    out = np.empty((len(pts), 2), dtype=np.int64)
    for i, (x, y) in enumerate(pts):
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        best = d2.min()
        if best > distance**2 * (1 + 1e-12):
            raise ValueError(f"outlet {i} ({x}, {y}) is farther than {distance} from every stream cell")
        cand = np.nonzero(d2 == best)[0]
        if acc is not None and cand.size > 1:
            a = acc[cand]
            cand = cand[a == np.nanmax(a)]
        out[i] = rows[cand[0]], cols[cand[0]]
    return out


def _cells_lonlat(s) -> bool:
    from ..cost._transition import _crs_lonlat

    return bool(_crs_lonlat(s.crs))


@threaded
def stream_links(dem, streams, *, outlets=None, snap: float | None = None, flats: bool = True, extent=None,
                 crs=None, band=None, nodata=None) -> tuple[xr.DataArray, pd.DataFrame]:
    """The links of a stream network: the stretches between sources, junctions and outlets.

    A link starts at a source (a stream cell nothing on the streams drains into), just
    below a junction (two or more streams meeting) or just below an outlet, and ends
    where the next one starts or where the stream leaves the map. Links are numbered
    1, 2, ... in the raster order of their first cells.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`): the D8 directions
        (:func:`flow_direction`) are taken from it.
    streams
        1 on the stream cells (:func:`streams`), on the DEM's grid.
    outlets
        Points where a link must end (gauges, dams, the watershed outlet): (x, y)
        pairs, an (n, 2) array or a GeoDataFrame; each snaps to the nearest stream
        cell (:func:`snap_outlets`, within ``snap``).
    snap
        The largest move of an outlet, in map units (default: 300 m).
    flats
        Drain flat areas (as :func:`flow_direction`).
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    links : xarray.DataArray
        The link of each stream cell (1..n), 0 elsewhere on the DEM, NaN where it is
        missing.
    table : pandas.DataFrame
        By link (index ``link``): ``downstream`` (the link its last cell drains into,
        0 for none), ``outlet`` (whether it ends at one of ``outlets``), the first
        (``start_row``, ``start_col``) and last cell (``end_row``, ``end_col``),
        ``cells``, ``length`` (along the D8 path, from the first cell's centre to the
        centre of the cell below the last one; a last cell that drains nowhere counts
        one cell width; metres for projected and unprojected grids), the elevations
        ``z_start`` and ``z_end`` (of the first cell and of the cell below the last
        one, or of the last one when nothing is below), and
        ``order`` (Strahler).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    >>> links, table = ls.hydro.stream_links(dem, st)
    >>> links.values
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 1., 0.]])
    """
    s = _read(dem, extent, crs, band, nodata)
    d = _d8(s, flats)
    st = read_surface(streams).values
    if st.shape != s.shape:
        raise ValueError("the streams differ in shape from the DEM")
    on = ((np.nan_to_num(st) == 1) & (d != -2)).astype(np.uint8)
    breaks = None
    if outlets is not None:
        rc = snap_outlets(outlets, s.raster(on.astype(np.float64), "streams"), distance=snap)
        breaks = np.zeros(s.shape, dtype=np.uint8)
        breaks[rc[:, 0], rc[:, 1]] = 1
    link, start, end, down = _core.hydro_stream_links(d, on, breaks)
    table = _link_table(s, d, link, start, end, down, breaks)
    out = np.where(d == -2, np.nan, link.astype(np.float64))
    return s.raster(out, "links"), table


def _link_table(s, d, link, start, end, down, breaks) -> pd.DataFrame:
    ny, nx = s.shape
    n = len(start)
    dx, dy = _cells(s)
    nxt = _down(d)
    flat_link = link.ravel()
    # the step out of every stream cell (to the cell below it), summed by link
    cells = np.nonzero(flat_link > 0)[0]
    # every cell's step along its direction; a last cell without one (an edge or a pit)
    # counts a straight cell (so that no link has no length)
    dc = d.ravel()[cells]
    step = np.where(dc >= 0, _steps(dx, dy)[np.clip(dc, 0, 7)], min(dx, dy))
    length = np.bincount(flat_link[cells], weights=step, minlength=n + 1)[1:]
    count = np.bincount(flat_link[cells], minlength=n + 1)[1:]
    z = s.values.ravel()
    below = nxt[end]
    z_end = np.where(below >= 0, z[np.maximum(below, 0)], z[end])
    # Strahler order, from the sources down
    order = np.zeros(n + 1, dtype=np.int64)
    ups: dict[int, list[int]] = {}
    for k in range(n):
        if down[k] > 0:
            ups.setdefault(int(down[k]), []).append(k + 1)
    pending = [k + 1 for k in range(n) if (k + 1) not in ups]
    remaining = {k: len(v) for k, v in ups.items()}
    while pending:
        k = pending.pop()
        u = [order[i] for i in ups.get(k, [])]
        if not u:
            order[k] = 1
        else:
            top = max(u)
            order[k] = top + 1 if u.count(top) > 1 else top
        dn = int(down[k - 1])
        if dn > 0:
            remaining[dn] -= 1
            if remaining[dn] == 0:
                pending.append(dn)
    table = pd.DataFrame({
        "downstream": down.astype(np.int64),
        "outlet": np.zeros(n, bool) if breaks is None else breaks.ravel()[end].astype(bool),
        "start_row": start // nx, "start_col": start % nx,
        "end_row": end // nx, "end_col": end % nx,
        "cells": count.astype(np.int64),
        "length": length,
        "z_start": z[start], "z_end": z_end,
        "order": order[1:],
    }, index=pd.RangeIndex(1, n + 1, name="link"))
    return table


@threaded
def watersheds(dem, seeds, *, flats: bool = True, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """The area draining to each seed: every cell labelled with the first seed below it.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    seeds
        Positive integer labels on the cells water drains to (stream links from
        :func:`stream_links`, outlet cells, lakes), 0 or NaN elsewhere; seed cells keep
        their own label.
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    xarray.DataArray
        The label of the seed each cell drains to, 0 where its path reaches none (it
        leaves the map or ends in a pit first), NaN where the DEM is missing.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[3, 2, 3], [3, 1, 3]], float)
    >>> ls.hydro.watersheds(dem, np.array([[0, 0, 0], [0, 7, 0]])).values
    array([[7., 7., 7.],
           [7., 7., 7.]])
    """
    s = _read(dem, extent, crs, band, nodata)
    d = _d8(s, flats)
    lab = np.nan_to_num(read_surface(seeds).values)
    if lab.shape != s.shape:
        raise ValueError("the seeds differ in shape from the DEM")
    if (lab < 0).any() or (lab != np.round(lab)).any():
        raise ValueError("seeds are non-negative integer labels")
    out = _core.hydro_watersheds(d, lab.astype(np.int32)).astype(np.float64)
    out[d == -2] = np.nan
    return s.raster(out, "watersheds")
