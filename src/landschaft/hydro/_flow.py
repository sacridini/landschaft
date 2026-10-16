# SPDX-License-Identifier: GPL-3.0-or-later
"""Depression filling, flow directions, flow accumulation, streams and HAND.

The algorithms follow WhiteboxTools 2.4 (J. Lindsay, MIT licence) cell for cell where it
has them: the same filled DEM as ``FillDepressions``, the same D8 directions as
``D8Pointer``, the same D-infinity azimuths as ``DInfPointer`` and the same
accumulations as ``D8FlowAccumulation``, ``DInfFlowAccumulation`` and
``FD8FlowAccumulation``. Flat areas drain as Barnes, Lehman & Mulla (2014) propose.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .. import _core
from .._threads import current, threaded
from ..cost._transition import _crs_lonlat
from ..surface._grid import Surface, read_surface

__all__ = ["D8_CODES", "breach", "fill", "flow_accumulation", "flow_direction", "hand", "streams"]

#: D8 codes of :func:`flow_direction` (ESRI's): the neighbour each cell drains to.
D8_CODES = {1: "E", 2: "SE", 4: "S", 8: "SW", 16: "W", 32: "NW", 64: "N", 128: "NE"}
_ESRI = np.array([128, 1, 2, 4, 8, 16, 32, 64], dtype=np.int16)   # by neighbour index 0 NE ... 7 N
_METHODS = ("d8", "dinf", "fd8", "quinn")


def _cells(s: Surface) -> tuple[float, float]:
    """Cell width and height in map units (metres at the centre latitude for longitude / latitude)."""
    dx, dy = (abs(float(v)) for v in s.res)
    if _crs_lonlat(s.crs):
        lat = math.radians((s.extent[2] + s.extent[3]) / 2)
        return dx * 111320.0 * math.cos(lat), dy * 110574.0
    return dx, dy


def _read(dem, extent, crs, band, nodata) -> Surface:
    s = read_surface(dem, extent=extent, crs=crs, band=band, nodata=nodata)
    if s.values.ndim != 2:
        raise ValueError("the DEM must have one band")
    return s


def _d8(s: Surface, flats: bool) -> np.ndarray:
    dx, dy = _cells(s)
    return _core.hydro_d8(s.values, dx, dy, bool(flats), current())


def _flat_dirs(s: Surface) -> np.ndarray:
    dx, dy = _cells(s)
    plain = _core.hydro_d8(s.values, dx, dy, False, current())
    return _core.hydro_flat_directions(s.values, plain, dx, dy)


@threaded
def fill(dem, *, drain_flats: bool = False, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Fill the depressions of a DEM, so that water can flow from every cell to the edge.

    Every depression (a cell or group of cells lower than all its surroundings) is
    raised to the elevation where it would spill over: the lowest DEM that keeps
    the original elevations everywhere else and drains to the map's edge or to a
    missing cell. The Priority-Flood algorithm of Barnes, Lehman & Mulla (2014),
    the same DEM as WhiteboxTools' ``FillDepressions``.

    Parameters
    ----------
    dem
        The elevations (DataArray, array or file); NaN or ``nodata`` mark missing cells.
    drain_flats
        Raise each filled cell by the smallest representable step above the cell it
        drains to (Priority-Flood+epsilon), so the filled areas slope towards their
        outlet. The steps are tiny: keep the result in float64. Without it, the
        filled areas are flat and :func:`flow_direction` drains them by the method
        of Barnes et al. (2014), which gives more natural flow paths.
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray
        The filled DEM, float64, on the DEM's grid.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[5, 5, 5, 5], [5, 1, 2, 5], [5, 2, 3, 4], [5, 5, 5, 5]], float)
    >>> ls.hydro.fill(dem).values
    array([[5., 5., 5., 5.],
           [5., 4., 4., 5.],
           [5., 4., 4., 4.],
           [5., 5., 5., 5.]])
    """
    s = _read(dem, extent, crs, band, nodata)
    return s.raster(_core.hydro_fill(s.values, bool(drain_flats)), "filled")


@threaded
def breach(dem, *, max_length: int = 100, max_cost: float | None = None, min_dist: bool = True,
           flat_increment: float | None = None, fill_rest: bool = True, extent=None, crs=None, band=None,
           nodata=None) -> xr.DataArray:
    """Remove the depressions of a DEM by cutting channels through their rims.

    Breaching often changes the terrain less than filling (:func:`fill`): a road
    embankment across a valley or a false rim of a few cells is cut, not the whole
    valley flooded behind it (Lindsay 2016). Each pit (a cell with no lower
    neighbour), from the lowest, is first raised to just below its lowest
    neighbour, then joined to the nearest cell it can drain to (lower than the
    channel, missing, or the map's edge) by the channel of least cost, cut to
    descend by ``flat_increment`` per cell. WhiteboxTools'
    ``BreachDepressionsLeastCost`` (after Lindsay & Dhun 2015), cell for cell,
    except that a channel that reaches the map's edge is cut (WhiteboxTools leaves
    it uncut).

    Parameters
    ----------
    dem
        The elevations; NaN or ``nodata`` mark missing cells.
    max_length
        The longest channel, in cells.
    max_cost
        The costliest channel to cut (``None``: any); pits that would need more are
        left (and filled with ``fill_rest``).
    min_dist
        Count the channel's length in its cost (the lowering times the distance),
        so shorter channels are preferred; otherwise only the lowering counts.
    flat_increment
        The drop per cell along a channel (default: from the elevations and the cell
        size, as WhiteboxTools chooses it). The steps are small: keep the result in
        float64.
    fill_rest
        Fill the depressions that no channel could solve (:func:`fill`).
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray
        The breached DEM, float64.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[9, 9, 9, 9, 9], [9, 5, 5, 5, 9], [9, 5, 1, 6, 4], [9, 5, 5, 5, 9], [9, 9, 9, 9, 9]], float)
    >>> b = ls.hydro.breach(dem, flat_increment=0.01)
    >>> float(b.values[2, 2])                       # the pit, raised below its lowest neighbour
    4.99
    >>> int((b.values != dem).sum())                # and one rim cell cut on the way to the 4 (east)
    2
    >>> bool(ls.hydro.flow_direction(b, flats=False).values[2, 2] > 0)
    True
    """
    s = _read(dem, extent, crs, band, nodata)
    dx, dy = _cells(s)
    cost = math.inf if max_cost is None else float(max_cost)
    inc = 0.0 if flat_increment is None else float(flat_increment)
    if flat_increment is not None and inc <= 0:
        raise ValueError("flat_increment must be positive")
    out = _core.hydro_breach(s.values, dx, dy, int(max_length), cost, bool(min_dist), inc, bool(fill_rest))
    return s.raster(out, "breached")


@threaded
def flow_direction(dem, *, method: str = "d8", flats: bool = True, extent=None, crs=None, band=None,
                   nodata=None) -> xr.DataArray:
    """The direction each cell drains to.

    Parameters
    ----------
    dem
        The elevations, best with its depressions filled (:func:`fill`).
    method : {"d8", "dinf"}
        ``"d8"`` (O'Callaghan & Mark 1984): the neighbour of steepest descent
        (drop over distance; of equal slopes the first from north-east clockwise),
        as an ESRI code (:data:`D8_CODES`: 1 east, 2 south-east, 4 south ... 128
        north-east), 0 where no neighbour is lower. ``"dinf"`` (Tarboton 1997):
        the azimuth of steepest descent over the eight triangular facets around
        the cell, in degrees clockwise from north, -1 where nothing is lower.
    flats
        Drain the flat areas towards lower terrain and away from higher terrain
        (Barnes, Lehman & Mulla 2014), so that every cell of a filled DEM has a
        direction; ``False`` leaves them without (as WhiteboxTools does).
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray
        The codes or azimuths (float64), NaN where the DEM is missing.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[3, 3, 3], [3, 2, 3], [3, 1, 3]], float)
    >>> ls.hydro.flow_direction(dem).values[1, 1]   # to the south
    np.float64(4.0)
    """
    s = _read(dem, extent, crs, band, nodata)
    if method == "d8":
        d = _d8(s, flats)
        out = np.where(d >= 0, _ESRI[np.clip(d, 0, 7)], 0).astype(np.float64)
        out[d == -2] = np.nan
        return s.raster(out, "d8", {"codes": "ESRI: 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE; 0 no flow"})
    if method == "dinf":
        dx, dy = _cells(s)
        return s.raster(_core.hydro_dinf(s.values, dx, dy, _flat_dirs(s) if flats else None, current()), "dinf",
                        {"units": "degrees clockwise from north; -1 no flow"})
    raise ValueError("method must be 'd8' or 'dinf'")


@threaded
def flow_accumulation(dem, *, method: str = "d8", units: str = "cells", weights=None, exponent: float | None = None,
                      flats: bool = True, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """The area that drains through each cell (the cell itself included).

    Parameters
    ----------
    dem
        The elevations, best with its depressions filled (:func:`fill`).
    method : {"d8", "dinf", "fd8", "quinn"}
        How a cell passes its flow on: all to its steepest neighbour (``"d8"``),
        split between the two cells of the D-infinity azimuth (``"dinf"``,
        Tarboton 1997), or among all lower neighbours in proportion to
        slope^exponent (``"fd8"``, Freeman 1991, exponent 1.1) or to slope times
        the contour length (0.5 cardinal, 0.354 diagonal; ``"quinn"``, Quinn et
        al. 1991, exponent 1). The multiple-flow methods avoid the parallel lines
        of D8 on hillslopes.
    units : {"cells", "area", "sca"}
        Number of cells, area (cells times the cell area, in map units squared;
        square metres for longitude / latitude), or specific contributing area
        (the area per unit contour width: area divided by the mean cell side, as
        WhiteboxTools does).
    weights
        A raster of what each cell contributes instead of 1 (runoff, sediment...).
    exponent
        The exponent of the multiple-flow methods (default 1.1 for ``"fd8"``, 1 for
        ``"quinn"``).
    flats
        Drain flat areas (see :func:`flow_direction`); without it their cells keep
        what they receive.
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray
        float64, NaN where the DEM is missing.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[4, 4, 4], [3, 3, 3], [2, 1, 2]], float)
    >>> ls.hydro.flow_accumulation(dem).values
    array([[1., 1., 1.],
           [2., 2., 2.],
           [1., 9., 1.]])
    """
    s = _read(dem, extent, crs, band, nodata)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if units not in ("cells", "area", "sca"):
        raise ValueError("units must be 'cells', 'area' or 'sca'")
    w = None
    if weights is not None:
        w = read_surface(weights).values
        if w.shape != s.shape:
            raise ValueError("the weights differ in shape from the DEM")
        w = np.where(np.isnan(w), 0.0, w)
    dx, dy = _cells(s)
    if method == "d8":
        acc = _core.hydro_d8_accumulation(_d8(s, flats), w)
    elif method == "dinf":
        angle = _core.hydro_dinf(s.values, dx, dy, _flat_dirs(s) if flats else None, current())
        acc = _core.hydro_dinf_accumulation(angle, w)
    else:
        p = float(exponent if exponent is not None else (1.1 if method == "fd8" else 1.0))
        acc = _core.hydro_mfd_accumulation(s.values, dx, dy, p, method == "quinn", _flat_dirs(s) if flats else None,
                                           w)
    if units != "cells":
        acc = acc * (dx * dy)
        if units == "sca":
            acc = acc / ((dx + dy) / 2.0)
    return s.raster(acc, "accumulation", {"method": method, "units": units})


def streams(accumulation, threshold: float) -> xr.DataArray:
    """The stream cells: those whose flow accumulation is at least ``threshold``.

    Parameters
    ----------
    accumulation
        A flow accumulation (:func:`flow_accumulation`).
    threshold
        The smallest accumulation of a stream, in its units.

    Returns
    -------
    xarray.DataArray
        1 on the streams, 0 elsewhere, NaN where the accumulation is missing.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> acc = ls.hydro.flow_accumulation(np.array([[4, 4, 4], [3, 3, 3], [2, 1, 2]], float))
    >>> ls.hydro.streams(acc, 4).values
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 1., 0.]])
    """
    s = read_surface(accumulation)
    a = s.values
    out = np.where(np.isnan(a), np.nan, (a >= threshold).astype(np.float64))
    return s.raster(out, "streams", {"threshold": float(threshold)})


@threaded
def hand(dem, streams=None, *, threshold: float | None = None, flats: bool = True, extent=None, crs=None, band=None,
         nodata=None) -> xr.DataArray:
    """Height Above the Nearest Drainage (HAND; Rennó et al. 2008, Nobre et al. 2011).

    The height of each cell above the stream cell where its water reaches the
    drainage, following the D8 directions downhill: 0 on the streams, small on
    floodplains and valley floors, large on hilltops. A terrain descriptor of
    soil water and flood susceptibility developed at INPE.

    Parameters
    ----------
    dem
        The elevations, with depressions filled (:func:`fill`), so that every cell
        reaches a stream.
    streams
        The drainage: a raster, non-zero on the stream cells (:func:`streams`).
    threshold
        Or the drainage as the cells of D8 accumulation (in cells) of at least this.
    flats
        Drain flat areas (see :func:`flow_direction`).
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray
        The height in the DEM's units, NaN where the DEM is missing or the cell
        reaches no stream.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.array([[9, 9, 9], [6, 5, 6], [4, 3, 4]], float)
    >>> ls.hydro.hand(dem, threshold=3).values
    array([[6., 6., 6.],
           [3., 2., 3.],
           [1., 0., 1.]])
    """
    s = _read(dem, extent, crs, band, nodata)
    d = _d8(s, flats)
    if streams is None:
        if threshold is None:
            raise ValueError("give the streams or a threshold of accumulation")
        mask = _core.hydro_d8_accumulation(d, None) >= threshold
    else:
        st = read_surface(streams).values
        if st.shape != s.shape:
            raise ValueError("the streams differ in shape from the DEM")
        mask = np.nan_to_num(st) != 0
    return s.raster(_core.hydro_hand(s.values, d, mask.astype(np.uint8)), "hand")
