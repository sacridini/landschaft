# SPDX-License-Identifier: GPL-3.0-or-later
"""Hilltops as permanent protection areas (APP de topo de morro, Lei 12.651/2012).

A clean-room, open implementation of the hilltop rule of Brazil's Forest Code,
in place of Dinamica EGO's ``CalcHilltop`` (which follows the older CONAMA
303/2002 rule on a DEM cut into elevation bands).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import threaded
from ..cost._transition import _crs_lonlat
from ._grid import read_surface

__all__ = ["Hilltop", "hilltop"]

_KINDS = np.array(["saddle", "plain", "lowest"])


@dataclass
class Hilltop(Saveable):
    """The hilltops of a DEM (:func:`hilltop`).

    Attributes
    ----------
    app : xarray.DataArray
        1 in the upper part of the hills that qualify (the APP), 0 elsewhere
        on the DEM, NaN where it has no value.
    hill : xarray.DataArray
        The id (``hills`` row) of the hill whose upper part holds each cell
        (of the highest threshold, where upper parts nest), NaN elsewhere.
    hills : pandas.DataFrame
        One row per summit, highest first: ``x``, ``y`` (the summit), ``top``,
        ``base``, ``base_kind`` (``"saddle"``: the key col to higher terrain;
        ``"plain"``: the plain or water body it meets; ``"lowest"``: the lowest
        cell of its connected area), ``base_x``, ``base_y`` (the saddle, the
        plain cell met, or the lowest cell nearest the summit), ``height``,
        ``mean_slope`` (degrees, of its cells above the base),
        ``line_slope`` (degrees, of the line from the summit to its foot: the
        nearest cell at or below its base), ``area`` (above the base, map
        units squared), ``area_ha`` (hectares; at the centre latitude for
        longitude / latitude), ``qualifies``, ``threshold`` (the elevation from which
        the APP starts) and ``app_cells``.
    """

    app: xr.DataArray
    hill: xr.DataArray
    hills: pd.DataFrame

    def __repr__(self) -> str:
        q = int(self.hills["qualifies"].sum())
        return f"Hilltop({len(self.hills)} summits, {q} hills qualify, {int(np.nansum(self.app.values))} APP cells)"


def _metres(s) -> tuple[float, float]:
    """Cell sizes in metres (map units; at the centre latitude for longitude / latitude)."""
    dx, dy = (abs(float(v)) for v in s.res)
    if _crs_lonlat(s.crs):
        lat = math.radians((s.extent[2] + s.extent[3]) / 2)
        return dx * 111320.0 * math.cos(lat), dy * 110574.0
    return dx, dy


@threaded
def hilltop(dem, *, min_height: float = 100.0, min_slope: float = 25.0, slope: str = "mean",
            fraction: float = 2 / 3, plain=None, extent=None, crs=None, band=None, nodata=None) -> Hilltop:
    """Hilltops as permanent protection areas (APP de topo de morro).

    Brazil's Forest Code (Lei 12.651/2012, art. 4, IX) protects the tops of
    hills, mounts, mountains and ranges at least 100 m high and of mean slope
    above 25 degrees: the areas above the contour at 2/3 of the height,
    measured from the base, which is the plane of the adjacent plain or water
    body or, in undulating relief, the elevation of the nearest saddle point.

    Every summit of the DEM is found with its base by sweeping the cells from
    the highest down and joining each to the regions of its 8 neighbours (a
    merge tree): where regions meet, the cell is a saddle, and the regions of
    lower summits end there (their base is the saddle: the key col of
    topographic prominence, so a range is one elevation above its lowest
    saddle and its peaks are hills of their own above theirs). A region that
    reaches a cell of ``plain`` gets that cell's elevation as its base; the
    highest summit of a connected area that meets neither gets its lowest
    elevation. A hill qualifies when its height is at least ``min_height``
    and its slope above ``min_slope``; its APP is the area joined to its
    summit through cells at least ``base + fraction x height`` high.

    Parameters
    ----------
    dem : str, path, xarray.DataArray or array_like
        The DEM (metres), projected or in longitude / latitude.
    min_height : float, default 100
        The least height of a hill (metres; CONAMA 303/2002: 50).
    min_slope : float, default 25
        The slope a hill must exceed (degrees; CONAMA 303/2002: about 16.7,
        30 %, along the line of steepest slope).
    slope : {"mean", "line"}, default "mean"
        ``"mean"``: the mean slope (Horn, degrees) of the hill's cells above
        its base (the law's "inclinação média"); ``"line"``: the slope of the
        line from the summit to the base cell.
    fraction : float, default 2/3
        Where the APP starts, as a fraction of the height above the base.
    plain : xarray.DataArray or array_like, optional
        Plains and water bodies (true / non-zero): a hill that reaches them
        takes their elevation as its base.
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    Hilltop

    Notes
    -----
    The law leaves room for interpretation and these are the choices made:
    the nearest saddle of an elevation is the col where its region meets
    higher terrain (the hill's prominence), so the highest peak of a range
    is measured from the range's base and lower peaks from their saddles;
    the mean slope is that of the cells of the hill above its base (its
    region in the merge tree); the APP is the part of that region above the
    threshold that is joined to the summit (a separate peak above the
    threshold, joined below it, is judged as a hill of its own). Plains are
    not inferred from the slope: give ``plain`` (flat floodplains, lakes,
    rivers, the sea). Cells on the DEM's edges have no slope and are left out
    of the mean. Results depend on the DEM's resolution and accuracy, and
    automatic delineation is a starting point for field and legal work.

    Dinamica EGO's ``CalcHilltop`` follows CONAMA 303/2002 on a DEM cut into
    elevation bands: bases are the band of a local minimum reached
    downhill, and its slope criterion is the angle between the summit and
    the nearest cell of the base band that borders other features or the
    map's edge (5 degrees for an isolated cone of 21.8 degrees in a plain,
    measured with Dinamica EGO 8.13), so its results differ.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> yy, xx = np.mgrid[0:100, 0:100] * 10.0
    >>> cone = 500 + np.clip(300 - 1.0 * np.hypot(yy - 500, xx - 500), 0, None)  # 300 m high, 45 degrees
    >>> h = ls.surface.hilltop(cone, extent=(0, 1000, 0, 1000))
    >>> top = h.hills.iloc[0]
    >>> float(top.base), float(top.height), round(float(top.mean_slope)), bool(top.qualifies)
    (500.0, 300.0, 44, True)
    >>> int(np.nansum(h.app)) == int(np.sum(cone >= 700))   # the APP: above 500 + 2/3 x 300
    True
    """
    if slope not in ("mean", "line"):
        raise ValueError(f"slope must be 'mean' or 'line', got {slope!r}")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    s = read_surface(dem, extent=extent, crs=crs, band=band, nodata=nodata)
    z = np.ascontiguousarray(s.values, dtype=np.float64)
    ext = tuple(map(float, s.extent))
    sl = np.asarray(_core.surf_terrain(z, ext, _crs_lonlat(s.crs), 8, True, False), dtype=np.float64)
    pm = None
    if plain is not None:
        pv = read_surface(plain, extent=extent, crs=crs).values
        if pv.shape != z.shape:
            raise ValueError(f"the plain mask has shape {pv.shape}, the DEM {z.shape}")
        pm = np.ascontiguousarray((~np.isnan(pv) & (pv != 0)).astype(np.uint8))
    dx, dy = _metres(s)
    r = _core.surf_hilltops(z, np.ascontiguousarray(sl), pm, dx, dy, float(min_height), float(min_slope),
                            slope == "line", float(fraction))
    label = np.asarray(r["label"])
    nan = np.isnan(z)
    app = np.where(nan, np.nan, (label >= 0).astype(np.float64))
    hill = np.where(label >= 0, label.astype(np.float64), np.nan)
    ny, nx = z.shape
    xmin, xmax, ymin, ymax = ext
    rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny

    def xy(cells):
        cells = np.asarray(cells)
        return xmin + (cells % nx + 0.5) * rx, ymax - (cells // nx + 0.5) * ry

    sx, sy = xy(r["summit"])
    bx, by = xy(r["base_cell"])
    counts = np.bincount(label[label >= 0], minlength=len(r["summit"])) if len(r["summit"]) else np.zeros(0, int)
    table = pd.DataFrame({
        "x": sx, "y": sy, "top": np.asarray(r["top"]), "base": np.asarray(r["base"]),
        "base_kind": _KINDS[np.asarray(r["kind"])], "base_x": bx, "base_y": by, "height": np.asarray(r["height"]),
        "mean_slope": np.asarray(r["mean_slope"]), "line_slope": np.asarray(r["line_slope"]),
        "area": np.asarray(r["cells"]) * abs(rx * ry), "area_ha": np.asarray(r["cells"]) * dx * dy / 1e4,
        "qualifies": np.asarray(r["qualifies"]),
        "threshold": np.asarray(r["threshold"]), "app_cells": counts.astype(np.int64)})
    table.index.name = "hill"
    return Hilltop(s.raster(app, "app"), s.raster(hill, "hill"), table)
