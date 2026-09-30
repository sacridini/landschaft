# SPDX-License-Identifier: GPL-3.0-or-later
"""Local features of a surface: peaks, valleys and what they add up to
(ISO 25178; geodiv's localsurface.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import threaded
from ._grid import Surface, read_surface
from ._hybrid import iso

__all__ = ["find_peaks", "find_valleys", "sds", "ssc", "s10z"]


def _extent(s: Surface) -> tuple[float, float, float, float]:
    return tuple(map(float, s.extent))


def _table(s: Surface, cells: np.ndarray) -> pd.DataFrame:
    ny, nx = s.shape
    xmin, xmax, ymin, ymax = s.extent
    row, col = np.divmod(cells, nx)
    return pd.DataFrame({
        "x": xmin + (col + 0.5) * ((xmax - xmin) / nx),
        "y": ymax - (row + 0.5) * ((ymax - ymin) / ny),
        "value": s.values.ravel()[cells],
        "row": row,
        "col": col,
    })


@threaded
def find_peaks(x, *, extent=None, crs=None, band=None, nodata=None) -> pd.DataFrame:
    """The peaks of a surface: cells higher than their 8 neighbours and than 0.

    Only cells off the edge whose 8 neighbours all have a value can be peaks;
    heights are compared with 0, so remove the mean or trend first
    (:func:`remove_plane`).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (a single band; NaN or ``nodata`` = missing).
    extent : tuple of float, optional
        ``(xmin, xmax, ymin, ymax)`` of an array (default: unit cells).
    crs : optional
        CRS of an array.
    band : int, optional
        Band of a multi-band file or DataArray (1-based).
    nodata : float, optional
        Value that marks missing cells.

    Returns
    -------
    pandas.DataFrame
        One row per peak, in row order: the coordinates of the cell centre
        (``x``, ``y``), its ``value``, its ``row`` and ``col`` (from 0).

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> peaks = ls.surface.find_peaks(z)
    >>> len(peaks), round(float(peaks.value.max()), 4)
    (6488, 0.1586)
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _table(s, _core.surf_find_peaks(s.values, _extent(s)))


@threaded
def find_valleys(x, convention: str = "geodiv", *, extent=None, crs=None, band=None,
                 nodata=None) -> pd.DataFrame:
    """The valleys of a surface: cells lower than their 8 neighbours and than 0.

    As :func:`find_peaks`, downwards.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`find_peaks`).
    convention : {"geodiv", "iso"}, default "geodiv"
        ``"geodiv"``: as geodiv, which reads each cell's neighbours from the
        values arranged by columns, not by rows: on a square raster it
        compares each cell with the neighbours of the transposed cell, on
        others with unrelated cells, and finds far more "valleys" than there
        are (below: 33041 against 5167); ``"iso"``: the cell's own neighbours.
    extent, crs, band, nodata
        As in :func:`find_peaks`.

    Returns
    -------
    pandas.DataFrame
        As :func:`find_peaks`.

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> len(ls.surface.find_valleys(z)), len(ls.surface.find_valleys(z, "iso"))
    (33041, 5167)
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _table(s, _core.surf_find_valleys(s.values, _extent(s), not iso(convention)))


@threaded
def sds(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Density of summits (Sds): peaks per cell.

    The number of :func:`find_peaks` over (rows - 1) (columns - 1).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`find_peaks`).
    extent, crs, band, nodata
        As in :func:`find_peaks`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sds(z), 6)
    0.047392
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_sds(s.values, _extent(s))


@threaded
def ssc(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Mean summit curvature (Ssc).

    Over the n :func:`find_peaks`, with heights rescaled to 0-1:
    -1 / (2n) times the sum of ((z - z[x+1]) + (z - z[x-1])) / 2 / dx and
    ((z - z[y+1]) + (z - z[y-1])) / 2 / dy, dx = 1 and dy the ratio of the
    cell's height to its width (geodiv's definition).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`find_peaks`).
    extent, crs, band, nodata
        As in :func:`find_peaks`.

    Returns
    -------
    float
        NaN without peaks.

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.ssc(z), 6)
    -0.021922
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_ssc(s.values, _extent(s))


@threaded
def s10z(x, convention: str = "geodiv", *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Ten-point height (S10z).

    (sum of the 5 highest peaks + sum of the absolute depths of the 5 deepest
    valleys) / 5, from :func:`find_peaks` and :func:`find_valleys` (fewer when
    there are fewer).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`find_peaks`).
    convention : {"geodiv", "iso"}, default "geodiv"
        Which valleys, as in :func:`find_valleys`.
    extent, crs, band, nodata
        As in :func:`find_peaks`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.s10z(z), 6), round(ls.surface.s10z(z, "iso"), 6)
    (0.718518, 0.665448)
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_s10z(s.values, _extent(s), not iso(convention))
