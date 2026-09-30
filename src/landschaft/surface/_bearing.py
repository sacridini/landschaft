# SPDX-License-Identifier: GPL-3.0-or-later
"""The bearing area (Abbott-Firestone) curve of a surface and its parameters
(ISO 25178; geodiv's bearingarea.R)."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from .. import _core
from .._threads import threaded
from ._grid import Surface, read_surface

__all__ = ["bearing_area", "height_ba", "sdc", "sbi", "svi", "sci", "sk", "spk", "svk"]


def bearing_parameters(s: Surface) -> dict:
    """sbi, svi, sci, sk, spk and svk of a surface, from one curve."""
    sq = _core.surf_amplitude(s.values, 2)["sq"]
    return _core.surf_bearing(s.values, None, sq, True)


def _read(x, extent, crs, band, nodata) -> Surface:
    return read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)


def _fraction(p, name: str) -> float:
    p = float(p)
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {p!r}")
    return p


@threaded
def bearing_area(x, *, extent=None, crs=None, band=None, nodata=None) -> pd.DataFrame:
    """The bearing area (Abbott-Firestone) curve: the share of the surface at
    or above each height.

    For every distinct height, from the highest, the fraction of cells whose
    height is at least that high (geodiv's curve is that of 1 - z, which puts
    rounding errors in the heights).

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
        Columns ``bearing`` (0-1, increasing) and ``height`` (decreasing), one
        row per distinct height: the points of geodiv's ``plot_ba_curve``.

    Examples
    --------
    >>> import landschaft as ls
    >>> curve = ls.surface.bearing_area(ls.surface.example_data("orelevation"))
    >>> curve.iloc[[0, -1]]
          bearing  height
    0    0.000015  1390.0
    953  1.000000   433.0
    """
    s = _read(x, extent, crs, band, nodata)
    b = _core.surf_bearing(s.values)
    return pd.DataFrame({"bearing": b["bearing"], "height": b["height"]})


@threaded
def height_ba(x, xval, *, extent=None, crs=None, band=None, nodata=None):
    """Height of the bearing area curve at a bearing fraction.

    The height above which a fraction ``xval`` of the surface lies: R's type-7
    quantile of z at 1 - ``xval``.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    xval : float or sequence of float
        Bearing fraction(s), 0 (the highest height) to 1 (the lowest).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float or numpy.ndarray
        One height per fraction.

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> ls.surface.height_ba(dem, [0, 0.5, 1]).tolist()
    [1390.0, 884.0, 433.0]
    """
    scalar = isinstance(xval, numbers.Real)
    probs = [_fraction(p, "xval") for p in ([xval] if scalar else xval)]
    s = _read(x, extent, crs, band, nodata)
    h = _core.surf_bearing(s.values, probs)["heights"]
    return float(h[0]) if scalar else np.asarray(h)


@threaded
def sdc(x, low: float, high: float, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Areal material ratio height difference (Sdc).

    The height difference between two bearing fractions:
    :func:`height_ba` at ``low`` minus at ``high``.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    low, high : float
        Bearing fractions, 0 <= low < high <= 1.
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.sdc(ls.surface.example_data("orelevation"), 0.1, 0.8)
    460.0
    """
    low, high = _fraction(low, "low"), _fraction(high, "high")
    if high <= low:
        raise ValueError("high must be greater than low")
    s = _read(x, extent, crs, band, nodata)
    h = _core.surf_bearing(s.values, [low, high])["heights"]
    return float(h[0] - h[1])


@threaded
def sbi(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Surface bearing index (Sbi): Sq divided by the height at 5 % bearing.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`), with its mean or trend removed.
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.sbi(ls.surface.example_data("normforest")), 6)
    0.801444
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["sbi"]


@threaded
def svi(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Valley fluid retention index (Svi): the void volume of the valley zone.

    The area between the bearing area curve and its top from 80 % to 100 %
    bearing (Simpson's rule on 500 intervals), in height units (geodiv's
    definition, corrected: its area depends on whether the curve dips below 0,
    which gives negative values on positive surfaces, and its Simpson's rule
    drops the last point).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.svi(ls.surface.example_data("normforest")), 6)
    0.012704
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["svi"]


@threaded
def sci(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Core fluid retention index (Sci): the void volume of the core zone.

    The area between the bearing area curve and its top from 5 % to 100 %
    bearing (1000 intervals) minus :func:`svi`.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.sci(ls.surface.example_data("normforest")), 6)
    0.069002
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["sci"]


@threaded
def sk(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Core roughness depth (Sk): the height range of the core line.

    The core line is the least-squares line through the bearing area curve
    over the flattest 40 % of it (geodiv's ``find_flat``: the curve's slopes by
    central differences at 10000 points, one-sided at the ends, their mean
    absolute value over each 40 % window); Sk is the difference of its heights
    at 0 % and 100 %.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.sk(ls.surface.example_data("normforest")), 6)
    0.102369
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["sk"]


@threaded
def spk(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Reduced peak height (Spk): the peaks above the core.

    The height difference between the top of the bearing area curve and the
    curve at the bearing where the core line (:func:`sk`) starts.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.spk(ls.surface.example_data("normforest")), 6)
    0.096705
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["spk"]


@threaded
def svk(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Reduced valley depth (Svk): the valleys below the core.

    The height difference between the bottom of the bearing area curve and the
    curve at the bearing where the core line (:func:`sk`) ends.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`bearing_area`).
    extent, crs, band, nodata
        As in :func:`bearing_area`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.svk(ls.surface.example_data("normforest")), 6)
    0.544987
    """
    return bearing_parameters(_read(x, extent, crs, band, nodata))["svk"]
