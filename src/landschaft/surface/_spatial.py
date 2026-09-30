# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial parameters of a surface: autocorrelation, texture direction and
wavelength, fractal dimension (ISO 25178; geodiv's aacf.R, fourier.R, sfd)."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .. import _core
from .._threads import threaded
from ..cost._transition import _crs_lonlat
from ._grid import Surface, read_surface

__all__ = ["aacf", "scl", "stxr", "std", "srw", "sfd"]

THRESHOLDS = (0.2, 1 / math.e)


def _args(s: Surface) -> tuple:
    return s.values, tuple(map(float, s.extent)), _crs_lonlat(s.crs)


def _thresholds(threshold) -> list[float]:
    t = [float(threshold)] if np.isscalar(threshold) else [float(v) for v in threshold]
    if not t or any(v < 0 for v in t):
        raise ValueError("threshold values must be 0 or more")
    return t


def spatial_parameters(s: Surface) -> dict:
    """sal, str, std, stdi, srw, srwi, shw and sfd of a surface."""
    z, e, ll = _args(s)
    lengths = _core.surf_scl(z, e, ll, list(THRESHOLDS))
    d = _core.surf_std(z, e, ll)
    w = _core.surf_srw(z, e, ll)
    fd, outside = _core.surf_sfd(z, e)
    return {"sal": lengths[0], "str": _ratio(lengths[0], lengths[2]), "std": d[0], "stdi": d[1],
            "srw": w[0], "srwi": w[1], "shw": w[2], "sfd": math.nan if outside else fd}


def _ratio(a: float, b: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


@threaded
def aacf(x, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Areal autocorrelation function of a surface.

    As geodiv: the surface times a two-dimensional Hann window, its power
    spectrum by FFT, the inverse FFT of that, shifted to put zero lag in the
    centre and divided by its maximum. With missing cells, the largest square
    without them around the centre is used (and the image covers it).

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
    xarray.DataArray
        The autocorrelation at every lag, 1 at zero lag.

    Examples
    --------
    >>> import landschaft as ls
    >>> a = ls.surface.aacf(ls.surface.example_data("normforest"))
    >>> a.shape, float(a.max())
    ((371, 371), 1.0)
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    z, e, ll = _args(s)
    values, ext = _core.surf_aacf(z, e, ll)
    if tuple(ext) == tuple(e) and values.shape == z.shape:
        return s.raster(values, "aacf")
    return Surface(values, tuple(ext), s.crs).raster(values, "aacf")


@threaded
def scl(x, threshold=THRESHOLDS, *, extent=None, crs=None, band=None, nodata=None) -> np.ndarray:
    """Correlation lengths: how far the autocorrelation takes to decay.

    On the upper half of :func:`aacf`, the distances (map units; metres on
    longitude/latitude rasters) from zero lag to the cells whose
    autocorrelation is at or below each threshold: the shortest (fastest
    decay, Sal for 0.2) and the longest (slowest decay).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`aacf`).
    threshold : float or sequence of float, default (0.2, 1/e)
        Autocorrelation value(s).
    extent, crs, band, nodata
        As in :func:`aacf`.

    Returns
    -------
    numpy.ndarray
        The fastest decay length for each threshold, then the slowest ones.

    Notes
    -----
    Distances as terra computes them for geodiv: on projected rasters GDAL's
    proximity (whole cells, in single precision, times the cell width), on
    longitude/latitude ones the haversine distance. With missing cells geodiv
    fails (it looks for the centre of the cropped image with the size of the
    whole raster); here the centre of the cropped image is used.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.scl(ls.surface.example_data("orelevation")).round(1).tolist()
    [4642.1, 3308.2, 6848.8, 6848.8]
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    z, e, ll = _args(s)
    return np.asarray(_core.surf_scl(z, e, ll, _thresholds(threshold)))


@threaded
def stxr(x, threshold=THRESHOLDS, *, extent=None, crs=None, band=None, nodata=None) -> np.ndarray:
    """Texture aspect ratio (Str): fastest over slowest decay length.

    From :func:`scl`, for each threshold: near 0 for a strongly oriented
    texture, near 1 for an isotropic one.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`aacf`).
    threshold : float or sequence of float, default (0.2, 1/e)
        Autocorrelation value(s).
    extent, crs, band, nodata
        As in :func:`aacf`.

    Returns
    -------
    numpy.ndarray
        One ratio per threshold.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.stxr(ls.surface.example_data("orelevation")).round(4).tolist()
    [0.6778, 0.483]
    """
    t = _thresholds(threshold)
    lengths = scl(x, t, extent=extent, crs=crs, band=band, nodata=nodata)
    k = len(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lengths[:k] / lengths[k:]


@threaded
def std(x, *, extent=None, crs=None, band=None, nodata=None) -> np.ndarray:
    """Texture direction (Std) and texture direction index (Stdi).

    From the amplitude of the surface's Fourier spectrum (upper half, within
    the largest circle around zero frequency that fits), summed over angles 0
    to 180 degrees in steps of 0.5 (+-0.25): Std is the angle with the largest
    sum, Stdi the mean sum over the largest (near 1: no dominant direction).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`aacf`).
    extent, crs, band, nodata
        As in :func:`aacf`.

    Returns
    -------
    numpy.ndarray
        ``[std, stdi]`` (degrees, ratio).

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.std(ls.surface.example_data("normforest")).round(4).tolist()
    [90.0, 0.1962]
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return np.asarray(_core.surf_std(*_args(s)))


@threaded
def srw(x, *, extent=None, crs=None, band=None, nodata=None) -> np.ndarray:
    """Dominant radial wavelength (Srw), its index (Srwi) and the half-power
    wavelength (Shw).

    The amplitude of the surface's Fourier spectrum (upper half) summed over
    circles around zero frequency (up to 100, evenly spaced): Srw is the
    radius with the largest sum, Srwi the mean sum over the largest, Shw the
    radius that splits the summed amplitude in half. NaN where geodiv returns
    NA (ties).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`aacf`).
    extent, crs, band, nodata
        As in :func:`aacf`.

    Returns
    -------
    numpy.ndarray
        ``[srw, srwi, shw]``.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.surface.srw(ls.surface.example_data("normforest")).round(4).tolist()
    [409.6657, 0.5874, 1638.6626]
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return np.asarray(_core.surf_srw(*_args(s)))


@threaded
def sfd(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Fractal dimension of the surface (Sfd), by triangular prisms.

    Clarke's (1986) method as geodiv implements it: heights rescaled to 0-1,
    the area of triangular prisms over the central square of a power-of-two
    side at cell sizes 1, 2, 4...; 2 minus the slope of log(area) against
    log(cell area).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`aacf`), without missing cells.
    extent, crs, band, nodata
        As in :func:`aacf`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        Where geodiv reads past the raster (when the central square reaches
        its last row or column, as on small or elongated rasters).

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.surface.sfd(ls.surface.example_data("orelevation")), 6)
    2.000007
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    fd, outside = _core.surf_sfd(s.values, tuple(map(float, s.extent)))
    if outside:
        raise ValueError("sfd: the central square reaches the edge of the raster (geodiv reads past it); "
                         "use a larger or squarer raster")
    return fd
