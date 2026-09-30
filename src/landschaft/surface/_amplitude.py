# SPDX-License-Identifier: GPL-3.0-or-later
"""Amplitude parameters of a surface (ISO 25178; geodiv's basics.R)."""

from __future__ import annotations

from .. import _core
from .._threads import threaded
from ._grid import read_surface

__all__ = ["sa", "sq", "ssk", "sku", "sv", "sph", "smean"]


# the parts of _core.surf_amplitude each metric needs
PARTS = {"sa": 1, "sq": 2, "ssk": 4, "sku": 8, "sv": 16, "sph": 16, "smean": 32}


def _amplitude(x, extent, crs, band, nodata, part: str) -> dict:
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_amplitude(s.values, PARTS[part])


@threaded
def sa(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Average roughness (Sa): the mean absolute deviation from the mean height.

    Sa = sum(|z - mean(z)|) / N over the N cells with a value.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (a single band; NaN or ``nodata`` = missing). Remove a trend
        first where there is one (:func:`remove_plane`).
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
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sa(z), 6)
    0.044294
    """
    return _amplitude(x, extent, crs, band, nodata, "sa")["sa"]


@threaded
def sq(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Root mean square roughness (Sq): the root mean square of the heights
    about their mean, sqrt(sum((z - mean(z))^2) / N) (ISO 25178; geodiv uses
    the sample standard deviation, over N - 1).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`).
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sq(z), 6)
    0.064846
    """
    return _amplitude(x, extent, crs, band, nodata, "sq")["sq"]


@threaded
def ssk(x, adj: bool = True, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Skewness of the height distribution (Ssk).

    (sum((z - mean(z))^3) / N) / Sq^3 with Sq from :func:`sq`; with ``adj``,
    multiplied by sqrt(N (N - 1)) / (N - 2) (the adjusted Fisher-Pearson
    coefficient). Negative: valleys dominate; positive: peaks dominate.
    (geodiv divides by the sample standard deviation cubed.)

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`).
    adj : bool, default True
        Adjusted coefficient.
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.ssk(z), 4), round(ls.surface.ssk(z, adj=False), 4)
    (-1.9341, -1.9341)
    """
    a = _amplitude(x, extent, crs, band, nodata, "ssk")
    return a["ssk"] if adj else a["ssk_unadj"]


@threaded
def sku(x, excess: bool = True, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Kurtosis of the height distribution (Sku).

    (sum((z - mean(z))^4) / N) / Sq^4 with Sq from :func:`sq`; with
    ``excess``, minus 3 (0 for a normal distribution). (geodiv divides by the
    sample standard deviation to the fourth.)

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`).
    excess : bool, default True
        The excess kurtosis.
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sku(z), 4), round(ls.surface.sku(z, excess=False), 4)
    (6.4979, 9.4979)
    """
    a = _amplitude(x, extent, crs, band, nodata, "sku")
    return a["sku"] if excess else a["sku_raw"]


@threaded
def sv(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Maximum valley depth (Sv): the absolute value of the lowest height.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`), usually with its mean or trend removed.
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sv(z), 6)
    0.585464
    """
    return _amplitude(x, extent, crs, band, nodata, "sv")["sv"]


@threaded
def sph(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Maximum peak height (Sp): the absolute value of the highest height.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`), usually with its mean or trend removed.
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sph(z), 6)
    0.158592
    """
    return _amplitude(x, extent, crs, band, nodata, "sph")["sph"]


@threaded
def smean(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Mean peak height: the mean of the positive heights.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sa`), with its mean or trend removed.
    extent, crs, band, nodata
        As in :func:`sa`.

    Returns
    -------
    float
        NaN without positive heights.

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.smean(z), 6)
    0.036899
    """
    return _amplitude(x, extent, crs, band, nodata, "smean")["smean"]
