# SPDX-License-Identifier: GPL-3.0-or-later
"""Trend surfaces: polynomial planes fitted to a surface (geodiv's fitplane,
bestfitplane and remove_plane, over spatial::surf.ls)."""

from __future__ import annotations

import xarray as xr

from .. import _core
from .._threads import threaded
from ._grid import read_surface

__all__ = ["fit_plane", "best_fit_plane", "remove_plane"]


def _extent(s) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in s.extent)


@threaded
def fit_plane(x, order: int, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """The least-squares polynomial surface of a given order.

    A polynomial in the cell-centre coordinates, x^i y^j with i + j <= ``order``,
    fitted to the cells with a value (coordinates scaled to [-1, 1] and a
    Householder QR, as ``spatial::surf.ls``) and evaluated in every cell.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (a single band; NaN or ``nodata`` = missing).
    order : int
        Order of the polynomial, 0 (the mean) to 6.
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
        The fitted surface on the grid of ``x`` (also where ``x`` is missing).

    Examples
    --------
    >>> import landschaft as ls
    >>> ndvi = ls.surface.example_data("orforest")
    >>> trend = ls.surface.fit_plane(ndvi, 1)
    >>> round(float(trend.min()), 4), round(float(trend.max()), 4)
    (0.6652, 0.6979)
    """
    order = _order(order)
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    _check_points(s, order)
    values, ifail = _core.surf_fit_plane(s.values, _extent(s), order)
    return s.raster(values, "fit_plane", {"order": order, "ifail": ifail})


@threaded
def best_fit_plane(x, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """The polynomial surface of order 0 to 3 that fits a surface best.

    As geodiv's ``bestfitplane``: of the four least-squares surfaces
    (:func:`fit_plane`), the one whose residuals have the smallest absolute
    mean, the lowest order on ties. The order is in ``attrs["order"]``.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`fit_plane`).
    extent, crs, band, nodata
        As in :func:`fit_plane`.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    Every least-squares fit with a constant term has residuals that average to
    zero, so the choice rests on rounding errors (about 1e-17): it reproduces
    geodiv, which is what this function is for. To remove a trend of a chosen
    order, subtract :func:`fit_plane`.

    Examples
    --------
    >>> import landschaft as ls
    >>> ndvi = ls.surface.example_data("orforest")
    >>> ls.surface.best_fit_plane(ndvi).attrs["order"]
    1
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    _check_points(s, 3)
    values, order = _core.surf_best_fit_plane(s.values, _extent(s))
    return s.raster(values, "best_fit_plane", {"order": int(order)})


@threaded
def remove_plane(x, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """A surface minus its best-fitting polynomial surface (geodiv's ``remove_plane``).

    The residuals from :func:`best_fit_plane`: positive above the trend,
    negative below; missing cells stay missing. The order removed is in
    ``attrs["order"]``.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`fit_plane`).
    extent, crs, band, nodata
        As in :func:`fit_plane`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> ndvi = ls.surface.example_data("orforest")
    >>> flat = ls.surface.remove_plane(ndvi)
    >>> flat.attrs["order"], round(float(flat.mean()), 12)
    (1, 0.0)
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    _check_points(s, 3)
    values, order = _core.surf_best_fit_plane(s.values, _extent(s))
    return s.raster(s.values - values, "remove_plane", {"order": int(order)})


def _order(order) -> int:
    k = int(order)
    if k != order or not 0 <= k <= 6:
        raise ValueError(f"order must be an integer from 0 to 6, got {order!r}")
    return k


def _check_points(s, order: int) -> None:
    import numpy as np

    npar = (order + 1) * (order + 2) // 2
    n = int(np.count_nonzero(~np.isnan(s.values)))
    if n < npar:
        raise ValueError(f"a polynomial of order {order} needs at least {npar} cells with a value, got {n}")
