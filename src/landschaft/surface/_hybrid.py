# SPDX-License-Identifier: GPL-3.0-or-later
"""Hybrid parameters of a surface: slopes and developed area (ISO 25178;
geodiv's sdq.R and surfacearea.R, corrected)."""

from __future__ import annotations

from .. import _core
from .._threads import threaded
from ._grid import read_surface

__all__ = ["sdq", "sdq6", "surface_area", "flat_area", "sdr"]


def _hybrid(x, extent, crs, band, nodata) -> dict:
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_hybrid(s.values, tuple(map(float, s.extent)))


@threaded
def sdq(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Root mean square slope (Sdq).

    The root of the mean of (z[r, c] - z[r, c+1])^2 + (z[r, c] - z[r+1, c])^2,
    the differences with the right and lower neighbours in height units per
    cell, over the cells that have both.

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
    float

    Notes
    -----
    geodiv divides by the number of cells, those without both neighbours too.

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sdq(z), 6)
    0.04389
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_sdq(s.values, tuple(map(float, s.extent)))


@threaded
def sdq6(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area root mean square slope (Sdq6), from seven-point derivatives.

    The slopes in x and y from ISO 25178's sixth-order central difference,
    (z[+3] - 9 z[+2] + 45 z[+1] - 45 z[-1] + 9 z[-2] - z[-3]) / 60 per cell,
    at the cells three or more cells away from the edges; the root of the mean
    of their squared sums.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    extent, crs, band, nodata
        As in :func:`sdq`.

    Returns
    -------
    float
        NaN on rasters with fewer than 7 rows or columns.

    Notes
    -----
    geodiv's stencil has ``- 9 z[+2]`` twice in place of ``+ 9 z[-2]`` (it does
    not cancel the height, so a high surface gives large values) and it takes
    the root of the sum, not of the mean.

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> round(ls.surface.sdq6(dem), 4)
    10.6134
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_sdq6(s.values, tuple(map(float, s.extent)))


@threaded
def surface_area(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area of the surface, triangulated, in geodiv's units.

    Every block of 2 x 2 cell centres is split into two triangles along each
    diagonal and the two areas are averaged; the sum over the blocks whose
    four values are all present. As geodiv, on the heights rescaled to 0-1 and
    cells 1 wide by their height-to-width ratio high, which makes the area
    independent of the heights' unit.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    extent, crs, band, nodata
        As in :func:`sdq`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.surface_area(z), 2)
    137137.26
    """
    return _hybrid(x, extent, crs, band, nodata)["surface_area"]


@threaded
def flat_area(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area of the surface's projection on a plane (geodiv's ``flatsa``).

    :func:`surface_area` of the same cells with every height 0.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    extent, crs, band, nodata
        As in :func:`sdq`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.flat_area(z), 2)
    136900.0
    """
    return _hybrid(x, extent, crs, band, nodata)["flat_area"]


@threaded
def sdr(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Developed interfacial area ratio (Sdr), in percent.

    How much larger the surface is than its projection:
    (:func:`surface_area` - :func:`flat_area`) / :func:`flat_area` x 100, with
    the heights rescaled to 0-1 as geodiv.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    extent, crs, band, nodata
        As in :func:`sdq`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> round(ls.surface.sdr(z), 4)
    0.1733
    """
    return _hybrid(x, extent, crs, band, nodata)["sdr"]
