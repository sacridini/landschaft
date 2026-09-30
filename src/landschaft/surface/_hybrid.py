# SPDX-License-Identifier: GPL-3.0-or-later
"""Hybrid parameters of a surface: slopes and developed area (ISO 25178;
geodiv's sdq.R and surfacearea.R)."""

from __future__ import annotations

from .. import _core
from .._threads import threaded
from ._grid import read_surface

__all__ = ["sdq", "sdq6", "surface_area", "flat_area", "sdr"]

CONVENTIONS = ("geodiv", "iso")


def iso(convention: str) -> bool:
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be 'geodiv' or 'iso', got {convention!r}")
    return convention == "iso"


def _hybrid(x, convention, extent, crs, band, nodata) -> dict:
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_hybrid(s.values, tuple(map(float, s.extent)), iso(convention))


@threaded
def sdq(x, *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Root mean square slope (Sdq).

    sqrt(sum((z[r, c] - z[r, c+1])^2 + (z[r, c] - z[r+1, c])^2) / n): the
    differences with the right and lower neighbours, in height units per cell,
    over the n cells that have both (pairs with a missing value add nothing but
    still count in n, as in geodiv).

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
def sdq6(x, convention: str = "geodiv", *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area root mean square slope (Sdq6), from seven-point derivatives.

    The slopes in x and y from the sixth-order central difference of ISO
    25178, (z[+3] - 9 z[+2] + 45 z[+1] - 45 z[-1] + 9 z[-2] - z[-3]) / 60 per
    cell, at the cells at least three cells away from the edges; the root of
    the mean of their squares.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    convention : {"geodiv", "iso"}, default "geodiv"
        ``"geodiv"``: as geodiv, whose stencil has ``- 9 z[+2]`` twice in place
        of ``+ 9 z[-2]`` (it does not cancel the height, so a tilted or high
        surface gives large values) and which takes the root of the sum of the
        squares, not of their mean (it divides by ``length(x)``, 1 for a
        raster); ``"iso"``: the definition above.
    extent, crs, band, nodata
        As in :func:`sdq`.

    Returns
    -------
    float

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> round(ls.surface.sdq6(dem, "iso"), 4)
    10.6134
    """
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    return _core.surf_sdq6(s.values, tuple(map(float, s.extent)), iso(convention))


@threaded
def surface_area(x, convention: str = "geodiv", *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area of the surface, triangulated.

    Every block of 2 x 2 cell centres is split into two triangles along each
    diagonal and the two areas are averaged; the sum over the blocks whose
    four values are all present.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    convention : {"geodiv", "iso"}, default "geodiv"
        ``"geodiv"``: heights rescaled to 0-1 and cells 1 wide by their
        height-to-width ratio high (the area is then in cells, and depends on
        the range of the heights); ``"iso"``: the heights and the cell size as
        they are (both must then be in the same unit).
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
    return _hybrid(x, convention, extent, crs, band, nodata)["surface_area"]


@threaded
def flat_area(x, convention: str = "geodiv", *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Area of the surface's projection on a plane (geodiv's ``flatsa``).

    :func:`surface_area` of the same cells with every height 0.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    convention : {"geodiv", "iso"}, default "geodiv"
        Units, as in :func:`surface_area`.
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
    return _hybrid(x, convention, extent, crs, band, nodata)["flat_area"]


@threaded
def sdr(x, convention: str = "geodiv", *, extent=None, crs=None, band=None, nodata=None) -> float:
    """Developed interfacial area ratio (Sdr), in percent.

    How much larger the surface is than its projection:
    (:func:`surface_area` - :func:`flat_area`) / :func:`flat_area` x 100.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (see :func:`sdq`).
    convention : {"geodiv", "iso"}, default "geodiv"
        Units, as in :func:`surface_area`: geodiv's Sdr is that of the heights
        rescaled to 0-1.
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
    return _hybrid(x, convention, extent, crs, band, nodata)["sdr"]
