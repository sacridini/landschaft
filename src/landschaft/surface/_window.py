# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface metrics in a moving window: texture images (geodiv's texture_image
and focal_metrics)."""

from __future__ import annotations

from typing import Iterable

import xarray as xr

from .. import _core
from .._threads import threaded
from ..cost._transition import _crs_lonlat
from ._grid import read_surface

__all__ = ["window_metrics"]

# In the order of the C++ WindowMetric ids.
WINDOW_METRICS = ("sa", "sq", "ssk", "sku", "sv", "sph", "smean", "sdq", "sdq6", "sdr", "sds", "ssc", "s10z",
                  "sbi", "sci", "svi", "sk", "spk", "svk", "sal", "str", "std", "stdi", "srw", "srwi", "shw", "sfd")


@threaded
def window_metrics(x, metrics: str | Iterable[str], window: int = 5, shape: str = "square", *, extent=None,
                   crs=None, band=None, nodata=None) -> xr.Dataset:
    """Surface metrics in a moving window: one texture image per metric.

    Every cell gets the value of each metric over the ``window`` x ``window``
    cells around it, computed exactly as :func:`metrics` computes it for a
    raster, on a raster that covers the window. Cells beyond the edges of the raster are
    missing in the windows that reach them.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The surface (a single band; NaN or ``nodata`` = missing).
    metrics : str or list of str
        Names from :func:`list_metrics`.
    window : int, default 5
        Side of the window in cells, odd.
    shape : {"square", "circle"}, default "square"
        ``"circle"``: only the cells within ``window // 2`` cells of the
        centre (the others are missing).
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
    xarray.Dataset
        One variable per metric on the grid of ``x``.

    Notes
    -----
    Windows run in parallel (``n_threads``). ``sv``, ``sph`` and ``sds`` on
    square windows take shortcuts (running extremes, a summed-area table of
    the peaks) that give the same values; the other metrics cost what they
    cost on a raster the size of the window, times the number of cells. The
    bearing-area metrics (``sbi``...``svk``) and the spatial ones (``sal``...
    ``sfd``) are expensive per window (``sk``, ``spk`` and ``svk`` average 6000
    runs of 4001 slopes each): use them on small rasters.

    geodiv's own ``texture_image`` does not work with current terra (it reads
    the raster as one column), its ``focal_metrics`` transposes every window
    and its circular windows mask nothing (their radius is the window's
    side): here each window gets the metric computed on it as on a raster.

    Examples
    --------
    >>> import landschaft as ls
    >>> z = ls.surface.example_data("normforest")
    >>> tex = ls.surface.window_metrics(z, ["sa", "sku", "sds"], window=7)
    >>> list(tex.data_vars), tex["sa"].shape
    (['sa', 'sku', 'sds'], (371, 371))
    """
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [n for n in names if n not in WINDOW_METRICS]
    if unknown or not names:
        raise ValueError(f"unknown surface metric(s) {', '.join(map(repr, unknown))}; see ls.surface.list_metrics()"
                         if unknown else "no metrics given")
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd number of cells, 3 or more, got {window}")
    if shape not in ("square", "circle"):
        raise ValueError(f"shape must be 'square' or 'circle', got {shape!r}")
    s = read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)
    ids = [WINDOW_METRICS.index(n) for n in names]
    images = _core.surf_window(s.values, tuple(map(float, s.extent)), _crs_lonlat(s.crs), window,
                               shape == "circle", ids)
    return xr.Dataset({n: s.raster(v, n) for n, v in zip(names, images)},
                      attrs={"window": window, "shape": shape})
