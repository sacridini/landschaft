# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistics of a stack's layers at every cell, and the correlation of two maps in a moving window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import read_values, wrap

__all__ = ["LocalCorrelation", "cell_stats", "local_correlation"]

_ARG = {"argmax": 14, "argmin": 15}


def cell_stats(maps, stat: str = "mean", *, nodata=None, n_threads: int | None = None) -> xr.DataArray:
    """A statistic of a stack's layers at every cell (ArcGIS's Cell Statistics, terra's ``app``).

    Over the layers with a value at the cell: the statistics of
    :func:`landschaft.zonal_stats` (``"mean"``, ``"std"``, ``"median"``,
    ``"majority"``, ``"p90"``...), or the layer (1.., in the stack's order) of
    the largest or smallest value (``"argmax"``, ``"argmin"``; a tie: the
    first). The month of the peak NDVI, the scenario of the highest
    suitability, the agreement of an ensemble. As Terraflux's
    ``stack.reduce``; C++ with OpenMP.

    Parameters
    ----------
    maps
        A ``(band, y, x)`` DataArray, or a list of maps on one grid.
    stat
        ``"mean"`` (default), ``"std"``, ``"var"`` (of the population),
        ``"min"``, ``"max"``, ``"range"``, ``"sum"``, ``"median"``, ``"pNN"``
        (a percentile, numpy's linear), ``"count"`` (layers with a value),
        ``"nodata"`` (without), ``"majority"``, ``"minority"`` (a tie: the
        smallest value), ``"unique"``, ``"argmax"``, ``"argmin"``.
    nodata
        Value marking missing cells (default: the rasters' no-data value).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        The statistic on the maps' grid (NaN where no layer has a value; 0 for
        ``count`` and ``unique``).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> s = np.array([[[1.0, 5.0]], [[3.0, np.nan]], [[2.0, 7.0]]])     # 3 layers of 1 x 2 cells
    >>> ls.stats.cell_stats(s, "argmax").values.tolist(), ls.stats.cell_stats(s, "mean").values.tolist()
    ([[2.0, 3.0]], [[2.0, 6.0]])
    """
    from .._zonal import _parse_stats
    from ..simulate._maps import read_series

    if isinstance(maps, np.ndarray) and maps.ndim == 3:
        maps = list(maps)
    layers, _ = read_series(maps, None, nodata)
    key = str(stat).lower()
    if key in _ARG:
        code, q = _ARG[key], 0.0
    else:
        _, codes, qs = _parse_stats([stat])
        code, q = codes[0], qs[0]
    shape = layers[0].shape
    data = np.ascontiguousarray(np.stack([m.as_float().ravel() for m in layers]))
    out = _core.stats_cell_stats(data, code, q, resolve(n_threads))
    return wrap(out.reshape(shape), layers[0].template, key)


@dataclass
class LocalCorrelation(Saveable):
    """The correlation and least squares line of two maps in a moving window (:func:`local_correlation`).

    Attributes
    ----------
    correlation : xarray.DataArray
        Pearson's correlation in the window.
    slope, intercept : xarray.DataArray
        The least squares line of ``y`` on ``x``.
    r2 : xarray.DataArray
        The share of ``y``'s variance the line explains.
    """

    correlation: xr.DataArray
    slope: xr.DataArray
    intercept: xr.DataArray
    r2: xr.DataArray

    def __repr__(self) -> str:
        r = np.asarray(self.correlation)
        return f"LocalCorrelation(mean correlation {np.nanmean(r) if np.isfinite(r).any() else np.nan:.3f})"


def local_correlation(x, y, *, radius: int = 3, shape: str = "square", min_cells: int = 3, nodata=None,
                      n_threads: int | None = None) -> LocalCorrelation:
    """The correlation and the least squares line between two maps in a moving window (terra's ``focalPairs``, ``focalReg``).

    Around every cell, over the cells of the window where both maps have a
    value: Pearson's correlation, the slope and intercept of ``y`` on ``x``
    and r² (geographically local regression). No value where a map has none
    at the cell, or the window fewer than ``min_cells`` pairs. As Terraflux's
    ``raster.local_correlation``; C++ with OpenMP.

    Parameters
    ----------
    x, y
        Two maps on one grid.
    radius
        The window's radius in cells (default 3: 7 x 7 cells).
    shape
        ``"square"`` (default) or ``"circle"`` (the cells within the radius).
    min_cells
        The fewest pairs of a window (default 3, at least 2).
    nodata
        Value marking missing cells.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    LocalCorrelation

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> x = np.arange(25.0).reshape(5, 5)
    >>> lc = ls.stats.local_correlation(x, 2 * x + 1, radius=1)
    >>> float(lc.correlation[2, 2]), float(lc.slope[2, 2]), float(lc.intercept[2, 2])
    (1.0, 2.0, 1.0)
    """
    if shape not in ("square", "circle"):
        raise ValueError(f"shape: square or circle, not {shape!r}")
    if int(radius) < 1:
        raise ValueError("radius: 1 or more")
    xv, tpl = read_values(x, nodata, "x")
    yv, _ = read_values(y, nodata, "y")
    if xv.shape != yv.shape:
        raise ValueError(f"x and y differ in shape: {xv.shape} and {yv.shape}")
    out = _core.stats_local_correlation(xv, yv, int(radius), shape == "circle", int(min_cells), resolve(n_threads))
    return LocalCorrelation(*(wrap(out[k], tpl, name) for k, name in
                              enumerate(("correlation", "slope", "intercept", "r2"))))
