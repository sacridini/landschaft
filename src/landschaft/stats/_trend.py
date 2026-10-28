# SPDX-License-Identifier: GPL-3.0-or-later
"""The monotonic trend of every cell along a series of maps: the Mann-Kendall test and Sen's slope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import wrap

__all__ = ["Trend", "mann_kendall", "trend"]


@dataclass
class Trend(Saveable):
    """Every cell's trend along a series of maps (:func:`trend`).

    Attributes
    ----------
    slope : xarray.DataArray
        Sen's slope, per step of the series (or per unit of ``times``).
    intercept : xarray.DataArray
        Its intercept, at the first step.
    p : xarray.DataArray
        The Mann-Kendall test's two-sided p.
    z : xarray.DataArray
        Its z.
    tau : xarray.DataArray
        Kendall's tau.
    trend : xarray.DataArray
        1 increasing, -1 decreasing, 0 no trend (at ``alpha``).
    """

    slope: xr.DataArray
    intercept: xr.DataArray
    p: xr.DataArray
    z: xr.DataArray
    tau: xr.DataArray
    trend: xr.DataArray

    def __repr__(self) -> str:
        t = np.asarray(self.trend)
        return (f"Trend({int((t == 1).sum())} cells increasing, {int((t == -1).sum())} decreasing, "
                f"{int((t == 0).sum())} without trend)")


def mann_kendall(series, *, alpha: float = 0.05) -> dict:
    """The Mann-Kendall test and Sen's slope of one series (pymannkendall's ``original_test``, the same bits).

    Missing values (NaN) are left out of the test; Sen's slope is the median of
    the slopes between every pair of positions of the series as given.

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.stats.mann_kendall([1.0, 2.1, 2.9, 4.2, 3.8, 5.5, 6.1])
    >>> r["trend"], round(r["slope"], 3), round(r["p"], 4)
    (1.0, 0.85, 0.0069)
    """
    names = ("s", "var_s", "z", "p", "tau", "slope", "intercept", "h", "trend")
    v = _core.stats_mann_kendall(np.ascontiguousarray(series, dtype=np.float64).ravel(), float(alpha))
    return dict(zip(names, v))


def trend(maps, *, alpha: float = 0.05, nodata=None, n_threads: int | None = None) -> Trend:
    """The monotonic trend of every cell along a series of maps (pymannkendall's ``original_test``, the same numbers).

    The Mann-Kendall test (S, its variance corrected for ties, z, the two-sided
    p, Kendall's tau, increasing or decreasing at ``alpha``) and Sen's slope
    (the median of the slopes between every pair of dates, per step of the
    series) with its intercept, cell by cell in C++; the missing values of a
    cell are left out (a cell with fewer than three values gets NaN).

    Parameters
    ----------
    maps
        The maps, a band per date: a 3-D ``xarray.DataArray`` (as
        ``ls.load_raster`` reads a multi-band file) or a list of 2-D maps.
    alpha
        The significance level of ``trend`` (default 0.05).
    nodata
        Value marking missing cells (default: the rasters' no-data value).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    Trend

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(1)
    >>> maps = [np.full((2, 2), float(t)) + rng.normal(0, 0.1, (2, 2)) for t in range(10)]
    >>> maps = [m * np.array([[1, -1], [0, 1]]) for m in maps]      # up, down, flat, up
    >>> t = ls.stats.trend(maps)
    >>> t.trend.values.tolist()
    [[1.0, -1.0], [0.0, 1.0]]
    """
    from ..simulate._maps import read_series

    ms, _ = read_series(maps, None, nodata)
    shape = ms[0].shape
    layers = np.ascontiguousarray(np.stack([m.as_float().ravel() for m in ms]))
    if len(ms) < 3:
        raise ValueError("three dates at least")
    out = _core.stats_trend(layers, float(alpha), 3, resolve(n_threads))
    tpl = ms[0].template
    names = ("slope", "intercept", "p", "z", "tau", "trend")
    return Trend(*(wrap(out[k].reshape(shape), tpl, names[k]) for k in range(6)))
