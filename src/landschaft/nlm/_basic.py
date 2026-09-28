# SPDX-License-Identifier: GPL-3.0-or-later
"""Random, percolation and gradient neutral landscape models (NLMR)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import as_values, check_resolution, check_size, grid, resolve_seed

__all__ = ["distance_gradient", "edge_gradient", "percolation", "planar_gradient", "random", "rescale"]


def rescale(x, *, n_threads: int | None = None):
    """Linearly rescale values to 0-1 (NLMR's ``util_rescale``).

    ``(x - min) / (max - min)`` over the non-missing cells; a raster whose minimum
    and maximum are equal within ``all.equal``'s tolerance (1.5e-8) becomes 0.

    Parameters
    ----------
    x : xarray.DataArray or numpy.ndarray
        Values; NaN (or the DataArray's nodata value) is missing and stays NaN.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The rescaled values (float64), of the same kind as ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> ls.nlm.rescale(np.array([[2.0, 4.0], [6.0, np.nan]]))
    array([[0. , 0.5],
           [1. , nan]])
    """
    values = as_values(x)
    with _threads.limit(n_threads):
        out = _core.nlm_rescale(values)
    if isinstance(x, xr.DataArray):
        da = x.copy(data=out)
        return da.rio.write_nodata(np.nan, encoded=False) if hasattr(da, "rio") else da
    return out


def random(ncol: int, nrow: int, resolution: float = 1.0, seed: int | None = None, rescale: bool = True,
           *, n_threads: int | None = None) -> xr.DataArray:
    """Spatially random neutral landscape model (NLMR's ``nlm_random``).

    Every cell gets an independent uniform value.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``): the same seed gives the same
        landscape as NLMR. ``None`` draws one, stored in ``attrs["seed"]``.
    rescale : bool, default True
        Rescale the values to 0-1 (:func:`rescale`).
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values on a grid with origin (0, 0) and the given resolution.

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.nlm.random(50, 40, seed=1)
    >>> r.shape, float(r.min()), float(r.max())
    ((40, 50), 0.0, 1.0)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_random(s, nrow, ncol)
        if rescale:
            values = _core.nlm_rescale(values)
    return grid(values, resolution, "random", {"seed": s, "model": "random"})


def percolation(ncol: int, nrow: int, resolution: float = 1.0, prob: float = 0.5, seed: int | None = None,
                *, n_threads: int | None = None) -> xr.DataArray:
    """Percolation neutral landscape model (NLMR's ``nlm_percolation``).

    Each cell is 1 with probability ``prob`` (a Bernoulli draw), else 0.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    prob : float, default 0.5
        Probability of a cell being 1.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``); ``None`` draws one.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        uint8 0/1 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> p = ls.nlm.percolation(100, 100, prob=0.3, seed=7)
    >>> round(float(p.mean()), 2)
    0.3
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    prob = float(prob)
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be in [0, 1], got {prob!r}")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_percolation(s, nrow, ncol, prob)
    return grid(values.astype(np.uint8), resolution, "percolation", {"seed": s, "model": "percolation", "prob": prob})


def _direction(direction, seed):
    s = resolve_seed(seed)
    if direction is None or (isinstance(direction, float) and np.isnan(direction)):
        # set.seed(user_seed); direction <- runif(1, 0, 360)
        (d,) = _core.nlm_r_sequence(s, [("runif", 1, 0.0, 360.0)])
        return float(d[0]), s
    return float(direction), s


def planar_gradient(ncol: int, nrow: int, resolution: float = 1.0, direction: float | None = None,
                    seed: int | None = None, rescale: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Planar gradient neutral landscape model (NLMR's ``nlm_planargradient``).

    Values increase linearly in one direction across the landscape.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    direction : float, optional
        Direction of the gradient in degrees, clockwise from north: the values
        grow northwards with 0 and eastwards with 90. ``None`` draws it with
        ``runif(1, 0, 360)`` after ``set.seed(seed)``.
    seed : int, optional
        Seed of R's generator, used only when ``direction`` is None.
    rescale : bool, default True
        Rescale the values to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values; ``attrs["direction"]`` holds the direction used.

    Examples
    --------
    >>> import landschaft as ls
    >>> g = ls.nlm.planar_gradient(5, 4, direction=90)
    >>> g.values[0].round(2)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    d, s = _direction(direction, seed)
    with _threads.limit(n_threads):
        values = _core.nlm_planar_gradient(nrow, ncol, d, bool(rescale), False)
    return grid(values, resolution, "planar_gradient", {"seed": s, "model": "planar_gradient", "direction": d})


def edge_gradient(ncol: int, nrow: int, resolution: float = 1.0, direction: float | None = None,
                  seed: int | None = None, rescale: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Edge gradient neutral landscape model (NLMR's ``nlm_edgegradient``).

    A planar gradient folded around its middle: values peak along a central
    band and fall to both edges (``1 - 2 |0.5 - g|`` of the rescaled planar
    gradient ``g``).

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    direction : float, optional
        Direction of the gradient in degrees; ``None`` draws it with
        ``runif(1, 0, 360)`` after ``set.seed(seed)``.
    seed : int, optional
        Seed of R's generator, used only when ``direction`` is None.
    rescale : bool, default True
        Rescale the values to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values; ``attrs["direction"]`` holds the direction used.

    Examples
    --------
    >>> import landschaft as ls
    >>> e = ls.nlm.edge_gradient(5, 3, direction=90)
    >>> e.values[0].round(2)
    array([0. , 0.5, 1. , 0.5, 0. ])
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    d, s = _direction(direction, seed)
    with _threads.limit(n_threads):
        values = _core.nlm_planar_gradient(nrow, ncol, d, bool(rescale), True)
    return grid(values, resolution, "edge_gradient", {"seed": s, "model": "edge_gradient", "direction": d})


def distance_gradient(ncol: int, nrow: int, origin, resolution: float = 1.0, rescale: bool = True,
                      nlmr_compat: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Distance gradient neutral landscape model (NLMR's ``nlm_distancegradient``).

    The distance of every cell to a rectangular origin.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    origin : sequence of 4 ints
        The origin rectangle, 1-based and inclusive. With ``nlmr_compat=True``
        as NLMR applies it: ``(first row, last row, first column, last column)``
        (NLMR documents ``xmin, xmax, ymin, ymax`` but indexes the raster as
        ``[rows, columns]``); otherwise ``(first column, last column, first row,
        last row)``.
    resolution : float, default 1
        Cell size.
    rescale : bool, default True
        Rescale the values to 0-1.
    nlmr_compat : bool, default True
        Reproduce NLMR exactly: ``terra::distance`` treats a raster of at most
        360 x 90 cells as longitude/latitude (terra's default CRS for such an
        extent) and returns great-circle distances in metres (haversine on a
        6378137 m sphere); larger rasters get GDAL's proximity algorithm (in
        cell units, float32). With False, exact Euclidean distances between
        cell centres times ``resolution``.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 distances (rescaled to 0-1 by default).

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.nlm.distance_gradient(5, 5, origin=(3, 3, 1, 1), nlmr_compat=False, rescale=False)
    >>> d.values[0]
    array([2., 1., 0., 1., 2.])
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    o = [int(v) for v in origin]
    if len(o) != 4:
        raise ValueError("origin must have 4 values")
    if nlmr_compat:
        r0, r1, c0, c1 = o
        if not (o[1] <= ncol and o[3] <= nrow):  # NLMR's checks
            raise ValueError("origin[2] must be <= ncol and origin[4] <= nrow")
    else:
        c0, c1, r0, r1 = o
    if not (1 <= r0 <= r1 <= nrow and 1 <= c0 <= c1 <= ncol):
        raise ValueError(f"origin {tuple(o)} is outside the {nrow} x {ncol} raster")
    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    mask[r0 - 1 : r1, c0 - 1 : c1] = 1
    with _threads.limit(n_threads):
        if not nlmr_compat:
            values = _core.nlm_distance(mask, "exact", resolution)
        elif ncol <= 360 and nrow <= 90:
            # terra::rast(ncol, nrow, xmin = 0, xmax = ncol, ...) is lon/lat
            values = _core.nlm_distance(mask, "haversine", 1.0, 0.0, float(nrow), 1.0, 1.0)
        else:
            values = _core.nlm_distance(mask, "gdal", 1.0)
        if rescale:
            values = _core.nlm_rescale(values)
    return grid(values, resolution, "distance_gradient",
                {"model": "distance_gradient", "origin": tuple(o), "nlmr_compat": int(bool(nlmr_compat))})
