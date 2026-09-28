# SPDX-License-Identifier: GPL-3.0-or-later
"""The models NLMR computes in C++: midpoint displacement, neighbourhood
clusters and random rectangular clusters (std::mt19937 with libstdc++'s
distributions, reproduced bit for bit)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import check_resolution, check_size, grid, resolve_seed

__all__ = ["mpd", "neigh", "random_rectangular_cluster"]


def _cxx_seed(seed) -> int:
    # NLMR hands user_seed to std::mt19937::seed as a 32-bit unsigned long
    return resolve_seed(seed) % 2**32


def mpd(ncol: int, nrow: int, resolution: float = 1.0, roughness: float = 0.5, rand_dev: float = 1.0,
        seed: int | None = None, torus: bool = False, rescale: bool = True, nlmr_compat: bool = True,
        *, n_threads: int | None = None) -> xr.DataArray:
    """Midpoint displacement neutral landscape model (NLMR's ``nlm_mpd``).

    The diamond-square algorithm: a fractal surface whose spatial autocorrelation
    is set by ``roughness`` (the random deviation shrinks by this factor at every
    halving of the grid).

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows asked for.
    resolution : float, default 1
        Cell size.
    roughness : float, default 0.5
        Between 0 and 1: low values give smooth, strongly autocorrelated surfaces,
        high values rough ones.
    rand_dev : float, default 1
        Standard deviation of the displacement at the first step.
    seed : int, optional
        Seed of NLMR's ``std::mt19937`` (NLMR's ``user_seed``): the same seed gives
        NLMR's landscape. ``None`` draws one, stored in ``attrs["seed"]``.
    torus : bool, default False
        Wrap the square steps around the edges (a periodic surface).
    rescale : bool, default True
        Rescale the values to 0-1.
    nlmr_compat : bool, default True
        Reproduce NLMR: its grid is a square of side ``max(nrow, ncol) + 1`` (made
        odd) and the result drops one cell on each side, so the raster is square
        and ``max(nrow, ncol) - 1`` or ``max(nrow, ncol)`` cells wide, not
        ``nrow x ncol`` (a warning says so). With False the grid is ``2**k + 1``
        cells wide and cropped to ``nrow x ncol``.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> m = ls.nlm.mpd(65, 65, roughness=0.4, seed=3)
    >>> m.shape
    (65, 65)
    >>> ls.nlm.mpd(100, 80, seed=3, nlmr_compat=False).shape
    (80, 100)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    roughness = float(roughness)
    if not 0 <= roughness <= 1:
        raise ValueError(f"roughness must be in [0, 1], got {roughness!r}")
    s = _cxx_seed(seed)
    if nlmr_compat:
        size = max(nrow + 1, ncol + 1)
        if size % 2 == 0:
            size += 1
    else:
        size = 3
        while size - 1 < max(nrow, ncol):
            size = 2 * (size - 1) + 1
    n_steps = math.ceil(math.log2(size - 1))
    devs = [float(rand_dev)]
    for _ in range(1, n_steps):
        devs.append(devs[-1] * roughness)
    with _threads.limit(n_threads):
        full = _core.nlm_mpd(size, devs, s, bool(torus))
        values = np.ascontiguousarray(full[1:-1, 1:-1] if nlmr_compat else full[:nrow, :ncol])
        if rescale:
            values = _core.nlm_rescale(values)
    if nlmr_compat and values.shape != (nrow, ncol):
        warnings.warn(f"mpd: NLMR's grid gives a {values.shape[0]} x {values.shape[1]} raster instead of "
                      f"{nrow} x {ncol}; use nlmr_compat=False for the size asked for", stacklevel=2)
    return grid(values, resolution, "mpd", {"seed": s, "model": "mpd", "roughness": roughness,
                                            "torus": int(bool(torus)), "nlmr_compat": int(bool(nlmr_compat))})


def neigh(ncol: int, nrow: int, p_neigh: float, p_empty: float, resolution: float = 1.0, categories: int = 3,
          neighbourhood: int = 4, proportions: Sequence[float] | None = None, seed: int | None = None,
          rescale: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Neighbourhood-based clusters (NLMR's ``nlm_neigh``, Scherer et al. 2016).

    Cells are visited in a random order and take the current category with
    probability ``p_neigh`` when a neighbour already holds one, ``p_empty``
    otherwise, which grows clusters of adjustable compactness. The edges wrap
    around.

    As in NLMR, categories are filled from ``categories - 1`` down to 0 and
    "holds one" means a neighbour with a non-zero category.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    p_neigh : float
        Probability of a cell joining the category when a neighbour holds one.
    p_empty : float
        Probability when no neighbour does.
    resolution : float, default 1
        Cell size.
    categories : int, default 3
        Number of categories (values 0 .. categories - 1).
    neighbourhood : {4, 8}, default 4
        Neighbours looked at (8 adds the diagonals when the 4 are empty).
    proportions : sequence of float, optional
        Share of the cells of each category (summing to 1); by default equal
        shares, ``floor(ncol * nrow / categories)`` cells each (the rest stays 0).
    seed : int, optional
        Seed of NLMR's ``std::mt19937`` (NLMR's ``user_seed``); ``None`` draws one.
    rescale : bool, default True
        Rescale the values to 0-1; else the categories as int32.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 (rescaled) or int32 categories.

    Examples
    --------
    >>> import landschaft as ls
    >>> n = ls.nlm.neigh(60, 40, p_neigh=0.7, p_empty=0.1, categories=4, seed=2, rescale=False)
    >>> sorted(int(v) for v in set(n.values.ravel()))
    [0, 1, 2, 3]
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    p_neigh, p_empty = float(p_neigh), float(p_empty)
    for name, p in (("p_neigh", p_neigh), ("p_empty", p_empty)):
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {p!r}")
    if p_neigh == 0 and p_empty == 0:
        raise ValueError("p_neigh and p_empty cannot both be 0: no cell would ever be filled")
    categories = int(categories)
    if categories < 1:
        raise ValueError("categories must be a positive integer")
    if neighbourhood not in (4, 8):
        raise ValueError("neighbourhood must be 4 or 8")
    if proportions is not None:
        props = np.asarray(proportions, dtype=np.float64)
        if props.size != categories:
            raise ValueError("proportions must have one value per category")
        if _core.nlm_r_sum(props) != 1:  # NLMR: sum(proportions) == 1
            raise ValueError("proportions must sum to 1")
        # proportions * nrow * ncol, truncated to integers
        cells = [int(p * nrow * ncol) for p in props]
    else:
        cells = [int(math.floor(nrow * ncol / categories))] * categories
    s = _cxx_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_neigh(nrow, ncol, categories - 1, cells, int(neighbourhood), np.float32(p_neigh),
                                 np.float32(p_empty), s)
        if rescale:
            values = _core.nlm_rescale(values)
        else:
            values = values.astype(np.int32)
    return grid(values, resolution, "neigh", {"seed": s, "model": "neigh", "p_neigh": p_neigh, "p_empty": p_empty,
                                              "categories": categories})


def random_rectangular_cluster(ncol: int, nrow: int, minl: int, maxl: int, resolution: float = 1.0,
                               seed: int | None = None, rescale: bool = True,
                               *, n_threads: int | None = None) -> xr.DataArray:
    """Random rectangular clusters (NLMR's ``nlm_randomrectangularcluster``, Gaucherel 2008).

    Rectangles with random sides between ``minl`` and ``maxl`` cells and a random
    value are laid over each other (wrapping around the edges) until every cell is
    covered.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    minl, maxl : int
        Smallest and largest side of the rectangles, in cells (at most ``ncol``
        and ``nrow``).
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of NLMR's ``std::mt19937`` (NLMR's ``user_seed``); ``None`` draws one.
    rescale : bool, default True
        Rescale the values to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.nlm.random_rectangular_cluster(50, 30, minl=3, maxl=8, seed=1)
    >>> r.shape, float(r.min()), float(r.max())
    ((30, 50), 0.0, 1.0)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    minl, maxl = int(minl), int(maxl)
    if not (1 <= minl <= maxl and maxl <= ncol and maxl <= nrow):
        raise ValueError("need 1 <= minl <= maxl <= min(ncol, nrow)")
    s = _cxx_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_random_rectangular_cluster(nrow, ncol, minl, maxl, s)
        if rescale:
            values = _core.nlm_rescale(values)
    return grid(values, resolution, "random_rectangular_cluster",
                {"seed": s, "model": "random_rectangular_cluster", "minl": minl, "maxl": maxl})
