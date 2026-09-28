# SPDX-License-Identifier: GPL-3.0-or-later
"""NLMR's cluster models written in R: random clusters and curds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import check_resolution, check_size, grid, resolve_seed

__all__ = ["curds", "random_cluster"]


def random_cluster(ncol: int, nrow: int, p: float, ai: Sequence[float] = (0.5, 0.5), neighbourhood: int = 4,
                   resolution: float = 1.0, seed: int | None = None, rescale: bool = True,
                   *, n_threads: int | None = None) -> xr.DataArray:
    """Random cluster neutral landscape model (NLMR's ``nlm_randomcluster``, Saura & Martínez-Millán 2000).

    A percolation map with probability ``p`` is split into clusters of adjacent
    cells, each cluster gets a type drawn with the weights ``ai``, and every
    remaining cell takes the most frequent type among its 8 neighbours (a random
    one of the tied types; a type drawn with ``ai`` when no neighbour has one).

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    p : float
        Share of the cells in the initial percolation map (up to 1).
    ai : sequence of float, default (0.5, 0.5)
        Weights of the types (their share of the clusters).
    neighbourhood : {4, 8}, default 4
        Adjacency of the cells of a cluster.
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``): the same seed gives NLMR's
        landscape. ``None`` draws one, stored in ``attrs["seed"]``.
    rescale : bool, default True
        Rescale the types to 0-1; else the types 1, 2, ... as int32.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 (rescaled) or int32 types.

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.nlm.random_cluster(60, 40, p=0.4, ai=[0.25, 0.25, 0.5], seed=4, rescale=False)
    >>> sorted(int(v) for v in set(r.values.ravel()))
    [1, 2, 3]
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    p = float(p)
    if not p <= 1:
        raise ValueError(f"p must be <= 1, got {p!r}")
    weights = [float(a) for a in ai]
    if not weights or any(not np.isfinite(a) or a < 0 for a in weights) or not any(a > 0 for a in weights):
        raise ValueError("ai must hold non-negative weights, at least one positive")
    if neighbourhood not in (4, 8):
        raise ValueError("neighbourhood must be 4 or 8")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_random_cluster(s, nrow, ncol, p, weights, int(neighbourhood))
        if rescale:
            values = _core.nlm_rescale(values)
        else:
            values = values.astype(np.int32)
    return grid(values, resolution, "random_cluster", {"seed": s, "model": "random_cluster", "p": p})


def curds(curds: Sequence[float], recursion_steps: Sequence[int], wheyes: Sequence[float] | None = None,
          resolution: float = 1.0, seed: int | None = None, *, n_threads: int | None = None) -> xr.DataArray:
    """Hierarchical curdling (NLMR's ``nlm_curds``, Keitt 2000).

    Starting from one cell, each step splits every cell into
    ``recursion_steps[i]`` x ``recursion_steps[i]`` cells and turns a share
    ``1 - curds[i]`` of the habitat cells into matrix; with ``wheyes``, a share
    ``wheyes[i]`` of the matrix cells goes back to habitat (the "wheyed curds"
    model). The raster has ``prod(recursion_steps)`` cells a side.

    Parameters
    ----------
    curds : sequence of float
        Share of the habitat kept at each step.
    recursion_steps : sequence of int
        Subdivision of the cells at each step (same length as ``curds``).
    wheyes : sequence of float, optional
        Share of the matrix turned back into habitat at each step.
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``); ``None`` draws one.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        uint8: 1 habitat (curds), 0 matrix.

    Examples
    --------
    >>> import landschaft as ls
    >>> c = ls.nlm.curds([0.5, 0.3, 0.6], [32, 6, 2], seed=1)
    >>> c.shape
    (384, 384)
    """
    resolution = check_resolution(resolution)
    p = [float(v) for v in curds]
    steps = [int(v) for v in recursion_steps]
    if len(p) != len(steps):
        raise ValueError("Length of p and s differs.")
    if any(v < 1 for v in steps):
        raise ValueError("recursion_steps must be positive integers")
    w = None if wheyes is None else [float(v) for v in wheyes]
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_curds(s, p, steps, w)
    return grid(values.astype(np.uint8), resolution, "curds", {"seed": s, "model": "curds"})
