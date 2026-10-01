# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers from landscapetools, NLMR's companion package: merging and
binarising neutral landscapes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._classify import classify
from ._common import as_values

__all__ = ["binarize", "merge"]


def merge(primary, secondary, scaling: float = 1.0, rescale: bool = True, *, n_threads: int | None = None):
    """Merge neutral landscapes (landscapetools' ``util_merge``).

    ``primary + scaling * sum(secondary)``: a coarse structure (say a gradient)
    with finer ones laid over it (say a random field), with ``scaling`` setting
    their weight.

    Parameters
    ----------
    primary : xarray.DataArray or numpy.ndarray
        The primary landscape.
    secondary : DataArray, ndarray or sequence of them
        One or more landscapes of the same shape.
    scaling : float, default 1
        Weight of the secondary landscapes.
    rescale : bool, default True
        Rescale the result to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The merged landscape, of the same kind as ``primary``.

    Examples
    --------
    >>> import landschaft as ls
    >>> g = ls.nlm.planar_gradient(100, 100, direction=45)
    >>> r = ls.nlm.random(100, 100, seed=1)
    >>> m = ls.nlm.merge(g, r, scaling=0.5)
    >>> float(m.min()), float(m.max())
    (0.0, 1.0)
    """
    p = as_values(primary)
    layers = [secondary] if isinstance(secondary, (np.ndarray, xr.DataArray)) else list(secondary)
    if not layers:
        raise ValueError("secondary must hold at least one landscape")
    stack = [as_values(layer) for layer in layers]
    for s in stack:
        if s.shape != p.shape:
            raise ValueError("the landscapes must have the same shape")
    total = stack[0].copy()
    for s in stack[1:]:
        total = total + s
    out = p + total * float(scaling)
    if rescale:
        with _threads.limit(n_threads):
            out = _core.nlm_rescale(out)
    if isinstance(primary, xr.DataArray):
        da = primary.copy(data=out)
        da.attrs = {k: v for k, v in primary.attrs.items() if k not in ("_FillValue",)}
        da.attrs["model"] = "merge"
        from ..io import _with_georef

        return _with_georef(da, nodata=np.nan)
    return out


def binarize(x, breaks: float | Sequence[float], *, n_threads: int | None = None):
    """Habitat / matrix maps at given habitat shares (landscapetools' ``util_binarize``).

    Each break ``b`` classifies ``x`` with :func:`classify` into the proportions
    ``1 - b`` (class 1, "Matrix", the lowest values) and ``b`` (class 2,
    "Habitat").

    Parameters
    ----------
    x : xarray.DataArray or numpy.ndarray
        A continuous landscape.
    breaks : float or sequence of float
        Habitat shares, between 0 and 1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        int32 classes 1 (matrix) and 2 (habitat); with several breaks, stacked
        along a new first dimension ``p`` (the breaks).

    Examples
    --------
    >>> import landschaft as ls
    >>> f = ls.nlm.fbm(100, 100, fract_dim=1.5, seed=4)
    >>> h = ls.nlm.binarize(f, [0.2, 0.5])
    >>> [int((h.sel(p=b) == 2).sum()) for b in (0.2, 0.5)]
    [2000, 5000]
    """
    multi = not np.isscalar(breaks)
    bs = [float(b) for b in (breaks if multi else [breaks])]
    if any(not 0 <= b <= 1 for b in bs):
        raise ValueError("breaks must be between 0 and 1")
    maps = [classify(x, weighting=[1 - b, b], level_names=["Matrix", "Habitat"], n_threads=n_threads) for b in bs]
    if not multi:
        return maps[0]
    if isinstance(x, xr.DataArray):
        return xr.concat(maps, dim="p").assign_coords(p=bs)
    return np.stack(maps)
