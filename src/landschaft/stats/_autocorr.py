# SPDX-License-Identifier: GPL-3.0-or-later
"""The spatial autocorrelation of a map over the contiguity of its cells: global and local Moran's I, Gi*."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import read_values, wrap

__all__ = ["LISA", "GetisOrd", "Moran", "getis_ord", "lisa", "moran"]


def _queen(contiguity: str) -> bool:
    if contiguity not in ("queen", "rook"):
        raise ValueError(f"contiguity: queen or rook, not {contiguity!r}")
    return contiguity == "queen"


@dataclass
class Moran(Saveable):
    """The global Moran's I of a map (:func:`moran`).

    Attributes
    ----------
    i : float
        Moran's I: how much neighbouring cells are alike (1 alike, 0 at random,
        -1 opposite).
    ei : float
        Its expectation under no autocorrelation, -1 / (n - 1).
    vi_norm, z_norm, p_norm : float
        Its variance, z and two-sided p under normality.
    vi_rand, z_rand, p_rand : float
        The same under randomisation.
    p_sim : float
        The pseudo p of the random permutations (NaN without them).
    n : int
        The cells with a value.
    """

    i: float
    ei: float
    vi_norm: float
    z_norm: float
    p_norm: float
    vi_rand: float
    z_rand: float
    p_rand: float
    p_sim: float
    n: int

    def __repr__(self) -> str:
        return f"Moran(I={self.i:.4f}, z_rand={self.z_rand:.3f}, p_rand={self.p_rand:.3g}, p_sim={self.p_sim:.3g})"


def moran(map, *, contiguity: str = "queen", permutations: int = 999, seed: int = 1, nodata=None,
          n_threads: int | None = None) -> Moran:
    """The global Moran's I of a map (PySAL's ``esda.Moran``, to rounding).

    How much neighbouring cells are alike, over the contiguity of the cells
    with a value (row-standardised weights); its expectation, variance, z and
    two-sided p under normality and under randomisation, and a pseudo p of
    random permutations of the values (numpy's ``default_rng(seed).shuffle``).

    Parameters
    ----------
    map
        The values (a 2-D DataArray or array; NaN and ``nodata`` are missing).
    contiguity
        The neighbours: ``"queen"`` (8, default) or ``"rook"`` (4).
    permutations
        The random permutations of the pseudo p (default 999; 0 none).
    seed
        The seed of the permutations.
    nodata
        Value marking missing cells.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    Moran

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> m = ls.stats.moran(np.add.outer(np.arange(20.0), np.arange(20.0)), permutations=99)
    >>> round(m.i, 3), m.p_sim
    (0.983, 0.01)
    """
    values, _ = read_values(map, nodata)
    v = _core.stats_moran(values, _queen(contiguity), int(permutations), int(seed), resolve(n_threads))
    return Moran(*v[:9], int(v[9]))


@dataclass
class LISA(Saveable):
    """The local Moran's I of every cell (:func:`lisa`).

    Attributes
    ----------
    i : xarray.DataArray
        The local Moran's I.
    quadrant : xarray.DataArray
        1 high among high, 2 low among high, 3 low among low, 4 high among low.
    z : xarray.DataArray
        Its z under randomisation.
    p : xarray.DataArray
        The pseudo p of the conditional permutations (NaN without them).
    clusters : xarray.DataArray
        The quadrant of the cells with ``p`` below ``alpha``, 0 elsewhere.
    ei, vi : xarray.DataArray
        Its expectation and variance under randomisation.
    """

    i: xr.DataArray
    quadrant: xr.DataArray
    z: xr.DataArray
    p: xr.DataArray
    clusters: xr.DataArray
    ei: xr.DataArray
    vi: xr.DataArray

    def __repr__(self) -> str:
        c = np.asarray(self.clusters)
        return (f"LISA({int((c == 1).sum())} high-high, {int((c == 3).sum())} low-low, "
                f"{int((c == 2).sum())} low-high, {int((c == 4).sum())} high-low cells)")


def lisa(map, *, contiguity: str = "queen", permutations: int = 99, alpha: float = 0.05, seed: int = 1,
         nodata=None, n_threads: int | None = None) -> LISA:
    """The local Moran's I of every cell (Anselin's LISA; PySAL's ``esda.Moran_Local``, to rounding).

    How much a cell and its neighbours are alike, its quadrant, its z under
    randomisation, a pseudo p of conditional permutations (its neighbours
    drawn at random among the other cells, a generator per cell from
    ``seed``: the same with any number of threads), and the clusters: the
    quadrant of the cells whose p is below ``alpha``.

    Parameters
    ----------
    map
        The values.
    contiguity
        ``"queen"`` (default) or ``"rook"``.
    permutations
        Per cell (default 99; 0 none).
    alpha
        The significance of the clusters (default 0.05).
    seed
        The seed of the permutations.
    nodata
        Value marking missing cells.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    LISA

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> x = np.zeros((20, 20)); x[:8, :8] = 10; x[12:, 12:] = -10
    >>> lm = ls.stats.lisa(x + np.random.default_rng(0).normal(0, 0.5, x.shape))
    >>> int((lm.clusters == 1).sum()) > 30, int((lm.clusters == 3).sum()) > 30
    (True, True)
    """
    values, tpl = read_values(map, nodata)
    i, q, ei, vi, z, p = _core.stats_lisa(values, _queen(contiguity), int(permutations), int(seed),
                                          resolve(n_threads))
    clusters = np.where(np.isnan(q), np.nan, np.where(p < alpha, q, 0.0))
    return LISA(wrap(i, tpl, "lisa"), wrap(q, tpl, "quadrant"), wrap(z, tpl, "z"), wrap(p, tpl, "p"),
                wrap(clusters, tpl, "clusters"), wrap(ei, tpl, "ei"), wrap(vi, tpl, "vi"))


@dataclass
class GetisOrd(Saveable):
    """Getis and Ord's Gi* of every cell (:func:`getis_ord`).

    Attributes
    ----------
    g : xarray.DataArray
        Gi*: the share of the map's total around the cell, itself included.
    z : xarray.DataArray
        Its z.
    p : xarray.DataArray
        Its one-sided p (of ``|z|``).
    spots : xarray.DataArray
        3, 2, 1: hot spots at 99, 95 and 90 % confidence; -1 to -3 cold; 0 none
        (as ArcGIS's ``Gi_Bin``).
    """

    g: xr.DataArray
    z: xr.DataArray
    p: xr.DataArray
    spots: xr.DataArray

    def __repr__(self) -> str:
        s = np.asarray(self.spots)
        return f"GetisOrd({int((s > 0).sum())} hot, {int((s < 0).sum())} cold cells)"


def getis_ord(map, *, contiguity: str = "queen", nodata=None, n_threads: int | None = None) -> GetisOrd:
    """Getis and Ord's Gi* of every cell (PySAL's ``esda.G_Local`` with ``star`` and row-standardised weights).

    The share of the map's total around a cell, itself included, its z and p,
    and the hot and cold spots at 90, 95 and 99 % confidence (as ArcGIS's
    ``Gi_Bin``). Meant for positive values.

    Parameters
    ----------
    map
        The values (positive).
    contiguity
        ``"queen"`` (default) or ``"rook"``.
    nodata
        Value marking missing cells.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    GetisOrd

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> x = np.ones((15, 15)); x[5:10, 5:10] = 5
    >>> int(ls.stats.getis_ord(x).spots.values[7, 7])
    3
    """
    values, tpl = read_values(map, nodata)
    g, z, p, s = _core.stats_getis_ord(values, _queen(contiguity), resolve(n_threads))
    return GetisOrd(wrap(g, tpl, "g"), wrap(z, tpl, "z"), wrap(p, tpl, "p"), wrap(s, tpl, "spots"))
