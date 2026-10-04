# SPDX-License-Identifier: GPL-3.0-or-later
"""Fuzzy similarity of categorical maps (Hagen 2003), as Dinamica EGO computes it.

Two maps agree at a cell to the degree that the cell's class is found
nearby in the other map: ``2^(-d/2)`` for the nearest cell of the class at
``d`` cells within a square window (Hagen's exponential decay with a
halving distance of 2), or 1 anywhere in the window (constant). Dinamica
EGO's validation compares the maps of differences (each cell that changed,
in its new class; the others empty) of the observed and the simulated
change, in windows of growing size, and keeps the smaller of the two
mean similarities. Measured on Dinamica EGO 8.13 (its code is closed):
the same maps and means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from .._threads import threaded
from ._maps import read_map

__all__ = ["FuzzySimilarity", "differences", "fuzzy_similarity", "similarity_of_differences"]


@dataclass
class FuzzySimilarity:
    """The fuzzy similarity of two maps (:func:`fuzzy_similarity`).

    Attributes
    ----------
    first, second
        The similarity at each cell with a class in the first (second) map
        (NaN elsewhere).
    first_mean, second_mean
        Their means (Dinamica's "first" and "second" similarity).
    window, decay
        The settings.
    """

    first: object
    second: object
    first_mean: float
    second_mean: float
    window: int
    decay: str

    @property
    def minimum(self) -> float:
        """The smaller mean: Dinamica's two-way similarity."""
        return float(np.fmin(self.first_mean, self.second_mean))

    def __repr__(self) -> str:
        return (f"FuzzySimilarity(window={self.window}, first={self.first_mean:.4f}, "
                f"second={self.second_mean:.4f})")


def _codes(a: np.ndarray, b: np.ndarray):
    classes = np.unique(np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]]))
    out = []
    for m in (a, b):
        c = np.full(m.shape, -1, dtype=np.int32)
        ok = ~np.isnan(m)
        c[ok] = np.searchsorted(classes, m[ok]).astype(np.int32)
        out.append(c)
    return out


def _like(values, template):
    return values if template is None else template.copy(data=values)


@threaded
def fuzzy_similarity(first, second, *, window: int = 5, decay: str = "exponential", nodata=None) -> FuzzySimilarity:
    """Fuzzy similarity of two categorical maps (Dinamica EGO's ``CalcReciprocalSimilarityMap``).

    At each cell with a class in ``first``: the membership of that class
    in ``second`` around the cell (``2^(-d/2)`` for the nearest cell of the
    class within the ``window x window`` square, ``d`` in cells; or 1 with
    ``decay="constant"``; 0 if none), the smaller of it and the reverse
    membership where ``second`` has a class too; and the same for the
    cells of ``second``.

    Parameters
    ----------
    first, second
        Categorical maps (cells without a value take no part).
    window
        Side of the square window, odd (1: only the cell itself).
    decay : {"exponential", "constant"}

    Returns
    -------
    FuzzySimilarity

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> a = np.full((5, 5), np.nan); a[2, 1] = 1
    >>> b = np.full((5, 5), np.nan); b[2, 3] = 1
    >>> s = ls.simulate.fuzzy_similarity(a, b, window=5)
    >>> s.first_mean, s.second_mean
    (0.5, 0.5)
    """
    if decay not in ("exponential", "constant"):
        raise ValueError("decay must be 'exponential' or 'constant'")
    if window < 1 or window % 2 == 0:
        raise ValueError("the window must be odd")
    ma, mb = read_map(first, nodata), read_map(second, nodata)
    a, b = ma.as_float(), mb.as_float()
    if a.shape != b.shape:
        raise ValueError("the maps differ in shape")
    ca, cb = _codes(a, b)
    r = _core.sim_similarity(ca, cb, int(window), decay == "exponential")
    f = np.asarray(r["first"], dtype=np.float64)
    s = np.asarray(r["second"], dtype=np.float64)
    return FuzzySimilarity(_like(f, ma.template), _like(s, mb.template), float(r["first_mean"]),
                           float(r["second_mean"]), int(window), decay)


def differences(t0, t1, *, nodata=None):
    """The map of differences: each cell that changed between the maps, in its new class; NaN elsewhere."""
    m0, m1 = read_map(t0, nodata), read_map(t1, nodata)
    a, b = m0.as_float(), m1.as_float()
    if a.shape != b.shape:
        raise ValueError("the maps differ in shape")
    out = np.where((a != b) & ~np.isnan(a) & ~np.isnan(b), b, np.nan)
    return _like(out, m1.template)


@threaded
def similarity_of_differences(t0, observed, simulated, *, windows=(1, 3, 5, 7, 9, 11), decay="exponential",
                              nodata=None) -> pd.DataFrame:
    """Dinamica EGO's validation: fuzzy similarity of the observed and the simulated change, per window.

    The maps of :func:`differences` of the observed and the simulated end
    are compared with :func:`fuzzy_similarity`.

    Returns
    -------
    pandas.DataFrame
        A row per window: ``simulated`` (the mean similarity over the
        simulated changes), ``observed`` (over the observed changes),
        ``minimum`` (the smaller: Dinamica's ``CalcSimilarityOfDifferences``
        result) and the cells of each.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> t0 = np.ones((6, 6))
    >>> obs = t0.copy(); obs[2, 2] = 2
    >>> sim = t0.copy(); sim[2, 3] = 2
    >>> ls.simulate.similarity_of_differences(t0, obs, sim, windows=[1, 3])["minimum"].round(4).tolist()
    [0.0, 0.7071]
    """
    d_obs = differences(t0, observed, nodata=nodata)
    d_sim = differences(t0, simulated, nodata=nodata)
    rows = []
    for w in windows:
        s = fuzzy_similarity(d_sim, d_obs, window=int(w), decay=decay)
        n_sim = int(np.sum(~np.isnan(np.asarray(d_sim, dtype=np.float64))))
        n_obs = int(np.sum(~np.isnan(np.asarray(d_obs, dtype=np.float64))))
        rows.append({"window": int(w), "simulated": s.first_mean, "observed": s.second_mean,
                     "minimum": s.minimum, "simulated_cells": n_sim, "observed_cells": n_obs})
    return pd.DataFrame(rows)

