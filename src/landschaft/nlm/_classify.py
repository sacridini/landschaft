# SPDX-License-Identifier: GPL-3.0-or-later
"""Classification of continuous neutral landscapes (NLMR's ``landscape_classify``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import as_values

__all__ = ["classify"]


def _cumulative_proportions(weighting) -> np.ndarray:
    # util_w2cp(): NA weights share what is left of 1
    # (R's sum() and cumsum() accumulate in 80 bits)
    w = np.array([np.nan if v is None else float(v) for v in weighting], dtype=np.float64)
    na = np.isnan(w)
    if na.any():
        w[na] = (1 - _core.nlm_r_sum(w[~na])) / na.sum()
    if np.any(w < 0):
        raise ValueError("weighting must be non-negative.")
    total = _core.nlm_r_sum(w)
    if abs(total) <= 1.5e-8:  # isTRUE(all.equal(total, 0))
        raise ValueError("weighting must sum to a positive value.")
    return _core.nlm_r_cumsum(w / total)


def _classify_weighting(values: np.ndarray, weighting) -> np.ndarray:
    cp = _cumulative_proportions(weighting)
    x = values[~np.isnan(values)]
    n = x.size
    # util_calc_boundaries(): the value at the cumulative proportion of the
    # sorted cells (as.integer truncates, clamped to 1..n)
    idx = np.clip(np.trunc(cp * n).astype(np.int64), 1, max(n, 1))
    boundaries = np.sort(x)[idx - 1]
    if np.unique(boundaries).size == 1:
        return np.ones_like(values)  # NLMR sets every cell to 1
    return _cut(values, np.concatenate([[0.0], boundaries]))


def _cut(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    breaks = np.sort(np.asarray(breaks, dtype=np.float64))
    if np.unique(breaks).size != breaks.size:
        raise ValueError("'breaks' are not unique (base::cut): classes would be empty")
    return _core.nlm_cut(values, breaks)


def classify(x, n: int | None = None, weighting: Sequence[float] | None = None,
             level_names: Sequence[str] | None = None, real_land=None, mask_val: float | None = None,
             *, n_threads: int | None = None):
    """Classify a continuous landscape into categories (NLMR's ``landscape_classify``).

    Three ways, as NLMR:

    - ``weighting``: class proportions (cumulative, relative to their sum; ``None``
      entries share what is left of 1); the class boundaries are the values at
      those proportions of the sorted cells, the first class starting at 0 (values
      below 0 are left out);
    - ``real_land``: the proportions of the classes of a real landscape (after
      removing ``mask_val``, whose cells are also removed from ``x``);
    - ``n``: Jenks natural breaks into ``n`` classes (at most the number of unique
      values). Jenks' algorithm costs O(n_classes x cells^2): keep it to small maps.

    ``weighting`` wins over ``n`` when both are given.

    Parameters
    ----------
    x : xarray.DataArray or numpy.ndarray
        Continuous values (NaN or the nodata value are missing).
    n : int, optional
        Number of classes for Jenks' breaks.
    weighting : sequence of float, optional
        Class proportions.
    level_names : sequence of str, optional
        Class names, stored in ``attrs["levels"]`` (class value -> name).
    real_land : xarray.DataArray or numpy.ndarray, optional
        A categorical landscape of the same shape whose class proportions are used.
    mask_val : float, optional
        Value of ``real_land`` to ignore (and to remove from ``x``).
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        int32 classes 1, 2, ...; 0 where ``x`` is missing or outside the classes
        (the DataArray's nodata value).

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.nlm.random(20, 20, seed=3)
    >>> c = ls.nlm.classify(r, weighting=[0.5, 0.25, 0.25])
    >>> [int((c == k).sum()) for k in (1, 2, 3)]
    [200, 100, 100]
    """
    values = as_values(x)
    if real_land is not None:
        real = as_values(real_land)
        if real.shape != values.shape:
            raise ValueError("x and real_land must have matching geometry.")
        if mask_val is not None:
            values = np.where(real == mask_val, np.nan, values)
            real = real[real != mask_val]
        real = real[~np.isnan(real)]
        if real.size == 0:
            raise ValueError("real_land contains no values after masking.")
        _, counts = np.unique(real, return_counts=True)
        weighting = counts / counts.sum()
    with _threads.limit(n_threads):
        if weighting is not None:
            classes = _classify_weighting(values, weighting)
        else:
            if n is None:
                raise ValueError("Either n, weighting, or real_land must be supplied.")
            k = int(n)
            if k < 1:
                raise ValueError("n must be a positive integer")
            v = values[~np.isnan(values)]
            k = min(k, np.unique(v).size)
            breaks = _core.nlm_jenks_breaks(np.sort(v), k)
            classes = _cut(values, breaks)
    out = np.where(np.isnan(classes), 0, classes).astype(np.int32)
    attrs = {}
    if level_names is not None:
        present = np.unique(out[out > 0])
        if len(level_names) < present.size:
            raise ValueError("level_names must have at least as many entries as classes.")
        attrs["levels"] = {int(c): str(level_names[int(c) - 1]) for c in present}
    if isinstance(x, xr.DataArray):
        da = x.copy(data=out)
        da.attrs = {**{k: v for k, v in x.attrs.items() if k not in ("_FillValue",)}, **attrs}
        from ..io import _with_georef

        return _with_georef(da, nodata=0)
    return out
