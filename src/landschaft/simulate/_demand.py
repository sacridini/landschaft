# SPDX-License-Identifier: GPL-3.0-or-later
"""How much land each class needs at each time step: demands for the allocation models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._change import _class_value, _rsum
from ._maps import read_series

__all__ = ["AreaDemand"]


def _approx_extrap(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """Hmisc::approxExtrap: R's approx (linear) inside, the end segments' lines outside."""
    keep = ~pd.Series(x).duplicated().to_numpy()
    x, y = x[keep], y[keep]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    n = x.size
    out = np.empty(xout.size)
    for k, v in enumerate(xout):
        if v < x[0]:
            out[k] = (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0]) + y[0]
        elif v > x[-1]:
            out[k] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) * (v - x[n - 2]) + y[n - 2]
        else:  # R's approx1: bisection, then y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
            i, j = 0, n - 1
            while i < j - 1:
                ij = (i + j) // 2
                if v < x[ij]:
                    j = ij
                else:
                    i = ij
            if v == x[j]:
                out[k] = y[j]
            elif v == x[i]:
                out[k] = y[i]
            else:
                out[k] = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    return out


def _round_sum(x: np.ndarray, ncell: int) -> np.ndarray:
    """lulcc's roundSum: each row scaled to sum to ncell, floored, the remainder to the largest fractions."""
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        y = x[r] / _rsum(x[r]) * ncell
        xint = np.floor(y)
        frac = y - np.floor(y)
        ix = np.argsort(frac, kind="stable")  # R's sort(index.return = TRUE): stable, increasing
        tot = ncell - _rsum(np.floor(y))
        if tot > 0:
            pick = ix[int(len(y) - tot):]
            xint[pick] += 1
        out[r] = xint
    return out


class AreaDemand:
    """Cells each class must cover at each time step (CLUE-S's demand).

    Parameters
    ----------
    table
        A DataFrame with a row per time step (the first is the starting map)
        and a column per class, in cells; or an array with ``classes``.
    classes
        The classes of an array's columns.

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.load_table(ls.simulate.example_data("sibuyan_demand"))
    >>> demand = ls.simulate.AreaDemand(d[d.scenario == "demand1"].drop(columns="scenario").set_index("time")
    ...                                 .rename(columns={"forest": 1, "coconut": 2, "grassland": 3, "rice": 4,
    ...                                                  "other": 5}))
    >>> demand.steps
    14
    """

    def __init__(self, table, classes=None):
        if isinstance(table, pd.DataFrame):
            self.table = table.copy()
            self.table.columns = [_class_value(c) for c in self.table.columns]
        else:
            a = np.asarray(table, dtype=np.float64)
            if classes is None or len(classes) != a.shape[1]:
                raise ValueError("give the class of each column")
            self.table = pd.DataFrame(a, columns=[_class_value(c) for c in classes])
        if len(self.table) < 2:
            raise ValueError("a demand needs at least the start and one step")

    @classmethod
    def from_maps(cls, maps, times=None, out_times=None, *, nodata=None) -> AreaDemand:
        """The areas of observed maps interpolated and extrapolated linearly (lulcc's ``approxExtrapDemand``).

        Each class's area is interpolated (or extrapolated from the first or
        last two maps) at ``out_times``, negative values set to 0, and each
        time's areas rounded to integers that add up to the number of cells
        (the largest fractions rounded up).

        Parameters
        ----------
        maps
            Observed maps (a band per date, or a list).
        times
            Their times (as in :func:`changes`).
        out_times
            Times of the demand (default: every integer time from the first to
            the last map).
        """
        from ._change import changes

        ms, ts = read_series(maps, times, nodata)
        ch = changes(ms, ts)
        areas = ch.areas
        tout = np.arange(ts[0], ts[-1] + 1) if out_times is None else np.asarray(list(out_times), dtype=np.float64)
        x = np.asarray(ts, dtype=np.float64)
        demand = np.empty((tout.size, areas.shape[1]))
        for k, c in enumerate(areas.columns):
            v = _approx_extrap(x, areas[c].to_numpy(dtype=np.float64), np.asarray(tout, dtype=np.float64))
            v[v < 0] = 0
            demand[:, k] = v
        ncell = int(areas.iloc[0].sum())
        demand = _round_sum(demand, ncell)
        return cls(pd.DataFrame(demand, index=pd.Index(tout, name="time"), columns=list(areas.columns)))

    @property
    def classes(self) -> list:
        return list(self.table.columns)

    @property
    def steps(self) -> int:
        return len(self.table) - 1

    def at(self, step: int, classes: list) -> np.ndarray:
        """The demand of each class at a step (0 = the start)."""
        row = self.table.iloc[step]
        return np.array([float(row.get(c, 0.0)) for c in classes])

    def __repr__(self) -> str:
        return f"AreaDemand({self.steps} steps, {len(self.classes)} classes)"
