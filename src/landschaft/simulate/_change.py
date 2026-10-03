# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysing land change between maps: transitions, areas, intensity, change counts.

The transition tables follow OpenLand's ``contingencyTable`` (Exavier &
Zeilhofer 2020), the intensity analysis its ``intensityAnalysis`` (Aldwaik
& Pontius 2012), the change count its ``acc_changes``, the class areas
lulcc's ``total``: the same numbers, sums included (R sums doubles in 80
bits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import threaded
from ._maps import Map, cell_area_of, read_map, read_series

__all__ = ["Changes", "IntensityAnalysis", "changes", "crosstab"]


def _rsum(x) -> float:
    """R's sum() of doubles (80-bit accumulation, in the given order)."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    return float(_core.nlm_r_sum(a)) if a.size else 0.0


def _class_value(v: float):
    return int(v) if float(v).is_integer() else float(v)


def _classes(maps: list[Map]) -> np.ndarray:
    found = [_core.sim_map_classes(m.values, m.nodata) for m in maps]
    return np.unique(np.concatenate(found)) if found else np.empty(0)


def _count(a: Map, b: Map, classes: np.ndarray) -> np.ndarray:
    return _core.sim_crosstab(a.values, b.values, classes, classes, a.nodata, b.nodata)


def _matrix_frame(counts: np.ndarray, classes: np.ndarray, labels: dict | None, drop_empty: bool) -> pd.DataFrame:
    keep = np.ones(classes.size, dtype=bool)
    if drop_empty:
        keep = (counts.sum(axis=1) > 0) | (counts.sum(axis=0) > 0)
    c = classes[keep]
    names = [labels.get(_class_value(v), _class_value(v)) if labels else _class_value(v) for v in c]
    df = pd.DataFrame(counts[np.ix_(keep, keep)], index=pd.Index(names, name="from"),
                      columns=pd.Index(names, name="to"))
    return df


@threaded
def crosstab(from_map, to_map, *, nodata=None, labels: dict | None = None) -> pd.DataFrame:
    """Cells of each class of one map that hold each class of another (a transition matrix of counts).

    Rows are the classes of ``from_map`` (the earlier map, or the
    comparison map), columns those of ``to_map`` (the later map, or the
    reference); only cells with a value in both maps count. The classes are
    those found in such cells, in increasing order (the set
    ``diffeR::crosstabm`` and ``raster::crosstab`` use).

    Parameters
    ----------
    from_map, to_map
        Categorical maps (``xarray.DataArray`` or 2-D arrays) on the same grid.
    nodata
        Value marking missing cells (default: the rasters' no-data value;
        NaN is always missing).
    labels
        Names of the classes (``{value: name}``) for the rows and columns.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    pandas.DataFrame
        Counts (int64), index ``from``, columns ``to``.

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    >>> m = ls.simulate.crosstab(lu[0], lu[1])
    >>> int(m.values.sum()), m.shape
    (7226, (5, 5))
    """
    a, b = read_map(from_map, nodata), read_map(to_map, nodata)
    if a.shape != b.shape:
        raise ValueError(f"the maps differ in shape: {a.shape} and {b.shape}")
    classes = _classes([a, b])
    return _matrix_frame(_count(a, b, classes), classes, labels, drop_empty=True)


@dataclass
class IntensityAnalysis:
    """Intensity analysis of land change (Aldwaik & Pontius 2012), as OpenLand computes it.

    Every rate is a percentage per time unit (per year when the times are
    years). Areas are in km² (``area=True``) or cells.

    Attributes
    ----------
    interval : pandas.DataFrame
        Per period: ``PercentChange`` (changed area, % of the study area),
        ``St`` (annual change intensity), ``U`` (uniform intensity over the
        whole series).
    category_gain, category_loss : pandas.DataFrame
        Per period and class: gross gain (``GG``) or loss (``GL``), its
        intensity ``Gtj`` / ``Lti`` and the period's ``St``.
    transition_gain : pandas.DataFrame
        Transitions to the class ``gain`` from each other class: area
        ``T_i2n``, intensity ``Rtin`` and the uniform intensity ``Wtn``.
    transition_loss : pandas.DataFrame
        Transitions from the class ``loss`` to each other class: area
        ``T_m2j``, intensity ``Qtmj`` and the uniform intensity ``Vtm``.
    category_gain_stationarity, category_loss_stationarity, transition_gain_stationarity,
    transition_loss_stationarity : pandas.DataFrame
        In how many periods each class is active / dormant (category level)
        or targeted / avoided (transition level), and ``Test`` = ``"Y"``
        when that holds in every period.
    """

    interval: pd.DataFrame
    category_gain: pd.DataFrame
    category_loss: pd.DataFrame
    transition_gain: pd.DataFrame
    transition_loss: pd.DataFrame
    category_gain_stationarity: pd.DataFrame
    category_loss_stationarity: pd.DataFrame
    transition_gain_stationarity: pd.DataFrame
    transition_loss_stationarity: pd.DataFrame
    gain: object = None
    loss: object = None


def _group_sum(df: pd.DataFrame, keys: list[str], col: str, name: str) -> pd.DataFrame:
    """dplyr's group_by(keys) %>% summarise(name = sum(col)): groups sorted, R sums in row order."""
    if df.empty:
        return pd.DataFrame(columns=[*keys, name])
    rows = []
    for key, g in df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append((*key, _rsum(g[col].to_numpy())))
    return pd.DataFrame(rows, columns=[*keys, name])


def _stationarity(df: pd.DataFrame, by: str, rate: str, ref: str, count: str, above: str, below: str) -> pd.DataFrame:
    n = df["Period"].nunique()
    parts = []
    for label, mask in ((above, df[rate] > df[ref]), (below, df[rate] < df[ref])):
        sub = df[mask]
        for key, g in sub.groupby(by, sort=True):
            parts.append({by: key, count: len(g), "N": n, "Stationarity": label, "Test": "Y" if len(g) == n else "N"})
    return pd.DataFrame(parts, columns=[by, count, "N", "Stationarity", "Test"])


@dataclass
class Changes:
    """Land change along a series of categorical maps (see :func:`changes`)."""

    maps: list[Map]
    times: list
    classes: np.ndarray
    cell_area: float
    labels: dict | None = None
    _pairs: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ transitions

    def _counts(self, i: int, j: int) -> np.ndarray:
        if (i, j) not in self._pairs:
            self._pairs[(i, j)] = _count(self.maps[i], self.maps[j], self.classes)
        return self._pairs[(i, j)]

    def _index(self, t) -> int:
        try:
            return self.times.index(t)
        except ValueError:
            raise ValueError(f"no map at time {t!r}; times: {self.times}") from None

    def _table(self, pairs) -> pd.DataFrame:
        rows = []
        for i, j in pairs:
            c = self._counts(i, j)
            t0, t1 = self.times[i], self.times[j]
            for r, s in zip(*np.nonzero(c)):
                rows.append((f"{t0}-{t1}", _class_value(self.classes[r]), _class_value(self.classes[s]),
                             int(c[r, s]), t1 - t0, t0, t1))
        df = pd.DataFrame(rows, columns=["Period", "From", "To", "cells", "Interval", "yearFrom", "yearTo"])
        df.insert(3, "km2", df["cells"].to_numpy(dtype=np.float64) * self.cell_area / 1e6)
        return df

    @property
    def table(self) -> pd.DataFrame:
        """Transitions between consecutive maps, one row per pair of classes (OpenLand's ``lulc_Multistep``).

        Columns ``Period`` ("t0-t1"), ``From``, ``To``, ``km2`` (cells times
        the cell area, over 10⁶), ``cells``, ``Interval``, ``yearFrom``,
        ``yearTo``; pairs without cells are left out.
        """
        return self._table([(k, k + 1) for k in range(len(self.maps) - 1)])

    @property
    def one_step(self) -> pd.DataFrame:
        """Transitions from the first map to the last (OpenLand's ``lulc_Onestep``)."""
        return self._table([(0, len(self.maps) - 1)])

    @property
    def total_area(self) -> dict:
        """Area of the first period's cells: ``{"km2": ..., "cells": ...}`` (OpenLand's ``totalArea``)."""
        t = self.table
        first = min(t["Period"].unique())
        g = t[t["Period"] == first]
        return {"km2": _rsum(g["km2"].to_numpy()), "cells": int(g["cells"].sum())}

    @property
    def total_interval(self):
        """From the first time to the last."""
        return self.times[-1] - self.times[0]

    def matrix(self, t0=None, t1=None) -> pd.DataFrame:
        """Transition matrix (cells) from the map at ``t0`` to the map at ``t1`` (default: first to last)."""
        i = 0 if t0 is None else self._index(t0)
        j = len(self.maps) - 1 if t1 is None else self._index(t1)
        return _matrix_frame(self._counts(i, j), self.classes, self.labels, drop_empty=True)

    @property
    def areas(self) -> pd.DataFrame:
        """Cells of each class (columns) in each map (rows, by time), as lulcc's ``total``."""
        rows = []
        for m in self.maps:
            c = _core.sim_crosstab(m.values, m.values, self.classes, self.classes, m.nodata, m.nodata)
            rows.append(np.diag(c))
        names = [self.labels.get(_class_value(v), _class_value(v)) if self.labels else _class_value(v)
                 for v in self.classes]
        return pd.DataFrame(np.array(rows), index=pd.Index(self.times, name="time"), columns=names)

    # ------------------------------------------------------------------ intensity

    def intensity(self, gain, loss, *, area: bool = True) -> IntensityAnalysis:
        """Intensity analysis at the interval, category and transition levels (OpenLand's ``intensityAnalysis``).

        Parameters
        ----------
        gain
            The class whose gains the transition level examines (OpenLand's
            ``category_n``): which classes it takes over more intensively
            than uniformly.
        loss
            The class whose losses the transition level examines
            (``category_m``): which classes take it over.
        area
            Areas in km² (default) or in cells.

        Returns
        -------
        IntensityAnalysis
        """
        lulc = self.table
        col = "km2" if area else "cells"
        unit = "km2" if area else "pixel"
        if not area:
            lulc = lulc.assign(cells=lulc["cells"].astype(np.float64))
        ta = self.total_area
        ae = ta["km2"] if area else float(ta["cells"])
        changed = lulc[lulc["From"] != lulc["To"]]

        # interval level
        eq1 = _group_sum(changed, ["Period", "Interval"], col, "intch")
        eq1["PercentChange"] = eq1["intch"] / ae * 100
        eq1["St"] = eq1["intch"] / (eq1["Interval"] * ae) * 100
        u = _rsum(changed[col].to_numpy()) / (self.total_interval * ae) * 100
        interval = eq1[["Period", "PercentChange", "St"]].assign(U=u)
        st = dict(zip(eq1["Period"], eq1["St"]))

        # category level
        def category(side: str, rate: str, name: str) -> pd.DataFrame:
            num = _group_sum(changed, ["Period", side, "Interval"], col, "num")
            den = _group_sum(lulc, ["Period", side], col, "den")
            df = num.merge(den, on=["Period", side], how="left")
            df[rate] = df["num"] / (df["den"] * df["Interval"]) * 100
            df["St"] = df["Period"].map(st)
            return df.rename(columns={"num": f"{name}_{unit}"})[["Period", side, "Interval", f"{name}_{unit}",
                                                                  rate, "St"]]

        cat_gain = category("To", "Gtj", "GG")
        cat_loss = category("From", "Lti", "GL")

        # transition level: gains of `gain` from each class
        to_n = changed[changed["To"] == gain]
        num05 = _group_sum(to_n, ["Period", "From", "Interval"], col, "num05")
        den05 = _group_sum(lulc[lulc["From"] != gain], ["Period", "From"], col, "den05")
        eq5 = num05.merge(den05, on=["Period", "From"], how="left")
        eq5["Rtin"] = eq5["num05"] / (eq5["Interval"] * eq5["den05"]) * 100
        num06 = _group_sum(to_n, ["Period", "To", "Interval"], col, "num06")
        den06 = _group_sum(lulc[lulc["From"] != gain], ["Period"], col, "den06")
        eq6 = num06.merge(den06, on="Period", how="left")
        eq6["Wtn"] = eq6["num06"] / (eq6["Interval"] * eq6["den06"]) * 100
        tr_gain = eq5.merge(eq6, on=["Period", "Interval"], how="left")
        tr_gain = tr_gain.rename(columns={"num05": f"T_i2n_{unit}"})[
            ["Period", "From", "To", "Interval", f"T_i2n_{unit}", "Rtin", "Wtn"]]

        # transition level: losses of `loss` to each class
        from_m = changed[changed["From"] == loss]
        num07 = _group_sum(from_m, ["Period", "To", "Interval"], col, "num07")
        den07 = _group_sum(lulc[lulc["To"] != loss], ["Period", "To"], col, "den07")
        eq7 = num07.merge(den07, on=["Period", "To"], how="left")
        eq7["Qtmj"] = eq7["num07"] / (eq7["Interval"] * eq7["den07"]) * 100
        num08 = _group_sum(from_m, ["Period", "From", "Interval"], col, "num08")
        den08 = _group_sum(lulc[lulc["To"] != loss], ["Period"], col, "den08")
        eq8 = num08.merge(den08, on="Period", how="left")
        eq8["Vtm"] = eq8["num08"] / (eq8["Interval"] * eq8["den08"]) * 100
        tr_loss = eq7.merge(eq8, on=["Period", "Interval"], how="left")
        tr_loss = tr_loss.rename(columns={"num07": f"T_m2j_{unit}"})[
            ["Period", "To", "From", "Interval", f"T_m2j_{unit}", "Qtmj", "Vtm"]]

        return IntensityAnalysis(
            interval=interval,
            category_gain=cat_gain,
            category_loss=cat_loss,
            transition_gain=tr_gain,
            transition_loss=tr_loss,
            category_gain_stationarity=_stationarity(cat_gain, "To", "Gtj", "St", "Gain", "Active Gain",
                                                     "Dormant Gain"),
            category_loss_stationarity=_stationarity(cat_loss, "From", "Lti", "St", "Loss", "Active Loss",
                                                     "Dormant Loss"),
            transition_gain_stationarity=_stationarity(tr_gain, "From", "Rtin", "Wtn", "Loss",
                                                       f"targeted by {gain}", f"avoided by {gain}"),
            transition_loss_stationarity=_stationarity(tr_loss, "To", "Qtmj", "Vtm", "Gain", f"targeted {loss}",
                                                       f"avoided {loss}"),
            gain=gain,
            loss=loss,
        )

    # ------------------------------------------------------------------ change count

    def change_count(self) -> tuple[xr.DataArray | np.ndarray, pd.DataFrame]:
        """How many times each cell changed along the series (OpenLand's ``acc_changes``).

        Returns
        -------
        (map, table)
            The count per cell (int32, -1 where any map is missing; a
            DataArray on the maps' grid when they came as rasters) and a
            table of ``PxValue``, ``Qt`` (cells) and ``Percent``.
        """
        n = _core.sim_change_count([m.values for m in self.maps],
                                   self.maps[0].nodata).reshape(self.maps[0].shape)
        values, counts = np.unique(n[n >= 0], return_counts=True)
        table = pd.DataFrame({"PxValue": values.astype(int), "Qt": counts.astype(int)})
        table["Percent"] = table["Qt"] / int(counts.sum()) * 100
        t = self.maps[0].template
        if t is not None:
            from ..io import _with_georef

            out = xr.DataArray(n, coords={k: t.coords[k] for k in t.dims}, dims=t.dims, name="changes")
            try:
                out = _with_georef(out, transform=t.rio.transform(), nodata=-1)
                if t.rio.crs is not None:
                    out = _with_georef(out, crs=t.rio.crs)
            except Exception:
                pass
            return out, table
        return n, table

    # ------------------------------------------------------------------ links

    def components(self, t0=None, t1=None, *, digits: int | None = None):
        """Gain, persistence, loss and the quantity, exchange and shift components of the change.

        :func:`difference_components` of :meth:`matrix` with ``analysis="change"``.
        """
        from ._difference import difference_components

        return difference_components(self.matrix(t0, t1), analysis="change", digits=digits)

    def markov(self, t0=None, t1=None, *, steps: int = 1):
        """The Markov chain of the change from ``t0`` to ``t1`` (see :class:`Markov`)."""
        from ._markov import Markov

        i = 0 if t0 is None else self._index(t0)
        j = len(self.maps) - 1 if t1 is None else self._index(t1)
        return Markov.from_counts(self._counts(i, j), self.classes, steps=steps)


@threaded
def changes(maps, times=None, *, cell_area: float | None = None, nodata=None, labels: dict | None = None) -> Changes:
    """Land change along a series of categorical maps.

    The transitions between consecutive maps and from the first to the last,
    the area of each class, the intensity analysis of Aldwaik & Pontius
    (2012) and the number of changes per cell: the analyses of OpenLand
    (``contingencyTable``, ``intensityAnalysis``, ``acc_changes``), lulcc
    (``crossTabulate``, ``total``) and diffeR (``crosstabm``,
    ``diffTablej``), with the same numbers.

    Parameters
    ----------
    maps
        A 3-D ``xarray.DataArray`` with a band per date (as
        ``ls.load_raster`` reads a multi-band file) or a list of 2-D maps
        on the same grid.
    times
        The date of each map (default: the band names when they are numbers,
        else 0, 1, 2...).
    cell_area
        Area of a cell in square map units (default: the product of the
        resolutions; ``km2`` columns are cells times this over 10⁶, so give
        it in m²).
    nodata
        Value marking missing cells (default: the rasters' no-data value).
    labels
        Names of the classes, ``{value: name}``, for matrices and tables.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    Changes
        With ``table``, ``one_step``, ``total_area``, ``areas``,
        ``matrix()``, ``intensity()``, ``components()``,
        ``change_count()`` and ``markov()``.

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    >>> ch = ls.simulate.changes(lu)
    >>> ch.times
    [1985, 1991, 1999]
    >>> ia = ch.intensity(gain=2, loss=1)       # gains of built land, losses of forest
    >>> ia.interval.columns.tolist()
    ['Period', 'PercentChange', 'St', 'U']
    """
    ms, ts = read_series(maps, times, nodata)
    return Changes(ms, ts, _classes(ms), cell_area_of(ms[0], cell_area), labels)
