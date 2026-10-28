# SPDX-License-Identifier: GPL-3.0-or-later
"""Carbon emissions and removals of land change over a series of maps (the IPCC's stock-difference method)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve

__all__ = ["Emissions", "emissions"]

_FORMS = {"chapman_richards": 1, "linear": 2}


@dataclass
class Emissions(Saveable):
    """The carbon emitted and removed along a series of land use maps (:func:`emissions`).

    Attributes
    ----------
    periods : pandas.DataFrame
        A row per period: ``year_from``, ``year_to``, ``emissions``, ``removals``
        and ``net`` (emissions minus removals), in tonnes of CO2 (or of carbon,
        with ``co2=False``).
    transitions : pandas.DataFrame
        A row per period and pair of classes with cells: ``year_from``,
        ``year_to``, ``from``, ``to``, ``area_ha``, ``emissions``, ``removals``.
    stocks : xarray.DataArray
        Every cell's stock at every date (``time``, y, x), tC/ha; NaN where a
        map has no class.
    net : xarray.DataArray
        Every cell's net emission over the series, tonnes (of CO2 or C) per
        hectare; NaN where no period has the cell at both dates.
    units : str
        ``"t CO2"`` or ``"t C"``.
    """

    periods: pd.DataFrame
    transitions: pd.DataFrame
    stocks: xr.DataArray
    net: xr.DataArray
    units: str = "t CO2"

    def __repr__(self) -> str:
        p = self.periods
        return (f"Emissions({len(p)} periods, emissions {p['emissions'].sum():.6g}, "
                f"removals {p['removals'].sum():.6g} {self.units})")


def _per_class(table, what: str, columns=None) -> dict:
    """``{class: value}`` (or ``{class: {column: value}}``) from a dict or a DataFrame."""
    if table is None:
        return {}
    if isinstance(table, pd.Series):
        return {float(k): float(v) for k, v in table.items()}
    if isinstance(table, pd.DataFrame):
        df = table.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        key = next((c for c in ("class", "lucode", "code") if c in df.columns), None)
        if key is not None:
            df = df.set_index(key)
        if columns is None:
            if df.shape[1] != 1:
                raise ValueError(f"{what}: a column of values by class (and the class column)")
            return {float(k): float(v) for k, v in df.iloc[:, 0].items()}
        return {float(k): {c: row[c] for c in df.columns if c in columns and pd.notna(row[c])}
                for k, row in df.iterrows()}
    if isinstance(table, dict):
        return {float(k): v for k, v in table.items()}
    raise TypeError(f"{what} is a dict by class or a DataFrame, not {type(table).__name__}")


def _label(v: float):
    return int(v) if float(v).is_integer() else float(v)


def emissions(maps, stock, *, regrowth=None, co2: bool = True, times=None, cell_area: float | None = None,
              nodata=None, n_threads: int | None = None) -> Emissions:
    """The carbon emitted and removed by land change over a series of maps (the IPCC's stock-difference method).

    Every cell's carbon stock at every date comes from its class: a stock per
    class (tC/ha), or, for classes that grow back (secondary forest), a curve
    of regrowth from the date the class appeared in the cell. Between two
    dates a fall of the stock is an emission and a rise a removal. Transitions
    happen at the later date of a period.

    Parameters
    ----------
    maps
        The land use maps: a 3-D ``xarray.DataArray`` with a band per date (as
        ``ls.load_raster`` reads a multi-band file) or a list of 2-D maps on one
        grid.
    stock
        The stock of each class, tC/ha: ``{class: stock}``, or a DataFrame with
        a ``class`` (or ``lucode``) column and a column of stocks.
    regrowth
        The classes that grow from the date they appear in a cell:
        ``{class: {"max": ..., "rate": ..., "shape": 1, "form": "chapman_richards"}}``
        (or a DataFrame with those columns and ``class``). Chapman-Richards:
        ``max (1 - exp(-rate age))^shape``; ``"linear"``: ``rate`` tC/ha a year
        up to ``max``. A class of regrowth already in a cell at the first date
        (or after a date without a class) has its ``stock`` entry, else its max.
    co2
        Tonnes of CO2 (carbon times 44/12; default), else of carbon.
    times
        The dates of the maps (default: the band names when they are numbers,
        else 0, 1, 2...).
    cell_area
        A cell's area in m² (default: the product of the resolutions).
    nodata
        Value marking missing cells (default: the rasters' no-data value).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    Emissions

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> y2000 = np.array([[1, 1], [2, 3]])             # 1 forest, 2 pasture, 3 secondary forest
    >>> y2010 = np.array([[2, 1], [3, 3]])             # a forest cleared, a pasture left to regrow
    >>> e = ls.services.emissions([y2000, y2010, y2010], {1: 150, 2: 10, 3: 40}, times=[2000, 2010, 2020],
    ...                           regrowth={3: {"max": 120, "rate": 0.05}}, co2=False, cell_area=10000)
    >>> e.periods.round(2)
       year_from  year_to  emissions  removals     net
    0       2000     2010      150.0      0.00  150.00
    1       2010     2020        0.0     47.22  -47.22
    >>> e.stocks.sel(time=2020).values.round(2).tolist()   # ten years of regrowth: 47.22 tC/ha
    [[10.0, 150.0], [47.22, 40.0]]
    """
    from ..simulate._maps import cell_area_of, read_series

    threads = resolve(n_threads)
    ms, ts = read_series(maps, times, nodata)
    stock = {k: float(v) for k, v in _per_class(stock, "stock").items()}
    regrowth = _per_class(regrowth, "regrowth", columns=("max", "rate", "shape", "form"))
    for c, r in regrowth.items():
        if not isinstance(r, dict) or "max" not in r or "rate" not in r:
            raise ValueError(f"regrowth of the class {_label(c)}: give max and rate")
        if r.get("form", "chapman_richards") not in _FORMS:
            raise ValueError(f"regrowth: form {r.get('form')!r} is not chapman_richards or linear")
    shape = ms[0].shape
    vals = [m.as_float() for m in ms]
    present = np.unique(np.concatenate([v[~np.isnan(v)] for v in vals])) if vals else np.empty(0)
    for c in present:
        if float(c) not in stock and float(c) not in regrowth:
            raise ValueError(f"the class {_label(c)} has no stock nor regrowth")
    classes = present.astype(np.float64)
    k = len(classes)
    cls = np.empty((len(ms), shape[0] * shape[1]), dtype=np.int32)
    for d, v in enumerate(vals):
        flat = v.ravel()
        idx = np.searchsorted(classes, flat)
        idx[np.isnan(flat)] = -1
        cls[d] = idx
    st = np.array([stock.get(float(c), np.nan) for c in classes])
    form = np.array([_FORMS[regrowth[float(c)].get("form", "chapman_richards")] if float(c) in regrowth else 0
                     for c in classes], dtype=np.int32)
    rmax, rate, rshape = (np.array([float(regrowth[float(c)].get(key, 1.0)) if float(c) in regrowth else 0.0
                                    for c in classes]) for key in ("max", "rate", "shape"))
    tarr = np.asarray(ts, dtype=np.float64)
    stocks = _core.serv_emission_stocks(cls, tarr, st, form, rmax, rate, rshape, threads)
    factor = 44.0 / 12.0 if co2 else 1.0
    ha = cell_area_of(ms[0], cell_area) / 10000.0
    net, per, trans = _core.serv_emission_flows(stocks, cls, k, factor, ha, threads)

    periods = pd.DataFrame({"year_from": ts[:-1], "year_to": ts[1:], "emissions": per[:, 0],
                            "removals": per[:, 1], "net": per[:, 0] - per[:, 1]})
    rows = []
    for d in range(len(ts) - 1):
        cells = trans[d, :, :, 0]
        for i, j in zip(*np.nonzero(cells)):
            rows.append((ts[d], ts[d + 1], _label(classes[i]), _label(classes[j]), cells[i, j] * ha,
                         trans[d, i, j, 1], trans[d, i, j, 2]))
    transitions = pd.DataFrame(rows, columns=["year_from", "year_to", "from", "to", "area_ha", "emissions",
                                              "removals"])
    tpl = ms[0].template
    if tpl is not None:
        layers = [tpl.copy(data=stocks[d].reshape(shape)) for d in range(len(ts))]
        stack = xr.concat(layers, dim=pd.Index(ts, name="time")).rename("stocks")
        net_da = tpl.copy(data=net.reshape(shape)).rename("net")
        for da in (stack, net_da):
            da.attrs = {}
        try:
            stack = stack.rio.write_nodata(np.nan)
            net_da = net_da.rio.write_nodata(np.nan)
        except Exception:  # noqa: BLE001, S110  (no rioxarray georeference)
            pass
    else:
        stack = xr.DataArray(stocks.reshape(len(ts), *shape), dims=("time", "y", "x"), coords={"time": ts},
                             name="stocks")
        net_da = xr.DataArray(net.reshape(shape), dims=("y", "x"), name="net")
    return Emissions(periods, transitions, stack, net_da, "t CO2" if co2 else "t C")
