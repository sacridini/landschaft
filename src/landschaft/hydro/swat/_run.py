# SPDX-License-Identifier: GPL-3.0-or-later
"""Running a SWAT+ project."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ... import _core
from ..._save import Saveable
from ._project import SwatProject
from ._tables import TABLES

__all__ = ["SwatResult", "run", "WEATHER_COLUMNS", "TABLES", "STEPS"]

# climate_module weather_daily, in the order the port (and the reference's dump) keeps them
WEATHER_COLUMNS = ("precip", "precip_next", "tmax", "tmin", "tave", "solrad", "solradmx", "rhum", "dewpt", "windsp",
                   "pet", "wndir", "phubase0", "ppet", "daylength", "precip_half_hr")

#: The print steps of SWAT+ (print.prt's columns), in the order of the record tags.
STEPS = ("day", "month", "year", "aa")


def _dates(keys: np.ndarray) -> pd.DatetimeIndex:
    """Dates of (year, day of year) pairs."""
    return pd.to_datetime(keys[:, 0].astype(str), format="%Y") + pd.to_timedelta(keys[:, 1] - 1, unit="D")


def _frame(keys: np.ndarray, vals: np.ndarray, columns, obj: str) -> pd.DataFrame:
    dates = _dates(keys)
    if obj == "basin":
        index = pd.DatetimeIndex(dates, name="date")
    else:
        index = pd.MultiIndex.from_arrays([dates, keys[:, 2]], names=["date", obj])
    return pd.DataFrame(vals[:, : len(columns)], index=index, columns=list(columns))


@dataclass
class SwatResult(Saveable):
    """What :func:`run` simulated.

    Attributes
    ----------
    weather : pandas.DataFrame
        The daily weather of each station (index ``date``, ``station``): precipitation
        (mm), maximum, minimum and mean temperature (deg C), solar radiation and its
        maximum (MJ/m2), relative humidity (fraction), dew point (deg C), wind speed
        (m/s), Hargreaves potential ET (mm), heat units, day length (h) and the
        fraction of the day's rain in its wettest half hour, measured or generated.
    tables : dict
        What SWAT+ printed (its print.prt, see :meth:`SwatProject.set_print`):
        ``{table: {step: DataFrame}}``, tables ``hru_wb``, ``hru_nb``, ``hru_ls``,
        ``hru_pw`` (water, nutrient, losses and plant/weather balances of each HRU),
        ``basin_wb``... (the same for the basin), ``channel_sd`` (SWAT-DEG channels:
        water body, storage, inflow and outflow, water temperature) and ``aquifer``;
        steps ``day``, ``month``, ``year``, ``aa`` (average annual). Columns as SWAT+
        names them in its output files (units in ``units``); index ``date`` (the day
        SWAT+ printed the row: the day, the last day of the month or year, the end of
        the simulation) and the object (``hru``, ``channel``, ``aquifer``: SWAT+'s
        numbers), the basin tables by date only. Monthly and yearly values are sums
        of the daily ones, or means for states (soil water, CN, LAI, biomass...), as
        SWAT+ prints them.
    units : dict
        ``{table: {column: unit}}``.
    warnings : list
        What SWAT+ would write to ``diagnostics.out`` (gages not found, ...).
    """

    weather: pd.DataFrame
    tables: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def table(self, name: str, step: str = "year") -> pd.DataFrame:
        """One printed table: ``result.table("hru_wb", "month")``."""
        if name not in TABLES:
            raise KeyError(f"unknown table {name!r}: one of {', '.join(TABLES)}")
        if step not in STEPS:
            raise KeyError(f"unknown step {step!r}: one of {', '.join(STEPS)}")
        got = self.tables.get(name, {})
        if step not in got:
            raise KeyError(f"SWAT+ did not print {name} by {step}: choose it with "
                           f"project.set_print({{{name!r}: {step!r}}}) or run(..., tables=...)")
        return got[step]

    def __repr__(self) -> str:
        days = self.weather.index.get_level_values("date").unique()
        printed = ", ".join(f"{t}[{'/'.join(s)}]" for t, s in self.tables.items())
        return f"SwatResult({len(days)} days, tables: {printed or 'none'})"


def _run_raw(project: SwatProject, dump=True) -> dict:
    if not isinstance(project, SwatProject):
        raise TypeError(f"expected a SwatProject (ls.hydro.swat.load_txtinout), got {type(project).__name__}")
    return _core.swat_run(project.files, dump)


def run(project: SwatProject, *, tables=None, nyskip: int | None = None) -> SwatResult:
    """Simulate a SWAT+ project, as SWAT+ 62.0.1 does.

    Parameters
    ----------
    project : SwatProject
        The model (:func:`load_txtinout`).
    tables : dict or list, optional
        What to print, overriding the project's print.prt for this run (see
        :meth:`SwatProject.set_print`), e.g. ``{"hru_wb": ["month", "year"],
        "channel_sd": "day"}``.
    nyskip : int, optional
        Years simulated before printing, overriding print.prt.

    Returns
    -------
    SwatResult

    Examples
    --------
    >>> import landschaft as ls
    >>> project = ls.hydro.swat.load_txtinout("TxtInOut")  # doctest: +SKIP
    >>> result = ls.hydro.swat.run(project, tables={"hru_ls": "year", "channel_sd": "day"})  # doctest: +SKIP
    >>> result.table("hru_ls", "year")["sedyld"]  # doctest: +SKIP
    """
    if tables is not None or nyskip is not None:
        project = project.copy()
        project.set_print(tables, nyskip=nyskip)
    raw = _run_raw(project, dump=[1])
    for w in raw["warnings"]:
        warnings.warn(f"SWAT+: {w}", stacklevel=2)
    keys, vals = raw["dump"].get(1, (np.zeros((0, 3), np.int32), np.zeros((0, len(WEATHER_COLUMNS)), np.float32)))
    stations = np.asarray(raw["stations"], dtype=object)
    index = pd.MultiIndex.from_arrays([_dates(keys), stations[keys[:, 2] - 1] if len(keys) else []],
                                      names=["date", "station"])
    weather = pd.DataFrame(vals[:, : len(WEATHER_COLUMNS)], index=index, columns=list(WEATHER_COLUMNS))
    out, units = {}, {}
    for name, (tag, obj, columns, us) in TABLES.items():
        for k, step in enumerate(STEPS):
            rec = raw["prints"].get(tag + k)
            if rec is not None:
                out.setdefault(name, {})[step] = _frame(rec[0], rec[1], columns, obj)
        if name in out:
            units[name] = dict(zip(columns, us))
    return SwatResult(weather=weather, tables=out, units=units, warnings=list(raw["warnings"]))
