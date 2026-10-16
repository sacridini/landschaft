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

__all__ = ["SwatProjectWarning", "SwatResult", "run", "run_many", "WEATHER_COLUMNS", "TABLES", "STEPS", "C_FACTORS"]

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

    Examples
    --------
    >>> import landschaft as ls
    >>> project = ls.hydro.swat.load_txtinout(ls.example_data("swat_hubbard_brook"))
    >>> result = ls.hydro.swat.run(project, tables={"basin_wb": ["month", "year"]}, nyskip=1)
    >>> result
    SwatResult(1096 days, tables: hru_wb[day/month/year/aa], hru_nb[day/month/year/aa], hru_ls[day/month/year/aa], hru_pw[day/month/year/aa], basin_wb[month/year], basin_nb[day/month/year/aa], basin_ls[day/month/year/aa], basin_pw[day/month/year/aa], channel_sd[day/month/year/aa], aquifer[day/month/year/aa])
    >>> result.table("basin_wb", "year")[["precip", "et", "surq_gen", "perc"]].astype(float).round(1)
                precip     et  surq_gen   perc
    date                                      
    2001-12-31  1004.1  128.3     356.0  536.7
    2002-12-31  1319.2  126.5     532.1  594.6
    >>> result.units["basin_wb"]["et"]
    'mm'
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


class SwatProjectWarning(UserWarning):
    """A project that SWAT+ runs without a word but whose input looks wrong (landschaft's
    own checks, not SWAT+'s diagnostics)."""


def _parameters_bsn(project: SwatProject) -> str:
    """The name of the basin parameters file (file.cio's basin line)."""
    if "file.cio" in project:
        for line in project.text("file.cio").splitlines():
            p = line.split()
            if len(p) >= 3 and p[0] == "basin":
                return p[2]
    return "parameters.bsn"


def _check(project: SwatProject, stacklevel: int) -> None:
    name = _parameters_bsn(project)
    if name not in project:
        return
    rows = [ln.split() for ln in project.text(name).splitlines()[2:3]]
    try:
        prf = float(rows[0][4])
    except (IndexError, ValueError):
        return
    if prf < 10:
        warnings.warn(f"the peak rate factor prf is {prf:g} (5th value of {name}); SWAT+ 62 takes 484 (the SWAT+ "
                      f"Editor 4 writes it): the file seems written for an earlier SWAT+, whose 5th value meant "
                      f"something else, and peak rates (hence erosion) come out about {484 / max(prf, 1e-6):.0f} "
                      f"times too low", SwatProjectWarning, stacklevel=stacklevel + 1)


#: The USLE C factors :func:`run` offers (``c_factor``).
C_FACTORS = ("swatplus", "plant")


def _run_raw(project: SwatProject, dump=True, c_factor: str = "swatplus") -> dict:
    if not isinstance(project, SwatProject):
        raise TypeError(f"expected a SwatProject (ls.hydro.swat.load_txtinout), got {type(project).__name__}")
    if c_factor not in C_FACTORS:
        raise ValueError(f"c_factor must be one of {', '.join(map(repr, C_FACTORS))}, got {c_factor!r}")
    return _core.swat_run(project.files, dump, c_factor_plant=c_factor == "plant")


def run(project: SwatProject, *, tables=None, nyskip: int | None = None, c_factor: str = "swatplus") -> SwatResult:
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
    c_factor : {"swatplus", "plant"}
        The cover and management factor C of MUSLE. ``"swatplus"`` (default), SWAT+
        62's: the cover of the residue times the cover of the growing biomass (from
        APEX), in which a tall canopy hardly protects the soil, so that a forest's C
        depends on its litter (0.5-0.9 under 1 t/ha of residue). ``"plant"``: SWAT+'s
        earlier method, which 62 keeps in its code but never uses (``codes.bsn``
        ``c_fact`` is ignored): C from the plants' minimum ``usle_c`` (plants.plt), raised
        towards 0.8 as cover falls (forest 0.001). Only sediment and what travels with it
        change; with ``"plant"`` the results are SWAT+'s with that method (checked
        against SWAT+ run so).

    Returns
    -------
    SwatResult

    Examples
    --------
    >>> import landschaft as ls
    >>> project = ls.hydro.swat.load_txtinout(ls.example_data("swat_hubbard_brook"))
    >>> result = ls.hydro.swat.run(project, tables={"hru_wb": "aa", "channel_sd": ["day", "aa"]}, nyskip=0)
    >>> wb = result.table("hru_wb", "aa")              # average annual water balance of each HRU
    >>> wb[["precip", "et", "surq_gen", "latq", "perc"]].astype(float).round(1).head(3)  # mm per year
                    precip     et  surq_gen  latq   perc
    date       hru
    2002-12-31 1    1243.8  314.2     444.1   1.3  448.8
               2    1243.8  314.2     440.0  10.9  443.4
               3    1243.8  314.2     436.9  18.0  439.5
    >>> q = result.table("channel_sd", "day")["flo_out"]  # m3/s leaving the channel
    >>> round(float(q.mean()), 3), result.units["channel_sd"]["flo_out"]
    (0.005, 'm^3/s')
    """
    _check(project, stacklevel=2)
    return _result(_raw_with(project, tables, nyskip, c_factor), stacklevel=3)


def run_many(projects, *, tables=None, nyskip: int | None = None, c_factor: str = "swatplus",
             n_threads: int | None = None) -> list:
    """Simulate several SWAT+ projects at once (scenarios, parameter sets), one per thread.

    Each simulation is :func:`run`'s, independent of the others: the same results as
    one by one, whatever the thread count.

    Parameters
    ----------
    projects : iterable of SwatProject
    tables, nyskip, c_factor
        As :func:`run`, for every project.
    n_threads : int, optional
        Simulations running together; default all cores but one.

    Returns
    -------
    list of SwatResult
        In the order of ``projects``.

    Examples
    --------
    Three rainfall scenarios of the Hubbard Brook example:

    >>> import landschaft as ls
    >>> base = ls.hydro.swat.load_txtinout(ls.example_data("swat_hubbard_brook"))
    >>> def scaled(project, factor):  # the measured rain of every gage times factor
    ...     p = project.copy()
    ...     for name in [n for n in p.files if n.endswith(".pcp")]:
    ...         lines = p.text(name).splitlines()
    ...         rows = [r.split() for r in lines[3:]]
    ...         p.set_text(name, "\\n".join(lines[:3] + [f"{r[0]} {r[1]} {float(r[2]) * factor:.3f}" for r in rows]))
    ...     return p
    >>> results = ls.hydro.swat.run_many([scaled(base, f) for f in (0.8, 1.0, 1.2)], tables={"hru_wb": "aa"})
    >>> [round(float(r.table("hru_wb", "aa")["wateryld"].mean())) for r in results]
    [324, 436, 564]
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..._threads import limit, resolve

    projects = list(projects)
    for project in projects:
        _check(project, stacklevel=2)
    n = min(resolve(n_threads), max(1, len(projects)))

    def one(project):
        with limit(1):
            return _raw_with(project, tables, nyskip, c_factor)

    with ThreadPoolExecutor(max_workers=n) as pool:
        raws = list(pool.map(one, projects))
    return [_result(raw, stacklevel=3) for raw in raws]


def _raw_with(project: SwatProject, tables, nyskip, c_factor: str = "swatplus") -> dict:
    if tables is not None or nyskip is not None:
        project = project.copy()
        project.set_print(tables, nyskip=nyskip)
    return _run_raw(project, dump=[1], c_factor=c_factor)


def _result(raw: dict, stacklevel: int = 2) -> SwatResult:
    for w in raw["warnings"]:
        warnings.warn(f"SWAT+: {w}", stacklevel=stacklevel)
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
