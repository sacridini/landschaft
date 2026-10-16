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

__all__ = ["SwatResult", "run", "WEATHER_COLUMNS"]

# climate_module weather_daily, in the order the port (and the reference's dump) keeps them
WEATHER_COLUMNS = ("precip", "precip_next", "tmax", "tmin", "tave", "solrad", "solradmx", "rhum", "dewpt", "windsp",
                   "pet", "wndir", "phubase0", "ppet", "daylength", "precip_half_hr")


def _dates(keys: np.ndarray) -> pd.DatetimeIndex:
    """Dates of (year, day of year) pairs."""
    return pd.to_datetime(keys[:, 0].astype(str), format="%Y") + pd.to_timedelta(keys[:, 1] - 1, unit="D")


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
    warnings : list
        What SWAT+ would write to ``diagnostics.out`` (gages not found, ...).
    """

    weather: pd.DataFrame
    warnings: list = field(default_factory=list)

    def __repr__(self) -> str:
        days = self.weather.index.get_level_values("date").unique()
        return f"SwatResult({len(days)} days, {self.weather.index.get_level_values('station').nunique()} stations)"


def _run_raw(project: SwatProject, dump: bool = True) -> dict:
    if not isinstance(project, SwatProject):
        raise TypeError(f"expected a SwatProject (ls.hydro.swat.load_txtinout), got {type(project).__name__}")
    return _core.swat_run(project.files, dump)


def run(project: SwatProject) -> SwatResult:
    """Simulate a SWAT+ project, as SWAT+ 62.0.1 does.

    Parameters
    ----------
    project : SwatProject
        The model (:func:`load_txtinout`).

    Returns
    -------
    SwatResult
    """
    raw = _run_raw(project)
    for w in raw["warnings"]:
        warnings.warn(f"SWAT+: {w}", stacklevel=2)
    keys, vals = raw["dump"].get(1, (np.zeros((0, 3), np.int32), np.zeros((0, len(WEATHER_COLUMNS)), np.float32)))
    stations = np.asarray(raw["stations"], dtype=object)
    index = pd.MultiIndex.from_arrays([_dates(keys), stations[keys[:, 2] - 1] if len(keys) else []],
                                      names=["date", "station"])
    weather = pd.DataFrame(vals[:, : len(WEATHER_COLUMNS)], index=index, columns=list(WEATHER_COLUMNS))
    return SwatResult(weather=weather, warnings=list(raw["warnings"]))
