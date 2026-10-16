# SPDX-License-Identifier: GPL-3.0-or-later
"""The landscape and the model: a SWAT+ project from rasters, and its results back on the map."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._editor import from_tables
from ._project import SwatProject
from ._qswat import SwatWatershed, watershed
from ._run import TABLES, run
from ._weather import SwatClimate

__all__ = ["HRU_COLUMNS", "CHANNEL_COLUMNS", "from_rasters", "hru_map", "channel_map", "load_qswat_project",
           "reach_discharge"]

#: The columns of the HRU tables a map can show (SWAT+'s names, see :data:`TABLES`).
HRU_COLUMNS = {name: TABLES[name][2] for name in ("hru_wb", "hru_nb", "hru_ls", "hru_pw")}
#: The columns of the SWAT-DEG channel table (``channel_sd``).
CHANNEL_COLUMNS = TABLES["channel_sd"][2]


def from_rasters(dem, landuse, soils, *, landuse_lookup, soil_lookup, soil_table, climate: SwatClimate, start=None,
                 end=None, codes: dict | None = None, name: str = "landschaft", **options) -> SwatProject:
    """A SWAT+ project from a DEM, land use and soil maps, tables and the weather.

    :func:`watershed` divides the landscape as QSWAT+ does, :func:`from_tables` writes
    the project as the SWAT+ Editor does; the project runs in :func:`run` (or in
    SWAT+ itself, :meth:`SwatProject.write`), and keeps the division
    (``project.watershed``) for :func:`hru_map` and :func:`channel_map`.

    Parameters
    ----------
    dem, landuse, soils, landuse_lookup, soil_lookup, soil_table
        As :func:`watershed`.
    climate : SwatClimate
        The weather (:func:`climate`).
    start, end, codes, name
        As :func:`from_tables`.
    **options
        Passed to :func:`watershed`: ``stream_threshold``, ``outlets``, ``snap``,
        ``slope_bands``, ``hru_method``, ``thresholds``, ``min_area``, ``area_in_ha``,
        ``exempt``.

    Returns
    -------
    SwatProject

    Examples
    --------
    >>> import landschaft as ls
    >>> wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin,  # doctest: +SKIP
    ...                            monthly={"pcp_hhr": 15.0, "slr_ave": 14.0, "dew_ave": 5.0, "wnd_ave": 3.0})
    >>> project = ls.hydro.swat.from_rasters(  # doctest: +SKIP
    ...     dem, landuse, soils, landuse_lookup={1: "FRSD", 2: "PAST"}, soil_lookup={1: "TX047"},
    ...     soil_table=usersoil, climate=wx, stream_threshold=2.0, hru_method="filter", thresholds=(10, 10, 10))
    >>> erosion = ls.hydro.swat.hru_map(project, "sedyld")  # t/ha per year  # doctest: +SKIP
    """
    w = watershed(dem, landuse, soils, landuse_lookup=landuse_lookup, soil_lookup=soil_lookup, soil_table=soil_table,
                  **options)
    project = from_tables(w, climate, start=start, end=end, codes=codes, name=name)
    project.watershed = w
    return project


def load_qswat_project(path) -> dict:
    """The tables of a QSWAT+ project database, for :func:`from_tables`.

    Reads ``gis_subbasins``, ``gis_channels``, ``gis_lsus``, ``gis_hrus``,
    ``gis_aquifers``, ``gis_deep_aquifers``, ``gis_points``, ``gis_routing``,
    ``soils_sol`` and ``soils_sol_layer`` from the ``.sqlite`` QSWAT+ writes (the
    projects without landscapes, lakes, reservoirs or point sources).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(os.fspath(path))
    names = ["gis_subbasins", "gis_channels", "gis_lsus", "gis_hrus", "gis_aquifers", "gis_deep_aquifers",
             "gis_points", "gis_routing", "soils_sol", "soils_sol_layer"]
    with sqlite3.connect(p) as conn:
        present = {r[0] for r in conn.execute("select name from sqlite_master where type = 'table'")}
        missing = [n for n in names if n not in present]
        if missing:
            raise ValueError(f"not a QSWAT+ project database (no {', '.join(missing)})")
        out = {n: pd.read_sql_query(f"select * from {n}", conn) for n in names}
    if "soil" in out["soils_sol_layer"] and "soil_id" not in out["soils_sol_layer"]:
        out["soils_sol_layer"] = out["soils_sol_layer"].rename(columns={"soil": "soil_id"})
    return out


def _table(project, result, table: str, step: str, c_factor: str = "swatplus") -> pd.DataFrame:
    """A printed table of ``result``, or of a run of ``project`` printing it."""
    if result is None:
        result = run(project, tables={table: ["day", "aa"]}, c_factor=c_factor)
    return result.table(table, step)


def _by_object(project, result, table: str, variable: str, level: str, how: str,
               c_factor: str = "swatplus") -> pd.Series:
    if how == "annual":
        return _table(project, result, table, "aa", c_factor)[variable].droplevel("date")
    daily = _table(project, result, table, "day", c_factor)[variable].groupby(level=level)
    if how in ("mean", "max", "sum"):
        return getattr(daily, how)()
    raise ValueError("how must be 'annual', 'mean', 'max' or 'sum'")


def _watershed_of(project) -> SwatWatershed:
    w = project if isinstance(project, SwatWatershed) else getattr(project, "watershed", None)
    if w is None:
        raise ValueError("the project was not made by from_rasters: it has no map of its HRUs and channels")
    return w


def hru_map(project, variable, *, how: str = "annual", result=None, c_factor: str = "swatplus") -> xr.DataArray:
    """An HRU result on the map: each cell takes the value of its HRU.

    Parameters
    ----------
    project : SwatProject
        A project of :func:`from_rasters`.
    variable : str or pandas.Series
        A column of an HRU table (:data:`HRU_COLUMNS`: e.g. ``"sedyld"``, the soil loss
        in t/ha; ``"wateryld"``, ``"surq_gen"``, ``"et"`` in mm; ``"bioms"`` in kg/ha),
        or values by HRU number (1..n).
    how : {"annual", "mean", "max", "sum"}
        ``"annual"``: SWAT+'s average annual value (the yearly total of a flux, the
        mean of a state, after the years print.prt skips); or of the daily values: the
        mean, the largest, the sum.
    result : SwatResult, optional
        A run of the project that printed the HRU table (``aa`` for ``"annual"``, ``day``
        otherwise), to map several variables of one run; by default the project runs.
    c_factor : {"swatplus", "plant"}
        The USLE C factor of that run (see :func:`run`); ignored with ``result``.

    Returns
    -------
    xarray.DataArray
        On the DEM's grid; NaN outside the watershed and on cells whose land use, soil
        and slope combination was folded into other HRUs.
    """
    w = _watershed_of(project)
    if isinstance(variable, str):
        table = next((k for k, cols in HRU_COLUMNS.items() if variable in cols), None)
        if table is None:
            raise ValueError(f"unknown HRU output {variable!r}")
        values = _by_object(project, result, table, variable, "hru", how, c_factor)
        name = variable
    else:
        values, name = pd.Series(variable), getattr(variable, "name", None) or "value"
    ids = w.hrus.values
    lut = np.full(int(np.nanmax(ids)) + 2 if np.isfinite(ids).any() else 1, np.nan)
    for k, v in values.items():
        if 0 <= int(k) < lut.size:
            lut[int(k)] = v
    out = np.full(ids.shape, np.nan)
    ok = np.isfinite(ids)
    out[ok] = lut[ids[ok].astype(np.int64)]
    da = w.hrus.copy(data=out)
    da.name = name
    da.attrs = {"how": how} if isinstance(variable, str) else {}
    return da


def reach_discharge(project, *, result=None) -> pd.DataFrame:
    """The daily discharge leaving each channel (m3/s, ``channel_sd``'s ``flo_out``):
    dates x channel number. ``result``: a run that printed ``channel_sd`` by day."""
    return _table(project, result, "channel_sd", "day")["flo_out"].unstack("channel")


def channel_map(project, values=None, *, how: str = "mean", result=None) -> xr.DataArray:
    """A channel result on the stream cells (by default the mean discharge, m3/s).

    ``values``: a column of the channel table (:data:`CHANNEL_COLUMNS`, default
    ``"flo_out"``) or values by channel number; ``how`` and ``result`` as
    :func:`hru_map`. NaN off the streams.
    """
    w = _watershed_of(project)
    if values is None or isinstance(values, str):
        col = values or "flo_out"
        if col not in CHANNEL_COLUMNS:
            raise ValueError(f"unknown channel output {col!r}")
        values = _by_object(project, result, "channel_sd", col, "channel", how).rename(col)
    values = pd.Series(values)
    ch = w.channels.values
    lut = np.full(int(np.nanmax(ch)) + 1, np.nan)
    for k, v in values.items():
        if 0 < int(k) < lut.size:
            lut[int(k)] = v
    out = np.full(ch.shape, np.nan)
    ok = np.isfinite(ch) & (ch > 0)
    out[ok] = lut[ch[ok].astype(np.int64)]
    da = w.channels.copy(data=out)
    da.name = values.name or "channel"
    return da
