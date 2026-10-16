# SPDX-License-Identifier: GPL-3.0-or-later
"""The landscape and the model: a SWAT+ project from rasters, and its results back on the map."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ... import _core
from ._editor import from_tables
from ._project import SwatProject
from ._qswat import SwatWatershed, watershed
from ._weather import SwatClimate

__all__ = ["HRU_COLUMNS", "CHANNEL_COLUMNS", "from_rasters", "hru_map", "channel_map", "load_qswat_project",
           "reach_discharge"]

#: The daily HRU outputs (the port's hwb_d, hnb_d, hls_d and hpw_d records, in order).
HRU_COLUMNS = {
    "wb": ("precip", "snofall", "snomlt", "surq_gen", "latq", "wateryld", "perc", "et", "ecanopy", "eplant", "esoil",
           "surq_cont", "cn", "sw_init", "sw_final", "sw", "sw_300", "sno_init", "sno_final", "snopack", "pet",
           "qtile", "irr", "surq_runon", "latq_runon", "overbank", "surq_cha", "surq_res", "surq_ls", "latq_cha",
           "latq_res", "latq_ls", "gwsoil", "satex", "satex_chan", "delsw", "lagsurf", "laglatq", "lagsatex",
           "wet_evap", "wet_out", "wet_stor"),
    "nb": ("grazn", "grazp", "lab_min_p", "act_sta_p", "fertn", "fertp", "fixn", "denit", "act_nit_n", "act_sta_n",
           "org_lab_p", "rsd_nitorg_n", "rsd_laborg_p", "no3atmo", "nh4atmo", "nuptake", "puptake", "gwsoiln",
           "gwsoilp"),
    "ls": ("sedyld", "sedorgn", "sedorgp", "surqno3", "latno3", "surqsolp", "usle", "sedminp", "tileno3", "lchlabp",
           "tilelabp", "satexn"),
    "pw": ("lai", "bioms", "yield", "residue", "sol_tmp", "strsw", "strsa", "strstmp", "strsn", "strsp", "strss",
           "nplnt", "percn", "pplnt", "tmx", "tmn", "tmpav", "solrad", "wndspd", "rhum", "phubase0", "lai_max",
           "bm_max", "bm_grow", "c_gro"),
}
#: The daily channel outputs (chsd_d of the SWAT-DEG channels).
CHANNEL_COLUMNS = ("flo_in", "aqu_in", "flo", "peakr", "sed_in", "sed_out", "washld", "bedld", "dep", "deg_btm",
                   "deg_bank", "hc_sed", "width", "depth", "slope", "deg_btm_m", "deg_bank_m", "hc_m", "flo_in_mm",
                   "aqu_in_mm", "flo_mm", "sed_stor", "n_tot", "p_tot", "dep_bf", "velav_bf")
_TAGS = {"wb": 10, "nb": 11, "ls": 12, "pw": 13}


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


def _daily(project: SwatProject, raw: dict | None = None) -> dict:
    """The port's daily records as DataFrames (internal: phase 8 replaces it by SwatResult)."""
    if raw is None:
        raw = _core.swat_run(project.files, True)
    dump = raw["dump"]

    def frame(tag, cols, level):
        keys, vals = dump.get(tag, (np.zeros((0, 3), np.int32), np.zeros((0, len(cols)), np.float32)))
        dates = pd.to_datetime(keys[:, 0].astype(str), format="%Y") + pd.to_timedelta(keys[:, 1] - 1, unit="D")
        idx = pd.MultiIndex.from_arrays([dates, keys[:, 2]], names=["date", level])
        return pd.DataFrame(vals[:, : len(cols)].astype(np.float64), index=idx, columns=list(cols))

    out = {k: frame(t, HRU_COLUMNS[k], "hru") for k, t in _TAGS.items()}
    out["channel"] = frame(30, CHANNEL_COLUMNS, "channel")
    return out


def _per_year(df: pd.DataFrame, level: str, how: str) -> pd.DataFrame:
    dates = df.index.get_level_values("date")
    g = df.groupby(level=level)
    if how == "annual":
        years = (dates.max() - dates.min()).days / 365.25 + 1 / 365.25
        return g.sum() / years
    if how == "mean":
        return g.mean()
    if how == "max":
        return g.max()
    if how == "sum":
        return g.sum()
    raise ValueError("how must be 'annual', 'mean', 'max' or 'sum'")


def _watershed_of(project) -> SwatWatershed:
    w = project if isinstance(project, SwatWatershed) else getattr(project, "watershed", None)
    if w is None:
        raise ValueError("the project was not made by from_rasters: it has no map of its HRUs and channels")
    return w


def hru_map(project, variable, *, how: str = "annual", outputs: dict | None = None) -> xr.DataArray:
    """An HRU result on the map: each cell takes the value of its HRU.

    Parameters
    ----------
    project : SwatProject
        A project of :func:`from_rasters`.
    variable : str or pandas.Series
        A daily HRU output (:data:`HRU_COLUMNS`: e.g. ``"sedyld"``, the soil loss in
        t/ha; ``"wateryld"``, ``"surq_gen"``, ``"et"`` in mm; ``"bioms"`` in kg/ha), or
        values by HRU number (1..n).
    how : {"annual", "mean", "max", "sum"}
        Of the daily values: the yearly total (the sum over the simulation divided by
        its length in years), the daily mean, the largest, the sum.
    outputs
        The daily outputs of an earlier run (internal; to map several variables of one
        run).

    Returns
    -------
    xarray.DataArray
        On the DEM's grid; NaN outside the watershed and on cells whose land use, soil
        and slope combination was folded into other HRUs.
    """
    w = _watershed_of(project)
    if isinstance(variable, str):
        group = next((k for k, cols in HRU_COLUMNS.items() if variable in cols), None)
        if group is None:
            raise ValueError(f"unknown HRU output {variable!r}")
        outputs = outputs or _daily(project)
        values = _per_year(outputs[group][[variable]], "hru", how)[variable]
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


def reach_discharge(project, *, outputs: dict | None = None) -> pd.DataFrame:
    """The daily discharge of each channel (m3/s): dates x channel number."""
    outputs = outputs or _daily(project)
    return outputs["channel"]["flo"].unstack("channel")


def channel_map(project, values=None, *, how: str = "mean", outputs: dict | None = None) -> xr.DataArray:
    """A channel result on the stream cells (by default the mean discharge, m3/s).

    ``values``: a daily channel output (:data:`CHANNEL_COLUMNS`) or values by channel
    number; ``how`` as :func:`hru_map`. NaN off the streams.
    """
    w = _watershed_of(project)
    if values is None or isinstance(values, str):
        col = values or "flo"
        if col not in CHANNEL_COLUMNS:
            raise ValueError(f"unknown channel output {col!r}")
        outputs = outputs or _daily(project)
        values = _per_year(outputs["channel"][[col]], "channel", how)[col]
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
