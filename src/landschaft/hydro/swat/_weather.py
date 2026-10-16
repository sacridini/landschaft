# SPDX-License-Identifier: GPL-3.0-or-later
"""Weather for a SWAT+ project from tables: observed daily series and the weather generator.

The files are those the SWAT+ Editor's weather import writes (``*.pcp``, ``*.tmp``,
``*.slr``, ``*.hmd``, ``*.wnd`` and their ``*.cli`` lists, -99 for missing days); the
weather generator statistics (``weather-wgn.cli``) are SWAT's WXGEN parameters,
computed from the series by :func:`wgn_from_series` or given.
"""

from __future__ import annotations


from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SwatClimate", "WGN_COLUMNS", "climate", "wgn_from_series"]

#: The monthly columns of ``weather-wgn.cli`` (SWAT+ ``weather_wgn_cli_mon``).
WGN_COLUMNS = ("tmp_max_ave", "tmp_min_ave", "tmp_max_sd", "tmp_min_sd", "pcp_ave", "pcp_sd", "pcp_skew", "wet_dry",
               "wet_wet", "pcp_days", "pcp_hhr", "slr_ave", "dew_ave", "wnd_ave")
_VARS = ("pcp", "tmax", "tmin", "slr", "hmd", "wnd")
_DESC = {"pcp": "Precipitation", "tmp": "Temperature", "slr": "Solar radiation", "hmd": "Relative humidity",
         "wnd": "Wind speed"}


def _frame(x, stations, what: str) -> pd.DataFrame | None:
    if x is None:
        return None
    df = pd.DataFrame(x).copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    if df.index.has_duplicates:
        raise ValueError(f"{what}: dates repeat")
    missing = [s for s in df.columns if s not in stations.index]
    if missing:
        raise ValueError(f"{what}: stations {missing} are not in the station table")
    return df.sort_index().astype(float)


@dataclass
class SwatClimate:
    """The weather of a SWAT+ project (made by :func:`climate`).

    Attributes
    ----------
    stations : pandas.DataFrame
        Index: the station names; ``lat``, ``lon`` (degrees), ``elev`` (m).
    series : dict
        ``{"pcp": ..., "tmax": ..., "tmin": ..., "slr": ..., "hmd": ..., "wnd": ...}``:
        daily DataFrames (dates x stations; mm, deg C, MJ/m2, fraction, m/s; NaN
        for missing days), any of them absent.
    wgn : pandas.DataFrame
        The weather generator: index (station, month 1-12), the :data:`WGN_COLUMNS`,
        plus ``rain_yrs`` (years of rain record).

    Examples
    --------
    >>> import landschaft as ls
    >>> ex = ls.hydro.swat.example_landscape()
    >>> wx = ls.hydro.swat.climate(ex["stations"], pcp=ex["pcp"], tmax=ex["tmax"], tmin=ex["tmin"], monthly=ex["monthly"])
    >>> sorted(wx.series)
    ['pcp', 'tmax', 'tmin']
    >>> wx.stations
                  lat         lon      elev
    valley  43.050142 -122.950156  899.0123
    """

    stations: pd.DataFrame
    series: dict
    wgn: pd.DataFrame
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SwatClimate({len(self.stations)} stations, {', '.join(sorted(self.series))})"

    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The days every series covers (the SWAT+ Editor's default simulation period)."""
        starts = [df.index[0] for df in self.series.values()]
        ends = [df.index[-1] for df in self.series.values()]
        return max(starts), min(ends)

    def files(self) -> tuple[dict, list]:
        """The weather files (name -> text) and, per file, (name, type, lat, lon)."""
        out, records = {}, []
        kinds = [("pcp", ["pcp"]), ("tmp", ["tmax", "tmin"]), ("slr", ["slr"]), ("hmd", ["hmd"]),
                 ("wnd", ["wnd"])]
        for ext, cols in kinds:
            if any(c not in self.series for c in cols):
                continue
            frames = [self.series[c] for c in cols]
            names = [s for s in self.stations.index if all(s in f.columns for f in frames)]
            if not names:
                continue
            listing = [f"{ext}.cli: {_DESC[ext]} file names - file written by landschaft", "filename"]
            for st in names:
                fname = f"{_file_stem(st)}.{ext}"
                listing.append(fname)
                info = self.stations.loc[st]
                idx = frames[0].index
                for f in frames[1:]:
                    idx = idx.union(f.index)
                days = pd.date_range(idx[0], idx[-1], freq="D")
                vals = [f[st].reindex(days) for f in frames]
                lines = [f"{fname}: {_DESC[ext]} data - file written by landschaft",
                         "nbyr".rjust(4) + "tstep".rjust(10) + "lat".rjust(10) + "lon".rjust(10) + "elev".rjust(10),
                         str(days[-1].year - days[0].year + 1).rjust(4) + "0".rjust(10)
                         + f"{float(info['lat']):.3f}".rjust(10) + f"{float(info['lon']):.3f}".rjust(10)
                         + f"{float(info['elev']):.3f}".rjust(10)]
                arr = np.column_stack([np.where(np.isnan(v.to_numpy()), -99.0, v.to_numpy()) for v in vals])
                for d, row in zip(days, arr):
                    lines.append(str(d.year) + str(d.dayofyear).rjust(5) + " "
                                 + "".join(f"{x:.5f}".rjust(10) for x in row))
                out[fname] = "\n".join(lines) + "\n"
                # the editor reads the coordinates back from the file's third line
                records.append((fname, ext, float(f"{float(info['lat']):.3f}"), float(f"{float(info['lon']):.3f}")))
            out[f"{ext}.cli"] = "\n".join(listing) + "\n"
        return out, records


def _file_stem(name: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "_." else "_" for ch in str(name))
    return stem.replace("-", "")


def wgn_from_series(pcp: pd.DataFrame, tmax: pd.DataFrame, tmin: pd.DataFrame, *, slr=None, hmd=None, wnd=None,
                    dewpoint=None, monthly: dict | None = None) -> pd.DataFrame:
    """SWAT's weather generator statistics (WXGEN) of each station, from daily series.

    By calendar month over all years: the means and sample standard deviations of
    the daily maximum and minimum temperature; the mean monthly precipitation
    (``pcp_ave``: the month's total, averaged over the years with data), the
    standard deviation and skew coefficient of the daily precipitation (all days),
    the probabilities of a wet day after a dry and after a wet day, the mean number
    of wet days (precipitation above 0); the mean daily solar radiation, dew point
    and wind speed when the series are given.

    Parameters
    ----------
    pcp, tmax, tmin
        Daily DataFrames (dates x stations): precipitation (mm), maximum and minimum
        temperature (deg C).
    slr, wnd, dewpoint
        Optional daily solar radiation (MJ/m2), wind speed (m/s) and dew point (deg C).
    hmd
        Daily relative humidity (fraction): with ``tmax`` and ``tmin`` it gives the
        dew point (Magnus' formula at the mean temperature) when ``dewpoint`` is not
        given.
    monthly
        Values not derived from daily data, by column (``pcp_hhr``: the largest
        half-hour rainfall of the month, mm; or ``slr_ave``, ``dew_ave``,
        ``wnd_ave``): a scalar, 12 monthly values, or a DataFrame (month x station).

    Returns
    -------
    pandas.DataFrame
        Index (station, month); the :data:`WGN_COLUMNS` and ``rain_yrs``.



    Examples
    --------
    >>> import landschaft as ls
    >>> ex = ls.hydro.swat.example_landscape()
    >>> wgn = ls.hydro.swat.wgn_from_series(ex["pcp"], ex["tmax"], ex["tmin"], monthly=ex["monthly"])
    >>> wgn.loc["valley", ["pcp_ave", "pcp_sd", "wet_dry", "wet_wet"]].round(3).head(3)
           pcp_ave  pcp_sd  wet_dry  wet_wet
    month                                   
    1      151.675   7.289    0.784    0.663
    2      172.350   9.764    0.522    0.612
    3      106.600   6.234    0.391    0.600
    """
    monthly = dict(monthly or {})
    pcp, tmax, tmin = (pd.DataFrame(x).astype(float) for x in (pcp, tmax, tmin))
    rows = []
    for st in pcp.columns:
        p = pcp[st].dropna()
        tx = tmax[st].dropna() if st in tmax else pd.Series(dtype=float)
        tn = tmin[st].dropna() if st in tmin else pd.Series(dtype=float)
        rain_yrs = float(len(set(p.index.year)))
        prev = p.shift(1, freq="D").reindex(p.index)
        for m in range(1, 13):
            pm = p[p.index.month == m]
            r = {"station": st, "month": m}
            txm, tnm = tx[tx.index.month == m], tn[tn.index.month == m]
            r["tmp_max_ave"], r["tmp_max_sd"] = float(txm.mean()), float(txm.std(ddof=1))
            r["tmp_min_ave"], r["tmp_min_sd"] = float(tnm.mean()), float(tnm.std(ddof=1))
            years = len(set(pm.index.year)) or 1
            r["pcp_ave"] = float(pm.sum() / years)
            r["pcp_sd"] = float(pm.std(ddof=1))
            n = len(pm)
            sd = pm.std(ddof=1)
            r["pcp_skew"] = float(n * (((pm - pm.mean()) / sd) ** 3).sum() / ((n - 1) * (n - 2))) if n > 2 and sd > 0 \
                else 0.0
            pv = prev[prev.index.month == m]
            ok = pv.notna()
            wet, was_wet = (pm[ok] > 0), (pv[ok] > 0)
            dry_days, wet_days = int((~was_wet).sum()), int(was_wet.sum())
            r["wet_dry"] = float((wet & ~was_wet).sum() / dry_days) if dry_days else 0.0
            r["wet_wet"] = float((wet & was_wet).sum() / wet_days) if wet_days else 0.0
            r["pcp_days"] = float((pm > 0).sum() / years)
            for col, series in (("slr_ave", slr), ("wnd_ave", wnd), ("dew_ave", dewpoint)):
                if series is not None and st in pd.DataFrame(series):
                    v = pd.DataFrame(series)[st].dropna()
                    r[col] = float(v[v.index.month == m].mean())
            if "dew_ave" not in r and hmd is not None and st in pd.DataFrame(hmd):
                rh = pd.DataFrame(hmd)[st]
                tav = (tmax[st] + tmin[st]) / 2
                g = np.log(np.clip(rh, 1e-6, None)) + 17.27 * tav / (237.3 + tav)
                dew = (237.3 * g / (17.27 - g)).dropna()
                r["dew_ave"] = float(dew[dew.index.month == m].mean())
            r["rain_yrs"] = rain_yrs
            rows.append(r)
    out = pd.DataFrame(rows).set_index(["station", "month"])
    for col, val in monthly.items():
        if col not in WGN_COLUMNS:
            raise ValueError(f"unknown weather generator column {col!r}")
        out[col] = _monthly(val, out.index)
    missing = [c for c in WGN_COLUMNS if c not in out or out[c].isna().any()]
    if missing:
        raise ValueError(f"the weather generator needs {missing}: give their daily series or monthly=")
    return out[list(WGN_COLUMNS) + ["rain_yrs"]]


def _monthly(val, index: pd.MultiIndex) -> np.ndarray:
    if isinstance(val, pd.DataFrame):
        return np.array([float(val.loc[m, s]) for s, m in index])
    a = np.asarray(val, dtype=float)
    if a.ndim == 0:
        return np.full(len(index), float(a))
    if a.shape == (12,):
        return np.array([a[m - 1] for _, m in index])
    raise ValueError("monthly values are a scalar, 12 values or a (month x station) DataFrame")


def climate(stations, *, pcp, tmax, tmin, slr=None, hmd=None, wnd=None, wgn=None, monthly: dict | None = None
            ) -> SwatClimate:
    """The weather of a SWAT+ project from tables.

    Parameters
    ----------
    stations : pandas.DataFrame
        One row per station (index: name), with ``lat``, ``lon`` (degrees) and
        ``elev`` (m).
    pcp, tmax, tmin, slr, hmd, wnd
        Daily DataFrames (index: dates; columns: station names): precipitation (mm),
        maximum and minimum temperature (deg C), solar radiation (MJ/m2), relative
        humidity (fraction) and wind speed (m/s). Missing days are NaN (-99 for
        SWAT+, which generates them); a variable SWAT+ does not get is generated.
    wgn : pandas.DataFrame, optional
        The weather generator statistics, as :func:`wgn_from_series` returns them
        (e.g. from SWAT+'s WGN database); computed from the series when omitted.
    monthly
        Passed to :func:`wgn_from_series` (``pcp_hhr`` at least: SWAT's weather
        generator needs the largest half-hour rainfall of each month).

    Returns
    -------
    SwatClimate



    Examples
    --------
    >>> import landschaft as ls
    >>> ex = ls.hydro.swat.example_landscape()
    >>> wx = ls.hydro.swat.climate(ex["stations"], pcp=ex["pcp"], tmax=ex["tmax"], tmin=ex["tmin"], monthly=ex["monthly"])
    >>> wx
    SwatClimate(1 stations, pcp, tmax, tmin)
    >>> wx.period()
    (Timestamp('2001-01-01 00:00:00'), Timestamp('2004-12-31 00:00:00'))
    >>> wx.wgn.loc["valley", ["tmp_max_ave", "tmp_min_ave", "pcp_ave", "wet_dry", "pcp_days"]].round(2).head(3)
           tmp_max_ave  tmp_min_ave  pcp_ave  wet_dry  pcp_days
    month                                                      
    1             5.59        -4.83   151.68     0.78     21.75
    2             7.40        -3.06   172.35     0.52     16.25
    3            10.59        -0.04   106.60     0.39     15.25
    """
    st = pd.DataFrame(stations).copy()
    for c in ("lat", "lon", "elev"):
        if c not in st:
            raise ValueError(f"the station table needs a {c!r} column")
    st.index = st.index.map(str)
    series = {}
    for name, x in (("pcp", pcp), ("tmax", tmax), ("tmin", tmin), ("slr", slr), ("hmd", hmd), ("wnd", wnd)):
        f = _frame(x, st, name)
        if f is not None:
            f.columns = f.columns.map(str)
            series[name] = f
    if wgn is None:
        wgn = wgn_from_series(series["pcp"], series["tmax"], series["tmin"], slr=series.get("slr"),
                              hmd=series.get("hmd"), wnd=series.get("wnd"), monthly=monthly)
    wgn = pd.DataFrame(wgn)
    return SwatClimate(stations=st, series=series, wgn=wgn)


def wgn_records(c: SwatClimate) -> list[dict]:
    """The weather generator stations as the editor stores them (one per station)."""
    out = []
    for st in dict.fromkeys(c.wgn.index.get_level_values(0)):
        info = c.stations.loc[st]
        months = c.wgn.loc[st].sort_index()
        out.append({"name": f"wgn_{_file_stem(st)}", "lat": float(info["lat"]), "lon": float(info["lon"]),
                    "elev": float(info["elev"]), "rain_yrs": int(months["rain_yrs"].iloc[0]),
                    "months": [{"month": int(m), **{k: float(r[k]) for k in WGN_COLUMNS}}
                               for m, r in months.iterrows()]})
    return out


def _time_sim(start, end) -> tuple[int, int, int, int]:
    """time.sim from a period, as the editor sets it (day 0: the year's first or last day)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    st = s.dayofyear if s.dayofyear > 1 else 0
    en = 0 if (e.month == 12 and e.day == 31) else e.dayofyear
    return st, s.year, en, e.year



