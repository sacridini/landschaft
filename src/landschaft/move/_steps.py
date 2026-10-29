# SPDX-License-Identifier: GPL-3.0-or-later
"""Steps of tracks and their summaries (amt's track_resample, steps, nsd, straightness & co., time_of_day)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from ._tracks import ATTRS, steps as _locations, time_seconds, track_codes

BURST = "burst_"


_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "min": 60.0, "mins": 60.0,
          "minute": 60.0, "minutes": 60.0, "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
          "d": 86400.0, "day": 86400.0, "days": 86400.0}


def _seconds(value, what: str) -> float:
    """A duration in seconds: a number, a string such as "2h" or "15min", or a timedelta."""
    import datetime
    import re

    if isinstance(value, np.timedelta64):  # a numpy integer too
        return float(value / np.timedelta64(1, "ns")) / 1e9
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, str):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-z]+)\s*", value.lower())
        if m and m.group(2) in _UNITS:
            return float(m.group(1)) * _UNITS[m.group(2)]
    try:
        return pd.Timedelta(value).total_seconds()
    except (TypeError, ValueError):
        raise ValueError(f"{what}: seconds, a string such as '2h' or '15min', or a timedelta") from None


def _groups(tracks, code: np.ndarray, by_burst: bool) -> np.ndarray:
    """Group codes: the tracks, or their bursts (a burst of one track is never joined to another's)."""
    if not by_burst:
        return code
    burst = track_codes(tracks[BURST])
    pairs = pd.MultiIndex.from_arrays([code, burst])
    group, _ = pd.factorize(pairs, sort=False)
    group = group.astype(np.int64)
    if len(group) > 1 and np.any(np.diff(group) < 0):
        raise ValueError(f"the locations of each burst must be consecutive (the column {BURST!r} of resample)")
    return group


def resample(tracks, rate, tolerance, *, start: int = 0, time: str | None = None, track: str | None = None):
    """Locations at a regular rate, in bursts (amt's ``track_resample``, the same locations and bursts).

    From the location ``start`` of each track, the next location kept is the
    first at least ``rate - tolerance`` later: in the same burst when it is at
    most ``rate + tolerance`` later, else it starts a new burst. The locations
    in between are dropped.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, ordered by track and time
        (:func:`landschaft.load_tracks`, :func:`as_tracks`).
    rate, tolerance
        Durations: seconds, strings such as ``"2h"`` and ``"15min"``, or timedeltas.
    start
        The location (0-based, in each track) where the first burst starts; the
        earlier ones are dropped.
    time, track
        The columns of the time and of the track, if not recorded
        (:func:`set_columns`) or with the usual names.

    Returns
    -------
    geopandas.GeoDataFrame
        The locations kept, with their burst (1, 2... within each track) in the
        column ``burst_``, which :func:`steps` and :func:`random_steps` follow.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian([0, 1, 2, 3, 7, 8, 9, 10], tracks=1, seed=1)
    >>> ls.move.resample(tr, rate=1, tolerance=0.1)[["time", "burst_"]].to_numpy().T
    array([[ 0.,  1.,  2.,  3.,  7.,  8.,  9., 10.],
           [ 1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.]])
    """
    s = _locations(tracks, time, track, need_time=True)
    r, tol = _seconds(rate, "rate"), _seconds(tolerance, "tolerance")
    if int(start) < 0:
        raise ValueError("start: a location, 0 or more")
    burst = _core.move_amt_resample(s.seconds, s.code, r, tol, int(start))
    keep = burst > 0
    out = tracks[keep].copy()
    out[BURST] = burst[keep]
    out.attrs = dict(tracks.attrs)
    return out


def filter_min_n_burst(tracks, min_n: int = 3, *, track: str | None = None):
    """The bursts (:func:`resample`) of at least ``min_n`` locations (amt's ``filter_min_n_burst``).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian([0, 1, 2, 3, 7, 8, 12, 13, 14], tracks=1, seed=1)
    >>> bursts = ls.move.resample(tr, rate=1, tolerance=0.1)
    >>> bursts["burst_"].tolist()
    [1, 1, 1, 1, 2, 2, 3, 3, 3]
    >>> ls.move.filter_min_n_burst(bursts)["burst_"].tolist()
    [1, 1, 1, 1, 3, 3, 3]
    """
    from ._tracks import columns

    if BURST not in tracks.columns:
        raise KeyError(f"column {BURST!r} not found: make bursts with ls.move.resample")
    k = columns(tracks, None, track)[1] if track is None else track
    size = tracks.groupby([tracks[k], tracks[BURST]], sort=False, dropna=False)[BURST].transform("size")
    out = tracks[size.to_numpy() >= int(min_n)].copy()
    out.attrs = dict(tracks.attrs)
    return out


def steps(tracks, *, lonlat: bool | None = None, keep_cols: str | None = None, time: str | None = None,
          track: str | None = None):
    """The steps between consecutive locations of each track (amt's ``steps``, ``steps_by_burst``).

    For each pair of consecutive locations of a track (of a burst, when the
    tracks have the column ``burst_`` of :func:`resample`): its start and end,
    length ``sl_``, direction ``direction_p`` (anticlockwise from east, in
    (-pi, pi]) and turn angle ``ta_`` from the previous step, with amt's
    numbers.

    In longitude / latitude (the default for a geographic CRS) the lengths are
    great-circle distances in metres (as ``sf::st_distance``) and the
    directions are geodesic bearings turned into angles from east. amt reads
    the bearing, clockwise from north, as if it were that angle; here it is
    converted, ``pi / 2 - bearing``.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, ordered by track and time.
    lonlat
        Lengths and directions on the sphere (default: when the CRS is geographic).
    keep_cols
        The other columns of the tracks to keep: ``"start"`` (of the location a
        step starts at), ``"end"``, ``"both"`` (suffixes ``_start`` and
        ``_end``) or ``None``.
    time, track
        The columns of the time and of the track, if not recorded
        (:func:`set_columns`) or with the usual names.

    Returns
    -------
    pandas.DataFrame
        One row per step: the track, ``burst_`` (with bursts), ``x1_``, ``x2_``,
        ``y1_``, ``y2_``, ``sl_``, ``direction_p``, ``ta_`` (NaN at the first
        step of each track or burst and after a step of length 0), ``t1_``,
        ``t2_`` and ``dt_``.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(5), tracks=1, seed=1)
    >>> ls.move.steps(tr)[["sl_", "direction_p", "ta_"]].round(3)
         sl_  direction_p    ta_
    0  0.708        2.657    NaN
    1  0.841       -1.351  2.275
    2  0.967        2.614 -2.319
    3  1.758        0.433 -2.180
    """
    s = _locations(tracks, time, track, need_time=True)
    by_burst = BURST in tracks.columns
    group = _groups(tracks, s.code, by_burst)
    ll = s.lonlat if lonlat is None else bool(lonlat)
    sl, direction, ta = _core.move_amt_steps(s.x, s.y, group, ll)
    n = len(group)
    last = np.ones(n, dtype=bool)
    if n:
        last[:-1] = group[1:] != group[:-1]
    i = np.flatnonzero(~last)  # the start of each step
    t = tracks[s.time_col]
    t1, t2 = t.iloc[i].array, t.iloc[i + 1].array  # the arrays keep time zones
    out = pd.DataFrame({s.track_col: tracks[s.track_col].iloc[i].array})
    if by_burst:
        out[BURST] = tracks[BURST].iloc[i].array
    out["x1_"], out["x2_"] = s.x[i], s.x[i + 1]
    out["y1_"], out["y2_"] = s.y[i], s.y[i + 1]
    out["sl_"], out["direction_p"], out["ta_"] = sl[i], direction[i], ta[i]
    out["t1_"], out["t2_"] = t1, t2
    out["dt_"] = t2 - t1
    if keep_cols is not None:
        if keep_cols not in ("start", "end", "both"):
            raise ValueError('keep_cols: "start", "end", "both" or None')
        other = [c for c in tracks.columns
                 if c not in (s.time_col, s.track_col, BURST) and c != tracks.geometry.name]
        for where, rows in (("start", i), ("end", i + 1)):
            if keep_cols in (where, "both"):
                for c in other:
                    out[c + ("_" + where if keep_cols == "both" else "")] = tracks[c].iloc[rows].array
    out.attrs[ATTRS] = {"time": s.time_col, "track": s.track_col, "crs": s.crs, "lonlat": ll}
    return out


def _per_track(tracks, time, track, fn) -> pd.Series:
    """``fn(locations, rows)`` for the rows of each track, as a Series indexed by the track."""
    s = _locations(tracks, time, track)
    ids = tracks[s.track_col].to_numpy()
    bounds = np.flatnonzero(np.r_[True, s.code[1:] != s.code[:-1], True]) if len(s.code) else np.array([0])
    keys, values = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        keys.append(ids[a])
        values.append(fn(s, np.arange(a, b)))
    return pd.Series(values, index=pd.Index(keys, name=s.track_col), dtype=np.float64)


def nsd(tracks, *, time: str | None = None, track: str | None = None) -> np.ndarray:
    """Net squared displacement: the squared distance of each location from the first of its track (amt's ``nsd``).

    In the units of the coordinates (squared degrees in longitude / latitude, as amt).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(4), tracks=2, seed=1)
    >>> ls.move.nsd(tr).round(3)
    array([0.   , 2.937, 3.901, 2.854, 0.   , 0.331, 2.958, 5.793])
    """
    s = _locations(tracks, time, track)
    first = np.flatnonzero(np.r_[True, s.code[1:] != s.code[:-1]]) if len(s.code) else np.array([], dtype=np.int64)
    start = np.repeat(first, np.diff(np.r_[first, len(s.code)]))
    dx, dy = s.x - s.x[start], s.y - s.y[start]
    return dx * dx + dy * dy


def cum_dist(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """The length of each track: the sum of its steps (amt's ``cum_dist``), metres in longitude / latitude.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)
    >>> ls.move.cum_dist(tr).round(3)
    track
    1    4.814
    2    6.667
    dtype: float64
    """
    def one(s, rows):
        d = _core.move_distances(s.x[rows], s.y[rows], np.zeros(len(rows), np.int64), s.lonlat, 1)[:-1]
        return _core.move_r_sum(np.ascontiguousarray(d)) if len(d) else 0.0
    return _per_track(tracks, time, track, one)


def tot_dist(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """The distance from the first to the last location of each track (amt's ``tot_dist``).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)
    >>> ls.move.tot_dist(tr).round(3)
    track
    1    0.935
    2    2.306
    dtype: float64
    """
    def one(s, rows):
        ends = rows[[0, -1]]
        return float(_core.move_distances(s.x[ends], s.y[ends], np.zeros(2, np.int64), s.lonlat, 1)[0])
    return _per_track(tracks, time, track, one)


def straightness(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """:func:`tot_dist` over :func:`cum_dist`: 1 for a straight track, near 0 for a tortuous one (amt's ``straightness``).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)
    >>> ls.move.straightness(tr).round(3)
    track
    1    0.194
    2    0.346
    dtype: float64
    """
    return tot_dist(tracks, time=time, track=track) / cum_dist(tracks, time=time, track=track)


def msd(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """Mean squared distance of the locations of each track from their centre (amt's ``msd``).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)
    >>> ls.move.msd(tr).round(3)
    track
    1    0.775
    2    1.835
    dtype: float64
    """
    return _per_track(tracks, time, track, lambda s, rows: _core.move_amt_msd(s.x[rows], s.y[rows]))


def _track_steps(s, rows):
    sl, _, ta = _core.move_amt_steps(s.x[rows], s.y[rows], np.zeros(len(rows), np.int64), s.lonlat)
    return np.ascontiguousarray(sl[:-1]), np.ascontiguousarray(ta[:-1])


def sinuosity(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """Sinuosity of each track (Benhamou 2004; amt's ``sinuosity``).

    ``2 / sqrt(p * ((1 - c^2 - s^2) / ((1 - c)^2 + s^2) + b^2))``, with ``p`` the
    mean step length, ``b`` its coefficient of variation and ``c``, ``s`` the
    means of the cosines and sines of the turn angles. amt takes the cosines
    and sines of the angles, in radians, as if they were degrees; here the
    radians are used. Steps as :func:`steps` without bursts, in metres in
    longitude / latitude (amt: in degrees).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(20), tracks=2, seed=1)
    >>> ls.move.sinuosity(tr).round(3)
    track
    1    1.538
    2    1.937
    dtype: float64
    """
    return _per_track(tracks, time, track, lambda s, rows: _core.move_amt_sinuosity(*_track_steps(s, rows)))


def tac(tracks, *, time: str | None = None, track: str | None = None) -> pd.Series:
    """Turn angle correlation of each track (Dray et al. 2010; amt's ``tac``).

    The sum of the squared differences of the cosines and sines of consecutive
    turn angles over the number of steps.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(20), tracks=2, seed=1)
    >>> ls.move.tac(tr).round(3)
    track
    1    1.699
    2    1.903
    dtype: float64
    """
    return _per_track(tracks, time, track, lambda s, rows: _core.move_amt_tac(_track_steps(s, rows)[1]))


def time_of_day(x, *, include_crepuscule: bool = False, where: str = "end", time: str | None = None,
                track: str | None = None) -> pd.Categorical:
    """Day or night at each location, or at the end or start of each step (amt's ``time_of_day``).

    Sunrise and sunset of the location's date (UTC) as suncalc computes them
    (the sun's upper edge at -0.833 degrees, with the refraction), dawn and
    dusk with the sun at -6 degrees (civil twilight). suncalc takes the noon of
    the machine's time zone; here, the noon of UTC (suncalc with ``TZ=UTC``).

    Parameters
    ----------
    x
        Tracks (a GeoDataFrame of points with a CRS) or their :func:`steps`.
    include_crepuscule
        Also ``"dawn"`` (dawn to sunrise) and ``"dusk"`` (sunset to dusk).
    where
        For steps: the location and time of their ``"end"`` or ``"start"``.
    time, track
        For tracks: the columns of the time and of the track.

    Returns
    -------
    pandas.Categorical
        ``"day"`` (sunrise to sunset), ``"night"``, and with
        ``include_crepuscule`` ``"dawn"`` and ``"dusk"``; missing where the sun
        does not rise or set that day.

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> t = pd.to_datetime(["2024-06-01 03:00", "2024-06-01 09:20", "2024-06-01 15:00", "2024-06-01 21:00"])
    >>> tr = ls.move.as_tracks(pd.DataFrame({"time": t, "track": 1, "x": -47.9, "y": -15.8}), x="x", y="y", crs=4326)
    >>> list(ls.move.time_of_day(tr, include_crepuscule=True))
    ['night', 'dawn', 'day', 'dusk']
    """
    if "x1_" in getattr(x, "columns", ()):
        if where not in ("end", "start"):
            raise ValueError('where: "end" or "start"')
        info = x.attrs.get(ATTRS, {})
        crs = info.get("crs")
        e = "2" if where == "end" else "1"
        xs, ys = x[f"x{e}_"].to_numpy(np.float64), x[f"y{e}_"].to_numpy(np.float64)
        t = time_seconds(x[f"t{e}_"])
    else:
        s = _locations(x, time, track, need_time=True)
        xs, ys, crs, t = s.x, s.y, s.crs, s.seconds
    if crs is None:
        raise ValueError("the locations have no CRS")
    from ..sdm._points import to_crs

    lon, lat = to_crs(np.asarray(xs, np.float64), np.asarray(ys, np.float64), crs, "EPSG:4326")
    noon = np.floor(t / 86400) * 86400 + 43200  # as.Date (UTC) at 12:00
    sunrise, sunset, dawn, dusk = _core.move_sun_times(np.ascontiguousarray(lon), np.ascontiguousarray(lat), noon)

    def within(start, end):
        # lubridate's %within%: 0 <= t - start <= end - start
        d = t - start
        with np.errstate(invalid="ignore"):
            return (d >= 0) & (d <= end - start), np.isnan(d) | np.isnan(end - start)

    day, missing = within(sunrise, sunset)
    tod = np.where(day, "day", "night").astype(object)
    tod[missing] = None
    if include_crepuscule:
        at_dawn, _ = within(dawn, sunrise)
        at_dusk, _ = within(sunset, dusk)
        tod[at_dawn] = "dawn"
        tod[at_dusk] = "dusk"
        return pd.Categorical(tod, categories=["day", "dusk", "night", "dawn"])
    return pd.Categorical(tod, categories=["day", "night"])
