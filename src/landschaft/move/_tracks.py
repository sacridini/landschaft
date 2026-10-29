# SPDX-License-Identifier: GPL-3.0-or-later
"""Tracks as GeoDataFrames: which columns hold the time and the track, their order, their coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ATTRS = "landschaft.move"
_TIME_NAMES = ("timestamp", "time", "datetime", "date_time", "t")
_TRACK_NAMES = ("individual-tag-local-identifier", "individual-local-identifier", "individual_local_identifier",
                "track", "track_id", "id", "individual", "animal")

_LENGTH = {"m": 1.0, "km": 1000.0}
_DURATION = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_SPEED = {"m/s": (1.0, 1.0), "km/h": (1000.0, 3600.0), "m/h": (1.0, 3600.0), "km/d": (1000.0, 86400.0)}


def _find(df, given, names, what):
    if given is not None:
        if given not in df.columns:
            raise KeyError(f"{what} column {given!r} not found")
        return given
    stored = df.attrs.get(ATTRS, {}).get(what)
    if stored is not None and stored in df.columns:
        return stored
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"which column holds the {what}? Pass {what}=..., or name it one of {', '.join(names[:4])}")


def columns(tracks, time: str | None = None, track: str | None = None) -> tuple[str, str]:
    """The (time, track) column names: the arguments, those set by ``set_columns`` / ``load_tracks``, or the usual names."""
    return _find(tracks, time, _TIME_NAMES, "time"), _find(tracks, track, _TRACK_NAMES, "track")


def set_columns(tracks, time: str | None = None, track: str | None = None):
    """Record which columns hold the time and the track (``tracks.attrs``), so later calls need not name them.

    Returns ``tracks`` (the same object). Every function of ``ls.move`` also takes
    ``time=`` and ``track=``, which win over what is recorded here; the record
    lives in ``attrs``, which some pandas operations drop.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1).rename(columns={"track": "animal"})
    >>> ls.move.set_columns(tr, track="animal").attrs["landschaft.move"]
    {'time': 'time', 'track': 'animal'}
    """
    t, k = columns(tracks, time, track)
    tracks.attrs[ATTRS] = {"time": t, "track": k}
    return tracks


@dataclass
class Steps:
    """The locations of a validated set of tracks, as arrays."""
    x: np.ndarray
    y: np.ndarray
    code: np.ndarray       # int64 track code per location (consecutive per track)
    seconds: np.ndarray | None  # time as float seconds (None when not asked)
    lonlat: bool
    a: float
    inv_flattening: float
    crs: object
    time_col: str
    track_col: str


def track_codes(values) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(values), sort=False, use_na_sentinel=False)
    return codes.astype(np.int64)


def _nanoseconds(s: pd.Series) -> np.ndarray:
    """Datetimes as int64 nanoseconds since the epoch (UTC), whatever the column's unit (pandas 3 reads
    ISO 8601 text to seconds, milliseconds or microseconds; before pandas 2 they are always nanoseconds)."""
    return (s.dt.as_unit("ns") if hasattr(s.dt, "as_unit") else s).array.asi8


def time_seconds(values) -> np.ndarray:
    """Times as float seconds; lags between datetimes are differences of int64 nanoseconds (exact)."""
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return _nanoseconds(s).astype(np.float64) / 1e9
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        return s.to_numpy("timedelta64[ns]").view(np.int64).astype(np.float64) / 1e9
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.to_numpy(np.float64)
    raise TypeError("the time column: datetimes or numbers")


def lags(values, code: np.ndarray) -> np.ndarray:
    """Time from each location to the next of its track, in seconds (NaN at the last of each track)."""
    s = pd.Series(values)
    n = len(s)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        d = np.diff(_nanoseconds(s)).astype(np.float64) / 1e9
        d[np.asarray(pd.isna(s))[1:] | np.asarray(pd.isna(s))[:-1]] = np.nan
    else:
        d = np.diff(time_seconds(s))
    d[code[1:] != code[:-1]] = np.nan
    out[:-1] = d
    return out


def check_order(code: np.ndarray, seconds: np.ndarray | None, strictly: bool = False) -> None:
    jumps = np.diff(code)
    if len(code) > 1 and (np.any(jumps < 0) or np.any(jumps > 1)):
        raise ValueError("the locations of each track must be consecutive: sort them by track and time first "
                         "(tracks.sort_values([track, time]))")
    if seconds is not None and len(seconds) > 1:
        d = np.diff(seconds)[code[1:] == code[:-1]]
        if np.any(d < 0):
            raise ValueError("the times must not decrease within a track: sort them by track and time first")
        if strictly and np.any(d == 0):
            raise ValueError("some locations of a track share their time (speeds would divide by 0): "
                             "drop them with ls.move.filter_unique")


def steps(tracks, time: str | None = None, track: str | None = None, *, need_time: bool = False,
          strictly: bool = False) -> Steps:
    """Validate ``tracks`` (points, ordered by track and time, none empty) and return their arrays."""
    t, k = columns(tracks, time, track)
    geom = getattr(tracks, "geometry", None)
    if geom is None:
        raise TypeError("tracks: a GeoDataFrame of points (ls.load_tracks, ls.move.as_tracks)")
    if len(geom) and (geom.is_empty.any() or geom.isna().any()):
        raise ValueError("some locations are empty: drop them first (tracks[~tracks.geometry.is_empty])")
    if len(geom) and not (geom.geom_type == "Point").all():
        raise ValueError("the geometry must be points")
    code = track_codes(tracks[k])
    seconds = time_seconds(tracks[t])
    check_order(code, seconds, strictly)
    crs = tracks.crs
    lonlat = bool(crs is not None and crs.is_geographic)
    a, rf = 6378137.0, 298.257223563
    if lonlat:
        e = crs.ellipsoid
        a, rf = float(e.semi_major_metre), float(e.inverse_flattening)
    return Steps(np.ascontiguousarray(geom.x.to_numpy(np.float64)), np.ascontiguousarray(geom.y.to_numpy(np.float64)),
                 code, seconds if need_time else None, lonlat, a, rf, crs, t, k)


def length_factor(units: str | None, crs, lonlat: bool) -> float:
    """Divide metres (or the CRS's units) by this to get ``units``."""
    if units is None:
        return 1.0
    if units not in _LENGTH:
        raise ValueError(f"units: one of {', '.join(_LENGTH)}")
    if not lonlat:
        unit = _crs_unit(crs)
        if unit is None:
            raise ValueError("the tracks have no CRS: their distances have no units to convert")
        return _LENGTH[units] / unit
    return _LENGTH[units]


def _crs_unit(crs) -> float | None:
    if crs is None:
        return None
    try:
        return float(crs.axis_info[0].unit_conversion_factor)
    except (IndexError, AttributeError):
        return None


def duration_factor(units: str | None) -> float:
    if units is None:
        return 1.0
    if units not in _DURATION:
        raise ValueError(f"units: one of {', '.join(_DURATION)}")
    return _DURATION[units]


def speed_factor(units: str | None, crs, lonlat: bool) -> tuple[float, float]:
    if units is None:
        return 1.0, 1.0
    if units not in _SPEED:
        raise ValueError(f"units: one of {', '.join(_SPEED)}")
    length, duration = _SPEED[units]
    if not lonlat:
        unit = _crs_unit(crs)
        if unit is None:
            raise ValueError("the tracks have no CRS: their speeds have no units to convert")
        length /= unit
    return length, duration
