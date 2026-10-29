# SPDX-License-Identifier: GPL-3.0-or-later
"""Duplicates, one location per interval and Movebank's outliers (move2's mt_filter_*)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import _core
from ._tracks import check_order, columns, time_seconds, track_codes

_OUTLIER_COLUMNS = ("manually-marked-outlier", "algorithm-marked-outlier", "import-marked-outlier",
                    "manually_marked_outlier", "algorithm_marked_outlier", "import_marked_outlier")
_VALID_COLUMNS = ("manually-marked-valid", "manually_marked_valid")


def _group_order(tracks, track_col: str, keys: list) -> list[np.ndarray]:
    """Row indices of each group of (track, *keys), in dplyr's order of the groups (sorted keys)."""
    tk = tracks[track_col]
    if isinstance(tk.dtype, pd.CategoricalDtype):
        tcode = tk.cat.codes.to_numpy(np.int64)
    else:
        uniq = np.array(sorted(pd.unique(tk.dropna()), key=lambda v: (str(type(v)), v)), dtype=object)
        lookup = {v: i for i, v in enumerate(uniq)}
        tcode = np.array([lookup.get(v, len(uniq)) for v in tk], dtype=np.int64)
    frame = pd.DataFrame({"__t": tcode, **{f"__k{i}": k for i, k in enumerate(keys)}})
    groups = frame.groupby(list(frame.columns), sort=True, dropna=False).indices
    return [np.sort(v) for _, v in sorted(groups.items(), key=lambda kv: kv[0])]


def _draw(sizes: list[int], seed: int | None) -> list[int]:
    """sample.int(n, 1) for each group in turn after set.seed(seed): 0-based picks."""
    if seed is None:
        rng = np.random.default_rng()
        return [int(rng.integers(n)) for n in sizes]
    ops = [("sample", int(n), 1, False, None) for n in sizes]
    return [int(np.asarray(v)[0]) - 1 for v in _core.nlm_r_sequence(int(seed), ops)]


def _equal(a, b, near: bool) -> bool:
    """identical(a, b) | is.na(b) for one value of two records (all.equal with ``near``)."""
    if b is None or (not isinstance(b, (str, bytes)) and pd.api.types.is_scalar(b) and pd.isna(b)):
        return True
    if a is None or (not isinstance(a, (str, bytes)) and pd.api.types.is_scalar(a) and pd.isna(a)):
        return False
    if hasattr(a, "equals") and hasattr(b, "geom_type"):  # geometries
        return bool(a.equals_exact(b, 0.0)) if not near else bool(a.equals_exact(b, 1.5e-8))
    try:
        if a == b:
            return True
    except (TypeError, ValueError):
        return False
    if near and isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        target, current = float(a), float(b)
        den = abs(target)
        diff = abs(target - current)
        return bool((diff / den if den > 0 and np.isfinite(diff / den) else diff) <= 1.5e-8)
    return False


def _subsets(rows: np.ndarray, records: list[tuple], near: bool) -> list[int]:
    """move2's slice_subsets: the rows of a group left once records that are subsets of others are dropped."""
    if len(rows) == 1:
        return [int(rows[0])]
    seen, kept = set(), []
    for r, rec in zip(rows, records):
        key = tuple("\0NA" if (v is None or (pd.api.types.is_scalar(v) and not isinstance(v, str) and pd.isna(v)))
                    else (v.wkb if hasattr(v, "wkb") else v) for v in rec)
        if key not in seen:
            seen.add(key)
            kept.append((int(r), rec))
    if len(kept) == 1:
        return [kept[0][0]]
    drop, pairs_seen = set(), set()
    # expand.grid's order: the second record outer, the first inner
    for j, (rj, b) in enumerate(kept):
        for i, (ri, a) in enumerate(kept):
            if i == j:
                continue
            if all(_equal(x, y, near) for x, y in zip(a, b)):  # the second (rj) is a subset of the first
                pair = (min(ri, rj), max(ri, rj))
                if pair in pairs_seen:  # mutual subsets: drop one of them only
                    continue
                pairs_seen.add(pair)
                drop.add(rj)
    return [r for r, _ in kept if r not in drop]


def unique(tracks, criterion: str = "subsets", *, columns_also: list[str] | None = None, seed: int | None = None,
           time: str | None = None, track: str | None = None) -> np.ndarray:
    """Which locations to keep so that no two of a track share their time (move2's ``mt_unique``).

    ``criterion``:

    * ``"subsets"`` (default): of the records of a track with the same time,
      drop those that only repeat another or hold less (each value equal to the
      other's or missing); a warning when some still remain;
    * ``"subsets_equal"``: the same, with numbers equal within R's ``all.equal``
      tolerance (1.5e-8);
    * ``"first"``, ``"last"``: the first or last record of each time;
    * ``"sample"``: one at random (R's ``sample.int`` after ``set.seed(seed)``, the
      same picks as move2 when ``seed`` is given).

    ``columns_also``: more columns whose values must also match for records to
    be duplicates (move2's ``additional_columns``). Returns a boolean mask.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> ls.move.unique(tr, "first")     # two records at 01:00
    array([ True,  True, False,  True,  True])
    """
    t, k = columns(tracks, time, track)
    keys = [tracks[t]] + [tracks[c] for c in (columns_also or [])]
    groups = _group_order(tracks, k, keys)
    keep = np.zeros(len(tracks), dtype=bool)
    if criterion in ("first", "last"):
        for g in groups:
            keep[g[0] if criterion == "first" else g[-1]] = True
        return keep
    if criterion == "sample":
        picks = _draw([len(g) for g in groups], seed)
        for g, p in zip(groups, picks):
            keep[g[p]] = True
        return keep
    if criterion not in ("subsets", "subsets_equal"):
        raise ValueError('criterion: "subsets", "subsets_equal", "first", "last" or "sample"')
    data = tracks.drop(columns=[k])
    values = list(data.itertuples(index=False, name=None))
    left = False
    for g in groups:
        if len(g) == 1:
            keep[g[0]] = True
            continue
        rows = _subsets(g, [values[i] for i in g], criterion == "subsets_equal")
        keep[rows] = True
        left |= len(rows) != 1
    if left:
        warnings.warn("after removing all records that are subsets of other records there are still remaining "
                      "duplicates", UserWarning, stacklevel=2)
    return keep


def filter_unique(tracks, criterion: str = "subsets", **kwargs):
    """``tracks`` without duplicated times within a track (move2's ``mt_filter_unique``; see :func:`unique`).

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> len(ls.move.filter_unique(tr))  # the second record at 01:00 only repeats the first
    4
    """
    return tracks[unique(tracks, criterion, **kwargs)]


def per_interval(tracks, unit: str = "hour", criterion: str = "first", *, seed: int | None = None,
                 time: str | None = None, track: str | None = None) -> np.ndarray:
    """Which locations to keep for one per time interval of each track (move2's ``mt_per_interval``).

    ``unit``: lubridate's units, e.g. ``"hour"``, ``"15 mins"``, ``"2 hours"``,
    ``"day"``, ``"week"`` (from Sunday), ``"month"``, ``"year"``; multiples
    count from the start of the next larger unit (minutes within the hour,
    hours within the day, days within the month), as ``lubridate::floor_date``.
    ``criterion``: ``"first"``, ``"last"`` or ``"random"`` (R's ``sample.int``
    after ``set.seed(seed)``). Returns a boolean mask.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> ls.move.per_interval(tr, "2 hours")
    array([ True, False, False,  True, False])
    """
    t, k = columns(tracks, time, track)
    check_order(track_codes(tracks[k]), time_seconds(tracks[t]))
    floored = floor_time(tracks[t], unit)
    groups = _group_order(tracks, k, [floored])
    keep = np.zeros(len(tracks), dtype=bool)
    if criterion == "random":
        picks = _draw([len(g) for g in groups], seed)
        for g, p in zip(groups, picks):
            keep[g[p]] = True
    elif criterion in ("first", "last"):
        for g in groups:
            keep[g[0] if criterion == "first" else g[-1]] = True
    else:
        raise ValueError('criterion: "first", "last" or "random"')
    return keep


def filter_per_interval(tracks, unit: str = "hour", criterion: str = "first", **kwargs):
    """``tracks`` with one location per time interval of each track (see :func:`per_interval`).

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> ls.move.filter_per_interval(tr, "2 hours", "last")["time"].dt.strftime("%H:%M").tolist()
    ['01:00', '03:20']
    """
    return tracks[per_interval(tracks, unit, criterion, **kwargs)]


_UNITS = {"second": "s", "seconds": "s", "sec": "s", "secs": "s", "s": "s",
          "minute": "min", "minutes": "min", "min": "min", "mins": "min",
          "hour": "h", "hours": "h", "h": "h",
          "day": "d", "days": "d", "d": "d",
          "week": "w", "weeks": "w",
          "month": "m", "months": "m",
          "year": "y", "years": "y"}


def floor_time(values, unit: str) -> pd.Series:
    """lubridate::floor_date(values, unit) for datetimes (UTC or their time zone)."""
    s = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s.dtype):
        raise TypeError("one location per interval needs datetimes in the time column")
    parts = str(unit).strip().split()
    n = 1
    if len(parts) == 2:
        n, name = float(parts[0]), parts[1]
        if n != int(n) or n < 1:
            raise ValueError("unit: a whole number of units, e.g. '15 mins'")
        n = int(n)
    else:
        name = parts[0]
    u = _UNITS.get(name.lower())
    if u is None:
        raise ValueError(f"unknown unit {unit!r}")
    dt = s.dt
    if u == "s":
        sec = dt.second + dt.microsecond / 1e6 + dt.nanosecond / 1e9
        return dt.floor("min") + pd.to_timedelta(np.floor(sec / n) * n, unit="s")
    if u == "min":
        return dt.floor("h") + pd.to_timedelta((dt.minute // n) * n, unit="min")
    if u == "h":
        return dt.floor("D") + pd.to_timedelta((dt.hour // n) * n, unit="h")
    day = dt.floor("D")
    if u == "d":
        return day - pd.to_timedelta((dt.day - 1) - ((dt.day - 1) // n) * n, unit="D")
    if u == "w":
        if n != 1:
            raise ValueError("unit: weeks one at a time")
        return day - pd.to_timedelta((dt.dayofweek + 1) % 7, unit="D")
    if u == "m":
        month = ((dt.month - 1) // n) * n + 1
        return pd.Series([pd.Timestamp(year=y, month=m, day=1, tz=dt.tz) for y, m in zip(dt.year, month)],
                         index=s.index)
    year = (dt.year // n) * n
    return pd.Series([pd.Timestamp(year=y, month=1, day=1, tz=dt.tz) for y in year], index=s.index)


def movebank_visible(tracks) -> np.ndarray:
    """Which locations Movebank shows (move2's ``mt_movebank_visible``): a boolean mask.

    The ``visible`` column when there is one; otherwise the records not marked
    as outliers (manually, by an algorithm or on import), and those marked
    valid by hand.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> tr["manually-marked-outlier"] = [False, True, False, False, False]
    >>> ls.move.movebank_visible(tr)
    array([ True, False,  True,  True,  True])
    """
    if "visible" in tracks.columns:
        v = tracks["visible"]
        if not pd.api.types.is_bool_dtype(v.dtype):
            raise TypeError("the values of 'visible' must be booleans")
        return v.to_numpy(bool)
    keep = np.ones(len(tracks), dtype=bool)
    for col in _OUTLIER_COLUMNS:
        if col in tracks.columns:
            v = tracks[col]
            if not (pd.api.types.is_bool_dtype(v.dtype) or v.isna().all()):
                raise TypeError(f"the values of {col!r} must be booleans")
            keep &= v.isna().to_numpy() | ~v.fillna(False).astype(bool).to_numpy()
    for col in _VALID_COLUMNS:
        if col in tracks.columns:
            v = tracks[col]
            if not (pd.api.types.is_bool_dtype(v.dtype) or v.isna().all()):
                raise TypeError(f"the values of {col!r} must be booleans")
            keep[v.fillna(False).astype(bool).to_numpy()] = True
    return keep


def filter_movebank_visible(tracks):
    """``tracks`` without the locations Movebank hides (see :func:`movebank_visible`).

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> tr["visible"] = [True, True, True, False, True]
    >>> len(ls.move.filter_movebank_visible(tr))
    4
    """
    return tracks[movebank_visible(tracks)]


__all__ = ["filter_movebank_visible", "filter_per_interval", "filter_unique", "floor_time", "movebank_visible",
           "per_interval", "time_seconds", "track_codes", "unique"]
