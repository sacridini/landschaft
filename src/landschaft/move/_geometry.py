# SPDX-License-Identifier: GPL-3.0-or-later
"""Tracks as lines and segments, locations interpolated in time, a local projection (move2's mt_track_lines & co.)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from ._filters import floor_time
from ._tracks import columns, lags, time_seconds, track_codes


def _ordered(tracks, time, track):
    t, k = columns(tracks, time, track)
    code = track_codes(tracks[k])
    jumps = np.diff(code)
    if len(code) > 1 and (np.any(jumps < 0) or np.any(jumps > 1)):
        raise ValueError("the locations of each track must be consecutive: sort them by track and time first")
    sec = time_seconds(tracks[t])
    d = np.diff(sec)[code[1:] == code[:-1]]
    if np.any(d < 0):
        raise ValueError("the times must not decrease within a track: sort them by track and time first")
    return t, k, code


def track_data(tracks, *, time: str | None = None, track: str | None = None) -> pd.DataFrame:
    """One row per track with the columns whose value does not change within a track (move2's ``mt_track_data``).

    Indexed by the track, in the order the tracks appear.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.track_data(tr).index.tolist()
    [1, 2]
    """
    t, k = columns(tracks, time, track)
    df = pd.DataFrame(tracks.drop(columns=[c for c in (t, tracks.geometry.name) if c in tracks.columns]))
    groups = df.groupby(k, sort=False, dropna=False, observed=True)
    keep = [c for c in df.columns if c != k and (groups[c].nunique(dropna=False) <= 1).all()]
    return groups[keep].first() if keep else pd.DataFrame(index=pd.Index(pd.unique(df[k]), name=k))


def track_lines(tracks, *, time: str | None = None, track: str | None = None):
    """Each track as a line through its locations in time order (move2's ``mt_track_lines``).

    A GeoDataFrame with one row per track, its id, the number of locations and
    the columns of :func:`track_data`. Empty locations are left out; a track
    with a single location gets no line (None).

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.track_lines(tr)[["track", "n_locations"]]
       track  n_locations
    0      1            6
    1      2            6
    """
    import geopandas as gpd
    from shapely import linestrings

    t, k = columns(tracks, time, track)
    geom = tracks.geometry
    ok = ~(geom.is_empty | geom.isna()).to_numpy()
    sub = tracks[ok]
    _ordered(sub, t, k)
    code = track_codes(sub[k])
    x, y = sub.geometry.x.to_numpy(), sub.geometry.y.to_numpy()
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    ends = np.r_[starts[1:], len(code)]
    lines = [linestrings(np.c_[x[a:b], y[a:b]]) if b - a >= 2 else None for a, b in zip(starts, ends)]
    ids = sub[k].to_numpy()[starts]
    out = gpd.GeoDataFrame({k: ids, "n_locations": ends - starts}, geometry=lines, crs=tracks.crs)
    data = track_data(sub, time=t, track=k).drop(columns=["n_locations"], errors="ignore")
    if len(data.columns):
        out = out.merge(data, left_on=k, right_index=True, how="left")
    return out


def segments(tracks, *, time: str | None = None, track: str | None = None):
    """The step from each location to the next of its track as a line (move2's ``mt_segments``).

    A GeoSeries aligned with ``tracks``: a two-point line per step, and the
    location itself (a point) at the last location of each track.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.segments(tr).geom_type.value_counts().to_dict()  # a point at the end of each track
    {'LineString': 10, 'Point': 2}
    """
    import geopandas as gpd
    from shapely import linestrings

    t, k, code = _ordered(tracks, time, track)
    geom = tracks.geometry
    if len(geom) and (geom.is_empty.any() or geom.isna().any()):
        raise ValueError("some locations are empty: drop them first")
    x, y = geom.x.to_numpy(), geom.y.to_numpy()
    n = len(x)
    out = np.array(list(geom), dtype=object)
    step = np.flatnonzero(code[1:] == code[:-1]) if n > 1 else np.array([], dtype=int)
    if len(step):
        coords = np.stack([np.c_[x[step], y[step]], np.c_[x[step + 1], y[step + 1]]], axis=1)
        out[step] = linestrings(coords)
    return gpd.GeoSeries(out, index=tracks.index, crs=tracks.crs)


def interpolate(tracks, times=None, *, max_time_lag=None, omit: bool = False, time: str | None = None,
                track: str | None = None):
    """Locations at given times between the recorded ones (move2's ``mt_interpolate``).

    Without ``times``, the records without a location (empty points) get one
    between the locations before and after them. With ``times`` (datetimes, or
    a unit such as ``"1 hour"`` or ``"15 mins"`` for a regular sequence from the
    first to the last time), a record is added for each time within the span of
    each track that the track does not have, and located. A location is placed
    on the great circle between its neighbours in proportion to the time (in
    longitude / latitude, and through it in a projected CRS, as move2 does with
    s2), on the straight line without a CRS. Gaps longer than ``max_time_lag``
    (seconds, or a ``pandas.Timedelta``) are left empty. ``omit``: keep only the
    records at ``times``.

    Returns a new GeoDataFrame, ordered by track and time.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> ls.move.interpolate(ls.move.filter_unique(tr)).geometry.to_wkt().tolist()  # the empty location
    ['POINT (-47.9 -15.8)', 'POINT (-47.8 -15.7)', 'POINT (-47.628559 -15.614297)', 'POINT (-47.6 -15.6)']
    """
    import geopandas as gpd

    t, k, _ = _ordered(tracks, time, track)
    geom = tracks.geometry
    if len(geom) and not (geom.is_empty | geom.isna() | (geom.geom_type == "Point")).all():
        raise ValueError("the geometry must be points")
    df = tracks
    wanted = None
    if times is not None:
        tcol = tracks[t]
        if isinstance(times, str):
            if not pd.api.types.is_datetime64_any_dtype(tcol.dtype):
                raise TypeError("a unit for times needs datetimes in the time column")
            lo = floor_time(pd.Series([tcol.min()]), times).iloc[0]
            hi = _ceiling_time(tcol.max(), times)
            step = _step(times)
            wanted = pd.Series(pd.date_range(lo, hi, freq=step))
        else:
            wanted = pd.Series(pd.to_datetime(times) if pd.api.types.is_datetime64_any_dtype(tcol.dtype)
                               and not pd.api.types.is_numeric_dtype(pd.Series(times).dtype) else times)
            if pd.api.types.is_datetime64_any_dtype(tcol.dtype) and getattr(tcol.dt, "tz", None) is not None:
                wanted = wanted.dt.tz_localize(tcol.dt.tz) if wanted.dt.tz is None else wanted.dt.tz_convert(tcol.dt.tz)
        add = []
        for tid, grp in tracks.groupby(k, sort=False, observed=True):
            tv = grp[t]
            s = wanted[(wanted >= tv.min()) & (wanted <= tv.max()) & ~wanted.isin(tv)]
            if len(s):
                add.append(pd.DataFrame({k: tid, t: s.to_numpy()}))
        if add:
            new = pd.concat(add, ignore_index=True)
            from shapely import Point

            new[tracks.geometry.name] = gpd.GeoSeries([Point()] * len(new), crs=tracks.crs)
            new = gpd.GeoDataFrame(new, geometry=tracks.geometry.name, crs=tracks.crs)
            df = pd.concat([tracks, new], ignore_index=True)
            df = gpd.GeoDataFrame(df, geometry=tracks.geometry.name, crs=tracks.crs)
            order = pd.DataFrame({"c": track_codes(df[k]), "t": time_seconds(df[t])})
            df = df.iloc[order.sort_values(["c", "t"], kind="mergesort").index].reset_index(drop=True)
        else:
            df = tracks.reset_index(drop=True).copy()
        target = df[t].isin(wanted).to_numpy() & df.geometry.is_empty.to_numpy()
    else:
        df = tracks.reset_index(drop=True).copy()
        target = np.ones(len(df), dtype=bool)
    sec = time_seconds(df[t])
    code = track_codes(df[k])
    empty = (df.geometry.is_empty | df.geometry.isna()).to_numpy()
    idx = np.where(~empty, np.arange(len(df)), -1)
    prev, nxt = _fill(idx, code, down=True), _fill(idx, code, down=False)
    has = (prev >= 0) & (nxt >= 0)
    tp = np.where(has, sec[np.maximum(prev, 0)], np.nan)
    tn = np.where(has, sec[np.maximum(nxt, 0)], np.nan)
    c = 1.0 / 60.0  # difftimes in seconds read as minutes (as.numeric(units = "mins"))
    with np.errstate(invalid="ignore", divide="ignore"):
        prop = ((sec - tp) * c) / ((tn - tp) * c)
    interval = tn - tp
    if max_time_lag is None:
        within = np.ones(len(df), dtype=bool)
    else:
        lag = max_time_lag.total_seconds() if hasattr(max_time_lag, "total_seconds") else float(max_time_lag)
        within = np.nan_to_num(interval, nan=np.inf) <= lag
    do = ~np.isnan(prop) & target & within & empty
    if do.any():
        gx = np.full(len(df), np.nan)
        gy = np.full(len(df), np.nan)
        gx[~empty] = df.geometry[~empty].x.to_numpy()
        gy[~empty] = df.geometry[~empty].y.to_numpy()
        p, q = prev[do], nxt[do]
        lonlat_crs = df.crs is not None and df.crs.is_geographic
        x0, y0, x1, y1 = gx[p], gy[p], gx[q], gy[q]
        if df.crs is not None and not lonlat_crs:
            from pyproj import Transformer

            fwd = Transformer.from_crs(df.crs, "EPSG:4326", always_xy=True)
            x0, y0 = fwd.transform(x0, y0)
            x1, y1 = fwd.transform(x1, y1)
        nx, ny = _core.move_interpolate(np.ascontiguousarray(x0, float), np.ascontiguousarray(y0, float),
                                        np.ascontiguousarray(x1, float), np.ascontiguousarray(y1, float),
                                        np.ascontiguousarray(prop[do]), df.crs is not None)
        if df.crs is not None and not lonlat_crs:
            nx, ny = Transformer.from_crs("EPSG:4326", df.crs, always_xy=True).transform(nx, ny)
        geoms = df.geometry.to_numpy().copy()
        geoms[do] = gpd.points_from_xy(nx, ny)
        df = df.set_geometry(gpd.GeoSeries(geoms, crs=df.crs))
    if omit and wanted is not None:
        df = df[df[t].isin(wanted).to_numpy()].reset_index(drop=True)
    df.attrs = dict(tracks.attrs)
    return df


def _fill(idx: np.ndarray, code: np.ndarray, down: bool) -> np.ndarray:
    """vctrs::vec_fill_missing within each track (-1 for missing)."""
    out = idx.copy()
    rng = range(1, len(idx)) if down else range(len(idx) - 2, -1, -1)
    for i in rng:
        j = i - 1 if down else i + 1
        if out[i] < 0 and code[j] == code[i]:
            out[i] = out[j]
    return out


def _step(unit: str):
    parts = unit.split()
    n, name = (int(float(parts[0])), parts[1]) if len(parts) == 2 else (1, parts[0])
    name = name.lower().rstrip("s") if name.lower() not in ("s", "secs") else "sec"
    table = {"sec": "s", "second": "s", "min": "min", "minute": "min", "hour": "h", "h": "h", "day": "D",
             "d": "D", "week": "7D"}
    if name in table:
        f = table[name]
        return f"{n * 7}D" if f == "7D" else f"{n}{f}"
    if name == "month":
        return pd.DateOffset(months=n)
    if name == "year":
        return pd.DateOffset(years=n)
    raise ValueError(f"unknown unit {unit!r}")


def _ceiling_time(value, unit: str):
    f = floor_time(pd.Series([value]), unit).iloc[0]
    if f == value:
        return f
    from pandas.tseries.frequencies import to_offset

    step = _step(unit)
    return f + (step if isinstance(step, pd.DateOffset) else to_offset(step))


def aeqd_crs(tracks, center="centroid", units: str = "m"):
    """An azimuthal equidistant projection centred on the tracks (move2's ``mt_aeqd_crs``).

    Distances and directions from the centre are true in it: a local metric
    projection for tracks anywhere. ``center``: ``"centroid"`` (of the
    locations, on the sphere for longitude / latitude), ``"center"`` (the middle
    of their longitude and latitude ranges) or a (longitude, latitude) pair.
    ``units``: ``"m"`` or ``"km"``. Returns a ``pyproj.CRS``.

    move2 transforms the centroid to EPSG:4236 (the Hu Tzu Shan datum) where
    WGS84 (EPSG:4326) is meant, which moves the centre by about a kilometre;
    here it stays in WGS84.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> crs = ls.move.aeqd_crs(tr)      # centred on the locations
    >>> crs.srs.split()[:3]
    ['+proj=aeqd', '+lat_0=-15.700027', '+lon_0=-47.774963']
    """
    from pyproj import CRS, Transformer

    if units not in ("m", "km"):
        raise ValueError('units: "m" or "km"')
    if isinstance(center, str):
        if center not in ("centroid", "center"):
            raise ValueError('center: "centroid", "center" or (longitude, latitude)')
        geom = tracks.geometry
        geom = geom[~(geom.is_empty | geom.isna())]
        if not len(geom):
            raise ValueError("no locations")
        if not (geom.geom_type == "Point").all():
            raise ValueError("the geometry must be points")
        x, y = geom.x.to_numpy(np.float64), geom.y.to_numpy(np.float64)
        crs = tracks.crs
        lonlat = crs is not None and crs.is_geographic
        if center == "centroid":
            if lonlat:
                lon, lat = _core.move_s2_centroid(np.ascontiguousarray(x), np.ascontiguousarray(y))
            else:
                cx, cy = _mean(x), _mean(y)
                lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy) \
                    if crs is not None else (cx, cy)
        else:
            if not lonlat and crs is not None:
                x, y = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
            lon, lat = (np.min(x) + np.max(x)) / 2, (np.min(y) + np.max(y)) / 2
    else:
        if hasattr(center, "x") and hasattr(center, "y"):
            lon, lat = center.x, center.y
        else:
            lon, lat = center
    return CRS.from_proj4(f"+proj=aeqd +lat_0={float(lat):.6f} +lon_0={float(lon):.6f} +x_0=0 +y_0=0 +ellps=WGS84 "
                          f"+datum=WGS84 +units={units} +no_defs")


def _mean(v: np.ndarray) -> float:
    s = 0.0
    for a in v:
        s += float(a)
    return s / len(v)


__all__ = ["aeqd_crs", "interpolate", "lags", "segments", "track_data", "track_lines"]
