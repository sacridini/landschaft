# SPDX-License-Identifier: GPL-3.0-or-later
"""Tracks from Movebank's files (move2's mt_read) and from any table of locations."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ._tracks import ATTRS, columns, time_seconds, track_codes

_BOOLEAN = ("visible", "manually-marked-outlier", "algorithm-marked-outlier", "import-marked-outlier",
            "manually-marked-valid", "manually_marked_outlier", "algorithm_marked_outlier", "import_marked_outlier",
            "manually_marked_valid")
_TEXT = ("individual-local-identifier", "tag-local-identifier", "individual-taxon-canonical-name",
         "study-name", "sensor-type", "comments")


def _read_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False, float_precision="round_trip",
                     dtype={c: "string" for c in _TEXT})
    for c in _BOOLEAN:
        if c in df.columns and not pd.api.types.is_bool_dtype(df[c].dtype):
            v = df[c].astype("string").str.strip().str.lower()
            df[c] = v.map({"true": True, "false": False}).astype("boolean")
    return df


def load_tracks(path, *, sort: bool = True):
    """Read a Movebank file into tracks (move2's ``mt_read``): a GeoDataFrame of points in EPSG:4326.

    Reads the CSV files Movebank exports (also ``.csv.gz``) and the zip files of
    Env-DATA (the CSV inside). The time is ``timestamp`` (UTC), the track the
    individual (``individual-local-identifier``; when an individual carried
    several tags, the individual and the tag, in a new
    ``individual-tag-local-identifier`` column, as move2 does). Records
    without coordinates (other sensors) keep an empty location.

    ``sort``: order the locations by track and time (stable), which every
    function of :mod:`landschaft.move` needs; move2 leaves the file's order.

    Examples
    --------
    >>> import landschaft as ls
    >>> tracks = ls.load_tracks("study.csv")                      # doctest: +SKIP
    >>> ls.move.speed(tracks)                                     # doctest: +SKIP
    """
    import geopandas as gpd

    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("the zip file should contain one csv file")
            df = _read_csv(io.BytesIO(z.read(names[0])))
    else:
        df = _read_csv(path)
    missing = [c for c in ("timestamp", "location-long", "location-lat") if c not in df.columns]
    if missing:
        raise ValueError(f"not a Movebank file: no {', '.join(missing)} column")
    if "individual-local-identifier" not in df.columns:
        raise ValueError("not a Movebank file: no individual-local-identifier column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    track = "individual-local-identifier"
    if "tag-local-identifier" in df.columns:
        pairs = df[["individual-local-identifier", "tag-local-identifier"]].drop_duplicates()
        if pairs["individual-local-identifier"].duplicated().any():
            df["individual-tag-local-identifier"] = (df["individual-local-identifier"].astype("string") + "_"
                                                     + df["tag-local-identifier"].astype("string"))
            track = "individual-tag-local-identifier"
    lon, lat = df.pop("location-long").to_numpy(float), df.pop("location-lat").to_numpy(float)
    from shapely import Point, points

    ok = ~(np.isnan(lon) | np.isnan(lat))
    geom = np.array([Point()] * len(df), dtype=object)
    geom[ok] = points(np.c_[lon[ok], lat[ok]])
    out = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(geom), crs="EPSG:4326")
    out = _sorted(out, "timestamp", track) if sort else out
    out.attrs[ATTRS] = {"time": "timestamp", "track": track}
    return out


def as_tracks(data, time: str | None = None, track: str | None = None, *, x: str | None = None, y: str | None = None,
              crs=None, sort: bool = True):
    """Tracks from a table of locations (move2's ``mt_as_move2``): a GeoDataFrame ``ls.move`` understands.

    ``data``: a GeoDataFrame of points, or a DataFrame with coordinate columns
    ``x`` and ``y`` (``crs``: theirs, e.g. ``"EPSG:4326"``). ``time``, ``track``:
    the columns of the time (datetimes, text read as datetimes, or numbers) and
    of the track. ``sort``: order by track and time (stable). The column names
    are recorded so that later calls need not repeat them.

    Examples
    --------
    >>> import landschaft as ls
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00",
    ...                     "2024-01-01 03:20"], utc=True)
    >>> pts = [Point(-47.9, -15.8), Point(-47.8, -15.7), Point(-47.8, -15.7), Point(), Point(-47.6, -15.6)]
    >>> tr = ls.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ["a"] * 5}, geometry=pts, crs=4326), "time", "id")
    >>> tr.attrs["landschaft.move"]       # the columns every function of ls.move reads
    {'time': 'time', 'track': 'id'}
    """
    import geopandas as gpd

    if isinstance(data, gpd.GeoDataFrame) and x is None:
        out = data.copy()
    else:
        if x is None or y is None:
            raise ValueError("x and y: the columns of the coordinates (or give a GeoDataFrame)")
        df = pd.DataFrame(data).copy()
        out = gpd.GeoDataFrame(df.drop(columns=[x, y]), geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)
    t, k = columns(out, time, track)
    if pd.api.types.is_object_dtype(out[t].dtype) or pd.api.types.is_string_dtype(out[t].dtype):
        out[t] = pd.to_datetime(out[t], utc=True)
    out = _sorted(out, t, k) if sort else out
    out.attrs[ATTRS] = {"time": t, "track": k}
    return out


def _sorted(df, time: str, track: str):
    # the tracks in their order of appearance, each by time (stable)
    order = np.lexsort((time_seconds(df[time]), track_codes(df[track])))
    return df.iloc[order].reset_index(drop=True)
