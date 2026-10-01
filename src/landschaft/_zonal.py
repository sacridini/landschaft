# SPDX-License-Identifier: GPL-3.0-or-later
"""Raster values by zone (``zonal_stats``) and at points (``extract``), computed in C++."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, _threads

__all__ = ["extract", "zonal_stats"]

_STATS = {"count": 0, "nodata": 1, "sum": 2, "mean": 3, "min": 4, "max": 5, "range": 6, "std": 7, "var": 8,
          "median": 9, "majority": 10, "minority": 11, "unique": 12}
_PERCENTILE = 13
_COUNTS = {"count", "nodata", "unique"}
_SUPPORTED = {np.dtype(t) for t in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "float32",
                                    "float64")}


def _parse_stats(stats) -> tuple[list[str], list[int], list[float]]:
    names = [stats] if isinstance(stats, str) else list(stats)
    if not names:
        raise ValueError("no statistic asked for")
    codes, qs = [], []
    for name in names:
        key = str(name).lower()
        m = re.fullmatch(r"(?:p|percentile_?)(\d+(?:\.\d+)?)", key)
        if m:
            q = float(m.group(1))
            if not 0 <= q <= 100:
                raise ValueError(f"percentile {name!r} outside 0-100")
            codes.append(_PERCENTILE)
            qs.append(q)
        elif key in _STATS:
            codes.append(_STATS[key])
            qs.append(0.0)
        else:
            raise ValueError(f"unknown statistic {name!r}; use {', '.join(_STATS)} or pNN (e.g. p90)")
    return [str(n) for n in names], codes, qs


def _raster(raster, what: str = "raster"):
    """(values (bands, y, x) or (y, x), has_nodata, nodata, geotransform, band labels, CRS) of a raster."""
    from .io import _transform

    if not isinstance(raster, xr.DataArray):
        raise TypeError(f"the {what} must be a georeferenced DataArray (see load_raster)")
    if raster.ndim not in (2, 3) or "x" not in raster.dims or "y" not in raster.dims:
        raise ValueError(f"the {what} must be (y, x) or (band, y, x), got {raster.dims}")
    if raster.ndim == 3 and raster.dims[0] in ("y", "x"):
        raster = raster.transpose(..., "y", "x")
    t = _transform(raster)
    if t.b or t.d:
        raise ValueError("rotated grids are not supported")
    gt = (t.c, t.a, t.b, t.f, t.d, t.e)
    nodata = raster.rio.nodata
    has_nodata = nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata))
    if raster.ndim == 3:
        dim = raster.dims[0]
        labels = [str(v) for v in raster[dim].values.tolist()] if dim in raster.coords else \
            [str(i + 1) for i in range(raster.shape[0])]
    else:
        labels = [raster.name if isinstance(raster.name, str) and raster.name else "value"]
    return raster, has_nodata, float(nodata) if has_nodata else 0.0, gt, labels, raster.rio.crs


def _values(data) -> np.ndarray:
    """C-contiguous values of a type the kernels read (dask arrays are computed)."""
    v = data.data if isinstance(data, xr.DataArray) else data
    if hasattr(v, "dask"):
        v = v.compute()
    v = np.asarray(v)
    if v.dtype == bool:
        v = v.view(np.uint8)
    elif v.dtype not in _SUPPORTED:
        v = v.astype(np.float32 if v.dtype == np.float16 else np.float64)
    return np.ascontiguousarray(v)


def _geodataframe(obj, crs):
    """A GeoDataFrame (or None for plain coordinates) in the CRS of the raster."""
    import geopandas as gpd

    if isinstance(obj, gpd.GeoSeries):
        obj = gpd.GeoDataFrame(geometry=obj)
    if isinstance(obj, gpd.GeoDataFrame):
        if obj.crs is not None and crs is not None and obj.crs != crs:
            from ._vector import to_crs

            obj = to_crs(obj, crs)
        return obj
    if hasattr(obj, "geom_type"):
        return gpd.GeoDataFrame(geometry=[obj], crs=crs)
    if isinstance(obj, (list, tuple)) and obj and all(hasattr(g, "geom_type") for g in obj):
        return gpd.GeoDataFrame(geometry=list(obj), crs=crs)
    return None


def _polygon_arrays(geoms):
    """Coordinates, ring offsets and the rings of every geometry (polygons and multipolygons), vectorised."""
    import shapely

    geoms = np.asarray(geoms, dtype=object)
    kinds = shapely.get_type_id(geoms)  # -1 missing, 3 Polygon, 6 MultiPolygon
    bad = ~np.isin(kinds, (-1, 3, 6)) & ~shapely.is_empty(geoms)
    if bad.any():
        kind = shapely.get_type_id(geoms[bad][0])
        raise ValueError(f"zones must be polygons, got {shapely.GeometryType(kind).name.lower()}s: buffer points "
                         "or lines first (or use extract for points)")
    parts, owner = shapely.get_parts(geoms, return_index=True)  # every polygon, in the geometries' order
    keep = ~shapely.is_empty(parts)
    parts, owner = parts[keep], owner[keep]
    if len(parts) == 0:
        return np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros(len(geoms) + 1, np.int64)
    _, coords, (ring_off, poly_off) = shapely.to_ragged_array(parts)
    poly_off = np.asarray(poly_off, dtype=np.int64)
    ring_off = np.asarray(ring_off, dtype=np.int64)
    first_part = np.searchsorted(owner, np.arange(len(geoms) + 1))  # the parts of geometry g: first_part[g]..[g+1]
    geom_rings = poly_off[first_part]
    xy = np.ascontiguousarray(coords[:, :2], dtype=np.float64)
    return _clockwise(xy, ring_off), ring_off, geom_rings


def _clockwise(xy: np.ndarray, ring_off: np.ndarray) -> np.ndarray:
    """Every ring clockwise, as GDAL's rasterizer takes them (GDALCollectRingsFromGeometry reverses the
    others): the orientation decides the cells whose centres lie on a horizontal edge."""
    n = len(xy)
    if n == 0:
        return xy
    lengths = np.diff(ring_off)
    ring = np.repeat(np.arange(len(lengths)), lengths)
    nxt = np.arange(1, n + 1)
    last = ring_off[1:] - 1
    nxt[last] = ring_off[:-1]  # wrap around within each ring
    cross = xy[:, 0] * xy[nxt, 1] - xy[nxt, 0] * xy[:, 1]
    area2 = np.bincount(ring, weights=cross, minlength=len(lengths))  # > 0: counter-clockwise
    flip = area2 > 0
    if not flip.any():
        return xy
    pos = np.arange(n)
    start = ring_off[:-1][ring]
    end = ring_off[1:][ring]
    idx = np.where(flip[ring], start + end - 1 - pos, pos)
    return np.ascontiguousarray(xy[idx])


def _columns(stat_names, labels, multi):
    if not multi:
        return list(stat_names)
    return [f"{b}_{s}" for b in labels for s in stat_names]


def _frame(out: np.ndarray, stat_names, labels, multi) -> pd.DataFrame:
    zones, bands, ns = out.shape
    df = pd.DataFrame(out.reshape(zones, bands * ns), columns=_columns(stat_names, labels, multi))
    for b, label in enumerate(labels):
        for k, s in enumerate(stat_names):
            if str(s).lower() in _COUNTS:
                col = _columns([s], [label], multi)[0]
                df[col] = df[col].fillna(0).astype(np.int64)
    return df


def zonal_stats(raster, zones, stats=("count", "mean", "min", "max", "std"), *, all_touched: bool = False,
                n_threads: int | None = None):
    """Statistics of a raster in every zone: polygons, or the labels of a raster.

    Mean NDVI per municipality, total population per watershed, the share of
    each land-cover class per property, the median priority rank of every
    protected area: computed in C++ in parallel, on every band of a stack at
    once. Polygons may overlap (e.g. buffers around points): each zone gets
    its own cells, the cells GDAL burns for it (:func:`rasterize`), reprojected
    to the raster's CRS first.

    Parameters
    ----------
    raster
        Georeferenced ``(y, x)`` or ``(band, y, x)`` DataArray (e.g. from
        :func:`load_raster`); NaN and its NoData are no data.
    zones
        Polygons (GeoDataFrame, GeoSeries, shapely geometries), or a raster of
        zone labels on the same grid (integers; its NoData and NaN are no zone),
        e.g. municipalities or land-cover classes.
    stats
        Any of ``"count"`` (cells with data), ``"nodata"`` (cells without),
        ``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"range"``, ``"std"``,
        ``"var"`` (population: over N), ``"median"``, ``"majority"`` and
        ``"minority"`` (the most and least frequent value; ties: the smallest),
        ``"unique"`` (distinct values) and percentiles ``"p10"``, ``"p90"``...
        (linear interpolation, as NumPy's).
    all_touched
        Polygons: every cell they touch (default: the cells whose centre is
        inside, as :func:`rasterize`).
    n_threads
        Threads (default: all cores but one); the result does not depend on it.

    Returns
    -------
    geopandas.GeoDataFrame or pandas.DataFrame
        For polygons, the zones with one column per statistic (per band and
        statistic, ``<band>_<stat>``, for a stack); for a raster of zones, one
        row per label (column ``zone``). Zones without data: count 0, NaN.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> import landschaft as ls
    >>> dem = ls.load_raster(ls.example_data("orelevation"))          # elevation (m)
    >>> x0, y0, x1, y1 = dem.rio.bounds()
    >>> halves = gpd.GeoDataFrame({"part": ["west", "east"]}, crs=dem.rio.crs,
    ...                           geometry=[box(x0, y0, (x0 + x1) / 2, y1), box((x0 + x1) / 2, y0, x1, y1)])
    >>> ls.zonal_stats(dem, halves, ["mean", "max", "p90"]).round(1)[["part", "mean", "max", "p90"]]
       part   mean     max     p90
    0  west  894.5  1252.0  1112.0
    1  east  903.5  1390.0  1228.0

    Zones from a raster: the cells of three elevation classes.

    >>> classes = ls.reclassify(dem, [(0, 800, 1), (800, 1200, 2), (1200, 3000, 3)])
    >>> ls.zonal_stats(dem, classes, ["count", "min", "max"])
       zone  count     min     max
    0     1  46678   433.0   799.0
    1     2  80653   800.0  1199.0
    2     3  10310  1200.0  1390.0
    """
    stat_names, codes, qs = _parse_stats(stats)
    raster, has_nodata, nodata, gt, labels, crs = _raster(raster)
    multi = raster.ndim == 3
    ny, nx = raster.sizes["y"], raster.sizes["x"]
    gdf = _geodataframe(zones, crs)
    with _threads.limit(n_threads):
        if gdf is None:
            from .io import match_grid

            if not isinstance(zones, (xr.DataArray, np.ndarray)) or np.ndim(zones) != 2:
                raise TypeError("zones must be polygons or a 2-D raster of zone labels")
            if isinstance(zones, xr.DataArray) and zones.rio.crs is not None and zones.shape != raster.shape[-2:]:
                zones = match_grid(zones, raster.isel({raster.dims[0]: 0}) if multi else raster, resampling="nearest")
            if tuple(np.shape(zones)) != (ny, nx):
                raise ValueError(f"the zones have {np.shape(zones)} cells, the raster {(ny, nx)}")
            znd = zones.rio.nodata if isinstance(zones, xr.DataArray) else None
            z_has = znd is not None and not (isinstance(znd, float) and np.isnan(znd))
            runs, zone_labels = _core.io_label_runs(_values(zones), z_has, float(znd) if z_has else 0.0)
        else:
            xy, ring_off, geom_rings = _polygon_arrays(gdf.geometry.values)
            runs = _core.io_polygon_runs(xy, ring_off, geom_rings, gt, nx, ny, all_touched)
        out = _core.io_zonal_stats(_values(raster), has_nodata, nodata, runs, codes, qs)
    table = _frame(out, stat_names, labels, multi)
    if gdf is None:
        zl = zone_labels
        zl = zl.astype(np.int64) if np.all(zl == np.round(zl)) else zl
        return pd.concat([pd.DataFrame({"zone": zl}), table], axis=1)
    result = gdf.reset_index(drop=True).copy()
    clash = [c for c in table.columns if c in result.columns]
    if clash:
        raise ValueError(f"the zones already have the columns {clash}; rename them")
    for c in table.columns:
        result[c] = table[c].to_numpy()
    return result


def _points(points, crs):
    """(x, y, the GeoDataFrame or None) of points in the raster's CRS."""
    gdf = _geodataframe(points, crs)
    if gdf is not None:
        kinds = set(gdf.geometry.geom_type.dropna())
        if kinds - {"Point"}:
            raise ValueError(f"extract takes points, got {', '.join(sorted(kinds - {'Point'}))}: use zonal_stats for "
                             "polygons, or extract(..., buffer=) for the cells around points")
        return gdf.geometry.x.to_numpy(np.float64), gdf.geometry.y.to_numpy(np.float64), gdf
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1]), None
    if arr.ndim == 2 and arr.shape[0] == 2:
        return np.ascontiguousarray(arr[0]), np.ascontiguousarray(arr[1]), None
    raise TypeError("points: a GeoDataFrame/GeoSeries of points, shapely points, an (n, 2) array or (x, y) arrays")


def extract(raster, points, *, method: str = "nearest", buffer: float | None = None, stats="mean",
            all_touched: bool = False, n_threads: int | None = None):
    """The values of a raster at points.

    Habitat suitability at the occurrence records, the resistance at the
    sampling sites, every band of a stack at a million points: computed in C++
    in parallel. The points are reprojected to the raster's CRS. A dask-backed
    raster (``load_raster(..., chunks="auto")``) is read only in the blocks of
    rows that hold points.

    Parameters
    ----------
    raster
        Georeferenced ``(y, x)`` or ``(band, y, x)`` DataArray; NaN and its
        NoData are no data.
    points
        GeoDataFrame or GeoSeries of points, shapely points, an ``(n, 2)`` array
        or a pair of ``x``, ``y`` arrays (in the raster's CRS).
    method
        ``"nearest"`` (default): the value of the cell holding the point;
        ``"bilinear"``: interpolated from the four nearest cell centres
        (weights renormalised over those with data).
    buffer
        A radius in the units of the raster's CRS: the statistics ``stats`` of
        the cells around every point (see :func:`zonal_stats`) instead of one
        value.
    stats
        With ``buffer``: the statistics, as :func:`zonal_stats` names them.
    all_touched
        With ``buffer``: every cell the circle touches.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    geopandas.GeoDataFrame or pandas.DataFrame
        The points with one column per band (named by the band, or ``value``),
        or per statistic with ``buffer``; NaN outside the raster or on no data.
        Coordinates give a DataFrame with columns ``x`` and ``y``.

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> dem = ls.load_raster(ls.example_data("maungawhau"))       # elevation (m), 10 m cells
    >>> table = pd.DataFrame({"site": ["summit", "slope"], "x": [2667705.0, 2667812.0], "y": [6478895.0, 6479004.0]})
    >>> sites = ls.load_vector(ls.write_table(table, "sites.csv"), crs=dem.rio.crs)
    >>> ls.extract(dem, sites)[["site", "maungawhau"]]
         site  maungawhau
    0  summit       195.0
    1   slope       168.0
    >>> ls.extract(dem, sites, method="bilinear")["maungawhau"].round(1).tolist()
    [195.0, 169.2]

    The cells within 50 m of every site:

    >>> ls.extract(dem, sites, buffer=50, stats=["mean", "max"])[["site", "mean", "max"]].round(1)
         site   mean    max
    0  summit  186.5  195.0
    1   slope  168.2  180.0
    """
    if method not in ("nearest", "bilinear"):
        raise ValueError("method must be 'nearest' or 'bilinear'")
    raster, has_nodata, nodata, gt, labels, crs = _raster(raster)
    if buffer is not None:
        x, y, gdf = _points(points, crs)
        import geopandas as gpd

        base = gdf if gdf is not None else gpd.GeoDataFrame({"x": x, "y": y}, geometry=gpd.points_from_xy(x, y),
                                                             crs=crs)
        circles = base.geometry.buffer(float(buffer))
        table = zonal_stats(raster, circles, stats, all_touched=all_touched, n_threads=n_threads)
        result = base.reset_index(drop=True).copy()
        for c in table.columns:
            if c != "geometry":
                result[c] = table[c].to_numpy()
        return result if gdf is not None else pd.DataFrame(result.drop(columns="geometry"))
    x, y, gdf = _points(points, crs)
    bilinear = method == "bilinear"
    data = raster.data
    bands = raster.shape[0] if raster.ndim == 3 else 1
    with _threads.limit(n_threads):
        if hasattr(data, "dask"):
            out = np.full((x.size, bands), np.nan)
            ny = raster.sizes["y"]
            row = np.floor((y - gt[3]) / gt[5])
            y0 = 0
            for nr in data.chunks[-2]:
                sel = np.flatnonzero((row >= y0) & (row < y0 + nr))  # the points of this block of rows
                if sel.size:
                    a, b = max(0, y0 - 1), min(ny, y0 + nr + 1)  # one row around, for bilinear
                    slab = _values(data[..., a:b, :])
                    out[sel] = _core.io_extract(slab, has_nodata, nodata, gt, x[sel], y[sel], bilinear, a, ny)
                y0 += nr
        else:
            out = _core.io_extract(_values(raster), has_nodata, nodata, gt, x, y, bilinear)
    if gdf is not None:
        result = gdf.reset_index(drop=True).copy()
        clash = [c for c in labels if c in result.columns]
        if clash:
            raise ValueError(f"the points already have the columns {clash}; rename them")
    else:
        result = pd.DataFrame({"x": x, "y": y})
    for b, label in enumerate(labels):
        result[label] = out[:, b]
    return result
