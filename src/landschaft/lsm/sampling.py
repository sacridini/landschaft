# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics of sample plots and moving windows (landscapemetrics'
``sample_lsm`` and ``window_lsm``)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import Affine

from .. import _core
from .catalog import add_full_name
from .utils import raster_extent
from .landscape import (
    _prepare,
    as_layers,
    chosen_metrics,
    core_options,
    long_frame,
    metric_options,
    metric_specs,
)

__all__ = ["sample_metrics", "scale_sample", "window_metrics"]


# -- sample plots ---------------------------------------------------------------


def _r_seq(start: float, stop: float, n: int) -> np.ndarray:
    """R's seq(start, stop, length.out = n)."""
    by = (stop - start) / (n - 1)
    return np.concatenate([[start], start + np.arange(1, n - 1) * by, [stop]])


_CIRCLE = _r_seq(0.0, 2 * np.pi, 100)
_CIRCLE_SIN, _CIRCLE_COS = _core.lsm_r_sincos(_CIRCLE)  # R's bits (x87 fsin / fcos)


def _buffer(x: float, y: float, size: float, shape: str) -> dict:
    """landscapemetrics' construct_buffer: a square (half side `size`) or a
    100-vertex circle (radius `size`) around (x, y), as a GeoJSON-like
    polygon whose ring is a closed (n, 2) array."""
    if shape == "square":
        ring = np.array([(x - size, y - size), (x - size, y + size), (x + size, y + size), (x + size, y - size),
                         (x - size, y - size)])
        return {"type": "Polygon", "coordinates": [ring], "rectangle": True}
    if shape == "circle":
        ring = np.empty((101, 2))
        ring[:100, 0] = _CIRCLE_SIN * size + x
        ring[:100, 1] = _CIRCLE_COS * size + y
        ring[100] = ring[0]
        return {"type": "Polygon", "coordinates": [ring]}
    raise ValueError(f"shape must be 'square' or 'circle', got {shape!r}")


def _geometries(y) -> list[dict]:
    """GeoJSON-like geometries of `y` (objects with __geo_interface__, dicts,
    or lists of them)."""
    if hasattr(y, "__geo_interface__"):
        gi = y.__geo_interface__
        if gi.get("type") == "FeatureCollection":
            return [f["geometry"] for f in gi["features"]]
        if gi.get("type") == "Feature":
            return [gi["geometry"]]
        return [gi]
    if isinstance(y, dict):
        if y.get("type") == "FeatureCollection":
            return [f["geometry"] for f in y["features"]]
        return [y.get("geometry", y)]
    if isinstance(y, Iterable) and not isinstance(y, (str, bytes)):
        out = []
        for g in y:
            out += _geometries(g)
        return out
    raise TypeError(f"expected coordinates or geometries, got {type(y).__name__}")


def _plots(y, size, shape) -> list[dict]:
    """Sample plots: buffers around points (coordinate matrix or point
    geometries) or polygons as given."""
    arr = None
    if not hasattr(y, "__geo_interface__") and not isinstance(y, dict):
        try:
            arr = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            arr = None
    if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
        points = [tuple(p) for p in arr]
        geoms = [{"type": "Point", "coordinates": p} for p in points]
    else:
        geoms = _geometries(y)
    plots = []
    for g in geoms:
        kind = g["type"]
        if kind in ("Point", "MultiPoint"):
            if size is None or size <= 0:
                raise ValueError("points need a buffer size > 0 (size=)")
            pts = [g["coordinates"]] if kind == "Point" else g["coordinates"]
            for p in pts:
                plots.append(_buffer(float(p[0]), float(p[1]), float(size), shape))
        elif kind in ("Polygon", "MultiPolygon"):
            plots.append(g)
        else:
            raise ValueError(f"sample plots must be points or polygons, got {kind}")
    return plots


def _rings(geom: dict):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        for k, ring in enumerate(poly):
            yield k == 0, np.asarray(ring, dtype=float)[:, :2]


def _area_function(crs):
    """Polygon area as terra::expanse(transform = TRUE): geodesic (WGS84
    ellipsoid) when the plots have a CRS, else planar."""
    if crs is None:

        def ring_area(r):
            return abs(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])) / 2

    else:
        from pyproj import CRS, Geod, Transformer

        to_lonlat = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
        geod = Geod(ellps="WGS84")

        def ring_area(r):
            lon, lat = to_lonlat.transform(r[:, 0], r[:, 1])
            return abs(geod.polygon_area_perimeter(lon, lat)[0])

    def area(geom: dict) -> float:
        total = 0.0
        for exterior, r in _rings(geom):
            a = ring_area(r)
            total += a if exterior else -a
        return total

    return area


def _crop(geom: dict, transform: Affine, shape: tuple[int, int]):
    """terra::crop(snap = "near", mask = TRUE, touches = TRUE): the grid window
    around the plot and the cells it touches (GDAL's all-touched
    rasterization, ported to C++). None if outside the raster."""
    coords = np.concatenate([r for _, r in _rings(geom)])
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    rx, ry = transform.a, -transform.e
    nrow, ncol = shape
    x0, ytop = transform.c, transform.f
    y0 = ytop - nrow * ry

    def near(v):
        return int(np.floor(v + 0.5))

    c0, c1 = near((xmin - x0) / rx), near((xmax - x0) / rx)
    b0, b1 = near((ymin - y0) / ry), near((ymax - y0) / ry)  # rows counted from the bottom
    collapsed = c0 == c1 or b0 == b1
    if c0 == c1:
        c0 -= 1
    if b0 == b1:
        b0 -= 1
    c0, c1 = max(c0, 0), min(c1, ncol)
    b0, b1 = max(b0, 0), min(b1, nrow)
    if c1 <= c0 or b1 <= b0:
        return None
    rows = slice(nrow - b1, nrow - b0)
    cols = slice(c0, c1)
    if not collapsed and (geom.get("rectangle") or _is_rectangle(geom)):
        # the window snapped to the nearest cell edges overlaps an
        # axis-aligned rectangle by at least half a cell in every cell
        inside = np.ones((b1 - b0, c1 - c0), dtype=bool)
    else:
        # GDAL's all-touched rasterization (C++ port: same cells, no GDAL call per plot)
        rings = [np.ascontiguousarray(r, dtype=float) for _, r in _rings(geom)]
        gt = (x0 + c0 * rx, rx, 0.0, y0 + b1 * ry, 0.0, -ry)
        inside = _core.lsm_rasterize(rings, gt, c1 - c0, b1 - b0).astype(bool)
    return rows, cols, inside


def _is_rectangle(geom: dict) -> bool:
    """An axis-aligned rectangle (e.g. a square buffer)."""
    if geom["type"] != "Polygon" or len(geom["coordinates"]) != 1:
        return False
    ring = np.asarray(geom["coordinates"][0], dtype=float)[:, :2]
    if len(ring) != 5 or not np.array_equal(ring[0], ring[-1]):
        return False
    xs, ys = np.unique(ring[:, 0]), np.unique(ring[:, 1])
    if len(xs) != 2 or len(ys) != 2:
        return False
    # consecutive vertices differ in exactly one coordinate
    step = np.diff(ring, axis=0) != 0
    return bool(np.all(step.sum(axis=1) == 1))


def _grid(data, res, nodata, band, with_crs: bool = False):
    """Class values, NA mask (bool), cell size and geotransform of a raster
    (and its CRS, or None, with `with_crs`)."""
    values, mask, r, template = _prepare(data, res, nodata, band)
    mask = np.zeros(values.shape, dtype=bool) if mask is None else mask.astype(bool)
    transform = crs = None
    if template is not None:
        try:
            transform = template.rio.transform()
            crs = template.rio.crs
        except Exception:
            transform = None
    if transform is None:
        # as terra::rast(matrix): origin at 0, first row on top
        transform = Affine(r[0], 0.0, 0.0, 0.0, -r[1], values.shape[0] * r[1])
    if transform.b != 0 or transform.d != 0:
        raise ValueError("rotated rasters are not supported")
    return (values, mask, r, transform, crs) if with_crs else (values, mask, r, transform)


def grid_array(data: np.ndarray, transform: Affine, crs=None, name: str | None = None) -> xr.DataArray:
    """A DataArray on the grid of `transform` (cell-centre coordinates), with
    NaN as nodata."""
    ny, nx = data.shape
    x = transform.c + (np.arange(nx) + 0.5) * transform.a
    y = transform.f + (np.arange(ny) + 0.5) * transform.e
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name=name)
    da = da.rio.write_transform(transform)
    if crs is not None:
        da = da.rio.write_crs(crs)
    return da.rio.write_nodata(np.nan, encoded=False)


def c_round(v: float) -> float:
    """C's round(): halves away from zero."""
    t = math.trunc(v)
    if abs(v - t) >= 0.5:
        t += 1 if v > 0 else -1
    return float(t)


def _is_equal(a: float, b: float, tolerance: float = 10.0) -> bool:
    """terra's is_equal (math_utils.cpp)."""
    tol = max(tolerance, abs(min(a, b))) * np.finfo(float).eps
    return a == b or abs(a - b) < tol


def terra_grid(data: np.ndarray, extent, crs=None) -> xr.DataArray:
    """A raster on terra's extent (xmin, xmax, ymin, ymax): the geotransform
    holds xmin, ymax and terra's resolution ((xmax - xmin) / ncol) exactly;
    xmax and ymin come back from it to within an ulp or so."""
    ny, nx = data.shape
    xmin, xmax, ymin, ymax = extent
    return grid_array(data, Affine((xmax - xmin) / nx, 0.0, xmin, 0.0, -(ymax - ymin) / ny, ymax), crs)


def terra_crop_extent(geom: dict, extent, shape) -> tuple[float, float, float, float]:
    """Extent (xmin, xmax, ymin, ymax) of ``terra::crop(x, plot, mask =
    TRUE)`` (SpatRaster::cropmask): the plot's extent snapped to the cell
    edges through terra's origin (align), cut to the raster, snapped again by
    setWindow, which keeps the whole raster when the window is within a tenth
    of a cell of it and otherwise rebuilds xmax and ymax from the cell counts
    (setExtent).

    terra's cropmask also rewrites the extent of the raster it crops
    (removeWindow rebuilds it with the window's cell size), so in
    ``sample_lsm`` the plots after the first one see an extent moved by a few
    ulps; that side effect is not reproduced (same cells and values)."""
    coords = np.concatenate([r for _, r in _rings(geom)])
    pxmin, pymin = coords.min(axis=0)
    pxmax, pymax = coords.max(axis=0)
    rxmin, rxmax, rymin, rymax = extent
    ny, nx = shape
    xres, yres = (rxmax - rxmin) / nx, (rymax - rymin) / ny
    ox = rxmin - xres * c_round(rxmin / xres)
    oy = rymax - yres * c_round(rymax / yres)
    if _is_equal(xres + ox, abs(ox)):
        ox = abs(ox)
    if _is_equal(yres + oy, abs(oy)):
        oy = abs(oy)

    def align(e):
        xmn = c_round((e[0] - ox) / xres) * xres + ox
        xmx = c_round((e[1] - ox) / xres) * xres + ox
        ymn = c_round((e[2] - oy) / yres) * yres + oy
        ymx = c_round((e[3] - oy) / yres) * yres + oy
        if xmn == xmx:
            if xmn < e[0]:
                xmx += xres
            else:
                xmn -= xres
        if ymn == ymx:
            if ymn < e[2]:
                ymx += yres
            else:
                ymn -= yres
        return xmn, xmx, ymn, ymx

    def intersect(e):
        return max(e[0], rxmin), min(e[1], rxmax), max(e[2], rymin), min(e[3], rymax)

    x = align(intersect(align((pxmin, pxmax, pymin, pymax))))
    tol = 0.1 * xres
    if all(abs(a - b) <= tol for a, b in zip(x, extent)):
        return tuple(extent)
    nc = max(1.0, c_round((x[1] - x[0]) / xres))
    nr = max(1.0, c_round((x[3] - x[2]) / yres))
    return x[0], x[0] + nc * xres, x[2], x[2] + nr * yres


def _crop_raster(values, mask, extent, crs, geom, crop) -> xr.DataArray:
    """The plot's raster as ``terra::crop(mask = TRUE)`` returns it: the
    snapped window (terra's extent), NA outside the plot."""
    rows, cols, inside = crop
    sub = values[rows, cols].astype(np.float64)
    sub[mask[rows, cols] | ~inside] = np.nan
    return terra_grid(sub, terra_crop_extent(geom, extent, values.shape), crs)


def sample_metrics(
    data,
    y,
    *,
    size: float | None = None,
    shape: str = "square",
    plot_id=None,
    all_classes: bool = False,
    level=None,
    metric=None,
    name=None,
    type=None,
    what=None,
    res=None,
    nodata=None,
    band: int | None = None,
    verbose: bool = True,
    extras: bool = False,
    full_name: bool = False,
    return_raster: bool = False,
    n_threads: int | None = None,
    **options,
) -> pd.DataFrame:
    """Landscape metrics of sample plots, like landscapemetrics' ``sample_lsm``.

    Each plot is cut from the raster as ``terra::crop(mask = TRUE)`` does
    (grid window snapped to the nearest cell edges; every cell the plot
    touches is kept, the others become NA) and its metrics are computed as
    a landscape of its own. All plots are computed in parallel.

    Parameters
    ----------
    data
        A raster (see :class:`Landscape`) or a list of rasters /
        :class:`~landschaft.RasterCollection` (``layer`` column).
    y
        Sample plots: an ``(n, 2)`` array of x/y coordinates or point
        geometries (buffers of ``shape`` and ``size`` around them), or
        polygons. Geometries can be shapely objects, a GeoSeries /
        GeoDataFrame (anything with ``__geo_interface__``) or GeoJSON-like
        dicts, in the raster's coordinates. As in landscapemetrics, the plot
        area behind ``percentage_inside`` is geodesic when ``y`` has a CRS
        (``y.crs``, e.g. a GeoDataFrame) and planar otherwise.
    size
        Buffer size around points: half the side of a square, or the radius
        of a circle (landscapemetrics' 100-vertex polygon).
    shape
        ``"square"`` (default) or ``"circle"``.
    plot_id
        Identifiers of the plots (default 1..n).
    all_classes
        Add class-level rows (NA) for the classes of the whole raster that are
        missing in a plot (with a row for class NA when the raster has NA
        cells, as landscapemetrics does).
    level, metric, name, type, what
        Metric selection, as in :func:`landschaft.list_metrics`.
    res, nodata, band
        Raster reading options (see :class:`Landscape`).
    verbose
        Warn when a plot lies less than 90 % inside the raster.
    extras
        Also select the FRAGSTATS metrics missing in landscapemetrics.
    full_name
        Add the columns ``name``, ``type`` and ``function_name`` after
        ``value`` (landscapemetrics' ``sample_lsm(full_name = TRUE)`` drops
        ``function_name`` when ``return_raster = FALSE``, by a fixed column
        index; it is kept here).
    return_raster
        Add a column ``raster_sample_plots`` with each plot's raster (an
        ``xarray.DataArray``: the snapped window, NaN outside the plot and on
        NA cells; ``None`` for a plot outside the raster).
    n_threads
        Threads of the C++ core (default: all cores but one).
    **options
        Metric options (``directions``, ``edge_depth``, ...).

    Returns
    -------
    pandas.DataFrame
        ``level``, ``class``, ``id``, ``metric``, ``value``, ``plot_id`` and
        ``percentage_inside`` (area of the plot's cells over the plot area,
        in %), ordered by plot, level, metric, class and id; plus ``layer``
        for several rasters. A plot outside the raster (an error in
        landscapemetrics) gives NA metrics and 0 %.
    """
    opts = core_options(metric_options(**options), n_threads)
    chosen = chosen_metrics(level, metric, name, type, what, extras)
    plots = _plots(y, size, shape)
    n_plots = len(plots)
    if plot_id is not None:
        plot_id = list(plot_id)
        if len(plot_id) != n_plots:
            warnings.warn("plot_id does not have one value per plot; using 1..n", stacklevel=2)
            plot_id = None
    ids = np.arange(1, n_plots + 1) if plot_id is None else np.asarray(plot_id)
    area = _area_function(getattr(y, "crs", None))
    areas = np.array([area(g) for g in plots])

    layers = as_layers(data)
    single = layers is None
    frames = []
    for layer_no, layer in enumerate([data] if single else layers, start=1):
        values, mask, r, transform, crs = _grid(layer, res, nodata, band, with_crs=True)
        rasters, inside_area = [], np.zeros(n_plots)
        crops = np.full(n_plots, None, dtype=object)
        extent = raster_extent(transform, values.shape, r)
        empty = (np.zeros((1, 1), np.int32), np.ones((1, 1), np.uint8), r[0], r[1])
        for k, geom in enumerate(plots):
            crop = _crop(geom, transform, values.shape)
            if crop is None:
                rasters.append(empty)
                continue
            if return_raster:
                crops[k] = _crop_raster(values, mask, extent, crs, geom, crop)
            rows, cols, inside = crop
            sub_mask = mask[rows, cols] | ~inside
            sub = np.ascontiguousarray(values[rows, cols])
            inside_area[k] = np.count_nonzero(~sub_mask) * r[0] * r[1] / 10000
            rasters.append((sub, sub_mask.astype(np.uint8), r[0], r[1]))
        table = _core.lsm_batch(rasters, opts, *metric_specs(chosen))
        df = long_frame(table, chosen)
        item = table["item"].astype(np.int64)
        df["plot_id"] = ids[item]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(inside_area > 0, inside_area / (areas / 10000) * 100, 0.0)
        df["percentage_inside"] = pct[item]
        if return_raster:
            df["raster_sample_plots"] = crops[item]
        if all_classes:
            df = _sort(_add_all_classes(df, values, mask), True)
        elif np.any(ids[1:] < ids[:-1]):
            # rows come per plot, each in sample_lsm's order: order the plots
            rank = np.empty(n_plots, dtype=np.int64)
            rank[np.argsort(ids, kind="stable")] = np.arange(n_plots)
            df = df.iloc[np.argsort(rank[item], kind="stable")].reset_index(drop=True)
        if not single:
            df.insert(0, "layer", layer_no)
        frames.append(df)
    out = frames[0] if single else pd.concat(frames, ignore_index=True)
    if verbose and (out["percentage_inside"] < 90).any():
        warnings.warn("percentage_inside is below 90% for at least one plot", stacklevel=2)
    return add_full_name(out) if full_name else out


def scale_sample(
    data,
    y,
    *,
    size,
    shape: str = "square",
    level=None,
    metric=None,
    name=None,
    type=None,
    what=None,
    verbose: bool = True,
    full_name: bool = False,
    **options,
) -> pd.DataFrame:
    """Landscape metrics of buffers of several sizes around points, like
    landscapemetrics' ``scale_sample``.

    :func:`sample_metrics` for each buffer size in ``size`` (the plots of all
    sizes are computed in parallel). ``y``: an ``(n, 2)`` array of
    coordinates or point geometries; other arguments (``full_name`` too) as
    in :func:`sample_metrics`.

    Returns
    -------
    pandas.DataFrame
        ``level``, ``class``, ``id``, ``metric``, ``value``, ``size``,
        ``plot_id``, ``percentage_inside`` (plus ``layer``), ordered by
        (layer,) plot, level, metric, class, id and size.
    """
    sizes = [float(s) for s in np.atleast_1d(size)]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("size must be one or more positive buffer sizes")
    try:
        coords = np.asarray(y, dtype=float)
        is_matrix = coords.ndim == 2 and coords.shape[1] == 2
    except (TypeError, ValueError):
        is_matrix = False
    if not is_matrix and any(g["type"] not in ("Point", "MultiPoint") for g in _geometries(y)):
        raise ValueError("scale_sample takes points")
    frames = []
    for s in sizes:
        df = sample_metrics(data, y, size=s, shape=shape, level=level, metric=metric, name=name, type=type,
                            what=what, verbose=False, **options)
        df.insert(df.columns.get_loc("plot_id"), "size", s)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    keys = (["layer"] if "layer" in out.columns else []) + ["plot_id", "level", "metric", "class", "id", "size"]
    tmp = out.assign(level=out["level"].astype(str), metric=out["metric"].astype(str))
    out = out.loc[tmp.sort_values(keys, kind="stable", na_position="last").index].reset_index(drop=True)
    if verbose and (out["percentage_inside"] < 90).any():
        warnings.warn("percentage_inside is below 90% for at least one buffer", stacklevel=2)
    return add_full_name(out) if full_name else out


def _add_all_classes(df: pd.DataFrame, values: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Class-level rows for every class of the raster (and NA if it has NA
    cells) in every plot, NA where the plot lacks the class."""
    is_class = (df["level"] == "class").to_numpy()
    if not is_class.any():
        return df
    cls_rows = df[is_class]
    classes = [int(c) for c in np.unique(values[~mask])]
    if mask.any():
        classes.append(None)
    known = {
        (None if pd.isna(c) else int(c), m, p): v
        for c, m, p, v in zip(cls_rows["class"], cls_rows["metric"], cls_rows["plot_id"], cls_rows["value"])
    }
    pct = dict(zip(df["plot_id"], df["percentage_inside"]))
    crops = dict(zip(df["plot_id"], df["raster_sample_plots"])) if "raster_sample_plots" in df.columns else None
    metrics = list(dict.fromkeys(cls_rows["metric"]))
    plots = list(dict.fromkeys(cls_rows["plot_id"]))
    grid = [(c, m, p) for c in classes for m in metrics for p in plots]
    added = pd.DataFrame(
        {
            "level": pd.Categorical(["class"] * len(grid), categories=df["level"].cat.categories),
            "class": pd.array([c for c, _, _ in grid], dtype="Int64"),
            "id": pd.array([None] * len(grid), dtype="Int64"),
            "metric": pd.Categorical([m for _, m, _ in grid], categories=df["metric"].cat.categories),
            "value": [known.get(key, np.nan) for key in grid],
            "plot_id": [p for _, _, p in grid],
            "percentage_inside": [pct[p] for _, _, p in grid],
        }
    )
    if crops is not None:
        column = np.empty(len(grid), dtype=object)
        for k, (_, _, p) in enumerate(grid):
            column[k] = crops[p]
        added["raster_sample_plots"] = column
    return pd.concat([df[~is_class], added], ignore_index=True)


def _sort(df: pd.DataFrame, single: bool) -> pd.DataFrame:
    """sample_lsm's order: (layer,) plot_id, level, metric, class, id (NA last)."""
    keys = [] if single else ["layer"]
    keys += ["plot_id", "level", "metric", "class", "id"]
    tmp = df.assign(
        level=df["level"].astype(str), metric=df["metric"].astype(str)
    )  # alphabetical, as R's order()
    order = tmp.sort_values(keys, kind="stable", na_position="last").index
    return df.loc[order].reset_index(drop=True)


# -- moving windows -------------------------------------------------------------


def _window(window) -> np.ndarray:
    if np.isscalar(window):
        side = int(window)
        if side != window or side < 1:
            raise ValueError(f"window size must be a positive odd integer, got {window!r}")
        w = np.ones((side, side), dtype=bool)
    else:
        w = np.asarray(window)
        if w.ndim != 2:
            raise ValueError("the window must be a 2-D array")
        w = ~np.isnan(w) & (w != 0) if np.issubdtype(w.dtype, np.floating) else w.astype(bool)
    if w.shape[0] % 2 == 0 or w.shape[1] % 2 == 0:
        raise ValueError(f"the window must have odd sides, got {w.shape}")
    return w


def window_metrics(
    data,
    window,
    level="landscape",
    metric=None,
    name=None,
    type=None,
    what=None,
    *,
    res=None,
    nodata=None,
    band: int | None = None,
    extras: bool = False,
    n_threads: int | None = None,
    **options,
):
    """Landscape-level metrics in a moving window, like landscapemetrics'
    ``window_lsm``.

    For every cell, the cells of the window centred on it form a landscape
    (cells outside the raster or outside the window shape are NA), and its
    metrics become the cell's values. Windows are computed in parallel.

    Parameters
    ----------
    data
        A raster (see :class:`Landscape`) or a list of rasters /
        :class:`~landschaft.RasterCollection`.
    window
        Odd side of a square window, or a 2-D array with odd sides whose
        cells that are True / non-zero (and not NaN) form the window, e.g. a
        circle.
    level, metric, name, type, what
        Metric selection, as in :func:`landschaft.list_metrics`; only
        landscape-level metrics are possible (all of them by default).
    res, nodata, band
        Raster reading options (see :class:`Landscape`).
    extras
        Also select the FRAGSTATS metrics missing in landscapemetrics.
    n_threads
        Threads of the C++ core (default: all cores but one).
    **options
        Metric options (``directions``, ``edge_depth``, ...).

    Returns
    -------
    dict
        ``{metric: raster}`` (a DataArray on the input grid, or a numpy
        array), or a list of such dicts for several rasters.
    """
    opts = core_options(metric_options(**options), n_threads)
    chosen = chosen_metrics(level, metric, name, type, what, extras)
    if set(chosen["level"]) != {"landscape"}:
        raise ValueError("window_metrics computes landscape-level metrics only")
    names = list(chosen["metric"])
    w = _window(window).astype(np.uint8)
    layers = as_layers(data)
    results = []
    for layer in [data] if layers is None else layers:
        values, mask, r, template = _prepare(layer, res, nodata, band)
        out = _core.lsm_window(values, mask, r[0], r[1], w, names, opts)
        maps = {}
        for k, m in enumerate(names):
            if template is None:
                maps[m] = out[k]
            else:
                da = template.copy(data=out[k])
                da.name = m
                da.attrs = {}
                da.encoding = {}
                maps[m] = da.rio.write_nodata(np.nan, encoded=False)
        results.append(maps)
    return results[0] if layers is None else results
