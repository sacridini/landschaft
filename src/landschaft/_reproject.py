# SPDX-License-Identifier: GPL-3.0-or-later
"""Reprojection of rasters and vectors (ls.reproject), as terra's project().

GDAL's warp of a whole raster depends on how GDAL splits it: its approximate coordinate
transformation (1/8 cell) interpolates along the rows of each chunk, and its kernels take
the scale of each chunk, whose size follows its memory limit (changing that limit alone
changes cells of a bilinear warp). Here the destination is always split in the same blocks,
anchored at its origin, each warped by GDAL on its own with the kernel scale of the whole
grid: in memory, lazily or from a file, with any number of threads, the same GDAL calls
give the same bits. tolerance=0 transforms every cell exactly (the bits of GDAL warping the
whole destination at once, but for the last bit of continuous values).
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import array_bounds

from .io import _RESAMPLING_NAMES, _grid_template, _load_single, _transform, _with_georef

__all__ = ["reproject"]

# rasterio silences this warning of the warp's in-memory datasets (which get their
# transform just after) with warnings.catch_warnings, which is not thread-safe: blocks
# warped in parallel let it through.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning, module=r"rasterio\.warp")

# Kernel radius of GDAL's resampling methods, in source cells at scale 1 (the window of a
# block is widened by it, scaled when downsampling); the statistics take the footprint.
_RADIUS = {"nearest": 1, "bilinear": 1, "cubic": 2, "cubic_spline": 2, "lanczos": 3}
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb", ".kml", ".gml", ".csv", ".txt", ".tsv",
                    ".parquet", ".geoparquet", ".pq", ".xlsx", ".xls"}
_BLOCK = 1024  # cells a side of the blocks the destination is warped in
_DENSIFY = 51  # points per edge when the bounds of the whole destination are transformed
_LATTICE = 11  # points a side of a block whose source window is computed


def _is_vector(obj) -> bool:
    gpd = _geopandas()
    if isinstance(obj, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return True
    if isinstance(obj, (str, os.PathLike)):
        if Path(os.fspath(obj)).suffix.lower() in _VECTOR_SUFFIXES:
            return True
        try:
            with rasterio.open(obj):
                return False
        except rasterio.errors.RasterioIOError:
            return True
    return False


def _geopandas():
    from ._vector import _gpd

    return _gpd()


def _parse_crs(crs) -> CRS:
    """A CRS from anything that names one: EPSG code, string, WKT, pyproj or rasterio CRS,
    or a raster or vector (object or path) whose CRS is taken."""
    from ._vector import _crs_of

    if isinstance(crs, xr.DataArray) or hasattr(crs, "geometry"):
        found = _crs_of(crs)
    elif isinstance(crs, os.PathLike) or (isinstance(crs, str) and Path(crs).is_file()):
        found = _crs_of(crs)
    else:
        try:
            return CRS.from_user_input(crs)
        except Exception as err:  # noqa: BLE001 - rasterio and pyproj raise different errors
            raise ValueError(f"cannot read a CRS from {crs!r}: {err}") from None
    if found is None:
        raise ValueError(f"{crs!r} has no CRS")
    return CRS.from_user_input(found)


def _resolution(res):
    if res is None:
        return None
    if isinstance(res, (int, float, np.integer, np.floating)):
        res = (float(res), float(res))
    res = tuple(float(abs(v)) for v in res)
    if len(res) != 2 or not all(v > 0 for v in res):
        raise ValueError(f"res must be a positive number or (xres, yres), got {res!r}")
    return res


def _grid_size(da: xr.DataArray) -> tuple[int, int]:
    return da.sizes["y"], da.sizes["x"]


def _source_window(src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs, pad=0):
    """The source cells (row0, row1, col0, col1) that a destination grid sees, widened by
    ``pad``; None when it sees none, the whole source when its bounds cannot be transformed."""
    from rasterio.warp import transform_bounds

    h, w = src_shape
    try:
        x0, y0, x1, y1 = transform_bounds(dst_crs, src_crs, *array_bounds(*dst_shape, dst_t),
                                          densify_pts=_DENSIFY)
    except Exception:  # noqa: BLE001 - outside the domain of the projection
        return 0, h, 0, w
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        return 0, h, 0, w
    cols = sorted(((x0 - src_t.c) / src_t.a, (x1 - src_t.c) / src_t.a))
    rows = sorted(((y1 - src_t.f) / src_t.e, (y0 - src_t.f) / src_t.e))
    c0, c1 = max(0, math.floor(cols[0]) - pad), min(w, math.ceil(cols[1]) + pad)
    r0, r1 = max(0, math.floor(rows[0]) - pad), min(h, math.ceil(rows[1]) + pad)
    if c1 <= c0 or r1 <= r0:
        return None
    return r0, r1, c0, c1


def _block_windows(src_shape, src_t, src_crs, dst_t, dst_crs, blocks, pad) -> list:
    """The source window of every block (row, col, height, width) of the destination, as
    ``_source_window``, from an 11 x 11 lattice of points of each block (its edges included)
    transformed at once (transform_bounds builds its transformation again on every call)."""
    from pyproj import Transformer

    h, w = src_shape
    k = np.linspace(0.0, 1.0, _LATTICE)
    b = np.asarray(blocks, dtype=np.float64)  # r0, c0, bh, bw
    cols = b[:, 1, None, None] + b[:, 3, None, None] * k[None, None, :]
    rows = b[:, 0, None, None] + b[:, 2, None, None] * k[None, :, None]
    cols, rows = np.broadcast_arrays(cols, rows)
    x = dst_t.c + dst_t.a * cols
    y = dst_t.f + dst_t.e * rows
    sx, sy = Transformer.from_crs(dst_crs, src_crs, always_xy=True).transform(x.ravel(), y.ravel(), errcheck=False)
    sc = ((np.asarray(sx) - src_t.c) / src_t.a).reshape(len(blocks), -1)
    sr = ((np.asarray(sy) - src_t.f) / src_t.e).reshape(len(blocks), -1)
    out = []
    for c, r in zip(sc, sr):
        if not (np.isfinite(c).all() and np.isfinite(r).all()):  # partly outside the projection's domain
            out.append((0, h, 0, w))
            continue
        c0, c1 = max(0, math.floor(c.min()) - pad), min(w, math.ceil(c.max()) + pad)
        r0, r1 = max(0, math.floor(r.min()) - pad), min(h, math.ceil(r.max()) + pad)
        out.append((r0, r1, c0, c1) if c1 > c0 and r1 > r0 else None)
    return out


def _scales(src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs) -> dict:
    """GDAL's XSCALE / YSCALE for the whole grid (destination cells per source cell, as GDAL
    takes them for one chunk holding the whole destination)."""
    win = _source_window(src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs)
    if win is None:
        return {}
    r0, r1, c0, c1 = win
    return {"XSCALE": repr(dst_shape[1] / (c1 - c0)), "YSCALE": repr(dst_shape[0] / (r1 - r0))}


def _nodata(data: xr.DataArray):
    """(source NoData, destination NoData, the type the warp works in)."""
    dtype = data.dtype
    nodata = data.rio.nodata
    if dtype == np.bool_:
        dtype = np.dtype(np.uint8)
    elif dtype == np.float16:
        dtype = np.dtype(np.float32)
    if np.issubdtype(dtype, np.floating):
        value = np.nan if nodata is None or np.isnan(nodata) else float(nodata)
        return value, value, dtype
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"reproject: cannot warp values of type {dtype}")
    if nodata is not None and not np.isnan(nodata):
        return nodata, nodata, dtype
    info = np.iinfo(dtype)
    return None, (info.max if info.min == 0 else info.min), dtype


def _target_grid(data: xr.DataArray, crs, like, res):
    """(transform, (height, width), CRS, x and y coordinates or None) of the destination."""
    from rasterio.warp import calculate_default_transform

    if like is not None and not _is_vector(like):
        if res is not None:
            raise ValueError("res= cannot be given with a raster like=: the result takes its grid")
        template = _grid_template(like)
        return (_transform(template), _grid_size(template), CRS.from_user_input(template.rio.crs),
                (template["x"].values, template["y"].values))
    target = _parse_crs(like if like is not None else crs) if (like is not None or crs is not None) \
        else CRS.from_user_input(data.rio.crs)
    src_t = _transform(data)
    h, w = _grid_size(data)
    resolution = _resolution(res)
    kwargs = {"resolution": resolution} if resolution is not None else {}
    dst_t, dw, dh = calculate_default_transform(CRS.from_user_input(data.rio.crs), target, w, h,
                                                *array_bounds(h, w, src_t), **kwargs)
    return dst_t, (dh, dw), target, None


class _Warp:
    """A warp split in fixed blocks of ``_BLOCK`` cells anchored at the destination's origin,
    each warped by GDAL on its own with the source window it sees: in memory (blocks over
    threads), lazily (tasks of whole blocks) or from a file, the same calls, the same bits."""

    def __init__(self, src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                 tolerance, n):
        self.src_t, self.src_crs, self.dst_t, self.dst_crs = src_t, src_crs, dst_t, dst_crs
        self.method, self.src_nodata, self.dst_nodata, self.dtype = method, src_nodata, dst_nodata, dtype
        self.tolerance, self.n = tolerance, n
        self.shape = dst_shape
        self.options = _scales(src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs)
        xs = float(self.options.get("XSCALE", 1.0))
        ys = float(self.options.get("YSCALE", 1.0))
        pad = math.ceil(_RADIUS.get(method, 1) * max(1.0, 1.0 / min(xs, ys))) + 2
        dh, dw = dst_shape
        cells = [(r0, c0, min(_BLOCK, dh - r0), min(_BLOCK, dw - c0))
                 for r0 in range(0, dh, _BLOCK) for c0 in range(0, dw, _BLOCK)]
        windows = _block_windows(src_shape, src_t, src_crs, dst_t, dst_crs, cells, pad)
        self.blocks = [(*c, win) for c, win in zip(cells, windows)]  # (row, col, height, width, window or None)

    def block(self, src, origin, r0, c0, bh, bw, win) -> np.ndarray:
        """One block from ``src``, the source cells from ``origin`` (row, col) on."""
        from rasterio.enums import Resampling
        from rasterio.warp import reproject as gdal_reproject

        a, b, c, d = win
        sub = np.ascontiguousarray(src[:, a - origin[0]:b - origin[0], c - origin[1]:d - origin[1]])
        dst = np.full((self.n, bh, bw), self.dst_nodata, dtype=self.dtype)
        gdal_reproject(sub, dst, src_transform=self.src_t * rasterio.Affine.translation(c, a), src_crs=self.src_crs,
                       src_nodata=self.src_nodata, dst_transform=self.dst_t * rasterio.Affine.translation(c0, r0),
                       dst_crs=self.dst_crs, dst_nodata=self.dst_nodata, resampling=Resampling[self.method],
                       num_threads=1, tolerance=self.tolerance, **self.options)
        return dst

    def region(self, src, origin, blocks, r0, c0, rh, cw) -> np.ndarray:
        """The blocks of a lazy task (rows r0.., columns c0..)."""
        out = np.full((self.n, rh, cw), self.dst_nodata, dtype=self.dtype)
        for br, bc, bh, bw, win in blocks:
            if win is not None:
                out[:, br - r0:br - r0 + bh, bc - c0:bc - c0 + bw] = self.block(src, origin, br, bc, bh, bw, win)
        return out

    def in_memory(self, src: np.ndarray, n_threads: int) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor

        out = np.full((self.n, *self.shape), self.dst_nodata, dtype=self.dtype)

        def run(blk):
            r0, c0, bh, bw, win = blk
            out[:, r0:r0 + bh, c0:c0 + bw] = self.block(src, (0, 0), r0, c0, bh, bw, win)

        jobs = [b for b in self.blocks if b[4] is not None]
        if n_threads <= 1 or len(jobs) <= 1:
            for blk in jobs:
                run(blk)
        else:
            with ThreadPoolExecutor(min(n_threads, len(jobs))) as pool:
                list(pool.map(run, jobs))  # GDAL releases the GIL
        return out

    def lazy(self, src, task: int):
        """A dask array of tasks of ``task`` cells a side (whole blocks), each reading the
        source window its blocks see."""
        import dask
        import dask.array as dsa

        dh, dw = self.shape
        by_task: dict = {}
        for blk in self.blocks:
            by_task.setdefault((blk[0] // task, blk[1] // task), []).append(blk)
        region = dask.delayed(self.region)
        rows = []
        for r0 in range(0, dh, task):
            row = []
            for c0 in range(0, dw, task):
                rh, cw = min(task, dh - r0), min(task, dw - c0)
                wins = [b[4] for b in by_task[(r0 // task, c0 // task)] if b[4] is not None]
                if not wins:
                    row.append(dsa.full((self.n, rh, cw), self.dst_nodata, dtype=self.dtype,
                                        chunks=(self.n, rh, cw)))
                    continue
                a, b = min(w[0] for w in wins), max(w[1] for w in wins)
                c, d = min(w[2] for w in wins), max(w[3] for w in wins)
                part = region(src[:, a:b, c:d], (a, c), by_task[(r0 // task, c0 // task)], r0, c0, rh, cw)
                row.append(dsa.from_delayed(part, shape=(self.n, rh, cw), dtype=self.dtype))
            rows.append(row)
        return dsa.block(rows)


def _reproject_raster(data: xr.DataArray, dst_t, dst_shape, dst_crs, coords, resampling: str,
                      n_threads: int | None, tolerance: float = 0.125) -> xr.DataArray:
    from ._threads import resolve

    if data.rio.crs is None:
        raise ValueError("reproject needs a georeferenced raster (see load_raster)")
    if "y" not in data.dims or "x" not in data.dims:
        raise ValueError("reproject needs a raster with y and x dimensions")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")
    src_t = _transform(data)
    if src_t.b or src_t.d or dst_t.b or dst_t.d:
        raise ValueError("rotated grids are not supported")
    if resampling == "auto":
        method = "bilinear" if np.issubdtype(data.dtype, np.floating) else "nearest"
    elif resampling in _RESAMPLING_NAMES:
        method = resampling
    else:
        raise ValueError(f"unknown resampling {resampling!r}; use 'auto' or one of {', '.join(_RESAMPLING_NAMES)}")
    src_crs = CRS.from_user_input(data.rio.crs)
    src_nodata, dst_nodata, dtype = _nodata(data)
    attrs = {k: v for k, v in data.attrs.items() if k not in ("_FillValue",)}
    lead = [d for d in data.dims if d not in ("y", "x")]
    data = data.transpose(*lead, "y", "x")
    lead_shape = tuple(data.sizes[d] for d in lead)
    n = int(np.prod(lead_shape)) if lead else 1
    h, w = _grid_size(data)
    dh, dw = dst_shape
    warp = _Warp((h, w), src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                 float(tolerance), n)
    values = data.data
    if values.dtype != dtype:
        values = values.astype(dtype)
    values = values.reshape((n, h, w))
    if hasattr(values, "dask"):
        out = warp.lazy(values, _task_size(data))
    else:
        out = warp.in_memory(values, resolve(n_threads))
    out = out.reshape(lead_shape + (dh, dw))
    if coords is None:
        x = dst_t.c + dst_t.a * (np.arange(dw) + 0.5)
        y = dst_t.f + dst_t.e * (np.arange(dh) + 0.5)
    else:
        x, y = coords
    da = xr.DataArray(out, dims=(*lead, "y", "x"), coords={**{d: data[d] for d in lead if d in data.coords},
                                                           "y": y, "x": x}, name=data.name, attrs=attrs)
    return _with_georef(da, crs=dst_crs, transform=dst_t, nodata=dst_nodata)


def _task_size(data: xr.DataArray) -> int:
    """Cells a side of the tasks of a lazy result: whole blocks, about the source's chunks."""
    sizes = dict(zip(data.dims, data.chunks))
    side = max(sizes["y"][0], sizes["x"][0])
    return _BLOCK * min(4, max(1, round(side / _BLOCK)))


def _reproject_vector(data, crs, like, densify):
    import shapely

    from ._vector import load_vector, to_crs

    gpd = _geopandas()
    series = isinstance(data, gpd.GeoSeries)
    gdf = data.to_frame() if series else data
    if isinstance(gdf, (str, os.PathLike)):
        gdf = load_vector(gdf)
    if gdf.crs is None:
        raise ValueError("reproject needs vector data with a CRS")
    if like is None and crs is None:
        raise ValueError("reproject: give the target as crs= or like=")
    target = _parse_crs(like if like is not None else crs)
    if densify is not None:
        if not densify > 0:
            raise ValueError(f"densify must be a positive length, got {densify!r}")
        geom = gdf.geometry.name
        gdf = gdf.set_geometry(shapely.segmentize(gdf.geometry.values, float(densify)), crs=gdf.crs)
        gdf = gdf.rename_geometry(geom) if gdf.geometry.name != geom else gdf
    out = to_crs(gdf, target) if gdf.crs != target else gdf
    return out.geometry if series else out


def reproject(data, crs=None, *, like=None, res=None, resampling: str = "auto", tolerance: float = 0.125,
              densify: float | None = None, n_threads: int | None = None):
    """Reproject a raster or vector to another CRS, or onto the grid of another raster.

    Like terra's ``project()``: a raster goes to a CRS (on the grid GDAL
    suggests, or with the cell size ``res``), or onto the grid of a reference
    raster (``like=``: its CRS, cells and extent, so that the two combine cell
    by cell); a vector goes to a CRS or to the CRS of a reference. With ``res``
    and no CRS, a raster is resampled in its own CRS.

    Rasters are warped by GDAL, always in the same blocks of the result: the
    values do not depend on the number of threads, nor on whether the raster
    is in memory, lazy (dask, ``load_raster(..., chunks=...)``) or a file, so
    a raster larger than memory gets the values it would get in memory. A
    lazy raster stays lazy; a path is read window by window, never whole.

    Parameters
    ----------
    data
        A georeferenced ``(y, x)``, ``(band, y, x)``... DataArray (any
        dimensions before ``y`` and ``x``), a GeoDataFrame or GeoSeries, or the
        path of a raster or vector file.
    crs
        The target CRS: ``"EPSG:4326"``, ``4326``, a WKT or PROJ string, a
        pyproj or rasterio CRS, or a raster or vector (object or path) whose CRS
        is taken (and only its CRS).
    like
        A reference: a raster (DataArray or path) whose grid a raster is put on
        (CRS, cells, extent and coordinates), or whose CRS a vector takes; or a
        vector (object or path) whose CRS is taken. Not with ``crs``.
    res
        Rasters: the cell size of the result, one number or ``(xres, yres)``, in
        the units of the target CRS (default: about the source's, as GDAL
        suggests). Not with a raster ``like``.
    resampling
        Rasters: ``"auto"`` (default) takes the nearest cell for integer
        rasters (classes) and interpolates floats bilinearly; or any GDAL
        method: ``"nearest"``, ``"bilinear"``, ``"cubic"``, ``"cubic_spline"``,
        ``"lanczos"``, ``"average"``, ``"mode"``, ``"min"``, ``"max"``,
        ``"med"``, ``"q1"``, ``"q3"``, ``"sum"``, ``"rms"``.
    tolerance
        Rasters: the largest error, in source cells, of GDAL's approximate
        coordinate transformation (default 0.125, as ``gdalwarp``); ``0``
        transforms every cell exactly, three to five times slower.
    densify
        Vectors: add vertices so that no segment is longer than this (in the
        units of the source CRS) before reprojecting, so that long straight
        edges follow the curvature of the target CRS. Default: the vertices as
        they are.
    n_threads
        Rasters in memory or from a file: threads (default: all cores but one);
        the result does not depend on it. A lazy raster stays lazy (its blocks
        run when computed, with dask's threads).

    Returns
    -------
    xarray.DataArray, geopandas.GeoDataFrame or geopandas.GeoSeries
        Rasters: the same type and dimensions; cells outside the source are
        NoData (NaN for floats, the NoData value of integer rasters, else the
        largest value of unsigned types and the smallest of signed ones).
        Boolean rasters become ``uint8``. Vectors: the same type, attributes
        kept.

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))            # 25 m, EPSG:3035
    >>> lonlat = ls.reproject(landcover, "EPSG:4326")
    >>> lonlat.rio.crs.to_epsg(), str(lonlat.dtype), int(lonlat.rio.nodata)   # classes kept
    (4326, 'uint8', 255)

    With a cell size, in metres of UTM 32N:

    >>> utm = ls.reproject(landcover, 32632, res=100)
    >>> utm.rio.resolution()
    (100.0, -100.0)

    Onto the grid of a reference raster, cell by cell:

    >>> back = ls.reproject(lonlat, like=landcover)
    >>> back.shape == landcover.shape, bool((back.x == landcover.x).all())
    (True, True)

    Vectors go to the CRS of a reference (raster or vector):

    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> x0, y0, x1, y1 = landcover.rio.bounds()
    >>> area = gpd.GeoDataFrame({"name": ["study"]}, geometry=[box(x0, y0, x1, y1)], crs=landcover.rio.crs)
    >>> ls.reproject(area, like=lonlat).crs.to_epsg()
    4326
    """
    if like is not None and crs is not None:
        raise ValueError("give the target as crs= or like=, not both")
    if _is_vector(data):
        if res is not None:
            raise ValueError("res= is for rasters")
        return _reproject_vector(data, crs, like, densify)
    if densify is not None:
        raise ValueError("densify= is for vectors")
    if like is None and crs is None and res is None:
        raise ValueError("reproject: give the target as crs=, like= or res=")
    path = None
    if isinstance(data, (str, os.PathLike)):
        path = data
        data = _load_single(path, band=None, masked="auto", chunks=True)
    if not isinstance(data, xr.DataArray):
        raise TypeError("reproject needs a georeferenced DataArray, a GeoDataFrame or GeoSeries, or a file path")
    if data.rio.crs is None:
        raise ValueError("reproject needs a georeferenced raster (see load_raster)")
    dst_t, dst_shape, dst_crs, coords = _target_grid(data, crs, like, res)
    if (dst_crs == CRS.from_user_input(data.rio.crs) and dst_shape == _grid_size(data)
            and dst_t.almost_equals(_transform(data))):  # already there
        out = data if coords is None else data.assign_coords(x=coords[0], y=coords[1])
        return out.compute() if path is not None else out
    out = _reproject_raster(data, dst_t, dst_shape, dst_crs, coords, resampling, n_threads, tolerance)
    if path is not None:
        import dask

        from ._threads import resolve

        with dask.config.set(scheduler="threads", num_workers=resolve(n_threads)):
            out = out.compute()
    return out
