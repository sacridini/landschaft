# SPDX-License-Identifier: GPL-3.0-or-later
"""Reprojection of rasters and vectors (ls.reproject), as terra's project().

Rasters are warped by a port of GDAL's warp kernel (``_core.io_warp``, from GDAL 3.12's
gdalwarpkernel.cpp) on source coordinates that do not depend on how the work is split.
GDAL's own warp does depend on it: its approximate coordinate transformation (1/8 cell)
interpolates along the rows of each chunk, and its kernels take the scale of each chunk,
whose size follows its memory limit (changing that limit alone changes cells of a bilinear
warp). Here the source coordinates of the destination's cells come from a lattice anchored
at its origin: exact (PROJ) every 16 cells, refined down to every cell where interpolating
them would be off by more than the tolerance, interpolated bilinearly in between; the
kernel scale is GDAL's for the whole destination in one chunk. In memory, lazily or from a
file, with any number of threads, every cell gets the same bits. tolerance=0 transforms
every cell exactly: GDAL's values for the whole destination at once (up to PROJ's last
bits, as pyproj's PROJ may not be GDAL's).

The sum method still runs GDAL's warp itself (fixed blocks of the destination, each with
the source window it sees and the kernel scale of the whole grid).
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
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

_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb", ".kml", ".gml", ".csv", ".txt", ".tsv",
                    ".parquet", ".geoparquet", ".pq", ".xlsx", ".xls"}

# GDAL's GDALResampleAlg numbers of the methods the kernel runs; the area methods take
# the source cells a destination cell covers (from its corners), the others sample at
# cell centres. GDAL's filter radius of the interpolating ones.
_METHODS = {"nearest": 0, "bilinear": 1, "cubic": 2, "cubic_spline": 3, "lanczos": 4, "average": 5, "mode": 6,
            "max": 8, "min": 9, "med": 10, "q1": 11, "q3": 12, "rms": 14}
_AREA_METHODS = {"average", "mode", "max", "min", "med", "q1", "q3", "rms"}
_FILTER_RADIUS = {"bilinear": 1, "cubic": 2, "cubic_spline": 2, "lanczos": 3}
_AFFINE, _LATTICE, _DENSE, _EXACT = 0, 1, 2, 3  # WarpCoords::Kind
_CELL = 16  # samples a side of a lattice cell (kWarpCell)
_LEVELS = 4  # refinements of a lattice cell: nodes every 16, 8, 4, 2 or 1 samples
_DENSE_POINTS = 1 << 22  # exact coordinates computed at once with tolerance=0
_STEP_COUNT = 21  # GDAL's DEFAULT_STEP_COUNT (sample points along each edge)


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


def _nodata(data: xr.DataArray):
    """(source NoData, destination NoData, the type the warp works in)."""
    dtype = data.dtype.newbyteorder("=")  # the kernel reads the machine's byte order
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


# --- source coordinates of destination points -------------------------------------


def _geotransform(t) -> tuple:
    return (t.c, t.a, t.b, t.f, t.d, t.e)


def _inv_geotransform(gt) -> tuple:
    """GDALInvGeoTransform of a grid without rotation."""
    return (-gt[0] / gt[1], 1.0 / gt[1], 0.0, -gt[3] / gt[5], 0.0, 1.0 / gt[5])


def _center_long(crs: CRS, t, shape):
    """GDAL's CENTER_LONG of a geographic source (InsertCenterLong): the middle of its
    longitudes, as "%g" writes it; the longitudes transformed to it wrap around it."""
    from pyproj import CRS as ProjCRS

    pc = ProjCRS.from_user_input(crs.to_wkt())
    if not pc.is_geographic or abs(pc.axis_info[0].unit_conversion_factor - math.pi / 180) > 1e-9:
        return None
    gt = _geotransform(t)
    h, w = shape
    corners = (gt[0] + 0 * gt[1] + 0 * gt[2], gt[0] + w * gt[1] + 0 * gt[2], gt[0] + 0 * gt[1] + h * gt[2],
               gt[0] + w * gt[1] + h * gt[2])
    lo, hi = min(corners), max(corners)
    if hi - lo > 360.0:
        return None
    return float("%g" % ((hi + lo) / 2.0))


def _is_geographic(crs: CRS) -> bool:
    from pyproj import CRS as ProjCRS

    return ProjCRS.from_user_input(crs.to_wkt()).is_geographic


class _Points:
    """The source pixel coordinates (column, row) of destination points given in destination
    pixel coordinates, by GDAL's own transformer (GDALGenImgProjTransform, from the GDAL
    rasterio loads, one per thread) or, without it, as GDAL computes them with pyproj's
    PROJ: destination geotransform, reprojection (longitudes wrapped around a geographic
    source's centre), inverse source geotransform. Failed points are infinite. ``same``:
    the same CRS, which GDAL does not reproject (but for a geographic one)."""

    def __init__(self, src_t, src_crs: CRS, src_shape, dst_t, dst_crs: CRS, dst_shape):
        from ._gdaltransform import GdalTransformer, open_library

        self._args = (src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape)
        self.gt = _geotransform(dst_t)
        self.src_gt = _geotransform(src_t)
        self.inv = _inv_geotransform(self.src_gt)
        self.src_crs, self.dst_crs = src_crs, dst_crs
        self.same = src_crs == dst_crs and not _is_geographic(src_crs)
        self.center = None if self.same else _center_long(src_crs, src_t, src_shape)
        self._local = threading.local()
        lib = None if self.same else open_library()
        self.gdal = GdalTransformer(lib, src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape) if lib else None

    def __getstate__(self):  # (dask's processes) GDAL's transformer is made again where unpickled
        return self._args

    def __setstate__(self, args):
        self.__init__(*args)

    def _transformer(self, forward: bool = False):
        name = "forward" if forward else "transformer"
        tr = getattr(self._local, name, None)
        if tr is None:
            from pyproj import Transformer

            a, b = (self.src_crs, self.dst_crs) if forward else (self.dst_crs, self.src_crs)
            tr = Transformer.from_crs(a.to_wkt(), b.to_wkt(), always_xy=True)
            setattr(self._local, name, tr)
        return tr

    def forward(self, x, y):
        """Source pixel coordinates to the destination's (GDAL's transformer the other way)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.gdal is not None:
            return self.gdal(x, y, dst_to_src=False)
        src_gt = self.src_gt
        dinv = _inv_geotransform(self.gt)
        gx = src_gt[0] + x * src_gt[1] + y * src_gt[2]
        gy = src_gt[3] + x * src_gt[4] + y * src_gt[5]
        if not self.same:
            gx, gy = self._transformer(forward=True).transform(gx, gy, errcheck=False)
            gx, gy = np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)
        return dinv[0] + gx * dinv[1] + gy * dinv[2], dinv[3] + gx * dinv[4] + gy * dinv[5]

    def special_points(self) -> list:
        """The poles (longitude 0, latitude +-89.9999) in destination pixel coordinates, as
        GDALTransformLonLatToDestGenImgProjTransformer gives them, with its bug fixed: from a
        geographic source GDAL (3.12, still in its master) skips the reprojection to the
        destination, and so misses a pole in the destination."""
        if self.same:
            return []
        from pyproj import CRS as ProjCRS
        from pyproj import Transformer

        src = ProjCRS.from_user_input(self.src_crs.to_wkt())
        out = []
        for lat in (-89.9999, 89.9999):
            gx, gy = 0.0, lat
            if not src.is_geographic:
                try:
                    tr = Transformer.from_crs(src.geodetic_crs, src, always_xy=True)
                    gx, gy = tr.transform(gx, gy, errcheck=True)
                except Exception:  # noqa: BLE001, S112 - outside the projection's domain
                    continue
            px = self.inv[0] + gx * self.inv[1] + gy * self.inv[2]
            py = self.inv[3] + gx * self.inv[4] + gy * self.inv[5]
            fx, fy = self.forward([px], [py])
            if np.isfinite(fx[0]) and np.isfinite(fy[0]):
                out.append((float(fx[0]), float(fy[0])))
        return out

    def handles(self, n: int):
        """(address of GDAL's transformer, n transformers of their own as an array) for the kernel,
        or (0, empty) without GDAL's; give them back with release()."""
        if self.gdal is None:
            return 0, np.empty(0, dtype=np.uintp)
        return self.gdal.address, np.array(self.gdal.take(n), dtype=np.uintp)

    def release(self, args) -> None:
        if self.gdal is not None and len(args):
            self.gdal.release([int(a) for a in args])

    def __call__(self, u, v):
        if self.gdal is not None:
            return self.gdal(u, v)
        gt, inv = self.gt, self.inv
        gx = gt[0] + u * gt[1] + v * gt[2]
        gy = gt[3] + u * gt[4] + v * gt[5]
        if not self.same:
            gx, gy = self._transformer().transform(gx, gy, errcheck=False)
            gx = np.asarray(gx, dtype=np.float64)
            gy = np.asarray(gy, dtype=np.float64)
            if self.center is not None:  # OGRProjCT's target wrap (x != HUGE_VAL and y != HUGE_VAL)
                ok = (gx != np.inf) & (gy != np.inf)
                low = ok & (gx < self.center - 180.0)
                high = ok & ~low & (gx > self.center + 180)
                gx = np.where(low, gx + 360.0, np.where(high, gx - 360.0, gx))
        return inv[0] + gx * inv[1] + gy * inv[2], inv[3] + gx * inv[4] + gy * inv[5]

    def many(self, u, v, n_threads: int, out=None):
        """__call__ on large flat arrays, in parallel (the transformations release the GIL);
        into out ((n, 2), x and y) when given."""
        n = u.size
        k = max(1, min(n_threads, n // 65536))
        if out is None:
            if k == 1:
                return self(u, v)
            out = np.empty((n, 2))
            split = True
        else:
            split = False
        cut = np.linspace(0, n, k + 1).astype(np.int64)

        def run(i):
            a, b = cut[i], cut[i + 1]
            out[a:b, 0], out[a:b, 1] = self(u[a:b], v[a:b])

        if k == 1:
            run(0)
        else:
            with ThreadPoolExecutor(k) as pool:
                list(pool.map(run, range(k)))
        return (out[:, 0].copy(), out[:, 1].copy()) if split else out

    def grid(self, cols, rows, offset: float, n_threads: int):
        """Coordinates of the samples (cols x rows, sample indices): (len(rows), len(cols), 2)."""
        u = np.broadcast_to(np.asarray(cols, dtype=np.float64)[None, :] + offset, (len(rows), len(cols)))
        v = np.broadcast_to(np.asarray(rows, dtype=np.float64)[:, None] + offset, (len(rows), len(cols)))
        out = np.empty((len(rows) * len(cols), 2))
        self.many(np.ascontiguousarray(u).ravel(), np.ascontiguousarray(v).ravel(), n_threads, out)
        return out.reshape(len(rows), len(cols), 2)


def _translation(points: _Points, src_t, dst_t) -> bool:
    """GDALTransformIsTranslationOnPixelBoundaries: the same CRS (not geographic, which GDAL
    reprojects anyway), cell size and a whole number of cells apart; GDAL then takes the
    nearest cell whatever the method."""
    if not points.same:
        return False
    s, d = _geotransform(src_t), _geotransform(dst_t)
    if not (s[1] == d[1] and s[5] == d[5] and s[2] == d[2] and s[4] == d[4]):
        return False
    inv = points.inv
    ox = inv[0] + d[0] * inv[1] + d[3] * inv[2]
    oy = inv[3] + d[0] * inv[4] + d[3] * inv[5]
    return abs(ox - round(ox)) <= 1e-6 and abs(oy - round(oy)) <= 1e-6


def _kernel_scales(points: _Points, src_shape, dst_shape, method: str, translation: bool) -> tuple:
    """GDAL's dfXScale, dfYScale for the whole destination in one chunk: the source window of
    GDALWarpOperation::ComputeSourceWindow (points along the edges; a grid when some fail or
    a pole is in the destination, widened by the source points that land in it), then
    GDALWarpKernel::PerformWarp's rules."""
    h, w = src_shape
    dh, dw = dst_shape
    step = 1.0 / (_STEP_COUNT - 1)

    def transform(u, v):
        x, y = points(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        return x, y

    def edges(every_cell):
        u, v = [], []
        if every_cell:
            for ix in range(dw + 1):
                u += [ix, ix]
                v += [0, dh]
            for iy in range(1, dh):
                u += [0, dw]
                v += [iy, iy]
        else:
            ratio = 0.0
            while ratio <= 1.0 + step * 0.5:
                u += [ratio * dw, ratio * dw, 0, dw]
                v += [0, dh, ratio * dh, ratio * dh]
                ratio += step
        return u, v

    def ratios(n):
        return [0.5 / n if i == 0 else ((i - 1) * step if i <= _STEP_COUNT else 1 - 0.5 / n)
                for i in range(_STEP_COUNT + 2)]

    def grid():
        u, v = [], []
        for ry in ratios(dw):  # (GDAL takes the x size for both)
            for rx in ratios(dw):
                u.append(rx * dw)
                v.append(ry * dh)
        return u, v

    def sample(u, v):
        x, y = transform(u, v)
        ok = np.isfinite(x) & np.isfinite(y)
        return x, y, ok

    cx, cy = transform([0, dw, 0, dw], [0, 0, dh, dh])
    every = not (np.isfinite(cx).all() and np.isfinite(cy).all())
    x, y, ok = sample(*edges(every))
    use_grid = False
    if any(0 <= px <= dw and 0 <= py <= dh for px, py in points.special_points()):
        use_grid = True
        x, y, ok = sample(*grid())
    if not use_grid and not ok.all():
        use_grid = True
        x, y, ok = sample(*grid())
    failed = int((~ok).sum())
    if failed > x.size - 5:
        return 1.0, 1.0
    min_x, max_x, min_y, max_y = x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max()
    if use_grid:  # ComputeSourceWindowStartingFromSource
        sx = np.array([rx * w for _ in ratios(h) for rx in ratios(w)])
        sy = np.array([ry * h for ry in ratios(h) for _ in ratios(w)])
        fx, fy = points.forward(sx, sy)
        inside = np.isfinite(fx) & np.isfinite(fy) & (fx >= 0) & (fx <= dw) & (fy >= 0) & (fy <= dh)
        if inside.any():
            min_x, max_x = min(min_x, sx[inside].min()), max(max_x, sx[inside].max())
            min_y, max_y = min(min_y, sy[inside].min()), max(max_y, sy[inside].max())
    if min_x > w or max_x < 0 or min_y > h or max_y < 0:
        return 1.0, 1.0

    def snap(v):
        r = float(np.round(v))
        return r if abs(r - v) < 1e-6 else float(v)

    min_x, max_x, min_y, max_y = snap(min_x), snap(max_x), snap(min_y), snap(max_y)
    res = 0 if translation else _FILTER_RADIUS.get(method, 0)

    def side(lo, hi, n_dst, n_src):
        scale0 = max(1e-3, n_dst / (hi - lo))
        radius = math.ceil(res / scale0) if scale0 < 0.95 else res
        if failed > 0:
            radius += 10
        lo_c = int(max(0.0, lo))
        hi_c = int(min(math.ceil(hi), float(n_src)))
        raw = max(0.0, min(float(n_src - lo_c), hi - lo))
        if hi_c - lo_c > 0.9 * n_src:
            size = n_src
        else:
            off = max(0, min(lo_c - radius, n_src))
            size = max(0, min(n_src - off, hi_c - off + radius))
        extra = size - raw
        # PerformWarp
        scale = n_dst / (size - extra)
        if size >= n_dst and size <= n_dst + extra:
            scale = 1.0
        if scale < 1.0:
            reciprocal = 1.0 / scale
            whole = int(reciprocal + 0.5)
            if abs(reciprocal - whole) < 0.05:
                scale = 1.0 / whole
        return scale

    xscale = side(min_x, max_x, dw, w)
    yscale = side(min_y, max_y, dh, h)
    if yscale / xscale > 100:
        # A wide source wrapped around the antimeridian: the mean scale at a few points.
        nx, ny = min(10, dw), min(10, dh)
        u, v = [], []
        for iy in range(ny):
            for ix in range(nx):
                fx = 0.0 if nx == 1 else float(ix) * dw / (nx - 1)
                fy = 0.0 if ny == 1 else float(iy) * dh / (ny - 1)
                u += [fx, fx - 1 if ix == nx - 1 else fx + 1, fx]
                v += [fy, fy, fy - 1 if iy == ny - 1 else fy + 1]
        px, py = transform(u, v)
        scales = []
        for i in range(0, px.size, 3):
            if all(np.isfinite(px[i:i + 3])) and all(np.isfinite(py[i:i + 3])):
                scales.append(1.0 / max(abs(px[i + 1] - px[i]), abs(px[i + 2] - px[i])))
        scales.sort()
        if scales:
            top = scales[-1]
            kept = [s for s in scales if s > top / 10]
            total = 0.0
            for s in kept:
                total += s
            xscale = total / len(kept)
    return float(xscale), float(yscale)


class _Lattice:
    """The source coordinates of the destination samples at the nodes of a lattice anchored
    at the destination's origin: cells of 16 x 16 samples with nodes every 16, 8, 4, 2 or 1
    samples, the coarsest whose interpolation is off by at most ``tolerance`` source cells
    (|dx| + |dy|, as GDAL's approximate transformer measures it) at the midpoints checked
    (``_core.io_lattice_checks``); cells with points the transformation fails on go down to
    every sample, cells well outside the source stay coarse. Samples (i, j) are the
    destination points (i + offset, j + offset). Every cell depends on itself only: the
    lattice of some of the cells (rows, cols: [first, end) cell ranges) has their values in
    the lattice of all."""

    def __init__(self, points: _Points, rows, cols, offset: float, tolerance: float, src_shape, n_threads: int):
        from . import _core

        h, w = src_shape
        (cy0, cy1), (cx0, cx1) = rows, cols
        cx, cy = cx1 - cx0, cy1 - cy0
        self.cells_x, self.cells_y, self.cell_x0, self.cell_y0 = cx, cy, cx0, cy0
        half = _CELL // 2
        g = points.grid((2 * cx0 + np.arange(2 * cx + 1)) * half, (2 * cy0 + np.arange(2 * cy + 1)) * half, offset,
                        n_threads)
        err, bad, outside, box = _core.io_lattice_checks(g[None], h, w)
        need = (((err[0] > tolerance) | bad[0]) & ~outside[0]).ravel()
        self.grid0 = np.ascontiguousarray(g[::2, ::2])
        level = np.zeros(cy * cx, dtype=np.int8)
        bbox = box[0].reshape(cy * cx, 4).copy()  # xmin, xmax, ymin, ymax of the finite nodes of every cell
        node_off = np.zeros(cy * cx, dtype=np.int64)
        blocks = []
        used = 0
        idx = np.flatnonzero(need)
        for lev in range(1, _LEVELS + 1):
            if idx.size == 0:
                break
            step = _CELL >> lev
            m = _CELL // step
            k = np.arange(2 * m + 1) * (step // 2) if step > 1 else np.arange(m + 1)
            ccx, ccy = idx % cx + cx0, idx // cx + cy0
            u = np.broadcast_to(ccx[:, None, None] * _CELL + k[None, None, :], (idx.size, k.size, k.size))
            v = np.broadcast_to(ccy[:, None, None] * _CELL + k[None, :, None], (idx.size, k.size, k.size))
            gl = np.empty((idx.size, k.size, k.size, 2))
            points.many(u.ravel() + offset, v.ravel() + offset, n_threads, gl.reshape(-1, 2))
            if step > 1:
                e, b, o, bx = _core.io_lattice_checks(gl, h, w)
                more = (((e > tolerance) | b) & ~o).any(axis=(1, 2))
                nodes = gl[:, ::2, ::2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cell_box = np.stack([np.nanmin(bx[..., 0], axis=(1, 2)), np.nanmax(bx[..., 1], axis=(1, 2)),
                                         np.nanmin(bx[..., 2], axis=(1, 2)), np.nanmax(bx[..., 3], axis=(1, 2))], -1)
            else:
                more = np.zeros(idx.size, dtype=bool)
                nodes = gl
                cell_box = _finite_box(gl.reshape(idx.size, -1, 2))
            done = idx[~more]
            if done.size:
                flat = np.ascontiguousarray(nodes[~more]).reshape(done.size, -1, 2)
                level[done] = lev
                bbox[done] = cell_box[~more]
                node_off[done] = used + np.arange(done.size) * flat.shape[1]
                used += flat.size // 2
                blocks.append(flat.reshape(-1, 2))
            idx = idx[more]
        self.level, self.node_off = level, node_off
        self.nodes = np.concatenate(blocks) if blocks else np.empty((0, 2))
        self.bbox = bbox.reshape(cy, cx, 4)

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the samples of the lattice's cells; None if the
        transformation fails on all of them."""
        b = self.bbox
        if not np.isfinite(b).any():
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanmin(b[..., 0]), np.nanmax(b[..., 1]), np.nanmin(b[..., 2]), np.nanmax(b[..., 3]))


def _finite_box(flat):
    """xmin, xmax, ymin, ymax of the finite points of every row of flat (k, n, 2); NaN for none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fin = np.isfinite(flat).all(-1)
        xs = np.where(fin, flat[..., 0], np.nan)
        ys = np.where(fin, flat[..., 1], np.nan)
        return np.stack([np.nanmin(xs, axis=1), np.nanmax(xs, axis=1), np.nanmin(ys, axis=1),
                         np.nanmax(ys, axis=1)], -1)


class _Warp:
    """A warp by landschaft's kernel: the whole destination in one call (in memory), or
    tasks of a lazy result reading the source window they see. The source coordinates of
    every destination sample are fixed beforehand (lattice, exact or affine), so any split
    gives the same bits."""

    def __init__(self, src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                 tolerance, n, n_threads):
        self.src_shape, self.shape, self.n = src_shape, dst_shape, n
        self.src_nodata, self.dst_nodata, self.dtype = src_nodata, dst_nodata, dtype
        self.tolerance = tolerance
        self.n_threads = n_threads
        self.points = _Points(src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape)
        translation = _translation(self.points, src_t, dst_t)
        if translation:
            method = "nearest"
        self.method = method
        self.area = method in _AREA_METHODS
        self.offset = 0.0 if self.area else 0.5
        self.extra = 1 if self.area else 0
        self.scales = _kernel_scales(self.points, src_shape, dst_shape, method, translation)
        if self.points.same:
            self.kind = _AFFINE
        elif tolerance == 0:
            self.kind = _EXACT if self.points.gdal is not None else _DENSE
        else:
            self.kind = _LATTICE
        radius = _FILTER_RADIUS.get(method, 0)
        xs, ys = self.scales
        rx = math.ceil(radius / xs) if xs < 1.0 else radius
        ry = math.ceil(radius / ys) if ys < 1.0 else radius
        margin = 2 if self.kind in (_DENSE, _EXACT) else 0  # exact points stray from the lattice's
        self.pad = (ry + 2 + margin, rx + 2 + margin)

    def lattice(self, r0, c0, rh, cw, n_threads: int) -> _Lattice:
        """The lattice of the cells holding the samples of a destination region (built with
        the default tolerance for the windows of exact coordinates)."""
        r1, c1 = r0 + rh + self.extra, c0 + cw + self.extra  # samples, exclusive
        tol = self.tolerance if self.kind == _LATTICE else 0.125
        return _Lattice(self.points, (r0 // _CELL, (r1 - 1) // _CELL + 1), (c0 // _CELL, (c1 - 1) // _CELL + 1),
                        self.offset, tol, self.src_shape, n_threads)

    def window(self, r0, c0, rh, cw, n_threads: int):
        """The source cells (row0, row1, col0, col1) a destination region needs; None if none."""
        r1, c1 = r0 + rh + self.extra, c0 + cw + self.extra
        if self.kind == _AFFINE:
            x, y = self.points(np.array([c0, c1 - 1, c0, c1 - 1], dtype=np.float64) + self.offset,
                               np.array([r0, r0, r1 - 1, r1 - 1], dtype=np.float64) + self.offset)
            box = (x.min(), x.max(), y.min(), y.max())
        else:
            box = self.lattice(r0, c0, rh, cw, n_threads).bounds()
            if box is None:
                return None
        h, w = self.src_shape
        pr, pc = self.pad
        a, b = max(0, math.floor(box[2]) - pr), min(h, math.ceil(box[3]) + pr)
        c, d = max(0, math.floor(box[0]) - pc), min(w, math.ceil(box[1]) + pc)
        if b <= a or d <= c:
            return None
        return a, b, c, d

    def _coords_args(self, coords, handles) -> tuple:
        """The coordinate arguments of _core.io_warp and io_warp_coords."""
        lat = coords if isinstance(coords, _Lattice) else None
        dense = coords if isinstance(coords, tuple) else None
        address, args = handles
        return (self.kind, self.offset, self.points.gt, self.points.inv,
                lat.level if lat is not None else np.empty(0, np.int8),
                lat.grid0 if lat is not None else np.empty((0, 2)),
                lat.node_off if lat is not None else np.empty(0, np.int64),
                lat.nodes if lat is not None else np.empty((0, 2)),
                (lat.cells_x, lat.cell_x0, lat.cell_y0) if lat is not None else (1, 0, 0),
                dense[0] if dense is not None else np.empty((0, 0, 2)), dense[1] if dense is not None else 0,
                dense[2] if dense is not None else 0, address, args)

    def _call(self, src, origin, r0, c0, rh, cw, coords, handles):
        from . import _core

        h, w = self.src_shape
        has_nodata = self.src_nodata is not None
        (kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense, dense_row0, dense_col0, address,
         args) = self._coords_args(coords, handles)
        return _core.io_warp(src, _METHODS[self.method], (h, w), origin, self.scales, has_nodata,
                             float(self.src_nodata) if has_nodata else 0.0, float(self.dst_nodata), (r0, c0, rh, cw),
                             kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense, dense_row0,
                             dense_col0, address, args)

    def _handles(self, n: int):
        """GDAL's transformer for the kernel (n threads), but for the same CRS (never needed)."""
        return self.points.handles(n) if self.kind != _AFFINE else (0, np.empty(0, dtype=np.uintp))

    def sample_coords(self, r0, c0, nr, nc, n_threads: int = 1) -> np.ndarray:
        """The source coordinates the kernel takes for samples rows r0.., columns c0..: (nr, nc, 2)."""
        from . import _core

        lat = self.lattice(r0, c0, nr - self.extra, nc - self.extra, n_threads) if self.kind == _LATTICE else None
        dense = (self.points.grid(np.arange(c0, c0 + nc), np.arange(r0, r0 + nr), self.offset, n_threads), r0, c0) \
            if self.kind == _DENSE else None
        handles = self._handles(1)
        try:
            return _core.io_warp_coords(*self._coords_args(lat if lat is not None else dense, handles),
                                        (r0, c0, nr, nc))
        finally:
            self.points.release(handles[1])

    def region(self, src, origin, r0, c0, rh, cw, n_threads: int) -> np.ndarray:
        """The destination rows r0.., columns c0.. from src (the source cells from origin on), on
        n_threads threads (the OpenMP limit set by the caller)."""
        src = np.ascontiguousarray(src)
        handles = self._handles(n_threads)
        try:
            if self.kind in (_AFFINE, _EXACT):
                return self._call(src, origin, r0, c0, rh, cw, None, handles)
            if self.kind == _LATTICE:
                return self._call(src, origin, r0, c0, rh, cw, self.lattice(r0, c0, rh, cw, n_threads), handles)
            out = np.empty((self.n, rh, cw), dtype=self.dtype)
            rows = max(1, _DENSE_POINTS // (cw + self.extra))
            for a in range(r0, r0 + rh, rows):
                nr = min(rows, r0 + rh - a)
                g = self.points.grid(np.arange(c0, c0 + cw + self.extra), np.arange(a, a + nr + self.extra),
                                     self.offset, n_threads)
                out[:, a - r0:a - r0 + nr] = self._call(src, origin, a, c0, nr, cw, (g, a, c0), handles)
            return out
        finally:
            self.points.release(handles[1])

    def in_memory(self, src: np.ndarray, n_threads: int) -> np.ndarray:
        from . import _threads

        dh, dw = self.shape
        with _threads.limit(n_threads):
            return self.region(src, (0, 0), 0, 0, dh, dw, n_threads)

    def task(self, src, origin, r0, c0, rh, cw) -> np.ndarray:
        from . import _threads

        with _threads.limit(1):
            return self.region(src, origin, r0, c0, rh, cw, 1)

    def lazy(self, src, task: int):
        """A dask array of tasks of ``task`` cells a side, each reading the source window it
        sees (from the lattice of its cells, which the task builds again)."""
        import dask
        import dask.array as dsa

        dh, dw = self.shape
        run = dask.delayed(self.task, pure=True)
        rows = []
        for r0 in range(0, dh, task):
            row = []
            for c0 in range(0, dw, task):
                rh, cw = min(task, dh - r0), min(task, dw - c0)
                win = self.window(r0, c0, rh, cw, self.n_threads)
                if win is None:
                    row.append(dsa.full((self.n, rh, cw), self.dst_nodata, dtype=self.dtype,
                                        chunks=(self.n, rh, cw)))
                    continue
                a, b, c, d = win
                part = run(src[:, a:b, c:d], (a, c), r0, c0, rh, cw)
                row.append(dsa.from_delayed(part, shape=(self.n, rh, cw), dtype=self.dtype))
            rows.append(row)
        return dsa.block(rows)


# --- GDAL's own warp (sum) -----------------------------------------------------------

# Kernel radius of GDAL's resampling methods, in source cells at scale 1 (the window of a
# block is widened by it, scaled when downsampling); the statistics take the footprint.
_RADIUS = {"nearest": 1, "bilinear": 1, "cubic": 2, "cubic_spline": 2, "lanczos": 3}
_BLOCK = 1024  # cells a side of the blocks the destination is warped in
_DENSIFY = 51  # points per edge when the bounds of the whole destination are transformed
_WINDOW_POINTS = 11  # points a side of a block whose source window is computed


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
    k = np.linspace(0.0, 1.0, _WINDOW_POINTS)
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


class _GdalWarp:
    """GDAL's warp split in fixed blocks of ``_BLOCK`` cells anchored at the destination's
    origin, each warped by GDAL on its own with the source window it sees: in memory (blocks
    over threads), lazily (tasks of whole blocks) or from a file, the same calls, the same bits."""

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
    threads = resolve(n_threads)
    if method == "sum":
        warp = _GdalWarp((h, w), src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                         float(tolerance), n)
    else:
        warp = _Warp((h, w), src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                     float(tolerance), n, threads)
    values = data.data
    if values.dtype != dtype:
        values = values.astype(dtype)
    values = values.reshape((n, h, w))
    if hasattr(values, "dask"):
        out = warp.lazy(values, _task_size(data))
    else:
        out = warp.in_memory(values, threads)
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

    Rasters are warped by GDAL's resampling kernels (ported to C++, run in
    parallel), on the source coordinates GDAL's own transformation gives for
    a fixed lattice of points of the result, interpolated in between: the
    values do not depend on the number of threads, nor on whether the raster
    is in memory, lazy (dask, ``load_raster(..., chunks=...)``) or a file, so
    a raster larger than memory gets the values it would get in memory. With
    ``tolerance=0`` they are GDAL's (``gdalwarp -et 0``), bit for bit. A lazy
    raster stays lazy; a path is read window by window, never whole.

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
        Rasters: the largest error, in source cells, of the coordinates
        interpolated between the points transformed exactly (default 0.125,
        as ``gdalwarp``; checked midway between them, finer where the
        projection bends, as GDAL's approximate transformation does). ``0``
        transforms every cell exactly, four to seven times slower: GDAL's values
        bit for bit.
    densify
        Vectors: add vertices so that no segment is longer than this (in the
        units of the source CRS) before reprojecting, so that long straight
        edges follow the curvature of the target CRS. Default: the vertices as
        they are.
    n_threads
        Rasters in memory or from a file: threads (default: all cores but one);
        the result does not depend on it. A lazy raster stays lazy (its tasks
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
