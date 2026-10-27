# SPDX-License-Identifier: GPL-3.0-or-later
"""Land use maps, parameter tables and the grid of the ecosystem service models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import resolve
from ..cost._transition import _crs_lonlat
from ..surface._grid import Surface, read_surface


class LandUse:
    """A land use map: its codes, which cells have one, and its grid."""

    def __init__(self, x, nodata=None, what: str = "the land use map"):
        s = read_surface(x, nodata=nodata) if not isinstance(x, Surface) else x
        self.surface = s
        v = s.values
        self.valid = ~np.isnan(v)
        codes = np.where(self.valid, v, 0)
        if np.any(codes != np.round(codes)):
            raise ValueError(f"{what} must hold integer class codes")
        self.codes = codes.astype(np.int64)
        self.what = what

    @property
    def shape(self):
        return self.surface.shape

    def pixel(self) -> tuple[float, float]:
        """Cell width and height in map units (metres: a projected map, as InVEST requires)."""
        if self.surface.crs is not None and _crs_lonlat(self.surface.crs):
            raise ValueError(f"{self.what} is in longitude / latitude: give it projected, in metres (as InVEST)")
        dx, dy = self.surface.res
        return abs(float(dx)), abs(float(dy))

    def classes(self, codes, n_threads=None) -> np.ndarray:
        """The row of ``codes`` (sorted) of every cell; -1 where the map has no value."""
        cls, first = _core.serv_classify(self.codes, self.valid.astype(np.uint8), np.asarray(codes, np.int64),
                                         resolve(n_threads))
        if first >= 0:
            missing = sorted(set(np.unique(self.codes[self.valid & (cls == -2)]).tolist()))
            raise ValueError(f"{self.what} has class(es) {missing} that the table lacks")
        return cls

    def raster(self, values, name: str):
        return self.surface.raster(values, name)


def same_grid(a: LandUse, b: LandUse) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{b.what} has {b.shape} cells, {a.what} {a.shape}: give them on the same grid")


def lookup_table(table, columns, what: str, index: str = "lucode") -> tuple[np.ndarray, pd.DataFrame]:
    """A table by land use code: ``(codes sorted, the columns as float64)``.

    ``table``: a DataFrame with a ``lucode`` column (or indexed by the codes), or a
    ``dict`` ``{code: {column: value}}``. Column names are not case-sensitive.
    """
    if isinstance(table, dict):
        df = pd.DataFrame.from_dict(table, orient="index")
    elif isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        raise TypeError(f"{what} is a DataFrame (or a dict by class code), not {type(table).__name__}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if index in df.columns:
        df = df.set_index(index)
    df.index = pd.Index([int(v) for v in df.index], name=index)
    if df.index.has_duplicates:
        raise ValueError(f"{what} lists a class more than once")
    missing = [c for c in columns if c.lower() not in df.columns]
    if missing:
        raise ValueError(f"{what} lacks the column(s) {missing}")
    df = df.sort_index()
    out = df[[c.lower() for c in columns]].astype(np.float64)
    if out.isna().any().any():
        raise ValueError(f"{what} has empty values")
    return df.index.to_numpy(np.int64), out


# --- inputs on different grids: InVEST's alignment ---------------------------------

# pygeoprocessing's resampling names, landschaft's (its warp kernel is GDAL's)
_RESAMPLING = {"near": "nearest", "bilinear": "bilinear", "mode": "mode"}


def _georeferenced(x):
    """``x`` as a georeferenced (y, x) DataArray in its own type, or None (a plain array)."""
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        from ..io import load_raster

        x = load_raster(x, masked=False)
    if not isinstance(x, xr.DataArray):
        return None
    if "band" in x.dims:
        if x.sizes["band"] != 1:
            return None
        x = x.squeeze("band", drop=True)
    if x.ndim != 2 or "x" not in x.dims or "y" not in x.dims:
        return None
    try:
        if x.rio.crs is None:
            return None
        x.rio.transform()
    except Exception:  # noqa: BLE001
        return None
    return x


def _bounds(t, shape) -> list:
    """pygeoprocessing's ``bounding_box`` of a grid: [minx, miny, maxx, maxy]."""
    ny, nx = shape
    xs = [t.c, t.c + nx * t.a + ny * t.b]
    ys = [t.f, t.f + nx * t.d + ny * t.e]
    return [min(xs), min(ys), max(xs), max(ys)]


def _transform_bounds(bb, src_crs, dst_crs) -> list:
    """pygeoprocessing's ``transform_bounding_box``: 11 points along each edge.

    Between two CRS that are the same the transformation is the identity (PROJ's
    no-op), but the points along the edges are still interpolated as there.
    """
    p = [np.array((bb[0], bb[3])), np.array((bb[0], bb[1])), np.array((bb[2], bb[1])), np.array((bb[2], bb[3]))]
    edges = [(p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0])]
    v = np.linspace(0, 1, 11)
    points = [[pa * s + pb * (1 - s) for s in v] for pa, pb in edges]
    if src_crs != dst_crs:
        from rasterio.warp import transform

        moved = []
        for pts in points:
            xs, ys = transform(src_crs, dst_crs, [float(q[0]) for q in pts], [float(q[1]) for q in pts])
            moved.append(list(zip(xs, ys)))
        points = moved
    raw = [min(q[0] for q in points[0]), min(q[1] for q in points[1]),
           max(q[0] for q in points[2]), max(q[1] for q in points[3])]
    minx, maxx = sorted([raw[0], raw[2]])
    miny, maxy = sorted([raw[1], raw[3]])
    out = [float(minx), float(miny), float(maxx), float(maxy)]
    if not all(np.isfinite(out)):
        raise ValueError(f"the bounds {bb} cannot be transformed to the CRS of the reference raster")
    return out


def _intersection(boxes) -> list:
    """pygeoprocessing's ``merge_bounding_box_list(..., 'intersection')``."""
    out = list(boxes[0])
    for b in boxes[1:]:
        out = [out[0] if out[0] > b[0] else b[0], out[1] if out[1] > b[1] else b[1],
               out[2] if out[2] <= b[2] else b[2], out[3] if out[3] <= b[3] else b[3]]
    if out[0] > out[2] or out[1] > out[3]:
        raise ValueError("the rasters (and the polygons) do not overlap")
    return out


def _warp_grid(bb, pixel):
    """The grid ``warp_raster`` gives gdal.Warp (``-te`` and ``-tr``): (transform, (rows, columns))."""
    from affine import Affine

    bb = list(bb)
    px, py = pixel
    nx = int(abs(float(bb[2] - bb[0]) / px))
    ny = int(abs(float(bb[3] - bb[1]) / py))
    if not np.isclose(abs(nx * px) - (bb[2] - bb[0]), 0.0):
        nx += 1
    if not np.isclose(abs(ny * py) - (bb[3] - bb[1]), 0.0):
        ny += 1
    nx, ny = max(nx, 1), max(ny, 1)
    bb[2] = bb[0] + abs(px * nx)
    bb[3] = bb[1] + abs(py * ny)
    xres, yres = abs(px), abs(py)
    # gdalwarp given both the bounds and the resolution
    w = int((bb[2] - bb[0] + xres / 2.0) / xres)
    h = int((bb[3] - bb[1] + yres / 2.0) / yres)
    return Affine(xres, 0.0, bb[0], 0.0, -yres, bb[3]), (h, w)


def _vector_bounds(v, crs):
    """The bounds of polygons in ``crs`` (pygeoprocessing reads the layer's extent)."""
    from .._vector import _as_geodataframe

    gdf = _as_geodataframe(v)
    bb = [float(b) for b in gdf.total_bounds]
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        from rasterio.crs import CRS

        bb = _transform_bounds(bb, CRS.from_user_input(gdf.crs), crs)
    return bb


def align_inputs(rasters: dict, *, reference: str, pixel: str = "own", bbox: str = "intersection",
                 vectors=(), snap: bool = True, transform_bounds: bool = False, n_threads=None) -> dict:
    """Put the input rasters of a model on one grid, as InVEST aligns them first.

    ``pygeoprocessing.align_and_resize_raster_stack`` (natcap.invest 3.20.2) with
    landschaft's warp kernel (GDAL's, bit for bit): the cell size of the
    ``reference`` raster (``pixel="square"``: its smaller side, both ways), its
    CRS, the intersection of the rasters' and the ``vectors``' bounds
    (``bbox="reference"``: the reference's own bounds), its lower left corner
    moved onto the reference's cells (``snap``), and each raster resampled by
    its method (``"near"``, ``"bilinear"``, ``"mode"``). ``transform_bounds``:
    the bounds go through ``transform_bounding_box`` as when InVEST gives a
    target projection (the edges sampled, which can move them by an ulp).

    ``rasters`` maps a name to ``(raster, method)`` (None: absent). A raster
    already on the grid is returned as it was given, no copy; when any input is
    a plain array (no grid), every input is returned as it was and the models
    require one grid, as before. Rasters in another CRS are warped to the
    reference's (InVEST does that only where it gives a target projection, and
    warps the others to their own CRS, which leaves them apart).
    """
    from rasterio.crs import CRS

    from .._reproject import _reproject_raster
    from ..io import _transform

    given = {k: v for k, v in rasters.items() if v[0] is not None}
    das = {k: _georeferenced(x) for k, (x, _) in given.items()}
    if any(d is None for d in das.values()):
        return {k: x for k, (x, _) in given.items()}
    ref = das[reference]
    ref_crs = CRS.from_user_input(ref.rio.crs)
    ref_t = _transform(ref)
    if ref_t.b or ref_t.d:
        raise ValueError("rotated rasters are not supported")
    ref_shape = ref.shape
    px, py = ref_t.a, ref_t.e
    if pixel == "square":
        m = min(abs(px), abs(py))
        px, py = m, -m
    grids = {k: (CRS.from_user_input(d.rio.crs), _transform(d), d.shape) for k, d in das.items()}
    if bbox == "reference":
        target = _bounds(ref_t, ref_shape)
    else:
        boxes = [_vector_bounds(v, ref_crs) for v in vectors if v is not None]
        for crs, t, shape in grids.values():
            b = _bounds(t, shape)
            if transform_bounds or crs != ref_crs:
                b = _transform_bounds(b, crs, ref_crs)
            boxes.append(b)
        target = _intersection(boxes)
        if snap:
            align = _bounds(ref_t, ref_shape)
            for i, size in ((0, ref_t.a), (1, ref_t.e)):
                n = int((target[i] - align[i]) / float(size))
                target[i] = n * size + align[i]
    dst_t, dst_shape = _warp_grid(target, (px, py))
    out = {}
    for k, (x, method) in given.items():
        crs, t, shape = grids[k]
        if crs == ref_crs and shape == dst_shape and tuple(t)[:6] == tuple(dst_t)[:6]:
            out[k] = x
            continue
        # pygeoprocessing keeps gdal.Warp's error threshold (0.125): the coordinates of GDAL's
        # approximate transformer, which move bilinear weights by an ulp now and then
        out[k] = _reproject_raster(das[k], dst_t, dst_shape, ref_crs, None, _RESAMPLING[method], n_threads,
                                   0.125, gdal_rows=True)
    return out
