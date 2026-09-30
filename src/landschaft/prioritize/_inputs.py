# SPDX-License-Identifier: GPL-3.0-or-later
"""The feature layers and auxiliary rasters of a prioritization, read by blocks of rows.

Sources: raster files (every band is a feature), ``xarray.DataArray`` (2-D, or
3-D with the features first), NumPy and dask arrays (2-D or 3-D). Values are
read as float32, as Zonation reads them through GDAL. Files are read in their
blocks of rows (the smallest block height of the files, as Zonation), which also
delimit the partial sums of the layers; arrays in blocks of about
``BLOCK_BYTES``, summed as one block.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

BLOCK_BYTES = 256 * 2**20


def _is_path(x) -> bool:
    return isinstance(x, (str, bytes)) or hasattr(x, "__fspath__")


def _is_dask(x) -> bool:
    return type(x).__module__.startswith("dask") and hasattr(x, "compute")


@dataclass
class Layer:
    """One feature layer: its name, a reader of rows and its NoData value."""

    name: str
    read: Callable[[int, int], np.ndarray]  # rows [y0, y1) -> float32 (y1 - y0, width)
    nodata: float = float("nan")  # the band's NoData as float32 (NaN: none)
    block_rows: int | None = None  # the file's block height
    source: object = None
    lazy: Callable[[int, int], object] | None = None  # rows [y0, y1) as a dask array (dask inputs)


@dataclass
class Grid:
    shape: tuple[int, int]
    transform: object = None
    crs: object = None
    template: object = None  # a 2-D DataArray on the grid, if any


@dataclass
class Sources:
    layers: list[Layer]
    grid: Grid
    _open: list = field(default_factory=list)

    def close(self) -> None:
        for ds in self._open:
            try:
                ds.close()
            except Exception:
                pass
        self._open.clear()

    @property
    def block_rows(self) -> int | None:
        rows = [ly.block_rows for ly in self.layers if ly.block_rows]
        return min(rows) if rows else None


def _f32(nodata) -> float:
    if nodata is None:
        return float("nan")
    return float(np.float32(nodata))


def _dataarray_nodata(da):
    try:
        nd = da.rio.nodata
        if nd is not None:
            return nd
    except Exception:
        pass
    for key in ("_FillValue", "nodata", "missing_value"):
        if key in da.attrs:
            return da.attrs[key]
        if key in getattr(da, "encoding", {}):
            return da.encoding[key]
    return None


def _grid_of_dataarray(da) -> Grid:
    t = crs = None
    try:
        t = da.rio.transform()
        crs = da.rio.crs
    except Exception:
        pass
    two = da
    while two.ndim > 2:
        two = two.isel({two.dims[0]: 0}, drop=True)
    return Grid(tuple(int(v) for v in da.shape[-2:]), t, crs, two)


def _open_file(path, sources: Sources, default_nodata) -> list[Layer]:
    import rasterio

    ds = rasterio.open(path)
    sources._open.append(ds)
    stem = Path(os.fsdecode(path)).stem
    bh = min(int(ds.block_shapes[i][0]) for i in range(ds.count))
    layers = []
    for b in range(1, ds.count + 1):
        nd = ds.nodatavals[b - 1]
        if nd is None:
            nd = default_nodata

        def read(y0, y1, b=b, ds=ds):
            from rasterio.windows import Window

            w = Window(0, y0, ds.width, y1 - y0)
            return ds.read(b, window=w, out_dtype="float32")

        name = stem if ds.count == 1 else f"{stem}_{b}"
        layers.append(Layer(name, read, _f32(nd), bh, path))
    if not sources.grid.shape[0]:
        sources.grid = Grid((ds.height, ds.width), ds.transform, ds.crs, None)
    return layers


def _array_layers(x, name: str, default_nodata) -> tuple[list[Layer], Grid]:
    import xarray as xr

    if isinstance(x, xr.DataArray):
        da = x
        nd = _dataarray_nodata(da)
        if nd is None:
            nd = default_nodata
        grid = _grid_of_dataarray(da)
        if da.ndim == 2:
            data = da.data
            stack = [(data, da.name or name)]
        elif da.ndim == 3:
            coord = da[da.dims[0]].values if da.dims[0] in da.coords else range(da.shape[0])
            stack = [(da.data[i], f"{da.name or name}_{c}") for i, c in enumerate(coord)]
        else:
            raise ValueError("expected a 2-D or 3-D DataArray")
    else:
        data = x
        grid = Grid(tuple(int(v) for v in np.shape(data)[-2:]))
        nd = default_nodata
        if np.ndim(data) == 2:
            stack = [(data, name)]
        elif np.ndim(data) == 3:
            stack = [(data[i], f"{name}_{i}") for i in range(np.shape(data)[0])]
        else:
            raise ValueError("expected a 2-D or 3-D array")
    layers = []
    for arr, nm in stack:
        def read(y0, y1, arr=arr):
            return _as_f32(arr[y0:y1])

        lazy = (lambda y0, y1, arr=arr: arr[y0:y1]) if _is_dask(arr) else None
        layers.append(Layer(str(nm), read, _f32(nd), None, x, lazy))
    return layers, grid


def _as_f32(v) -> np.ndarray:
    if _is_dask(v):
        v = v.compute()
    v = np.ma.filled(np.ma.asarray(v), np.nan) if np.ma.isMaskedArray(v) else np.asarray(v)
    return np.asarray(v, dtype=np.float32)


def open_features(features, nodata=None) -> Sources:
    """The feature layers of ``features`` (see ``zonation``) on one grid."""
    items = features if isinstance(features, (list, tuple)) else [features]
    sources = Sources([], Grid((0, 0)))
    for i, item in enumerate(items):
        if _is_path(item):
            layers = _open_file(item, sources, nodata)
        else:
            layers, grid = _array_layers(item, f"feature{i}" if len(items) > 1 else "feature", nodata)
            if not sources.grid.shape[0] or (sources.grid.template is None and grid.template is not None
                                            and sources.grid.transform is None):
                sources.grid = grid
        sources.layers.extend(layers)
    if not sources.layers:
        sources.close()
        raise ValueError("no feature layers")
    shape = sources.grid.shape
    for ly in sources.layers:
        probe = ly.read(0, 1)
        if probe.shape != (1, shape[1]):
            sources.close()
            raise ValueError(f"feature {ly.name!r} has {probe.shape[1]} columns, the first {shape[1]}")
    return sources


def open_aux(x, grid: Grid, what: str, sources: Sources, nodata=None) -> Layer | None:
    """A single-band auxiliary raster (cost, masks, solution) on the features' grid."""
    if x is None:
        return None
    if _is_path(x):
        tmp = Sources([], Grid((0, 0)))
        layers = _open_file(x, tmp, nodata)
        sources._open.extend(tmp._open)
        shape = tmp.grid.shape
    else:
        layers, g = _array_layers(x, what, nodata)
        shape = g.shape
    if len(layers) != 1:
        raise ValueError(f"the {what} must have one band")
    if shape != grid.shape:
        raise ValueError(f"the {what} has {shape[0]} x {shape[1]} cells, the features {grid.shape[0]} x "
                         f"{grid.shape[1]}")
    return layers[0]


def blocks(height: int, width: int, layers: int, block_rows: int | None) -> Iterator[tuple[int, int, bool]]:
    """Blocks of rows (y0, y1, ends_block): Zonation's blocks for files, split to bound memory."""
    per_row = max(1, width * max(layers, 1) * 4)
    step = max(1, BLOCK_BYTES // per_row)
    z = block_rows or height
    y = 0
    while y < height:
        zend = min(height, (y // z + 1) * z)
        y1 = min(zend, y + step)
        yield y, y1, y1 == zend
        y = y1


def read_rows(layers: list[Layer], y0: int, y1: int, width: int) -> np.ndarray:
    """Rows [y0, y1) of the layers: dask layers computed together (one graph), the
    files read in parallel threads (one per file; GDAL releases the GIL)."""
    from .. import _threads

    out = np.empty((len(layers), y1 - y0, width), dtype=np.float32)
    lazy = [i for i, ly in enumerate(layers) if ly.lazy is not None]
    if lazy:
        import dask

        for i, v in zip(lazy, dask.compute(*[layers[i].lazy(y0, y1) for i in lazy])):
            out[i] = _as_f32(v)
    by_source: dict = {}
    for i, ly in enumerate(layers):
        if ly.lazy is None:
            key = id(ly.source) if not _is_path(ly.source) else os.fspath(ly.source)
            by_source.setdefault(key, []).append(i)

    def read_group(idx):
        for i in idx:
            out[i] = layers[i].read(y0, y1)

    groups = list(by_source.values())
    n = min(len(groups), _threads.current())
    if n > 1 and any(_is_path(layers[g[0]].source) for g in groups):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(n) as ex:
            list(ex.map(read_group, groups))
    else:
        for g in groups:
            read_group(g)
    return out


def area_rows(area: Layer | None, y0: int, y1: int, which: int) -> np.ndarray | None:
    """The analysis-area mask of rows [y0, y1): cells != 0 (``which`` = 0) or == ``which``."""
    if area is None:
        return None
    v = area.read(y0, y1).astype(np.float64)
    nd = area.nodata
    v[np.isnan(v) | (v == nd)] = 0
    v = np.clip(np.rint(v), 0, 2**32 - 1)
    return (v != 0 if which == 0 else v == which).astype(np.uint8)


def gather(layer: Layer | None, cell_index: np.ndarray, grid: Grid, *, lo=None, hi=None, round_int=False):
    """The values of a raster at the data cells (Zonation's readImageMatchingMapDataIndexes):
    NoData as 0, clipped to [lo, hi]."""
    if layer is None:
        return None
    h, w = grid.shape
    out = np.zeros(cell_index.size, dtype=np.float64)
    rows = cell_index // w
    for y0, y1, _ in blocks(h, w, 1, None):
        a, b = np.searchsorted(rows, [y0, y1])
        if a == b:
            continue
        v = layer.read(y0, y1).astype(np.float64).ravel()
        nd = np.isnan(v) | (v == layer.nodata)
        if round_int:
            v = np.rint(v)
        if lo is not None or hi is not None:
            v = np.clip(v, lo if lo is not None else -math.inf, hi if hi is not None else math.inf)
        v[nd] = 0.0
        out[a:b] = v[cell_index[a:b] - y0 * w]
    return out
