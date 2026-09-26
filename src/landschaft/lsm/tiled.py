# SPDX-License-Identifier: GPL-3.0-or-later
"""Out-of-core landscapes: a raster file read by blocks of rows.

The file is read three times at most: once for its class values, once for
the patches and class adjacencies (enough for most metrics), and once more
for core areas, gyrate, circle and enn when those are asked for (plus once
for gyrate with ``cell_center``). Only one block (and a few context rows) is
in memory at a time; the patch tables are.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window

from .. import _core

__all__ = ["TiledReader", "DETAIL_FAMILIES"]

_INT32 = np.iinfo(np.int32)
_STRIP = 64  # the C++ strip height: blocks are multiples of it
_CORE = {"core", "ncore", "cai", "tca", "cpland", "ndca", "dcad", "dcore"}

# metric family -> detail computed by the extra pass
DETAIL_FAMILIES = {**{f: "core" for f in _CORE}, "gyrate": "gyrate", "circle": "circle", "enn": "enn"}


def block_size(block_rows, width: int) -> int:
    """Rows per block, a multiple of 64. "auto": 64 rows per thread (the
    strips labelled in parallel) and at least 4 million cells, but at most
    32 million cells."""
    width = max(width, 1)
    if block_rows == "auto" or block_rows is True:
        rows = max(_STRIP * _core.max_threads(), (1 << 22) // width)
        rows = min(rows, (1 << 25) // width)
    else:
        rows = int(block_rows)
        if rows <= 0:
            raise ValueError(f"block_rows must be positive, got {block_rows!r}")
    return max(_STRIP, -(-rows // _STRIP) * _STRIP)


class TiledReader:
    """Reads a raster file by blocks and feeds the C++ out-of-core engine."""

    def __init__(self, path, *, band, nodata, res, block_rows, options: dict):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("block_rows needs a raster file (path) as input")
        self.path = path
        with rasterio.open(path) as src:
            if src.count > 1 and band is None:
                raise ValueError("the raster has several bands; select one with band=")
            self.band = 1 if band is None else int(band)
            self.shape = (src.height, src.width)
            if res is None:
                rx, ry = src.res
                res = (abs(float(rx)), abs(float(ry)))
            elif np.isscalar(res):
                res = (float(res), float(res))
            self.res = (float(res[0]), float(res[1]))
            if self.res[0] <= 0 or self.res[1] <= 0:
                raise ValueError(f"cell size must be positive, got {self.res}")
            if src.crs is not None and src.crs.is_geographic:
                warnings.warn(
                    "the raster has a geographic CRS: areas and distances are in squared degrees and degrees "
                    "(as in landscapemetrics); project it for metric units",
                    stacklevel=4,
                )
            self.nodata = src.nodata if nodata is None else nodata
            self.float = np.issubdtype(np.dtype(src.dtypes[self.band - 1]), np.floating)
        self.block_rows = block_size(block_rows, self.shape[1])
        self._warned = False
        self.done: set[str] = set()
        classes = self._class_values()
        ny, nx = self.shape
        self.engine = _core.LsmTiled(nx, ny, self.res[0], self.res[1], options, classes)
        for block in self._blocks(1):
            self.engine.add_block(*block)
        self.engine.finish()
        self.options = options

    # -- reading -------------------------------------------------------------

    def _read(self, src, row0: int, row1: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows [row0, row1): class values (int32) and NA mask (uint8)."""
        arr = src.read(self.band, window=Window(0, row0, self.shape[1], row1 - row0))
        nd = self.nodata
        has_nodata = nd is not None and not (isinstance(nd, float) and np.isnan(nd))
        if self.float:
            mask = np.isnan(arr)
            if has_nodata:
                mask |= arr == nd
            valid = arr[~mask]
            if not self._warned and valid.size and not np.array_equal(valid, np.trunc(valid)):
                warnings.warn("non-integer values are truncated to integers (as terra::as.int)", stacklevel=5)
                self._warned = True
            if valid.size and (valid.min() < _INT32.min + 1 or valid.max() >= _INT32.max + 1):
                raise ValueError("class values must fit in a 32-bit integer")
            arr = np.trunc(np.where(mask, 0, arr))
        elif np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
            mask = arr == nd if has_nodata else np.zeros(arr.shape, dtype=bool)
            if arr.dtype.itemsize >= 4 and arr.dtype != np.int32:  # may not fit in int32
                valid = arr[~mask]
                if valid.size and (valid.min() < _INT32.min + 1 or valid.max() > _INT32.max):
                    raise ValueError("class values must fit in a 32-bit integer")
            elif arr.dtype == np.int32 and np.any((arr == _INT32.min) & ~mask):
                raise ValueError("class values must fit in a 32-bit integer")
        else:
            raise TypeError(f"expected a numeric raster, got dtype {arr.dtype}")
        return np.ascontiguousarray(arr, dtype=np.int32), mask.view(np.uint8)

    def _blocks(self, halo: int):
        """(values, mask, y0, rows, above, below) of every block with `halo`
        context rows."""
        ny = self.shape[0]
        # GDAL decodes compressed blocks on all cores
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(self.path) as src:
            for y0 in range(0, ny, self.block_rows):
                rows = min(self.block_rows, ny - y0)
                above = min(halo, y0)
                below = min(halo, ny - y0 - rows)
                values, mask = self._read(src, y0 - above, y0 + rows + below)
                yield values, mask, y0, rows, above, below

    def _class_values(self) -> np.ndarray:
        """The class values of the file (pass 0)."""
        with rasterio.open(self.path) as src:
            dtype = np.dtype(src.dtypes[self.band - 1])
        if dtype.kind in "ui" and dtype.itemsize <= 2:
            # small integers: count every value of the raw blocks
            info = np.iinfo(dtype)
            counts = np.zeros(int(info.max) - int(info.min) + 1, dtype=np.int64)
            ny = self.shape[0]
            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(self.path) as src:
                for y0 in range(0, ny, self.block_rows):
                    rows = min(self.block_rows, ny - y0)
                    arr = src.read(self.band, window=Window(0, y0, self.shape[1], rows)).ravel()
                    if info.min < 0:
                        arr = arr.astype(np.int32) - int(info.min)
                    counts += np.bincount(arr, minlength=counts.size)
            nd = self.nodata
            if nd is not None and not (isinstance(nd, float) and np.isnan(nd)) and float(nd).is_integer():
                if info.min <= nd <= info.max:
                    counts[int(nd) - int(info.min)] = 0
            return (np.flatnonzero(counts) + int(info.min)).astype(np.int32)
        found: set[int] = set()
        for values, mask, *_ in self._blocks(0):
            valid = values[mask == 0]
            if not valid.size:
                continue
            lo, hi = int(valid.min()), int(valid.max())
            if hi - lo < 1 << 20:  # small range: count instead of sorting
                found.update((np.flatnonzero(np.bincount(valid - lo, minlength=hi - lo + 1)) + lo).tolist())
            else:
                found.update(np.unique(valid).tolist())
        return np.array(sorted(found), dtype=np.int32)

    # -- detail passes -------------------------------------------------------

    def ensure(self, metric_names) -> None:
        """Run the detail pass for the metrics that need it (once)."""
        families = {name.split("_")[0] for name in metric_names}
        need = {DETAIL_FAMILIES[f] for f in families if f in DETAIL_FAMILIES} - self.done
        if not need:
            return
        core, gyrate, circle, enn = ("core" in need), ("gyrate" in need), ("circle" in need), ("enn" in need)
        e = self.engine
        e.begin_detail(core, gyrate, circle, enn)
        for block in self._blocks(e.detail_halo(core, enn)):
            e.detail_block(*block)
        if e.end_detail():
            for block in self._blocks(0):
                e.tie_block(*block)
            e.end_ties()
        self.done |= need
