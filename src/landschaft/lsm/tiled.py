# SPDX-License-Identifier: GPL-3.0-or-later
"""Out-of-core landscapes: a raster file read by blocks of rows.

The file is read three times at most: once for its class values, once for
the patches and class adjacencies (enough for most metrics), and once more
for core areas, gyrate, circle and enn when those are asked for (plus once
for gyrate with ``cell_center``). Blocks are read ahead on a thread, and
blocks of few rows are prepared several at a time (each on some of the
threads) and merged in order, so the results do not depend on it; only those
blocks (with a few context rows) are in memory at a time, and the patch
tables. Maps of patch ids or metrics are lazy (dask): every block is read and
labelled again when it is computed.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.windows import Window

from .. import _core
from .._threads import resolve

__all__ = ["TiledReader", "DETAIL_FAMILIES"]

_INT32 = np.iinfo(np.int32)
_STRIP = 64  # the C++ strip height: blocks are multiples of it
_AHEAD = 2  # blocks read ahead while the engine works on the current one
_CORE = {"core", "ncore", "cai", "tca", "cpland", "ndca", "dcad", "dcore"}

# metric family -> detail computed by the extra pass
DETAIL_FAMILIES = {**{f: "core" for f in _CORE}, "gyrate": "gyrate", "circle": "circle", "enn": "enn"}


def block_size(block_rows, width: int, n_threads: int = 0) -> int:
    """Rows per block, a multiple of 64. "auto": one strip of 64 rows (the
    blocks are worked on side by side, see `TiledReader._workers`), but at
    least 32,768 cells."""
    width = max(width, 1)
    if block_rows == "auto" or block_rows is True:
        rows = (1 << 15) // width
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
            self.transform = src.transform
            self.crs = src.crs
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
        self.n_threads = int(options.get("n_threads") or 0)
        self.block_rows = block_size(block_rows, self.shape[1], self.n_threads)
        self._warned = False
        self.done: set[str] = set()
        classes = self._class_values()
        ny, nx = self.shape
        self.engine = _core.LsmTiled(nx, ny, self.res[0], self.res[1], options, classes)
        self._run(1, self.engine.prepare_block, self.engine.merge_block)
        self.engine.finish()
        self.options = options

    # -- reading -------------------------------------------------------------

    @property
    def _gdal_threads(self) -> str:
        return str(resolve(self.n_threads))

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

    def _spans(self, halo: int) -> list[tuple[int, int, int, int]]:
        """(y0, rows, above, below) of every block with `halo` context rows."""
        ny = self.shape[0]
        return [(y0, rows, min(halo, y0), min(halo, ny - y0 - rows)) for y0, rows in self.blocks()]

    def _workers(self) -> tuple[int, int]:
        """Blocks prepared at once and the threads of each: blocks side by
        side, one thread each (a block's own parallel work has serial parts,
        and small parallel regions cost more than they give), within about
        64 rows (or 32,768 cells) per thread in all and at most 32 million
        cells; a block larger than that alone, on every thread."""
        threads = resolve(self.n_threads)
        width = self.shape[1]
        block = self.block_rows * width
        budget = min(threads * max(_STRIP * width, 1 << 15), 1 << 25)
        workers = max(1, min(threads, len(self.blocks()), budget // block))
        return workers, 1 if workers > 1 else threads

    def _run(self, halo: int, prepare, merge) -> None:
        """Every block read (ahead, see `_blocks`), prepared (several at once,
        see `_workers`) and merged in block order."""
        workers, threads = self._workers()
        if workers == 1:
            for block in self._blocks(halo):
                merge(prepare(*block, threads))
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            try:
                for block in self._blocks(halo):
                    pending.append(pool.submit(prepare, *block, threads))
                    if len(pending) > workers:
                        merge(pending.popleft().result())
                while pending:
                    merge(pending.popleft().result())
            finally:
                for future in pending:
                    future.cancel()

    def _blocks(self, halo: int):
        """(values, mask, y0, rows, above, below) of every block with `halo`
        context rows. The next blocks are read (and converted) on a thread of
        their own while the caller works on the current one."""
        spans = self._spans(halo)
        local = threading.local()

        def read(span):
            y0, rows, above, below = span
            if getattr(local, "src", None) is None:
                # GDAL decodes compressed blocks on the same threads (the
                # environment is the reading thread's, for all its blocks)
                local.env = rasterio.Env(GDAL_NUM_THREADS=self._gdal_threads)
                local.env.__enter__()
                local.src = rasterio.open(self.path)
            values, mask = self._read(local.src, y0 - above, y0 + rows + below)
            return values, mask, y0, rows, above, below

        def close():
            if getattr(local, "src", None) is not None:
                local.src.close()
                local.env.__exit__(None, None, None)
                local.src = local.env = None

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = deque(pool.submit(read, span) for span in spans[:_AHEAD])
            following = len(pending)
            try:
                while pending:
                    block = pending.popleft().result()
                    if following < len(spans):
                        pending.append(pool.submit(read, spans[following]))
                        following += 1
                    yield block
            finally:
                for future in pending:
                    future.cancel()
                pool.submit(close).result()

    def _class_values(self) -> np.ndarray:
        """The class values of the file (pass 0)."""
        found = np.empty(0, dtype=np.int32)
        for values, mask, *_ in self._blocks(0):
            found = np.union1d(found, _core.lsm_unique_values(values, mask))
        return found.astype(np.int32)

    # -- maps ------------------------------------------------------------------

    def blocks(self) -> list[tuple[int, int]]:
        """(first row, rows) of the blocks of the first pass."""
        ny = self.shape[0]
        return [(y0, min(self.block_rows, ny - y0)) for y0 in range(0, ny, self.block_rows)]

    def chunks(self) -> list[list[tuple[int, int]]]:
        """The blocks grouped into the chunks of the lazy maps: rows in
        multiples of 512 (the tiles ``write_raster`` writes) and at least 4
        million cells, unless a block is larger."""
        blocks = self.blocks()
        target = max(512, -(-(1 << 22) // self.shape[1]))
        out: list[list[tuple[int, int]]] = []
        for y0, rows in blocks:
            if out:
                first = out[-1][0][0]
                height = y0 - first
                if height < target or height % 512:
                    out[-1].append((y0, rows))
                    continue
            out.append([(y0, rows)])
        return out

    def chunk_labels(self, chunk: list[tuple[int, int]], n_threads: int = 0) -> np.ndarray:
        """Patch ids (1..n, 0 = NA) of the rows of consecutive blocks, read again."""
        y0 = chunk[0][0]
        y1 = chunk[-1][0] + chunk[-1][1]
        with rasterio.Env(GDAL_NUM_THREADS=str(n_threads or self._gdal_threads)), rasterio.open(self.path) as src:
            values, mask = self._read(src, y0, y1)
        out = np.empty(values.shape, dtype=np.int32)
        for b0, rows in chunk:
            a = b0 - y0
            out[a:a + rows] = self.engine.block_labels(values[a:a + rows], mask[a:a + rows], b0, rows, 0, 0, n_threads)
        return out

    def lazy_map(self, values: np.ndarray | None = None):
        """The patch ids as a dask array (see `chunks`), or ``values[ids]``
        with ``values[0]`` for NA."""
        import dask
        import dask.array as dsa

        dtype = np.int32 if values is None else values.dtype
        # dask computes the chunks side by side: each gets its share of the threads
        chunks = self.chunks()
        threads = resolve(self.n_threads)
        share = max(1, threads // min(threads, len(chunks)))

        def chunk_map(chunk) -> np.ndarray:
            ids = self.chunk_labels(chunk, share)
            return ids if values is None else values[ids]

        parts = [
            dsa.from_delayed(dask.delayed(chunk_map, pure=False)(c), dtype=dtype,
                             shape=(c[-1][0] + c[-1][1] - c[0][0], self.shape[1]))
            for c in chunks
        ]
        return dsa.concatenate(parts, axis=0)

    # -- detail passes -------------------------------------------------------

    def ensure(self, metric_names) -> None:
        """Run the detail pass for the metrics that need it (once)."""
        families = {name.split("_")[0] for name in metric_names}
        raster_only = families & {"prox", "simi", "econ", "connect"}
        if self.options.get("convention") == "fragstats" and "dcore" in families:
            raster_only.add("dcore (convention='fragstats')")
        if raster_only:
            raise NotImplementedError(
                f"{sorted(raster_only)} are not available for an out-of-core landscape (block_rows)"
            )
        need = {DETAIL_FAMILIES[f] for f in families if f in DETAIL_FAMILIES} - self.done
        if not need:
            return
        core, gyrate, circle, enn = ("core" in need), ("gyrate" in need), ("circle" in need), ("enn" in need)
        e = self.engine
        e.begin_detail(core, gyrate, circle, enn)
        self._run(e.detail_halo(core, enn), e.prepare_detail, e.merge_detail)
        if e.end_detail():
            for block in self._blocks(0):
                e.tie_block(*block)
            e.end_ties()
        self.done |= need
