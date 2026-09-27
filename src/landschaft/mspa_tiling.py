# SPDX-License-Identifier: GPL-3.0-or-later
"""MSPA Tiling (GuidosToolbox *Image Analysis -> Pattern -> Morphological -> MSPA Tiling*).

GTB's way of running MSPA on maps too large for memory: the map is cut into
overlapping tiles (``tile_size`` pixels including an ``overlap``), MSPA runs
on every tile, and the tiles are put back together keeping, of each overlap,
the half nearest to each tile. As GTB warns, this is an approximation: MSPA
classes depend on whole objects (e.g. loops, bridges, islets, openings), so
near the seams they can differ from MSPA of the whole map when objects are
larger than the overlap. :func:`mspa_tiled` reproduces GTB's tiling exactly,
including its shortcuts for tiles without background or without foreground.

With a raster path the map is processed out of core: every tile is read from
and written to disk by window, so memory holds only the tiles in progress.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from .mspa import _check_params, _mspa_2d
from .validation import InvalidInputError, validate_mspa_input

__all__ = ["mspa_tiled", "tile_layout"]

PathLike = str | os.PathLike


def tile_layout(size: int, tile_size: int = 6500, overlap: int = 2000) -> list[tuple[int, int, int, int]]:
    """GTB's tiles along one axis of ``size`` pixels.

    Returns ``(read_start, read_stop, keep_start, keep_stop)`` per tile: the
    tile read for MSPA and the part of it kept in the result (map
    coordinates). Tiles start every ``tile_size - overlap`` pixels; the last
    one ends at the map border. Of each overlap, the first ``overlap // 2``
    pixels go to the earlier tile.
    """
    if overlap < 0 or tile_size <= overlap:
        raise ValueError(f"tile_size ({tile_size}) must be larger than overlap ({overlap}) >= 0")
    step, border = tile_size - overlap, overlap // 2
    starts = [0]
    while starts[-1] + tile_size < size:
        starts.append(starts[-1] + step)
    n = len(starts)
    tiles = []
    for k, s in enumerate(starts):
        stop = min(s + tile_size, size)
        keep_start = 0 if k == 0 else s + border
        keep_stop = size if k == n - 1 else s + border + step
        tiles.append((s, stop, keep_start, keep_stop))
    return tiles


def _tile_mspa(tile: np.ndarray, kw: dict) -> np.ndarray:
    """MSPA of one tile, with GTB's shortcuts for tiles lacking background or foreground."""
    hi, lo = int(tile.max()), int(tile.min())
    if hi > 2:
        raise InvalidInputError(f"MSPA input values must be 0 (missing), 1 (background) or 2 (foreground), got {hi}")
    if hi == 0:  # all missing
        return np.full_like(tile, 129)
    if lo == 2:  # all foreground
        return np.full_like(tile, 17)
    if hi == 1:  # background and missing
        return (tile == 0).astype(np.uint8) * np.uint8(129)
    if not (tile == 1).any():  # foreground and missing
        return np.where(tile == 2, np.uint8(17), np.uint8(129)).astype(np.uint8)
    return _mspa_2d(np.ascontiguousarray(tile), **kw)


def _gtb_suffix(connectivity: int, edge_width: int, transition: bool, internal: bool) -> str:
    return f"{connectivity}_{edge_width}_{int(bool(transition))}_{int(bool(internal))}"


def _run_tiles(shape, read, write, kw: dict, tile_size: int, overlap: int, n_workers: int) -> None:
    ny, nx = shape
    rows, cols = tile_layout(ny, tile_size, overlap), tile_layout(nx, tile_size, overlap)
    if len(rows) == 1 and len(cols) == 1:  # one tile: MSPA of the whole map
        rows, cols = [(0, ny, 0, ny)], [(0, nx, 0, nx)]
    jobs = [(r, c) for r in rows for c in cols]

    def job(rc) -> None:
        (y0, y1, ky0, ky1), (x0, x1, kx0, kx1) = rc
        out = _tile_mspa(read(y0, y1, x0, x1), kw)
        write(ky0, ky1, kx0, kx1, out[ky0 - y0:ky1 - y0, kx0 - x0:kx1 - x0])

    if n_workers <= 1:
        for rc in jobs:
            job(rc)
    else:
        with ThreadPoolExecutor(n_workers) as pool:
            for f in [pool.submit(job, rc) for rc in jobs]:
                f.result()


def mspa_tiled(
    data,
    output: PathLike | None = None,
    *,
    connectivity: int = 8,
    edge_width: int = 1,
    transition: bool = True,
    internal: bool = True,
    tile_size: int = 6500,
    overlap: int = 2000,
    n_workers: int = 1,
    n_threads: int | None = None,
    overwrite: bool = False,
):
    """MSPA of a large map by overlapping tiles, identical to GTB's MSPA Tiling.

    Parameters
    ----------
    data
        MSPA input (0 missing, 1 background, 2 foreground): a 2-D array or
        DataArray (processed in memory), or the path of a GeoTIFF (processed
        out of core, tile by tile, into ``output``).
    output
        GeoTIFF to write. For a path input it defaults to GTB's name next to
        the input, ``<name>_<FGconn>_<EdgeWidth>_<Transition>_<Intext>.tif``.
    connectivity, edge_width, transition, internal
        MSPA parameters, see :func:`landschaft.mspa`. GTB's MSPA Tiling fixes
        the edge width to 1.
    tile_size, overlap
        Tile side including the overlap, and the overlap [pixels]. GTB uses
        2000 and a tile size of 6500 on MS-Windows (9500 on Linux, 8500 on
        macOS).
    n_workers
        Tiles processed concurrently; each needs about 32 bytes per tile pixel.
    n_threads
        Threads per tile (default: all cores but one, shared among the workers).
    overwrite
        Replace an existing ``output``.

    Returns
    -------
    The MSPA map (array or DataArray) for in-memory input; the path of the
    written GeoTIFF for a path input.
    """
    _check_params(connectivity, edge_width)
    tile_layout(1, tile_size, overlap)  # checks the tiling parameters
    n_workers = max(1, int(n_workers))
    if n_threads is None:
        from ._threads import default_threads

        n_threads = max(1, default_threads() // n_workers)
    kw = dict(connectivity=int(connectivity), edge_width=int(edge_width), internal=bool(internal),
              n_threads=int(n_threads))
    attrs = {
        "long_name": "MSPA classes",
        "landschaft_product": "MSPA",
        "mspa_connectivity": kw["connectivity"],
        "mspa_edge_width": kw["edge_width"],
        "mspa_transition": int(bool(transition)),
        "mspa_internal": int(kw["internal"]),
        "mspa_tile_size": int(tile_size),
        "mspa_tile_overlap": int(overlap),
    }

    if isinstance(data, (str, os.PathLike)):
        return _mspa_tiled_file(Path(data), output, kw, attrs, tile_size, overlap, n_workers, overwrite)

    template = data if isinstance(data, xr.DataArray) else None
    arr = np.asarray(data.values if template is not None else data)
    if arr.ndim != 2:
        raise ValueError(f"mspa_tiled expects a 2-D map, got shape {arr.shape}")
    validate_mspa_input(arr)
    result = np.empty_like(arr, dtype=np.uint8)

    def read(y0, y1, x0, x1):
        return arr[y0:y1, x0:x1]

    def write(y0, y1, x0, x1, block):
        result[y0:y1, x0:x1] = block

    _run_tiles(arr.shape, read, write, kw, tile_size, overlap, n_workers)
    if template is not None:
        out = template.copy(data=result)
        out.name = "mspa"
        out.attrs = attrs
        out.encoding = {}
    else:
        out = result
    if output is not None:
        from .io import mspa_colormap, write_raster

        if Path(output).exists() and not overwrite:
            raise FileExistsError(f"{output} exists (pass overwrite=True)")
        write_raster(out, output, colormap=mspa_colormap(bool(transition)))
    return out


def _mspa_tiled_file(src_path: Path, output, kw: dict, attrs: dict, tile_size: int, overlap: int,
                     n_workers: int, overwrite: bool) -> Path:
    import rasterio
    from rasterio.windows import Window

    from .io import mspa_colormap

    suffix = _gtb_suffix(attrs["mspa_connectivity"], attrs["mspa_edge_width"], attrs["mspa_transition"],
                         attrs["mspa_internal"])
    out_path = Path(output) if output is not None else src_path.with_name(f"{src_path.stem}_{suffix}.tif")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (pass overwrite=True)")
    if out_path.resolve() == src_path.resolve():
        raise ValueError("output must differ from the input")
    lock = threading.Lock()
    with rasterio.open(src_path) as src:
        if src.count != 1 or src.dtypes[0] != "uint8":
            raise InvalidInputError(f"{src_path}: MSPA needs a single-band uint8 map")
        profile = {
            "driver": "GTiff", "height": src.height, "width": src.width, "count": 1, "dtype": "uint8",
            "crs": src.crs, "transform": src.transform, "compress": "lzw", "BIGTIFF": "IF_SAFER",
            "tiled": True, "blockxsize": 512, "blockysize": 512,
        }
        tmp_path = out_path.with_name(out_path.name + ".part")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            def read(y0, y1, x0, x1):
                with lock:
                    return src.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))

            def write(y0, y1, x0, x1, block):
                with lock:
                    dst.write(np.ascontiguousarray(block), 1, window=Window(x0, y0, x1 - x0, y1 - y0))

            _run_tiles((src.height, src.width), read, write, kw, tile_size, overlap, n_workers)
            dst.write_colormap(1, mspa_colormap(bool(attrs["mspa_transition"])))
            dst.update_tags(
                TIFFTAG_IMAGEDESCRIPTION="GTB_MSPA (landschaft), MSPA Tiling, MSPA parameters "
                f"(FGconn_EdgeWidth_Transition_Intext): {suffix}",
                **{k: str(v) for k, v in attrs.items() if k.startswith(("mspa_", "landschaft_"))},
            )
    os.replace(tmp_path, out_path)
    return out_path
