# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel processing of many rasters, one thread per raster.

:func:`batch` runs any landschaft tool on many rasters (GuidosToolbox's
*File -> Batch Process*); :func:`mspa_batch` is the MSPA-specific version with
GTB's output naming.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import os
import re
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

from ._threads import default_threads
from .collection import RasterCollection, RasterInfo, expand_sources
from .io import PathLike, _load_map, write_raster
from .mspa import mspa

__all__ = ["BATCH_TOOLS", "BatchResult", "batch", "mspa_batch"]

T = TypeVar("T")

#: Peak memory of MSPA per pixel (C++ core ~30 B plus input and output).
BYTES_PER_PIXEL = 32


@dataclass
class BatchResult:
    """Outcome of one input of :func:`batch` / :func:`mspa_batch`.

    Attributes
    ----------
    input
        The input raster.
    output
        The main output file (``None`` on error).
    seconds
        Run time.
    error
        The error message, or ``None``.
    outputs
        Every file written.
    """

    input: Path
    output: Path | None
    seconds: float
    error: str | None = None
    outputs: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the input was processed without error.
        """
        return self.error is None


def parse_memory(value: int | str | None) -> int | None:
    """``16_000_000_000``, ``"16GB"``, ``"500 MiB"`` -> bytes."""
    if value is None or isinstance(value, int):
        return value
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)(i?)B?\s*", str(value), re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot parse memory size {value!r} (use e.g. '16GB')")
    number, unit, binary = float(m.group(1)), m.group(2).upper(), bool(m.group(3))
    base = 1024 if binary else 1000
    return int(number * base ** " KMGT".index(unit or " "))


class _MemoryBudget:
    """Blocks a worker until its estimated memory fits in the budget."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0
        self._cond = threading.Condition()

    @contextmanager
    def reserve(self, amount: int):
        if self.limit is None:
            yield
            return
        with self._cond:
            # A raster larger than the whole budget still runs, alone.
            self._cond.wait_for(lambda: self.used == 0 or self.used + amount <= self.limit)
            self.used += amount
        try:
            yield
        finally:
            with self._cond:
                self.used -= amount
                self._cond.notify_all()


def run_parallel(
    items: list[T],
    work: Callable[[T], object],
    *,
    pixels: Callable[[T], int],
    n_workers: int | None,
    threads_per_worker: int,
    max_memory: int | str | None,
) -> list:
    """Run ``work`` on every item with a thread pool and a memory budget.

    Results come back in input order. ``work`` must not raise if the caller
    wants every item processed.
    """
    if n_workers is None:
        n_workers = max(1, default_threads() // max(1, threads_per_worker))
    budget = _MemoryBudget(parse_memory(max_memory))

    def task(item: T):
        with budget.reserve(pixels(item) * BYTES_PER_PIXEL):
            return work(item)

    results: dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, max(1, len(items)))) as pool:
        futures = {pool.submit(task, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [results[i] for i in range(len(items))]


def _output_name(src: Path, connectivity, edge_width, transition, internal) -> str:
    # GuidosToolbox naming: <name>_<FGconn>_<EdgeWidth>_<Transition>_<Intext>.tif
    return f"{src.stem}_{connectivity}_{edge_width}_{int(transition)}_{int(internal)}.tif"


def _header_pixels(path: Path) -> int:
    try:
        return RasterInfo.read(path).pixels
    except Exception:  # noqa: BLE001 - the worker reports unreadable files
        return 0


def mspa_batch(
    inputs,
    output_dir: PathLike,
    *,
    connectivity: int = 8,
    edge_width: int = 1,
    transition: bool = True,
    internal: bool = True,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    max_memory: int | str | None = None,
    overwrite: bool = False,
    raise_errors: bool = False,
) -> list[BatchResult]:
    """Run MSPA on many rasters in parallel and write the results.

    Each worker thread loads one raster, runs MSPA and writes
    ``<output_dir>/<name>_<connectivity>_<edge_width>_<transition>_<internal>.tif``
    (GuidosToolbox naming), then moves to the next one. The C++ core releases
    the GIL, so the threads run fully in parallel, and memory stays bounded by
    the rasters being processed at the same time.

    Parameters
    ----------
    inputs
        A :class:`RasterCollection`, a directory, a glob pattern or a list of
        raster paths (single band, coded 0/1/2).
    output_dir
        Directory for the results (created if needed).
    connectivity, edge_width, transition, internal
        MSPA parameters, see :func:`landschaft.mspa`.
    n_workers
        Rasters processed concurrently. Default: all cores but one, divided
        by ``threads_per_worker``.
    threads_per_worker
        Threads inside each raster (default 1: best throughput for many
        rasters; raise it when processing a few very large ones).
    max_memory
        Memory budget, e.g. ``"16GB"``. Workers wait until the raster they
        take fits (about 32 bytes per pixel). ``None``: no limit.
    overwrite
        Recompute outputs that already exist.
    raise_errors
        Raise the first error instead of recording it in the results.

    Returns
    -------
    list of BatchResult
        One entry per input, in input order.

    Examples
    --------
    >>> from pathlib import Path
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> Path("tiles").mkdir(exist_ok=True)
    >>> for row in (0, 500):
    ...     _ = ls.write_raster(forest[row:row + 500, :500], f"tiles/forest_{row}.tif")
    >>> runs = ls.mspa_batch("tiles", "mspa_out", edge_width=1)
    >>> [run.output.name for run in runs]
    ['forest_0_8_1_1_1.tif', 'forest_500_8_1_1_1.tif']
    """
    if isinstance(inputs, RasterCollection):
        paths, band = inputs.paths, inputs.band
        size = {i.path: i.pixels for i in inputs}
    else:
        paths, band, size = expand_sources(inputs), None, {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def process(src: Path) -> BatchResult:
        start = time.perf_counter()
        dst = out_dir / _output_name(src, connectivity, edge_width, transition, internal)
        try:
            if dst.exists() and not overwrite:
                return BatchResult(src, dst, 0.0)
            data = _load_map(src, band=band)
            result = mspa(
                data,
                connectivity=connectivity,
                edge_width=edge_width,
                transition=transition,
                internal=internal,
                n_threads=threads_per_worker,
                validate=False,  # load_raster already validated
            )
            write_raster(result, dst)
            return BatchResult(src, dst, time.perf_counter() - start)
        except Exception as err:  # noqa: BLE001 - reported per file
            if raise_errors:
                raise
            return BatchResult(src, None, time.perf_counter() - start, f"{type(err).__name__}: {err}")

    return run_parallel(
        paths,
        process,
        pixels=lambda p: size.get(p) or _header_pixels(p),
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        max_memory=max_memory,
    )


def mspa_collection(
    collection: RasterCollection,
    *,
    output_dir: PathLike | None = None,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    max_memory: int | str | None = None,
    overwrite: bool = False,
    **params,
):
    """:func:`landschaft.mspa` for a :class:`RasterCollection`."""
    if output_dir is not None:
        return mspa_batch(
            collection,
            output_dir,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            max_memory=max_memory,
            overwrite=overwrite,
            **params,
        )
    if collection.total_pixels > 4_000_000_000:
        warnings.warn(
            f"keeping {collection.total_pixels / 1e9:.1f} Gpx of results in memory; "
            "pass output_dir= to write them to disk instead",
            stacklevel=3,
        )

    def process(index: int):
        data = collection.load(index)
        result = mspa(data, n_threads=threads_per_worker, validate=False, **params)
        result.name = data.name  # file stem: tells the results apart
        result.attrs["source"] = data.attrs["source"]
        return result

    return run_parallel(
        list(range(len(collection))),
        process,
        pixels=lambda i: collection[i].pixels,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        max_memory=max_memory,
    )


# --- any tool -------------------------------------------------------------------------

#: Tools :func:`batch` knows by name (any other function of one raster works too).
BATCH_TOOLS: tuple[str, ...] = (
    "accounting", "clustering_fac", "conefor_inputs", "contagion_pff", "contortion", "distance_to_resistance",
    "euclidean_distance", "fad", "fad_multiscale", "fos", "gray_spatial_convolution", "gtn_components", "gtn_importance",
    "influence_zones", "landscape_mosaic", "landscape_mosaic_heatmap", "landscape_mosaic_stats", "mspa",
    "parcellation", "proximity", "reclassify", "restoration_status", "spa", "spatial_entropy",
    "spatial_entropy_map", "landscape_metrics", "window_metrics",
)

# Peak memory per pixel of the tools that need more than MSPA's 32 bytes.
_TOOL_BYTES_PER_PIXEL = {"fad_multiscale": 64, "landscape_mosaic": 48, "landscape_mosaic_heatmap": 48,
                         "influence_zones": 80, "proximity": 96, "gtn_importance": 64, "euclidean_distance": 48,
                         "landscape_metrics": 64, "window_metrics": 48}


def _resolve_tool(tool) -> tuple[str, Callable]:
    if callable(tool):
        return getattr(tool, "__name__", "tool"), tool
    if tool not in BATCH_TOOLS:
        raise ValueError(f"unknown tool {tool!r}; known tools: {', '.join(BATCH_TOOLS)} (or pass a function)")
    import landschaft

    return tool, getattr(landschaft, tool)


def _thread_keyword(fn: Callable) -> str | None:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    return next((k for k in ("n_threads", "threads") if k in params), None)


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _is_map(value) -> bool:
    return isinstance(value, (xr.DataArray, np.ndarray)) and np.ndim(value) == 2 and min(np.shape(value)) > 1


def _writable(value):
    """The map with a GeoTIFF-friendly data type."""
    arr = value.values if isinstance(value, xr.DataArray) else np.asarray(value)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    elif arr.dtype == np.int64:
        info = np.iinfo(np.int32)
        arr = arr.astype(np.int32 if arr.size == 0 or (arr.min() >= info.min and arr.max() <= info.max) else np.float64)
    elif arr.dtype == np.uint64:
        arr = arr.astype(np.float64)
    return value.copy(data=arr) if isinstance(value, xr.DataArray) else arr


def _stream_landscape_metrics(data, params: dict, base: Path, table_format: str) -> Path:
    """landscape_metrics' long table, written by Landscape.write_metrics."""
    from .lsm.landscape import Landscape

    selection = ("level", "metric", "name", "type", "what", "extras")
    landscape = Landscape(data, **{k: v for k, v in params.items() if k not in selection})
    path = base.with_name(base.name + (".parquet" if table_format == "parquet" else ".csv"))
    return landscape.write_metrics(path, **{k: v for k, v in params.items() if k in selection})


def _write_outputs(result, base: Path, table_format: str = "csv") -> tuple[list[Path], dict[str, Any]]:
    """Write every map (GeoTIFF), table (CSV or Parquet) and report (TXT) of a tool result."""
    outputs: list[Path] = []
    values: dict[str, Any] = {}

    def name_for(suffix: str, ext: str) -> Path:
        return base.with_name(f"{base.name}_{suffix}{ext}" if suffix else f"{base.name}{ext}")

    def put(key: str, value) -> None:
        if value is None:
            return
        if _is_map(value):
            outputs.append(write_raster(_writable(value), name_for(key, ".tif")))
        elif isinstance(value, pd.DataFrame):
            if table_format == "parquet":
                path = name_for(key, ".parquet")
                frame = value if isinstance(value.index, pd.RangeIndex) else value.reset_index()
                frame.to_parquet(path, index=False)
            else:
                path = name_for(key, ".csv")
                value.to_csv(path, index=not isinstance(value.index, pd.RangeIndex))
            outputs.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                if _is_map(v) or isinstance(v, pd.DataFrame):
                    put(f"{key}_{k}" if key else str(k), v)
                elif not isinstance(v, (xr.DataArray, np.ndarray)):
                    values[f"{key}.{k}" if key else str(k)] = _json_value(v)
        elif isinstance(value, (bool, int, float, str, np.generic)):
            values[key or "value"] = _json_value(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (bool, int, float, str, np.generic)) for v in value):
            values[key or "value"] = _json_value(list(value))

    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        for f in dataclasses.fields(result):
            put(f.name, getattr(result, f.name))
        if callable(getattr(result, "summary", None)):
            path = name_for("", ".txt")
            path.write_text(result.summary())
            outputs.append(path)
        if callable(getattr(result, "csv_text", None)):
            path = name_for("gtb", ".csv")
            path.write_text(result.csv_text())
            outputs.append(path)
    elif isinstance(result, tuple):
        for i, v in enumerate(result):
            put(str(i), v)
    else:
        put("", result)
    return outputs, values


def batch(
    tool,
    inputs,
    output_dir: PathLike,
    *,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    max_memory: int | str | None = None,
    overwrite: bool = False,
    raise_errors: bool = False,
    table_format: str = "csv",
    **params,
) -> list[BatchResult]:
    """Run a landschaft tool on many rasters in parallel and write its outputs.

    GuidosToolbox's *Batch Process* for every tool: each worker thread loads
    one raster, runs ``tool(data, **params)`` and writes what it returns to
    ``output_dir``, named after the input and the tool:

    - maps: ``<name>_<tool>[_<field>].tif`` (GeoTIFF with the input's
      georeferencing and, for GTB products, GTB's colour table);
    - tables: ``<name>_<tool>_<field>.csv`` (``.parquet`` with
      ``table_format="parquet"``; ``"landscape_metrics"`` writes its long
      table as ``<name>_landscape_metrics.csv``); the text report (``summary()``):
      ``<name>_<tool>.txt``; GTB's CSV report when the tool has one:
      ``<name>_<tool>_gtb.csv``;
    - ``<name>_<tool>.json``: tool, parameters, output files, scalar results
      and run time. An input whose JSON exists is skipped unless ``overwrite``.

    MSPA is written with GTB's naming (see :func:`mspa_batch`).

    Parameters
    ----------
    tool
        A name from :data:`BATCH_TOOLS` (e.g. ``"accounting"``, ``"spa"``,
        ``"fad_multiscale"``) or any function taking a raster as first argument.
    inputs
        A :class:`RasterCollection`, a directory, a glob pattern or a list of
        raster paths.
    output_dir
        Directory for the results (created if needed).
    n_workers, threads_per_worker, max_memory, overwrite, raise_errors
        As in :func:`mspa_batch`. ``threads_per_worker`` is passed to the
        tool's ``n_threads``/``threads`` argument.
    table_format
        "csv" (default) or "parquet" (needs pyarrow) for the tables.
    **params
        Parameters of the tool, e.g. ``variant=6`` for ``"spa"``.

    Returns
    -------
    list of BatchResult
        One entry per input, in input order; ``outputs`` lists every file written.

    Examples
    --------
    Accounting of every raster of a folder:

    >>> from pathlib import Path
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> Path("tiles").mkdir(exist_ok=True)
    >>> for row in (0, 500):
    ...     _ = ls.write_raster(forest[row:row + 500, :500], f"tiles/forest_{row}.tif")
    >>> runs = ls.batch("accounting", "tiles", "accounting_out")
    >>> [run.ok for run in runs]
    [True, True]
    >>> [path.name for path in runs[0].outputs]  # doctest: +NORMALIZE_WHITESPACE
    ['forest_0_accounting_classes.tif', 'forest_0_accounting_ids.tif',
     'forest_0_accounting_area.tif', 'forest_0_accounting_stats.csv',
     'forest_0_accounting.txt', 'forest_0_accounting.json']
    """
    name, fn = _resolve_tool(tool)
    if table_format not in ("csv", "parquet"):
        raise ValueError("table_format must be 'csv' or 'parquet'")
    if name == "mspa":
        return mspa_batch(inputs, output_dir, n_workers=n_workers, threads_per_worker=threads_per_worker,
                          max_memory=max_memory, overwrite=overwrite, raise_errors=raise_errors, **params)
    if isinstance(inputs, RasterCollection):
        paths, band = inputs.paths, inputs.band
        size = {i.path: i.pixels for i in inputs}
    else:
        paths, band, size = expand_sources(inputs), None, {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thread_kw = _thread_keyword(fn)
    call_params = dict(params)
    if thread_kw and thread_kw not in call_params:
        call_params[thread_kw] = threads_per_worker
    from . import __version__

    def process(src: Path) -> BatchResult:
        start = time.perf_counter()
        base = out_dir / f"{src.stem}_{name}"
        meta_path = base.with_name(base.name + ".json")
        try:
            if meta_path.exists() and not overwrite:
                files = [Path(f) for f in json.loads(meta_path.read_text()).get("outputs", [])]
                return BatchResult(src, files[0] if files else meta_path, 0.0, outputs=files + [meta_path])
            if name == "landscape_metrics" and call_params.get("block_rows") is not None:
                data = src  # read by blocks, not into memory
            else:
                data = _load_map(src, band=band)
            if name == "landscape_metrics":
                # written metric by metric, never as one table in memory
                outputs, values = [_stream_landscape_metrics(data, call_params, base, table_format)], {}
            else:
                result = fn(data, **call_params)
                outputs, values = _write_outputs(result, base, table_format)
            seconds = time.perf_counter() - start
            meta_path.write_text(json.dumps({
                "tool": name, "input": str(src), "params": _json_value(params),
                "outputs": [str(p) for p in outputs], "values": values,
                "seconds": round(seconds, 3), "landschaft": __version__,
            }, indent=1))
            return BatchResult(src, outputs[0] if outputs else meta_path, seconds, outputs=outputs + [meta_path])
        except Exception as err:  # noqa: BLE001 - reported per file
            if raise_errors:
                raise
            return BatchResult(src, None, time.perf_counter() - start, f"{type(err).__name__}: {err}")

    per_pixel = _TOOL_BYTES_PER_PIXEL.get(name, BYTES_PER_PIXEL)
    return run_parallel(
        paths,
        process,
        pixels=lambda p: (size.get(p) or _header_pixels(p)) * per_pixel // BYTES_PER_PIXEL,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        max_memory=max_memory,
    )
