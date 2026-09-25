# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel processing of many rasters, one thread per raster."""

from __future__ import annotations

import os
import re
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .collection import RasterCollection, RasterInfo, expand_sources
from .io import PathLike, load_raster, write_raster
from .mspa import mspa

__all__ = ["BatchResult", "mspa_batch"]

T = TypeVar("T")

#: Peak memory of MSPA per pixel (C++ core ~30 B plus input and output).
BYTES_PER_PIXEL = 32


@dataclass
class BatchResult:
    """Outcome of processing one input file."""

    input: Path
    output: Path | None
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
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
        n_workers = max(1, (os.cpu_count() or 1) // max(1, threads_per_worker))
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
        Rasters processed concurrently. Default: number of CPU cores divided
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
            data = load_raster(src, band=band)
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
