# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch processing of many rasters."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .io import PathLike, load_raster, write_raster
from .mspa import mspa

__all__ = ["BatchResult", "mspa_batch"]


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


def _output_name(src: Path, connectivity, edge_width, transition, internal) -> str:
    # GuidosToolbox naming: <name>_<FGconn>_<EdgeWidth>_<Transition>_<Intext>.tif
    return f"{src.stem}_{connectivity}_{edge_width}_{int(transition)}_{int(internal)}.tif"


def mspa_batch(
    inputs: Iterable[PathLike],
    output_dir: PathLike,
    *,
    connectivity: int = 8,
    edge_width: int = 1,
    transition: bool = True,
    internal: bool = True,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    overwrite: bool = False,
    raise_errors: bool = False,
) -> list[BatchResult]:
    """Run MSPA on many rasters in parallel and write the results.

    Each worker loads a file, runs MSPA and writes
    ``<output_dir>/<name>_<connectivity>_<edge_width>_<transition>_<internal>.tif``.
    The C++ core releases the GIL, so workers are threads that run fully in
    parallel.

    Parameters
    ----------
    inputs
        Raster paths (single band, coded 0/1/2).
    output_dir
        Directory for the results (created if needed).
    connectivity, edge_width, transition, internal
        MSPA parameters, see :func:`guidospy.mspa`.
    n_workers
        Maps processed concurrently. Default: number of CPU cores divided by
        ``threads_per_worker``. Each worker needs about 30 bytes of RAM per
        pixel of the map it is processing.
    threads_per_worker
        OpenMP threads inside each map (default 1: best throughput for many
        maps; raise it when processing a few very large maps).
    overwrite
        Recompute outputs that already exist.
    raise_errors
        Raise the first error instead of recording it in the results.

    Returns
    -------
    list of BatchResult
        One entry per input, in input order.
    """
    paths = [Path(p) for p in inputs]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // max(1, threads_per_worker))

    def process(src: Path) -> BatchResult:
        start = time.perf_counter()
        dst = out_dir / _output_name(src, connectivity, edge_width, transition, internal)
        try:
            if dst.exists() and not overwrite:
                return BatchResult(src, dst, 0.0)
            data = load_raster(src)
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

    results: dict[Path, BatchResult] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process, p): i for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [results[i] for i in range(len(paths))]
