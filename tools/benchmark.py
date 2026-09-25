# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark guidospy's MSPA against the original GuidosToolbox executable.

Usage::

    python tools/benchmark.py path/to/GTB [--exe path/to/mspa_win64.exe]

Uses GTB's sample maps: ``data/input.tif`` (1000 x 1000) and ``data/clc.tif``
(2000 x 2000 CORINE land cover, forest classes as foreground).
The original's time includes its TIFF I/O; guidospy is timed in memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import guidospy as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from gtb_reference import run_original_mspa  # noqa: E402


def best_of(f, repeats=3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gtb", type=Path, help="GTB repository (for data/input.tif and data/clc.tif)")
    ap.add_argument("--exe", type=Path, help="original mspa executable")
    args = ap.parse_args()

    maps = {
        "input.tif 1000x1000": gp.load_raster(args.gtb / "data" / "input.tif").values,
        "CORINE 2000x2000": gp.reclassify(
            gp.load_raster(args.gtb / "data" / "clc.tif", validate=None).values, foreground=[23, 24, 25]
        ),
    }
    print(f"threads available: {gp.max_threads()}")
    print(f"{'map':22s} {'params':12s} {'original':>9s} {'1 thread':>9s} {'all':>9s} {'identical':>9s}")
    for name, data in maps.items():
        for conn, eew in [(8, 1), (8, 5), (4, 20)]:
            ours = gp.mspa(data, connectivity=conn, edge_width=eew)
            t1 = best_of(lambda: gp.mspa(data, connectivity=conn, edge_width=eew, n_threads=1))
            tn = best_of(lambda: gp.mspa(data, connectivity=conn, edge_width=eew))
            if args.exe:
                ref = None

                def run():
                    nonlocal ref
                    ref = run_original_mspa(args.exe, data, connectivity=conn, edge_width=eew)

                to = best_of(run, repeats=2)
                same = "yes" if np.array_equal(ref, ours) else "NO"
                orig = f"{to:8.2f}s"
            else:
                orig, same = "-", "-"
            print(f"{name:22s} {f'{conn}_{eew}':12s} {orig:>9s} {t1:8.2f}s {tn:8.2f}s {same:>9s}")


if __name__ == "__main__":
    main()
