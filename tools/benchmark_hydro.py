# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro against WhiteboxTools 2.4 on a large DEM, both on the same cores.

Usage::

    python tools/benchmark_hydro.py [--size 3000] [--cores 4]

A rough terrain of ``size`` x ``size`` cells (fixed seed). Times are the best of
three runs; WhiteboxTools' include reading and writing its GeoTIFFs, so the
landschaft column includes writing its result too.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_reference_hydro import read, wbt_exe, write  # noqa: E402


def best(fn, n=3):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=3000)
    ap.add_argument("--cores", type=int, default=4)
    a = ap.parse_args()
    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(a.cores)))
    except Exception:  # noqa: BLE001
        pass
    import landschaft as ls

    ls.set_default_threads(a.cores)
    rng = np.random.default_rng(1)
    dem = np.cumsum(np.cumsum(rng.uniform(-1.0, 1.0, (a.size, a.size)), axis=0), axis=1) / a.size
    dem += rng.uniform(0.0, 0.5, dem.shape)
    exe = wbt_exe()
    with tempfile.TemporaryDirectory(prefix="ls-hydro-bench-") as tmp:
        wd = Path(tmp)
        write(wd / "dem.tif", dem, 30.0, 30.0)
        da = ls.load_raster(wd / "dem.tif")

        def wbt(tool, *args):
            return lambda: subprocess.run([str(exe), f"-r={tool}", f"--wd={wd}", f"--max_procs={a.cores}", *args],
                                          capture_output=True, check=True)

        wbt("FillDepressions", "--dem=dem.tif", "-o=filled.tif")()
        filled = ls.load_raster(wd / "filled.tif")
        rows = [
            ("fill", lambda: ls.write_raster(ls.hydro.fill(da), wd / "o.tif"),
             wbt("FillDepressions", "--dem=dem.tif", "-o=w.tif")),
            ("D8 accumulation", lambda: ls.write_raster(ls.hydro.flow_accumulation(filled, flats=False), wd / "o.tif"),
             wbt("D8FlowAccumulation", "--input=filled.tif", "-o=w.tif", "--out_type=cells")),
            ("D-inf accumulation",
             lambda: ls.write_raster(ls.hydro.flow_accumulation(filled, method="dinf", flats=False), wd / "o.tif"),
             wbt("DInfFlowAccumulation", "--input=filled.tif", "-o=w.tif", "--out_type=cells")),
            ("FD8 accumulation",
             lambda: ls.write_raster(ls.hydro.flow_accumulation(filled, method="fd8", flats=False), wd / "o.tif"),
             wbt("FD8FlowAccumulation", "--dem=filled.tif", "-o=w.tif", "--out_type=cells")),
        ]
        print(f"{a.size} x {a.size} cells, {a.cores} cores")
        print(f"{'':20s} {'landschaft':>11s} {'WhiteboxTools':>14s}")
        for name, ours, theirs in rows:
            t1, t2 = best(ours), best(theirs)
            print(f"{name:20s} {t1:10.2f}s {t2:13.2f}s   x{t2 / t1:.1f}")


if __name__ == "__main__":
    main()
