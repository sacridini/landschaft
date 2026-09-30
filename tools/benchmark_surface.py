# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.surface against geodiv and spatialEco (the tasks of tools/surface_benchmark.R).

Usage::

    python tools/benchmark_surface.py [--reps N] [--no-r]

Prints a Markdown table: each task's best time in R (geodiv / spatialEco, on
their default settings) and in Landschaft (default threads, best of ``reps``).
Needs Rscript and the library of ``tools/make_reference_surface.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import landschaft as ls  # noqa: E402
import lsm_reference  # noqa: E402
import make_reference_surface  # noqa: E402

DATA = ROOT / "src" / "landschaft" / "surface" / "data"


def tasks() -> dict:
    s = ls.surface
    nf = ls.load_raster(DATA / "normforest.tif", validate=None)
    ndvi = ls.load_raster(DATA / "orforest.tif", validate=None)
    dem = ls.load_raster(DATA / "orelevation.tif", validate=None)
    small, tiny = nf[100:200, 100:200], nf[100:130, 100:130]
    return {
        "amplitude (sa, sq, ssk, sku)": lambda: s.metrics(nf, ["sa", "sq", "ssk", "sku"]),
        "best trend surface (bestfitplane)": lambda: s.best_fit_plane(ndvi),
        "slopes and area (sdq, sdq6, sdr)": lambda: s.metrics(nf, ["sdq", "sdq6", "sdr"]),
        "peaks, valleys (sds, ssc, s10z)": lambda: s.metrics(nf, ["sds", "ssc", "s10z"]),
        "bearing area (sbi, sci, svi, sk, spk, svk)": lambda: s.metrics(nf, ["sbi", "sci", "svi", "sk", "spk", "svk"]),
        "autocorrelation lengths (scl, stxr)": lambda: (s.scl(nf), s.stxr(nf)),
        "direction, wavelength (std, srw)": lambda: (s.std(nf), s.srw(nf)),
        "fractal dimension (sfd)": lambda: s.sfd(nf),
        "texture image, sa in 5 x 5 windows, 100 x 100": lambda: s.window_metrics(small, "sa", 5),
        "texture image, sk in 5 x 5 windows, 30 x 30": lambda: s.window_metrics(tiny, "sk", 5),
        "tpi, 5 x 5": lambda: s.tpi(dem, 5),
        "tri": lambda: s.tri(dem),
        "vrm": lambda: s.vrm(dem),
        "curvature (total)": lambda: s.curvature(dem, "total"),
        "hsp, scales 3-27": lambda: s.hsp(dem),
        "spherical sd": lambda: s.spherical_sd(dem, 3),
    }


def best(f, reps: int) -> float:
    f()  # warm up
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def run_r(reps: int) -> dict:
    rscript = lsm_reference.find_rscript()
    out = subprocess.run([str(rscript), str(ROOT / "tools" / "surface_benchmark.R"), str(DATA), str(reps)],
                         capture_output=True, text=True, check=True, env=make_reference_surface.r_environment())
    res = {}
    for line in out.stdout.splitlines():
        if "\t" in line:
            name, t = line.rsplit("\t", 1)
            res[name] = float(t)
    return res


def fmt(t: float) -> str:
    return f"{t:.3g} s" if t >= 0.01 else f"{t * 1000:.2g} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--no-r", action="store_true")
    args = parser.parse_args()
    ours = {name: best(f, args.reps) for name, f in tasks().items()}
    theirs = {} if args.no_r else run_r(3)
    print("| Task | R | Landschaft | Times faster |")
    print("|---|---|---|---|")
    for name, t in ours.items():
        r = theirs.get(name, np.nan)
        speed = f"{r / t:,.0f}" if np.isfinite(r) else ""
        print(f"| {name} | {fmt(r) if np.isfinite(r) else ''} | {fmt(t)} | {speed} |")


if __name__ == "__main__":
    main()
