# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.prioritize.zonation against Zonation 5 (z5.exe, z5_16bit.exe).

Usage::

    python tools/benchmark_prioritize.py [--cases a,b] [--work DIR] [--reps N] [--threads N] [--cpus 0-11]

For every case, writes synthetic features (Gaussian random fields above a
threshold: species-like distributions, seeded) as one-strip GeoTIFFs and a z5
project, then times z5 on one thread and on ``--threads``, and landschaft on
one thread and on ``--threads`` (the fastest of ``reps`` runs; the z5 runs
once each, as they are long). Checks that landschaft's rank map is the same, bit
for bit, as z5's on one thread (z5 on several threads may differ: it adds up
the layers per thread). Prints a Markdown table. The z5 binaries are looked for
in ``LANDSCHAFT_Z5_DIR`` (as ``tools/make_reference_prioritize.py``). ``--cpus``
pins both programs to those processors (on hybrid CPUs, the performance cores).
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
import prioritize_cases  # noqa: E402
from make_reference_prioritize import Z5_DIR, write_raster  # noqa: E402

# name: (cells per side, features, share of the landscape each occupies, z5 mode, bits)
CASES = {
    "caz2_1000_30": (1000, 30, 0.6, "CAZ2", 8),
    "caz2_2000_30": (2000, 30, 0.6, "CAZ2", 8),
    "abf_2000_30": (2000, 30, 0.6, "ABF 0.25", 8),
    "caz2_2000_30_16bit": (2000, 30, 0.6, "CAZ2", 16),
    "caz2_1000_200_sparse": (1000, 200, 0.1, "CAZ2", 8),
}


def features(work: Path, side: int, n: int, occupied: float) -> list[Path]:
    """Species-like features: seeded Gaussian fields, 0 below their (1 - occupied) quantile."""
    files = []
    for i in range(n):
        f = work / f"f{side}_{n}_{occupied:g}_{i}.tif"
        if not f.exists():
            field = ls.nlm.gaussian_field(side, side, autocorr_range=side / 25, seed=1000 + i).values
            cut = np.quantile(field, 1 - occupied)
            v = np.where(field > cut, field - cut, 0.0)
            write_raster(f, prioritize_cases.RasterSpec(v, "float32", -9999.0))
        files.append(f)
    return files


def run_z5(exe: Path, mode: str, threads: int, cwd: Path) -> float:
    t = time.perf_counter()
    res = subprocess.run([str(exe), f"--mode={mode}", f"--maxthreads={threads}", "settings.z5", f"out{threads}"],
                         cwd=cwd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-2000:] + res.stderr[-2000:])
    return time.perf_counter() - t


def run_ours(files: list[Path], rule: str, param: float | None, bits: int, threads: int, reps: int):
    best, res = np.inf, None
    for _ in range(reps):
        t = time.perf_counter()
        res = ls.prioritize.zonation([str(f) for f in files], rule=rule, rule_param=param, bits=bits,
                                     n_threads=threads)
        best = min(best, time.perf_counter() - t)
    return best, res


def main() -> None:
    import rasterio

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--work", default=str(ROOT / "build" / "benchmark_prioritize"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=ls.default_threads())
    ap.add_argument("--cpus", default=None, help="processors to run on, as 0-11 or 0,2,4")
    args = ap.parse_args()
    if args.cpus:
        import psutil

        cpus = [int(c) for part in args.cpus.split(",") for c in (
            range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]
        psutil.Process().cpu_affinity(cpus)  # z5 inherits it
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases.split(","):
        side, n, occupied, mode, bits = CASES[name]
        files = features(work, side, n, occupied)
        case_dir = work / name
        case_dir.mkdir(exist_ok=True)
        (case_dir / "features.txt").write_text("filename\n" + "".join(f'"{f.resolve()}"\n' for f in files),
                                              encoding="utf-8")
        (case_dir / "settings.z5").write_text("feature list file = features.txt\n", encoding="utf-8")
        exe = Z5_DIR / ("z5_16bit.exe" if bits == 16 else "z5.exe")
        z1 = run_z5(exe, mode, 1, case_dir)
        zt = run_z5(exe, mode, args.threads, case_dir)
        rule, *p = mode.split()
        param = float(p[0]) if p else None
        o1, res = run_ours(files, rule, param, bits, 1, 1)
        ot, res = run_ours(files, rule, param, bits, args.threads, args.reps)
        with rasterio.open(case_dir / "out1" / "rankmap.tif") as src:
            ref = src.read(1)
        ours = np.asarray(res.rank)
        same = bool(((ours == ref) | (np.isnan(ours) & ~np.isfinite(ref))).all())
        cells = int(np.isfinite(ours).sum())
        rows.append((name, side, n, cells, mode, bits, res.iterations, z1, zt, o1, ot, same))
        print(f"{name}: z5 {z1:.1f} s / {zt:.1f} s, landschaft {o1:.1f} s / {ot:.2f} s, "
              f"{res.iterations} iterations, identical: {same}", flush=True)
    print()
    print(f"| Case | Cells with data | Features | Rule | Iterations | z5, 1 thread | z5, {args.threads} threads "
          f"| Landschaft, 1 thread | Landschaft, {args.threads} threads | Same rank map |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, side, n, cells, mode, bits, it, z1, zt, o1, ot, same in rows:
        rule = mode.split()[0] + (", 16 bits" if bits == 16 else "")
        print(f"| {side} × {side} | {cells / 1e6:.1f} M | {n} | {rule} | {it} | {z1:.1f} s | {zt:.1f} s | {o1:.1f} s "
              f"| {ot:.2f} s | {'yes' if same else 'no'} |")


if __name__ == "__main__":
    main()
