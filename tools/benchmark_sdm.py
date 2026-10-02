# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.sdm against the R packages it reproduces (the tasks of tools/sdm_benchmark.R).

Usage::

    python tools/benchmark_sdm.py [--reps N] [--cpus 0-11] [--no-r]

Writes the inputs (synthetic records over the example climate, the Bradypus
presences with 10 000 background points, predictions to evaluate) to
``build/benchmark_sdm`` and prints a Markdown table: each task's best time in
R and in Landschaft (default threads, best of ``reps``). Needs Rscript and the
library of ``tools/make_reference_sdm.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import landschaft as ls  # noqa: E402
import lsm_reference  # noqa: E402
import make_reference_sdm  # noqa: E402

LABELS = {
    "clean records": ("Clean 5 000 records: capitals, centroids, equal, GBIF, institutions, outliers, zeros",
                      "CoordinateCleaner `clean_coordinates`"),
    "thin": ("Thin 2 000 records at 10 km, 10 repetitions", "spThin `thin.algorithm`"),
    "vif selection": ("VIF selection, 9 layers", "usdm `vifstep`"),
    "background": ("10 000 background points", "dismo `randomPoints`"),
    "spatial blocks": ("Hexagonal blocks of 500 km, 10 116 points, 100 iterations", "blockCV `cv_spatial`"),
    "spatial clusters": ("k-means folds, 10 116 points, 25 starts", "blockCV `cv_cluster`"),
    "autocorrelation range": ("Autocorrelation range, 9 layers, 5 000 cells each", "blockCV `cv_spatial_autocor`"),
    "evaluate": ("Evaluate 1 000 presences against 10 000 background points", "predicts `pa_evaluate`"),
    "boyce": ("Boyce index, the same predictions", "ecospat `ecospat.boyce`"),
    "mess": ("MESS, 9 layers, 10 116 reference points", "predicts `mess`"),
    "maxnet": ("MaxNet, 116 presences and 10 000 background points, 9 predictors", "maxnet `maxnet`"),
}


def inputs(work: Path):
    """Write the inputs for R and return them for Landschaft."""
    rng = np.random.default_rng(1)
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    cells = np.flatnonzero(~np.isnan(env.values[0]).ravel())
    from landschaft.sdm._background import _cell_centres

    pick = rng.choice(cells, 5000)
    x, y = _cell_centres(env, pick)
    records = pd.DataFrame({"species": rng.choice([f"sp{i}" for i in range(20)], 5000),
                            "decimalLongitude": x + rng.uniform(-0.25, 0.25, 5000),
                            "decimalLatitude": y + rng.uniform(-0.25, 0.25, 5000)})
    centres = rng.uniform([-70, -25], [-45, 0], (40, 2))
    t = centres[rng.integers(0, 40, 2000)] + rng.normal(0, 0.5, (2000, 2))
    thin = pd.DataFrame({"lon": t[:, 0], "lat": t[:, 1]})
    bg = ls.sdm.background(env, 10_000, occ=occ, seed=1)
    data = ls.sdm.dataset(occ, bg, env)
    pts = pd.DataFrame({"lon": data.coords.geometry.x, "lat": data.coords.geometry.y, "occ": data.y.astype(float)})
    p = np.round(rng.beta(4, 1.5, 1000), 4)
    a = np.round(rng.uniform(0, 1, 10_000), 4)
    ev = pd.DataFrame({"pred": np.concatenate([p, a]), "y": [1] * 1000 + [0] * 10_000})
    mx = data.X.copy()
    mx["y"] = data.y
    work.mkdir(parents=True, exist_ok=True)
    for name, df in [("records", records), ("thin", thin), ("points", pts), ("evaluation", ev), ("maxnet", mx)]:
        df.to_csv(work / f"{name}.csv", index=False, float_format="%.17g")
    return env, occ, records, thin, data, p, a


def tasks(env, occ, records, thin, data, p, a) -> dict:
    sdm = ls.sdm
    pts = data.coords.copy()
    pts["occ"] = data.y.astype(float)
    off = dict(seas=False, urban=False, countries=False, duplicates=False, aohi=False)
    return {
        "clean records": lambda: sdm.clean_occurrences(records, action="flag", **off),
        "thin": lambda: sdm.thin(thin, 10_000, reps=10, seed=1),
        "vif selection": lambda: sdm.select_predictors(env, vif=10, seed=1),
        "background": lambda: sdm.background(env, 10_000, occ=occ, seed=1),
        "spatial blocks": lambda: sdm.spatial_blocks(pts, column="occ", size=500_000, k=5, iteration=100, seed=1),
        "spatial clusters": lambda: sdm.spatial_clusters(pts, 5, seed=1),
        "autocorrelation range": lambda: sdm.autocorrelation_range(env, n=5000, seed=1),
        "evaluate": lambda: sdm.evaluate(p, a),
        "boyce": lambda: sdm.boyce(p, np.concatenate([a, p])),
        "mess": lambda: sdm.mess(data, env),
        "maxnet": lambda: sdm.MaxNet().fit(data.X, data.y),
    }


def best(f, reps: int) -> float:
    f()  # warm up (and the first download of the reference data)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def run_r(work: Path, reps: int) -> dict:
    rscript = lsm_reference.find_rscript()
    out = subprocess.run([str(rscript), str(ROOT / "tools" / "sdm_benchmark.R"), str(work),
                          str(ls.sdm.example_data("bioclim")), str(reps)],
                         capture_output=True, text=True, check=True, env=make_reference_sdm.r_environment())
    res = {}
    for line in out.stdout.splitlines():
        if "\t" in line:
            name, t = line.rsplit("\t", 1)
            res[name] = float(t)
    return res


def fmt(t: float | None) -> str:
    if t is None:
        return "—"
    return f"{t:.2f} s" if t >= 1 else f"{t * 1000:.1f} ms" if t < 0.1 else f"{t * 1000:.0f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--r-reps", type=int, default=3)
    ap.add_argument("--cpus", default=None, help="processors to run on, as 0-11 or 0,2,4")
    ap.add_argument("--no-r", action="store_true")
    args = ap.parse_args()
    if args.cpus:
        import psutil

        cpus = [int(c) for part in args.cpus.split(",") for c in (
            range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]
        psutil.Process().cpu_affinity(cpus)  # Rscript inherits it
    work = ROOT / "build" / "benchmark_sdm"
    warnings.simplefilter("ignore")
    values = inputs(work)
    ours = {name: best(f, args.reps) for name, f in tasks(*values).items()}
    theirs = {} if args.no_r else run_r(work, args.r_reps)
    print("| Task | R | Landschaft | Speed-up |")
    print("|---|---|---|---|")
    for name, (label, ref) in LABELS.items():
        r, o = theirs.get(name), ours[name]
        speed = f"{r / o:.0f}×" if r else "—"
        print(f"| {label} ({ref}) | {fmt(r)} | {fmt(o)} | {speed} |")


if __name__ == "__main__":
    main()
