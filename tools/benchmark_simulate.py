# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.simulate against the land change tools it reproduces.

Usage::

    python tools/benchmark_simulate.py [--reps N] [--only lulcc,molusce,dinamica]

Prints a Markdown table: each task's best time in the original and in
Landschaft (default threads, best of ``reps``). The originals run as in
``tools/make_reference_simulate.py``: lulcc in R (only the timed call is
measured, with ``system.time``), MOLUSCE's algorithms imported in this
process, Dinamica EGO's console (the time of a script that only loads the
maps is subtracted, so its start-up and reading do not count).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import make_reference_simulate as mrs
import simulate_cases as sc
import simulate_engines as se

import landschaft as ls

S = ls.simulate


def tile(world, k: int):
    """The bands of a world repeated k x k times, on a grid of their own."""
    import xarray as xr

    bands = [str(b) for b in world.band.values]
    return xr.concat([sc.grid(np.tile(world.sel(band=b).values, (k, k))) for b in bands],
                     dim="band").assign_coords(band=bands)


def best(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def r_elapsed(code: str, inputs: tuple[str, ...]) -> float:
    case = sc.Case("bench", "r", inputs, r=code)
    arrays, _, _ = mrs.run_r({"bench": case})
    return float(np.asarray(arrays["bench/elapsed"]).ravel()[0])


# ----------------------------------------------------------------------------- lulcc

_LULCC_SETUP = sc._ALLOC_R.split("model <- {model}")[0]


def lulcc_rows(reps: int) -> list:
    rows = []
    pie, drivers = sc.build_input("pie"), sc.build_input("pie_drivers")
    for label, model_r, alloc in (
        ("CLUE-S allocation, Plum Island, 14 steps", sc._clues_r((0.2, 0.2, 0.2), sc._VIGNETTE),
         S.CLUE(jitter=0.0002, scale=0.000001, max_iter=1000, max_diff=50, mean_diff=50)),
        ("Ordered allocation, Plum Island, 14 steps",
         "OrderedModel(obs = obs, ef = ef, models = models, time = 0:14, demand = dmd, order = c(2, 1, 3))",
         S.Ordered(order=[2, 1, 3])),
    ):
        code = (_LULCC_SETUP + f"model <- {model_r}\n"
                f"el <- min(replicate({reps}, {{ set.seed(1); system.time(allocate(model))[['elapsed']] }}))\n"
                "list(elapsed = el)\n")
        r = r_elapsed(code, ("pie", "pie_drivers"))
        rules = S.Rules(elasticity=[0.2, 0.2, 0.2])
        lr = S.LogisticRegression(sc._PIE_FORMS).fit(pie[0], drivers)
        demand = S.AreaDemand.from_maps(pie, times=[0, 6, 14])
        model = S.Model(lr, demand, alloc, rules)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = best(lambda m=model: m.run(pie[0], drivers, seed=1), reps)
        rows.append((label, "lulcc `allocate`", r, ours))
    code = """
x <- val_pie_raster
el <- min(replicate(REPS, system.time({
  tmc <- ThreeMapComparison(x = x[[1]], x1 = x[[2]], y1 = x[[3]], factors = c(1, 2, 4, 8, 16, 32),
                            categories = c(1, 2, 3), labels = c("a", "b", "c"))
  ab <- AgreementBudget(tmc); fom <- FigureOfMerit(tmc)
})[["elapsed"]]))
list(elapsed = el)
""".replace("REPS", str(reps))
    r = r_elapsed(code, ("val_pie",))
    v = sc.build_input("val_pie")

    def tmc():
        t = S.three_map_comparison(v[0], v[1], v[2], factors=[1, 2, 4, 8, 16, 32])
        t.agreement_budget()
        t.figure_of_merit()
    rows.append(("Three-map comparison, budget and figure of merit, Plum Island, 6 resolutions",
                 "lulcc `ThreeMapComparison`", r, best(tmc, reps)))
    return rows


# ----------------------------------------------------------------------------- MOLUSCE

def molusce_rows(reps: int) -> list:
    w = sc.molusce_world()
    tiled = tile(w, 4)  # 180 x 240 cells
    job = sc.molusce_job(tiled)
    se.run_molusce(lambda im: None, Path("."), se.MOLUSCE_FIXED)  # import once
    ref = best(lambda: se.run_molusce(job, Path("."), se.MOLUSCE_FIXED), max(1, reps // 2))
    ours = best(lambda: sc.molusce_py(tiled), reps)
    return [("Weights of evidence and 3 simulator iterations, 180 × 240", "MOLUSCE `WoeManager`, `Simulator`",
             ref, ours)]


# ----------------------------------------------------------------------------- Dinamica EGO

def dinamica_rows(reps: int) -> list:
    w = sc.sim_world()
    big = tile(w, 8)  # 480 x 640 cells
    rasters = {k: big.sel(band=k).fillna(255).astype("int32").rio.write_nodata(255)
               for k in ("t0", "observed", "simulated")}
    load = "\n".join(f'  {k} := LoadCategoricalMap "{k}.tif" 255 .default 0;' for k in rasters)
    calls = "\n".join(f"  d{k} v{k} := CalcSimilarityOfDifferences t0 observed simulated .yes {k};\n"
                      f'  SaveMap d{k} "d{k}.tif";' for k in (1, 3, 5, 7, 9, 11))
    saves = "\n".join(f'  SaveMap {k} "copy_{k}.tif";' for k in rasters)

    def run(script):
        with tempfile.TemporaryDirectory() as tmp:
            t = time.perf_counter()
            se.run_dinamica({"rasters": rasters, "script": script, "outputs": []}, Path(tmp))
            return time.perf_counter() - t

    base = min(run("Script {{\n" + load + "\n" + saves + "\n}};\n") for _ in range(reps))
    full = min(run("Script {{\n" + load + "\n" + calls + "\n}};\n") for _ in range(reps))
    ours = best(lambda: S.similarity_of_differences(big.sel(band="t0"), big.sel(band="observed"),
                                                    big.sel(band="simulated")), reps)
    return [("Fuzzy similarity of the differences, windows 1-11, 480 × 640", "Dinamica EGO `CalcSimilarityOfDifferences`",
             max(full - base, 0.0), ours)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only", default="lulcc,molusce,dinamica")
    args = p.parse_args()
    groups = {"lulcc": lulcc_rows, "molusce": molusce_rows, "dinamica": dinamica_rows}
    print("| Task | Original | Original's time | Landschaft | Speed-up |")
    print("|---|---|---|---|---|", flush=True)
    for name in args.only.split(","):
        for task, tool, ref, ours in groups[name](args.reps):
            print(f"| {task} | {tool} | {ref:.3g} s | {ours:.3g} s | {ref / ours:.0f} × |", flush=True)


if __name__ == "__main__":
    main()
