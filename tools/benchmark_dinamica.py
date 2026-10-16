# SPDX-License-Identifier: GPL-3.0-or-later
"""Time the Dinamica EGO functors of version 1.46 against Dinamica EGO's console, on 4 cores.

Usage::

    python tools/benchmark_dinamica.py [--reps N] [--cores 0,2,4,6]

The process (and Dinamica, its child) is pinned to the given cores and both
sides use as many threads (Dinamica's ``-processors``). Dinamica's time is
the functor's own, from its log; Landschaft's the best of ``reps`` calls.
Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import simulate_engines as se  # noqa: E402


def grid(values, cell=30.0):
    import rioxarray  # noqa: F401
    import xarray as xr

    ny, nx = values.shape
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": 7.5e6 - cell * (np.arange(ny) + 0.5),
                                                       "x": 5e5 + cell * (np.arange(nx) + 0.5)})
    return da.rio.write_crs("EPSG:31983")


def dinamica(rasters: dict, script: str, functor: str, cores: int) -> float:
    """The elapsed time Dinamica logs for ``functor`` (its display name)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (da, categorical) in rasters.items():
            se.write_dinamica_raster(tmp / f"{name}.tif", da.assign_attrs(categorical=categorical), name)
        (tmp / "model.ego").write_text(script, newline="\n")
        p = subprocess.run([str(se.DINAMICA), "-processors", str(cores), "model.ego"], cwd=tmp,
                           capture_output=True, text=True, check=False)
        log = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)
        found = re.findall(rf'"{functor}" ran successfully \(elapsed ([0-9.]+) s', log)
        if not found:
            raise RuntimeError(log[-2000:])
        return float(found[0])


def best(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cores", default="0,2,4,6")
    a = ap.parse_args()
    cores = [int(c) for c in a.cores.split(",")]
    import psutil

    psutil.Process().cpu_affinity(cores)
    import landschaft as ls

    ls.set_default_threads(len(cores))
    S = ls.simulate
    rng = np.random.default_rng(0)
    rows = []

    n = 1000
    m = np.full((n, n), np.nan)
    m.flat[rng.choice(n * n, 1000, replace=False)] = rng.random(1000) * 100
    src = grid(m)
    d = dinamica({"src": (src, False)}, 'Script {{\n  s := LoadMap "src.tif";\n'
                 '  a := CalcInteractionPotentialMap s .none 2 .float32 .default;\n  SaveMap a "a.tif";\n}};\n',
                 "Calc Interaction Potential Map", len(cores))
    rows.append(("Gravity potential, 1000 × 1000, 1000 sources", d,
                 best(lambda: ls.surface.gravity(src, 2.0), a.reps)))

    n = 3000
    stock, prob = grid(rng.random((n, n)) * 200), grid(rng.random((n, n)))
    d = dinamica({"s": (stock, False), "p": (prob, False)},
                 'Script {{\n  s := LoadMap "s.tif";\n  p := LoadMap "p.tif";\n'
                 '  ml mp rq := PatcherContinuousState s p 450000 0.09 0 1.5 3 3 10;\n  SaveMap ml "ml.tif";\n}};\n',
                 "Patcher Continuous State", len(cores))
    cp = S.ContinuousPatcher(mean=0.09, isometry=1.5)
    rows.append(("Stock taken by patches, 3000 × 3000", d, best(lambda: cp.allocate(stock, prob, 450000, seed=1),
                                                               a.reps)))

    fr = grid(1 + rng.random((n, n)) * 9)
    net = np.zeros((n, n))
    net[0, :] = net[:, 0] = 1
    net[1500, 1000:2000] = 1
    d = dinamica({"net": (grid(np.where(net > 0, 1.0, np.nan)), True), "fr": (fr, False)},
                 'Script {{\n  n := LoadCategoricalMap "net.tif" .none .default 0;\n  f := LoadMap "fr.tif";\n'
                 '  c d := CalcCostMap n f .no .float64 -1 .yes 0 .no;\n  SaveMap c "c.tif";\n}};\n',
                 "Calc Cost Map", len(cores))
    rows.append(("Cost surface to a road network, 3000 × 3000 (passes until exact)", d,
                 best(lambda: S.friction_cost(net, fr), a.reps)))

    print(f"| Task | Dinamica EGO {se.dinamica_version()} | Landschaft {ls.__version__} |")
    print("|---|---|---|")
    for task, d, o in rows:
        print(f"| {task} | {d:.3g} s | {o:.3g} s |")


if __name__ == "__main__":
    main()
