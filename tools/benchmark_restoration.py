# SPDX-License-Identifier: GPL-3.0-or-later
"""Time the exact restoration planner at a global scale, on synthetic data.

Usage::

    python tools/benchmark_restoration.py [--side N] [--species N] [--range-share F] [--regions N] [--profile]

``side`` x ``side`` cells with random anthropic shares, ``species`` disk ranges covering about
``range-share`` of the grid each (random habitat shares), one landscape area target of 5 % of the
restorable area (or that target split among ``regions`` bands of rows). Prints the times of building
the data, the problem and the extinction-risk benefit, and of the solve (with its status, rounds and
gap); ``--profile`` adds cProfile's top functions. Pins itself to the P-cores of the dev machine
(logical CPUs 0-11) when psutil is there.

Measured (2026-10-09, i5-13600K, 11 threads): 10^6 cells x 2000 species (3.8 x 10^7 gains) 0.7 s;
4 x 10^6 x 4000 (1.5 x 10^8) 3.4 s; 9 x 10^6 x 5000 (4.3 x 10^8) 9.9 s, all proven within 1e-7.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


def synthetic(side: int, n_species: int, share: float, seed: int = 1):
    """Anthropic shares and a HabitatAreas of disk ranges (gain = habitat share x anthropic share)."""
    from landschaft.sdm._habitat import HabitatAreas

    rng = np.random.default_rng(seed)
    anth = rng.random((side, side)) * 0.7
    a_flat = anth.ravel()
    indptr, idx, vals, cur, pot = [0], [], [], [], []
    for _ in range(n_species):
        r = np.sqrt(share * rng.uniform(0.5, 1.5) / np.pi) * side
        cy, cx = rng.random(2) * side
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, side)
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, side)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        cells = (yy[inside] * side + xx[inside]).astype(np.int64)
        hab = rng.random(cells.size)
        a = a_flat[cells]
        idx.append(cells)
        vals.append(hab * a)
        cur.append(float(np.sum((1 - a) * hab)))
        pot.append(cur[-1] + float(np.sum(a * hab)))
        indptr.append(indptr[-1] + cells.size)
    gain = sp.csr_matrix((np.concatenate(vals), np.concatenate(idx), np.asarray(indptr)),
                         shape=(n_species, side * side))
    table = pd.DataFrame({"species": [f"s{k}" for k in range(n_species)], "current": cur, "potential": pot,
                          "range_cells": np.diff(indptr), "classes": "", "valid": True})
    return anth, HabitatAreas(table=table, gain=gain, _template=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=int, default=1000)
    parser.add_argument("--species", type=int, default=2000)
    parser.add_argument("--range-share", type=float, default=0.02)
    parser.add_argument("--regions", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(12)))
    except Exception:  # noqa: BLE001, S110 - another machine: no pinning
        pass
    import landschaft as ls

    t = time.perf_counter()
    anth, aoh = synthetic(args.side, args.species, args.range_share)
    print(f"data: {anth.size} cells, {args.species} species, {aoh.gain.nnz:.3g} gains, {time.perf_counter() - t:.1f} s")
    t = time.perf_counter()
    p = ls.prioritize.restoration_problem(anth, cell_area=1.0)
    print(f"problem: {time.perf_counter() - t:.2f} s")
    t = time.perf_counter()
    p = p.add_extinction_risk_benefit(aoh)
    print(f"risk benefit: {time.perf_counter() - t:.2f} s")
    target = 0.05 * float(p.upper @ p.area)
    if args.regions:
        reg = np.arange(args.side)[:, None] * args.regions // args.side * np.ones((1, args.side))
        p = p.add_area_target({float(k): target / args.regions for k in range(args.regions)}, regions=reg)
    else:
        p = p.add_area_target(target)
    prof = cProfile.Profile() if args.profile else None
    t = time.perf_counter()
    if prof:
        prof.enable()
    sol = p.solve()
    if prof:
        prof.disable()
    print(f"solve: {time.perf_counter() - t:.2f} s, status {sol.status}, rounds {sol.rounds}, "
          f"gap {(sol.bound - sol.benefit) / sol.bound:.2e}")
    if prof:
        pstats.Stats(prof).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
