# SPDX-License-Identifier: GPL-3.0-or-later
"""Objectives that reward the targets met (max features, max phylogenetic diversity and endemism) in
ls.prioritize against Gurobi.

The instances of the MILP prioritization's phase 9: side x side cells, 30 Gaussian features (each over 30 % of
the area: the field above its 70 % quantile, minus it), costs from 1 to 10 (a field), relative targets
(``--target``, 30 % by default), a budget of 10 % of the total cost and, for the phylogenetic objectives, a
random tree of the features (branch lengths 0.5 to 3).

Solvers: ``landschaft`` (``Problem.solve`` on HiGHS: the wide engine, then the search over the targets met) and
``gurobi`` (the same compiled model). Every run: objective, bound, gap and seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import ssl  # noqa: F401  (Windows: load Python's OpenSSL before GDAL's)
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import landschaft as ls  # noqa: E402

P = ls.prioritize


def _field(side: int, seed: int) -> np.ndarray:
    return ls.nlm.gaussian_field(side, side, autocorr_range=max(side / 25, 2), seed=seed).values


def tree(names, seed=7) -> str:
    rng = np.random.default_rng(seed)
    nodes = list(names)
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append(f"({a}:{rng.uniform(0.5, 3):.3f},{b}:{rng.uniform(0.5, 3):.3f})")
    return nodes[0] + ";"


def problem(side: int, objective: str, target: float, budget_share: float = 0.1, n_features: int = 30):
    from optimize_cases import grid

    c = _field(side, 1)
    cost = 1 + 9 * (c - c.min()) / (c.max() - c.min())
    feats = []
    for k in range(n_features):
        v = _field(side, 1000 + k)
        cut = np.quantile(v, 0.7)
        feats.append(np.where(v > cut, v - cut, 0.0))
    names = [f"f{k + 1}" for k in range(n_features)]
    p = P.problem(grid(cost, res=1.0), grid(np.stack(feats), res=1.0, names=names)).add_relative_targets(target) \
        .add_binary_decisions()
    budget = budget_share * float(cost.sum())
    if objective == "max_features":
        return p.add_max_features_objective(budget)
    if objective == "max_phylo_div":
        return p.add_max_phylo_div_objective(budget, tree(names))
    return p.add_max_phylo_end_objective(budget, tree(names))


def run(p, solver: str, gap: float, time_limit: float, threads: int) -> dict:
    q = (p.add_highs_solver if solver == "landschaft" else p.add_gurobi_solver)(gap=gap, time_limit=time_limit,
                                                                                 n_threads=threads)
    t0 = time.perf_counter()
    s = q.solve()
    return {"objective": s.objective, "bound": s.bound, "gap": s.gap, "status": s.status,
            "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", nargs="+", type=int, default=[100])
    ap.add_argument("--objectives", nargs="+", default=["max_features", "max_phylo_div", "max_phylo_end"])
    ap.add_argument("--solvers", nargs="+", default=["landschaft", "gurobi"])
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="JSON lines file to append the results to")
    a = ap.parse_args()
    for side in a.sides:
        for objective in a.objectives:
            p = problem(side, objective, a.target)
            for solver in a.solvers:
                r = run(p, solver, a.gap, a.time_limit, a.threads)
                r.update(units=side * side, objective_name=objective, solver=solver, target=a.target,
                         gap_asked=a.gap, time_limit=a.time_limit, threads=a.threads)
                print(f"{side * side:7d} {objective:14s} {solver:11s} obj {r['objective']:12.4f} "
                      f"bound {r['bound']:12.4f} gap {100 * r['gap']:7.2f} % {r['seconds']:7.1f} s {r['status']}",
                      flush=True)
                if a.out is not None:
                    with a.out.open("a") as fh:
                        fh.write(json.dumps({k: (v if not (isinstance(v, float) and math.isnan(v)) else None)
                                             for k, v in r.items()}) + "\n")


if __name__ == "__main__":
    main()
