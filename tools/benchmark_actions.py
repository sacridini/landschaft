# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-action problems with boundaries in ls.prioritize against Gurobi, to gap 0.

The inputs of ``tools/benchmark_optimize.py``'s ``actions`` family (prioriactions tables on a side x side
grid: 10 features and 3 threats, fields above a quantile; monitoring costs 1 to 3, action costs 1 to 5;
recovery targets 30 % of every feature's threatened amount), built here with ``ls.prioritize.actions_data``,
plus the grid's boundaries (shared edges of length 1, every cell's perimeter of 4), ``blm`` and every
threat's ``blm_actions``.

Solvers: ``landschaft`` (``ActionsProblem.solve`` on HiGHS: the closure engine, then the MIP) and ``gurobi``
(the same equivalent model on Gurobi). Every run: objective, bound, gap and seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import ssl  # noqa: F401  (Windows: load Python's OpenSSL before GDAL's)
import time
from pathlib import Path

import numpy as np
import pandas as pd

import landschaft as ls

P = ls.prioritize


def _field(side: int, seed: int) -> np.ndarray:
    return ls.nlm.gaussian_field(side, side, autocorr_range=max(side / 25, 2), seed=seed).values


def _presence(side: int, seed: int, share: float) -> np.ndarray:
    v = _field(side, seed).ravel()
    return v > np.quantile(v, 1 - share)


def boundary_table(side: int) -> pd.DataFrame:
    """Shared edges between 4-neighbours (1) and every cell's perimeter (4) on the diagonal."""
    ids = np.arange(1, side * side + 1).reshape(side, side)
    i1 = [ids[:, :-1].ravel(), ids[:-1, :].ravel(), ids.ravel()]
    i2 = [ids[:, 1:].ravel(), ids[1:, :].ravel(), ids.ravel()]
    b = [np.ones(i1[0].size), np.ones(i1[1].size), np.full(ids.size, 4.0)]
    return pd.DataFrame({"id1": np.concatenate(i1), "id2": np.concatenate(i2), "boundary": np.concatenate(b)})


def actions_problem(side: int, blm: float, blm_actions: float):
    rng = np.random.default_rng(5)
    n = side * side
    feat = np.stack([_presence(side, 2000 + k, 0.3) for k in range(10)])
    thr = np.stack([_presence(side, 3000 + k, 0.4) for k in range(3)])
    ids = np.arange(1, n + 1)
    fp, fk = np.nonzero(feat.T)
    tp, tk = np.nonzero(thr.T)
    recoverable = (feat & thr.any(axis=0)).sum(axis=1)
    pu = pd.DataFrame({"id": ids, "monitoring_cost": rng.integers(1, 4, n).astype(float), "status": 0})
    features = pd.DataFrame({"id": np.arange(1, 11), "target_recovery": np.floor(0.3 * recoverable)})
    dist_features = pd.DataFrame({"pu": fp + 1, "feature": fk + 1, "amount": 1.0})
    threats = pd.DataFrame({"id": np.arange(1, 4), "blm_actions": blm_actions})
    dist_threats = pd.DataFrame({"pu": tp + 1, "threat": tk + 1, "amount": 1.0,
                                 "action_cost": rng.integers(1, 6, tp.size).astype(float), "status": 0})
    sf, st = np.meshgrid(np.arange(1, 11), np.arange(1, 4))
    sensitivity = pd.DataFrame({"feature": sf.ravel(), "threat": st.ravel()})
    d = P.actions_data(pu, features, dist_features, threats, dist_threats, sensitivity, boundary_table(side))
    return d.problem("minimizeCosts", blm=blm)


def run(p, solver: str, gap: float, time_limit: float, threads: int) -> dict:
    q = (p.add_highs_solver if solver == "landschaft" else p.add_gurobi_solver)(gap=gap, time_limit=time_limit,
                                                                                 n_threads=threads)
    t0 = time.perf_counter()
    s = q.solve()
    return {"objective": s.objective, "bound": s.bound, "gap": s.gap, "status": s.status,
            "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", nargs="+", type=int, default=[32, 100])
    ap.add_argument("--solvers", nargs="+", default=["landschaft", "gurobi"])
    ap.add_argument("--blm", type=float, default=1.0)
    ap.add_argument("--blm-actions", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="JSON lines file to append the results to")
    a = ap.parse_args()
    for side in a.sides:
        p = actions_problem(side, a.blm, a.blm_actions)
        for solver in a.solvers:
            r = run(p, solver, a.gap, a.time_limit, a.threads)
            r.update(units=side * side, solver=solver, gap_asked=a.gap, time_limit=a.time_limit, threads=a.threads)
            print(f"{side * side:7d} {solver:11s} obj {r['objective']:12.4f} bound {r['bound']:12.4f} "
                  f"gap {100 * r['gap']:6.3f} % {r['seconds']:7.1f} s {r['status']}", flush=True)
            if a.out is not None:
                with a.out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
            if not math.isfinite(r["objective"]):
                continue


if __name__ == "__main__":
    main()
