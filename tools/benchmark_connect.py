# SPDX-License-Identifier: GPL-3.0-or-later
"""Contiguity and corridors in ls.prioritize against Gurobi.

Instances (``--instances``):

- ``contiguity_raster``, ``contiguity_matrix``, ``feature_contiguity_raster``: the test problems of
  ``tests/optimize_cases.py`` (166 units);
- ``synthetic13``: a 13 x 13 grid, 10 features (presence 30 %), relative targets of 20 %, contiguity;
- ``serra_verde_3km``: the Serra Verde example in blocks of 3 x 3 km (``coarsen``), 8 features, relative
  targets of 20 %, contiguity.

Solvers (``--solvers``):

- ``landschaft``: ``Problem.solve`` on HiGHS (the first patch, then the branch-and-cut);
- ``gurobi``: Gurobi on the same compiled (exact) model;
- ``gurobi_cuts``: the same with landschaft's cuts (``opt_separate_connected``) added at Gurobi's nodes as
  user cuts.

Every run: objective, bound, gap and seconds, with the gap and time limit given.
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


def synthetic13():
    import optimize_cases as oc

    rng = np.random.default_rng(13)
    side, nf = 13, 10
    cost = rng.uniform(1.0, 10.0, (side, side))
    feats = (rng.random((nf, side, side)) < 0.3) * rng.uniform(0.5, 1.5, (nf, side, side))
    return (P.problem(oc.grid(cost), oc.grid(feats)).add_min_set_objective().add_relative_targets(0.2)
            .add_binary_decisions().add_contiguity_constraints())


def serra_verde_3km():
    cost = ls.load_raster(P.example_data("cost"))
    feats = ls.load_raster(P.example_data("features"))
    c, f = P.coarsen(cost, feats, 3)
    return P.problem(c, f).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions() \
        .add_contiguity_constraints()


def case(name):
    import optimize_cases as oc

    return lambda: oc.CASES[name].py()


INSTANCES = {
    "contiguity_raster": case("contiguity_raster"),
    "contiguity_matrix": case("contiguity_matrix"),
    "feature_contiguity_raster": case("feature_contiguity_raster"),
    "synthetic13": synthetic13,
    "serra_verde_3km": serra_verde_3km,
}


def run_landschaft(p, gap, time_limit, threads):
    t0 = time.perf_counter()
    s = p.add_highs_solver(gap=gap, time_limit=time_limit, n_threads=threads).solve()
    return {"objective": s.objective, "bound": s.bound, "gap": s.gap, "status": s.status,
            "seconds": time.perf_counter() - t0}


def run_gurobi(p, gap, time_limit, threads, cuts: bool):
    import gurobipy as gp

    from landschaft import _core
    from landschaft.prioritize._milp_connect import structures

    m = p.compile()
    n_units = p.units.n
    conn, cor = structures(m, n_units)
    vt = {"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER}
    t0 = time.perf_counter()
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as g:
        x = g.addMVar(m.ncol, lb=m.lb, ub=np.where(np.isinf(m.ub), gp.GRB.INFINITY, m.ub),
                      vtype=[vt.get(t, gp.GRB.CONTINUOUS) for t in m.vtype])
        g.addMConstr(m.A, x, np.asarray([s[0] for s in m.sense]), m.rhs)
        g.setObjective(m.obj @ x, gp.GRB.MAXIMIZE if m.modelsense == "max" else gp.GRB.MINIMIZE)
        g.Params.MIPGap = gap
        g.Params.Threads = threads
        g.Params.TimeLimit = time_limit
        xs = x.tolist()
        if cuts:
            g.Params.PreCrush = 1

            def cb(model, where):
                if where != gp.GRB.Callback.MIPNODE:
                    return
                if model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) != gp.GRB.OPTIMAL:
                    return
                z = np.asarray(model.cbGetNodeRel(xs))
                start, index, value, upper = _core.opt_separate_connected(z, conn, cor, 200)
                for k in range(upper.size):
                    a, b = start[k], start[k + 1]
                    model.cbCut(gp.quicksum(value[q] * xs[index[q]] for q in range(a, b)) <= upper[k])

            g.optimize(cb)
        else:
            g.optimize()
        ok = g.SolCount > 0
        return {"objective": g.ObjVal if ok else math.nan, "bound": g.ObjBound,
                "gap": g.MIPGap if ok else math.nan, "status": str(g.Status),
                "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=list(INSTANCES))
    ap.add_argument("--solvers", nargs="+", default=["landschaft", "gurobi", "gurobi_cuts"])
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="JSON lines file to append the results to")
    a = ap.parse_args()
    for name in a.instances:
        p = INSTANCES[name]()
        for solver in a.solvers:
            if solver == "landschaft":
                r = run_landschaft(p, a.gap, a.time_limit, a.threads)
            else:
                r = run_gurobi(p, a.gap, a.time_limit, a.threads, solver == "gurobi_cuts")
            r.update(instance=name, solver=solver, units=p.units.n, gap_asked=a.gap, time_limit=a.time_limit,
                     threads=a.threads)
            print(f"{name:26s} {solver:12s} obj {r['objective']:12.4f} bound {r['bound']:12.4f} "
                  f"gap {100 * r['gap']:6.2f} % {r['seconds']:7.1f} s {r['status']}", flush=True)
            if a.out is not None:
                with a.out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
