# SPDX-License-Identifier: GPL-3.0-or-later
"""Contiguity and corridors on HiGHS: the exact models of
:func:`~landschaft.prioritize._milp_spatial._add_connected` and ``_apply_corridors`` solved in three
steps.

1. A first patch: the model with every arc running from a higher node to a lower one only (prioritizr's
   formulation, a restriction: every patch it finds is a patch, but it misses some), solved to the gap
   asked (at least 1 %) within 30 % of the time (two minutes without a time limit).
2. The bound: the linear relaxation of the exact model with directed cuts, r(W) + y(in(W)) >= x_k for
   every set W of nodes holding node k (a patch reaches k from its root, inside W or through an arc
   into W), separated by maximum flows from a super root (arcs to every node with capacity r*, the
   arcs with capacity y*) to the most selected nodes, in rounds while the bound rises.
   Corridors get node cuts, x(S) >= 1 for every set S of units that separates a terminal from the
   source (a selected path crosses it), separated by maximum flows through the units (capacity x*).
3. HiGHS's MIP on the exact model with those cuts, from the first patch.

The cuts are valid for every patch, so the bound proves the gap of the best patch found against the
best patch there is.
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp

from ._milp import OptimizationProblem

__all__ = ["solve_connected"]

FLOW_SCALE = 1.0e7  # scipy's maximum_flow wants integer capacities
MAX_CANDIDATES = 200  # nodes tried per structure and round
MIN_RISE = 1.0e-3  # rounds stop when the bound rises less than this (relative) three times running
FIRST_PATCH_SECONDS = 120.0  # the first patch's time without a time limit (the exact MIP goes on)


def _restricted_bounds(m: OptimizationProblem) -> np.ndarray:
    """Upper bounds with the arcs from a lower node to a higher one fixed at 0 (prioritizr's restriction)."""
    ub = m.ub.copy()
    for c in m.connected:
        up = np.flatnonzero(c.tail < c.head)
        ub[c.arc + up] = 0.0
    return ub


def _cuts(m: OptimizationProblem, z: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Violated directed cuts of a fractional solution z: (columns, values, 0) of rows ``>= 0``."""
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    out = []
    for c in m.connected:
        n = c.nodes.size
        xv = z[c.nodes]
        rv = z[c.root:c.root + n]
        yv = z[c.arc:c.arc + c.tail.size]
        keep = yv > 1.0e-9
        rk = rv > 1.0e-9
        src = np.concatenate([c.tail[keep], np.full(int(rk.sum()), n)])
        dst = np.concatenate([c.head[keep], np.flatnonzero(rk)])
        cap = np.concatenate([yv[keep], rv[rk]])
        g = sp.csr_matrix((np.maximum(np.round(cap * FLOW_SCALE), 1).astype(np.int32), (src, dst)),
                          shape=(n + 1, n + 1))
        seen: set = set()
        for k in np.argsort(-xv, kind="stable")[:MAX_CANDIDATES]:
            if xv[k] < 0.05:
                break
            f = maximum_flow(g, n, int(k))
            if f.flow_value / FLOW_SCALE >= xv[k] - 1.0e-4:
                continue
            res = (g - f.flow).tocsr()
            res.data = np.maximum(res.data, 0)
            res.eliminate_zeros()
            w = breadth_first_order(res.T.tocsr(), int(k), directed=True, return_predecessors=False)
            w = np.sort(w[w != n])
            key = w.tobytes()
            if key in seen:
                continue
            seen.add(key)
            inside = np.zeros(n, dtype=bool)
            inside[w] = True
            arcs = np.flatnonzero(inside[c.head] & ~inside[c.tail])
            cols = np.concatenate([c.arc + arcs, c.root + w, [c.nodes[k]]])
            vals = np.concatenate([np.ones(arcs.size + w.size), [-1.0]])
            out.append((cols.astype(np.int64), vals, 0.0))
    return out


def _corridor_cuts(m: OptimizationProblem, z: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Violated node cuts x(S) >= 1 of the corridor networks: (columns, values, 1) of rows ``>= 1``."""
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    out = []
    # "infinite" capacity: 10 is plenty to decide whether a unit of flow gets through, and keeps the
    # flows within scipy's 32-bit integers
    big = np.int64(10 * FLOW_SCALE)
    for c in m.corridors:
        n = int(max(c.tail.max(initial=-1), c.head.max(initial=-1), c.source, c.sinks.max(initial=-1)) + 1)
        x = np.clip(z[:n], 0.0, 1.0)
        cap = np.maximum(np.round(x * FLOW_SCALE), 0).astype(np.int64)
        cap[c.source] = big
        cap[c.sinks] = big
        # node v: in = v, out = n + v; an edge both ways out -> in
        src = np.concatenate([np.arange(n), n + c.tail, n + c.head])
        dst = np.concatenate([n + np.arange(n), c.head, c.tail])
        w = np.concatenate([np.minimum(cap, big), np.full(2 * c.tail.size, big)]).astype(np.int32)
        g = sp.csr_matrix((w, (src, dst)), shape=(2 * n, 2 * n))
        g.sum_duplicates()
        for t in c.sinks:
            f = maximum_flow(g, n + c.source, int(t))
            if f.flow_value / FLOW_SCALE >= 1.0 - 1.0e-4:
                continue
            res = (g - f.flow).tocsr()
            res.data = np.maximum(res.data, 0)
            res.eliminate_zeros()
            reach = np.zeros(2 * n, dtype=bool)
            reach[breadth_first_order(res, n + c.source, directed=True, return_predecessors=False)] = True
            s = np.flatnonzero(reach[:n] & ~reach[n:])  # units whose in-node is reached and out-node not
            if s.size:
                out.append((s.astype(np.int64), np.ones(s.size), 1.0))
    return out


def _with_rows(m: OptimizationProblem, cuts: list) -> OptimizationProblem:
    if not cuts:
        return m
    r = np.concatenate([np.full(c.size, i) for i, (c, _, _) in enumerate(cuts)])
    a = sp.csr_matrix((np.concatenate([v for _, v, _ in cuts]), (r, np.concatenate([c for c, _, _ in cuts]))),
                      shape=(len(cuts), m.ncol))
    return OptimizationProblem(A=sp.vstack([m.A, a]).tocsr(), obj=m.obj, lb=m.lb, ub=m.ub,
                               rhs=np.concatenate([m.rhs, [h for _, _, h in cuts]]),
                               sense=m.sense + (">=",) * len(cuts), vtype=m.vtype, modelsense=m.modelsense,
                               row_ids=m.row_ids + ("cut",) * len(cuts), col_ids=m.col_ids, connected=m.connected,
                               corridors=m.corridors)


def _better(a: float, b: float, sense: str) -> bool:
    return a < b if sense == "min" else a > b


def solve_connected(m: OptimizationProblem, gap: float, time_limit: float, threads: int, presolve: str,
                    verbose: bool, start: np.ndarray | None, n_units: int | None) -> dict:
    from ._milp_solve import _highs

    t0 = time.perf_counter()
    sense = m.modelsense
    sign = 1.0 if sense == "min" else -1.0

    def left() -> float:
        return time_limit - (time.perf_counter() - t0)

    best_x, best = None, math.inf * sign
    if start is not None:
        best_x = np.asarray(start, dtype=np.float64)
        best = float(m.obj @ best_x)
    # 1. a first patch from the restriction
    if m.connected:
        share = left() * 0.3 if math.isfinite(time_limit) else FIRST_PATCH_SECONDS
        restricted = OptimizationProblem(A=m.A, obj=m.obj, lb=m.lb, ub=_restricted_bounds(m), rhs=m.rhs,
                                         sense=m.sense, vtype=m.vtype, modelsense=sense, row_ids=m.row_ids,
                                         col_ids=m.col_ids)
        r0 = _highs(restricted, max(gap, 0.01), share, threads, presolve, verbose, best_x, n_units=n_units)
        if r0["has_solution"] and (best_x is None or _better(r0["objective"], best, sense)):
            best_x, best = np.asarray(r0["x"], dtype=np.float64), float(r0["objective"])

    # 2. the bound: the exact relaxation with directed cuts
    cuts: list = []
    bound = -math.inf * sign
    rises = 0
    budget = left() * 0.4 if math.isfinite(time_limit) else math.inf
    tl0 = time.perf_counter()
    for _ in range(200):
        if time.perf_counter() - tl0 >= budget or left() <= 0:
            break
        lp = _highs(_with_rows(m, cuts), gap, max(left(), 1.0e-3), threads, presolve, False, None,
                    relaxation=True, n_units=n_units)
        if lp["status"] != "optimal":
            if lp["status"] == "infeasible":
                return {"status": "infeasible", "has_solution": False, "x": None, "objective": math.nan,
                        "bound": math.nan, "gap": math.nan, "nodes": 0, "seconds": time.perf_counter() - t0}
            break
        rise = sign * (lp["objective"] - bound) if math.isfinite(bound) else math.inf
        bound = lp["objective"] if not math.isfinite(bound) else (max(bound, lp["objective"]) if sense == "min"
                                                                  else min(bound, lp["objective"]))
        if best_x is not None and sign * (best - bound) <= gap * max(abs(best), 1.0e-10):
            break
        z = np.asarray(lp["x"])
        new = _cuts(m, z) + _corridor_cuts(m, z)
        if not new:
            break
        cuts += new
        rises = rises + 1 if rise < MIN_RISE * max(abs(bound), 1.0) else 0
        if rises >= 3:
            break

    out = {"x": best_x, "has_solution": best_x is not None, "objective": best if best_x is not None else math.nan,
           "bound": bound, "nodes": 0}
    closed = best_x is not None and math.isfinite(bound) and sign * (best - bound) <= gap * max(abs(best), 1.0e-10)
    # 3. the exact MIP with the cuts, from the first patch
    if not closed and left() > 0:
        r = _highs(_with_rows(m, cuts), gap, left(), threads, presolve, verbose, best_x, n_units=n_units)
        if r["has_solution"] and (best_x is None or _better(r["objective"], best, sense)):
            out.update(x=np.asarray(r["x"], dtype=np.float64), has_solution=True, objective=float(r["objective"]))
        if math.isfinite(r.get("bound", math.nan)):
            out["bound"] = r["bound"] if not math.isfinite(bound) else (
                max(bound, r["bound"]) if sense == "min" else min(bound, r["bound"]))
        out["nodes"] = r.get("nodes", 0)
        if r["status"] == "infeasible" and not out["has_solution"]:
            out["status"] = "infeasible"
    if out.get("status") != "infeasible":
        if out["has_solution"] and math.isfinite(out["bound"]):
            g = sign * (out["objective"] - out["bound"])
            out["gap"] = 0.0 if g <= 0 else g / max(abs(out["objective"]), 1.0e-10)
        else:
            out["gap"] = math.nan
        if out["has_solution"] and out["gap"] <= gap:
            out["status"] = "optimal"
        else:
            out["status"] = "time_limit"
    else:
        out["gap"] = math.nan
    out["seconds"] = time.perf_counter() - t0
    return out
