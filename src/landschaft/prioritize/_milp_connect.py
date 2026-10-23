# SPDX-License-Identifier: GPL-3.0-or-later
"""Contiguity and corridors on HiGHS: the exact models of
:func:`~landschaft.prioritize._milp_spatial._add_connected` and ``_apply_corridors`` solved in steps.

1. On minimum sets with contiguity, a patch heuristic (``opt/patch.cpp``: patches grown from every unit,
   pruned, improved by moves, then perturbed and regrown), 10 % of the time.
2. A first patch from the model with every arc running from a higher node to a lower one only
   (prioritizr's formulation, a restriction: every patch it finds is a patch, but it misses some), from
   the heuristic's patch, to the gap asked (at least 1 %) within 20 % of the time (two minutes without
   a time limit); the heuristic again from its patch.
3. Cuts at the root, by rounds on the linear relaxation re-solved warm (``opt::separate_root``), 40 % of
   the time left: directed cuts r(W) + y(in(W)) >= x_k for every set W of nodes holding node k (a patch
   reaches k from its root, inside W or through an arc into W), separated by maximum flows from a super
   root (arcs to every node with capacity r*, the arcs with capacity y*) to the most selected nodes
   (``opt/connect.cpp``); corridors get node cuts, x(S) >= 1 for every set S of units that separates a
   terminal from the source (a selected path crosses it), by maximum flows through the units (capacity
   x*). They join the model as rows.
4. HiGHS's MIP on the exact model with them, from the best patch. With one patch (contiguity) or
   corridors the same cuts are separated at every node of its branch-and-bound too (the user's separator
   of ``opt/highs_patch.cmake``; presolve off); with a patch per feature, HiGHS's presolve does more.

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

MAX_CANDIDATES = 200  # nodes tried per structure and separation call
FIRST_PATCH_SHARE = 0.2  # the restriction's share of the time limit
HEURISTIC_SHARE = 0.1  # the heuristic's share of the time limit (minimum sets with contiguity)
HEURISTIC_SECONDS = 10.0  # ... and its time at most without a time limit
STALL_ROUNDS = 50  # without a time limit, its perturbations stop after this many rounds without a cheaper patch
ROOT_SHARE = 0.4  # the root cuts' share of the time left
TREE_CUTS = True  # the cuts separated at the nodes too, with one patch or corridors (presolve off)
FIRST_PATCH_SECONDS = 120.0  # the first patch's time without a time limit (the exact MIP goes on)


def _restricted_bounds(m: OptimizationProblem) -> np.ndarray:
    """Upper bounds with the arcs from a lower node to a higher one fixed at 0 (prioritizr's restriction)."""
    ub = m.ub.copy()
    for c in m.connected:
        up = np.flatnonzero(c.tail < c.head)
        ub[c.arc + up] = 0.0
    return ub


def structures(m: OptimizationProblem, n_units: int | None) -> tuple[list, list]:
    """The patches and corridor networks of a model for the C++ separator (``opt_solve_highs``)."""
    conn = [{"nodes": c.nodes, "root": int(c.root), "arc": int(c.arc), "tail": c.tail, "head": c.head}
            for c in m.connected]
    cor = []
    for c in m.corridors:
        n = n_units if n_units is not None else int(
            max(c.tail.max(initial=-1), c.head.max(initial=-1), c.source, c.sinks.max(initial=-1)) + 1)
        cor.append({"n": int(n), "source": int(c.source), "sinks": c.sinks, "tail": c.tail, "head": c.head})
    return conn, cor


def separate(m: OptimizationProblem, z: np.ndarray, n_units: int | None = None,
             max_candidates: int = MAX_CANDIDATES) -> sp.csr_matrix:
    """The violated cuts of a solution z (directed cuts of the patches, node cuts of the corridors) as
    rows ``a z <= upper``: (a, upper)."""
    from .. import _core

    conn, cor = structures(m, n_units)
    start, index, value, upper = _core.opt_separate_connected(np.asarray(z, dtype=np.float64), conn, cor,
                                                              max_candidates)
    return sp.csr_matrix((value, index, start), shape=(upper.size, m.ncol)), upper


def _patch_problem(m: OptimizationProblem, n_units: int | None) -> dict | None:
    """The arrays of :func:`landschaft._core.opt_improve_patch` for a minimum set with contiguity and
    nothing else (target rows >= on the units, the patch's columns and rows); None for other models."""
    if (n_units is None or m.modelsense != "min" or len(m.connected) != 1 or m.corridors
            or not np.array_equal(m.connected[0].nodes, np.arange(n_units))):
        return None
    n = n_units
    if np.any(m.obj[n:] != 0) or np.any(m.obj[:n] < 0) or not all(c.startswith("c_") for c in m.col_ids[n:]):
        return None
    a = m.A.tocsr()
    touches_other = np.asarray((a[:, n:] != 0).sum(axis=1)).ravel() > 0
    target_rows = []
    for r in range(m.nrow):
        if touches_other[r]:
            if not m.row_ids[r].startswith("c_"):
                return None
        elif m.sense[r] == ">=":
            target_rows.append(r)
        else:
            return None
    t = a[target_rows][:, :n].tocsc()
    t.sort_indices()
    c = m.connected[0]
    adj = sp.csr_matrix((np.ones(c.tail.size), (c.tail, c.head)), shape=(n, n))
    adj.sort_indices()
    return {"cost": m.obj[:n], "start": t.indptr.astype(np.int64), "feature": t.indices.astype(np.int64),
            "amount": t.data, "target": m.rhs[target_rows], "adj_start": adj.indptr.astype(np.int64),
            "adj": adj.indices.astype(np.int64), "locked_in": (m.lb[:n] > 0.5).astype(np.uint8),
            "locked_out": (m.ub[:n] < 0.5).astype(np.uint8), "rows": np.asarray(target_rows, dtype=np.int64)}


def patch_columns(m: OptimizationProblem, selection: np.ndarray) -> np.ndarray:
    """A solution of the whole model from a selection that is one patch: its root (the first unit), the arcs
    of a breadth-first tree from it, and the flows (every arc carries the units below it)."""
    from collections import deque

    c = m.connected[0]
    n = c.nodes.size
    na = c.tail.size
    x = np.zeros(m.ncol)
    sel = np.flatnonzero(np.asarray(selection) > 0.5)
    x[c.nodes[sel]] = 1.0
    if not sel.size:
        return x
    adj = sp.csr_matrix((np.arange(na) + 1.0, (c.tail, c.head)), shape=(n, n))
    on = np.zeros(n, dtype=bool)
    on[sel] = True
    root = int(sel[0])
    parent_arc = np.full(n, -1)
    seen = np.zeros(n, dtype=bool)
    seen[root] = True
    order = [root]
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for k in range(adj.indptr[u], adj.indptr[u + 1]):
            v = int(adj.indices[k])
            if on[v] and not seen[v]:
                seen[v] = True
                parent_arc[v] = int(adj.data[k]) - 1
                order.append(v)
                queue.append(v)
    below = np.ones(n)
    for v in reversed(order[1:]):
        below[c.tail[parent_arc[v]]] += below[v]
    x[c.root + root] = 1.0
    arcs = parent_arc[order[1:]]
    x[c.arc + arcs] = 1.0
    x[c.arc + na + arcs] = below[order[1:]]
    return x


def _first_patch(m: OptimizationProblem, pp: dict, n_units: int, seconds: float, threads: int,
                 start: np.ndarray | None, timed: bool = True) -> np.ndarray | None:
    """A patch by the heuristic (:func:`landschaft._core.opt_improve_patch`: grown from every unit, and from
    ``start``): the whole model's solution, or None. ``timed``: the perturbations go on for all ``seconds``
    (with a time limit); otherwise they stop after rounds that find nothing cheaper."""
    from .. import _core

    s0 = np.zeros(0, dtype=np.uint8) if start is None else (np.asarray(start)[:n_units] > 0.5).astype(np.uint8)
    kw = {k: v for k, v in pp.items() if k != "rows"}
    h = _core.opt_improve_patch(**kw, initial=s0, time_limit=max(seconds, 1.0e-3), threads=threads,
                                stall_rounds=2**31 - 1 if timed else STALL_ROUNDS)
    if not h["has_solution"]:
        return None
    return patch_columns(m, h["selection"])


def _better(a: float, b: float, sense: str) -> bool:
    return a < b if sense == "min" else a > b


def root_cuts(m: OptimizationProblem, st: tuple[list, list], seconds: float, threads: int) -> dict:
    """The root's cuts by rounds on the LP relaxation (:func:`landschaft._core.opt_root_cuts`): the model
    with them as rows (``model``), the LP's bound and status."""
    from .. import _core
    from ._milp_solve import _integer

    lo, hi = m.row_bounds()
    a = m.A
    rc = _core.opt_root_cuts(start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data,
                             obj=m.obj, lb=m.lb, ub=m.ub, row_lo=lo, row_hi=hi, integer=_integer(m.vtype),
                             maximize=m.modelsense == "max", time_limit=seconds, threads=threads,
                             connected=st[0], corridors=st[1], max_candidates=MAX_CANDIDATES)
    k = rc["upper"].size
    model = m
    if k:
        cuts = sp.csr_matrix((rc["value"], rc["index"], rc["start"]), shape=(k, m.ncol))
        model = OptimizationProblem(A=sp.vstack([m.A, cuts]).tocsr(), obj=m.obj, lb=m.lb, ub=m.ub,
                                    rhs=np.concatenate([m.rhs, rc["upper"]]), sense=m.sense + ("<=",) * k,
                                    vtype=m.vtype, modelsense=m.modelsense, row_ids=m.row_ids + ("cut",) * k,
                                    col_ids=m.col_ids, connected=m.connected, corridors=m.corridors)
    return {"model": model, "bound": rc["bound"], "status": rc["status"], "rounds": rc["rounds"]}


def solve_connected(m: OptimizationProblem, gap: float, time_limit: float, threads: int, presolve: str,
                    verbose: bool, start: np.ndarray | None, n_units: int | None) -> dict:
    from ._milp_solve import _highs

    t0 = time.perf_counter()
    sense = m.modelsense
    sign = 1.0 if sense == "min" else -1.0
    st = structures(m, n_units)
    timed = math.isfinite(time_limit)

    def left() -> float:
        return time_limit - (time.perf_counter() - t0)

    best_x, best = None, math.inf * sign
    if start is not None:
        best_x = np.asarray(start, dtype=np.float64)
        best = float(m.obj @ best_x)

    def keep(x) -> None:
        nonlocal best_x, best
        if x is not None and (best_x is None or _better(float(m.obj @ x), best, sense)):
            best_x, best = np.asarray(x, dtype=np.float64), float(m.obj @ x)

    # 1. a first patch: the heuristic on minimum sets with contiguity; then the restriction from the best patch,
    # and the heuristic again from the restriction's
    pp = _patch_problem(m, n_units)
    if pp is not None:
        keep(_first_patch(m, pp, n_units, left() * HEURISTIC_SHARE if timed else HEURISTIC_SECONDS, threads,
                          best_x, timed))
    if m.connected:
        share = left() * FIRST_PATCH_SHARE if timed else FIRST_PATCH_SECONDS
        restricted = OptimizationProblem(A=m.A, obj=m.obj, lb=m.lb, ub=_restricted_bounds(m), rhs=m.rhs,
                                         sense=m.sense, vtype=m.vtype, modelsense=sense, row_ids=m.row_ids,
                                         col_ids=m.col_ids)
        r0 = _highs(restricted, max(gap, 0.01), share, threads, presolve, verbose, best_x, n_units=n_units)
        if r0["has_solution"]:
            x0 = np.asarray(r0["x"], dtype=np.float64)
            if pp is not None and (best_x is None or _better(float(m.obj @ x0), best, sense)):
                keep(_first_patch(m, pp, n_units, left() * HEURISTIC_SHARE / 2 if timed else HEURISTIC_SECONDS / 2,
                                  threads, x0, timed))
            keep(x0)

    # 2. the root's cuts, as rows of the model
    out = {"x": best_x, "has_solution": best_x is not None, "objective": best if best_x is not None else math.nan,
           "bound": -math.inf * sign, "nodes": 0}
    model = m
    if left() > 0:
        rc = root_cuts(m, st, left() * ROOT_SHARE if timed else math.inf, threads)
        if rc["status"] == "infeasible":
            return {**out, "status": "infeasible", "has_solution": False, "x": None, "objective": math.nan,
                    "bound": math.nan, "gap": math.nan, "seconds": time.perf_counter() - t0}
        model = rc["model"]
        if rc["status"] == "optimal":
            out["bound"] = rc["bound"]

    # 3. the exact MIP with them, more cuts separated at its nodes, from the best patch
    closed = best_x is not None and math.isfinite(out["bound"]) and \
        sign * (best - out["bound"]) <= gap * max(abs(best), 1.0e-10)
    if not closed and left() > 0:
        # the cuts at the nodes too with one patch (contiguity) or corridors; with a patch per feature,
        # HiGHS's presolve does more for the bound than the cuts (measured: 421 against 401 in 120 s)
        tree = TREE_CUTS and len(m.connected) <= 1
        r = _highs(model, gap, left(), threads, presolve, verbose, best_x, n_units=n_units,
                   structures=st if tree else None, method="highs")
        if r["has_solution"] and (best_x is None or _better(r["objective"], best, sense)):
            out.update(x=np.asarray(r["x"], dtype=np.float64), has_solution=True, objective=float(r["objective"]))
        if math.isfinite(r.get("bound", math.nan)):
            out["bound"] = r["bound"] if not math.isfinite(out["bound"]) else (
                max(out["bound"], r["bound"]) if sense == "min" else min(out["bound"], r["bound"]))
        out["nodes"] = r.get("nodes", 0)
        if r["status"] == "infeasible" and not out["has_solution"]:
            out["status"] = "infeasible"
    if out.get("status") != "infeasible":
        if out["has_solution"] and math.isfinite(out["bound"]):
            g = sign * (out["objective"] - out["bound"])
            out["gap"] = 0.0 if g <= 0 else g / max(abs(out["objective"]), 1.0e-10)
        else:
            out["gap"] = math.nan
        out["status"] = "optimal" if out["has_solution"] and out["gap"] <= gap else "time_limit"
    else:
        out["gap"] = math.nan
    out["seconds"] = time.perf_counter() - t0
    return out
