# SPDX-License-Identifier: GPL-3.0-or-later
"""The anneal solver: Marxan's simulated annealing (``opt/marxan``) on a compiled
minimum-set model.

The model is read as Marxan's problem, exactly: the units' objective
coefficients, every pair column (``b = x_i x_j``, linearised by prioritizr's rows
``b <= x_i``, ``b <= x_j`` and ``b >= x_i + x_j - 1``) with objective ``q``
becomes a Marxan boundary of ``-q / 2`` between the two units and ``q / 2`` on
each unit's cost, since

    q x_i x_j = q / 2 (x_i + x_j) - q / 2 (x_i + x_j - 2 x_i x_j),

and the last bracket is 1 exactly when one of the two is selected (Marxan's
boundary, BLM 1). prioritizr's boundary penalty (``penalty`` x the selection's
boundary, ``edge_factor`` of the study area's edge) is thus Marxan's with BLM =
``penalty`` and every unit's fixed boundary its exposed edge x ``edge_factor``.
Rows ``A x >= rhs`` on the units are Marxan's features (``A``'s rows the
amounts, ``rhs`` the targets) and the columns' bounds its locked units.

Every run is then repaired to meet every target and trimmed (``opt/marxan``:
``repair``); the solution is the run with the lowest objective. The bound is
the linear relaxation of the model without the boundaries' cut terms (each at
least ``min(0, boundary) x (x_i + x_j)``), a valid lower bound on the optimum.
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp

from ._milp import OptimizationProblem

__all__ = ["marxan_model", "solve_anneal"]

_SUPPORTED = ("the anneal solver solves minimum-set problems with binary decisions: targets (and other "
              "'>=' rows on the planning units with amounts >= 0), locked units, and penalties on units and "
              "on pairs of units (boundary, connectivity, linear)")


def marxan_model(m: OptimizationProblem, n: int) -> dict:
    """The compiled model as Marxan's problem (BLM 1, no fixed boundaries): ``cost``, ``status`` (2 locked
    in, 3 locked out), the features' amounts by unit (``offset``, ``spindex``, ``amount``) and ``target``,
    and the boundaries (``pi``, ``pj``, ``bound``, every pair once)."""
    if m.modelsense != "min":
        raise ValueError(f"{_SUPPORTED}; this one maximises")
    vt = np.asarray(m.vtype)
    if not (vt[:n] == "B").all():
        raise ValueError(f"{_SUPPORTED}; this one has proportion or semi-continuous decisions")
    a = m.A.tocsr()
    a.sort_indices()
    nnz = np.diff(a.indptr)
    row = np.repeat(np.arange(a.shape[0]), nnz)
    col = a.indices
    val = a.data
    sense = np.asarray(m.sense)
    extra = col >= n
    pair_row = np.zeros(a.shape[0], dtype=bool)
    pair_row[row[extra]] = True
    # every row on a pair column: b - x_i <= 0 (two of them) or b - x_i - x_j >= -1
    rr = np.flatnonzero(pair_row)
    le = (sense[rr] == "<=") & (nnz[rr] == 2) & (m.rhs[rr] == 0)
    ge = (sense[rr] == ">=") & (nnz[rr] == 3) & (m.rhs[rr] == -1)
    if not (le | ge).all():
        raise ValueError(f"{_SUPPORTED}; this one has rows on other variables ({_ids(m, rr[~(le | ge)])})")
    on_pair = pair_row[row]
    ok = np.where(extra, val == 1.0, val == -1.0)
    if not ok[on_pair].all():
        raise ValueError(f"{_SUPPORTED}; this one has rows on other variables")
    le_rows = rr[le]
    first = a.indptr[le_rows]
    # the unit and the pair column of every b <= x_i row (the unit's column is the smaller)
    unit, pcol = col[first], col[first + 1]
    if (unit >= n).any() or (pcol < n).any():
        raise ValueError(f"{_SUPPORTED}; this one has rows on other variables")
    order = np.lexsort((unit, pcol))
    unit, pcol = unit[order], pcol[order]
    cols, counts = np.unique(pcol, return_counts=True)
    if (counts != 2).any() or (cols.size != m.ncol - n) or (unit[0::2] == unit[1::2]).any():
        raise ValueError(f"{_SUPPORTED}; this one has variables other than the planning units and their pairs")
    pi, pj = unit[0::2], unit[1::2]
    q = m.obj[cols]
    if not np.isfinite(m.lb[cols]).all() or (m.lb[cols] > 0).any() or (m.ub[cols] < 1).any():
        raise ValueError(f"{_SUPPORTED}; this one bounds a pair column")
    # the features: the rows on the units alone, all '>='
    feat = np.flatnonzero(~pair_row)
    bad = sense[feat] != ">="
    if bad.any():
        raise ValueError(f"{_SUPPORTED}; this one has '<=' or '=' rows ({_ids(m, feat[bad])})")
    f = a[feat][:, :n]
    if f.nnz and (f.data < 0).any():
        raise ValueError(f"{_SUPPORTED}; this one has a '>=' row with negative amounts")
    rhs = m.rhs[feat]
    keep = rhs > 0  # rows met by any selection
    f = f[np.flatnonzero(keep)]
    f.eliminate_zeros()
    by_unit = f.T.tocsr()
    by_unit.sort_indices()
    cost = np.asarray(m.obj[:n], dtype=np.float64).copy()
    np.add.at(cost, pi, q / 2)
    np.add.at(cost, pj, q / 2)
    lb, ub = m.lb[:n], m.ub[:n]
    status = np.zeros(n, dtype=np.int32)
    status[lb >= 0.5] = 2
    status[ub <= 0.5] = 3
    if ((lb >= 0.5) & (ub <= 0.5)).any():
        raise ValueError("a planning unit is locked in and out")
    return {"cost": cost, "status": status, "offset": by_unit.indptr.astype(np.int64),
            "spindex": by_unit.indices.astype(np.int32), "amount": by_unit.data.astype(np.float64),
            "target": np.asarray(rhs[keep], dtype=np.float64), "features": f, "pi": pi, "pj": pj,
            "bound": -q / 2}


def _ids(m: OptimizationProblem, rows: np.ndarray) -> str:
    return ", ".join(sorted({m.row_ids[r] for r in rows[:1000]}))


def _neighbours(n: int, pi: np.ndarray, pj: np.ndarray, bound: np.ndarray):
    """Every unit's neighbours and boundaries (both directions; repeated pairs summed)."""
    s = sp.coo_matrix((np.concatenate([bound, bound]), (np.concatenate([pi, pj]), np.concatenate([pj, pi]))),
                      shape=(n, n)).tocsr()
    s.sum_duplicates()
    s.sort_indices()
    return s.indptr.astype(np.int64), s.indices.astype(np.int32), s.data.astype(np.float64)


def _pair_values(m: OptimizationProblem, n: int, x: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    full = np.zeros(m.ncol)
    full[:n] = x
    full[n:] = x[pi] * x[pj] if pi.size else 0.0
    return full


def solve_anneal(m: OptimizationProblem, n: int, options: dict, start: np.ndarray | None = None) -> dict:
    from .. import _core, _threads
    from ._milp_solve import _highs

    t0 = time.perf_counter()
    mm = marxan_model(m, n)
    f, target = mm["features"], mm["target"]
    status = mm["status"]
    # infeasible targets (the units not locked out cannot meet them)
    avail = (status != 3).astype(np.float64)
    reach = f @ avail if f.shape[0] else np.zeros(0)
    if (reach < target - 1e-9 * np.maximum(1.0, np.abs(target))).any():
        return {"status": "infeasible", "has_solution": False, "x": None, "objective": math.nan,
                "bound": math.nan, "gap": math.nan, "nodes": 0, "seconds": time.perf_counter() - t0}
    if start is not None:
        s = np.asarray(start, dtype=np.float64)[:n]
        status = status.copy()
        status[(status == 0) & (s >= 0.5)] = 1
    nb_start, nb, nb_cost = _neighbours(n, mm["pi"], mm["pj"], mm["bound"])
    threads = _threads.resolve(options.get("n_threads"))
    tl = options.get("time_limit")
    num_iters = options.get("num_iters")
    if num_iters is None:
        num_iters = max(1_000_000, 1000 * n)
    spf = np.atleast_1d(np.asarray(options.get("spf", (1.0, 2.0)), dtype=np.float64))
    opts = {"blm": 1.0, "iterations": int(num_iters), "tinit": float(options.get("start_temp", -1.0)),
            "tcool": float(options.get("cool_fac", 0.0)), "titns": int(options.get("num_temp", 10_000)),
            "repeats": int(options.get("num_reps", 10)), "runmode": 1,
            "itimptype": int(options.get("itimp_type", 0)), "seed": int(options.get("seed", 1)) & 0xFFFFFFFF,
            "threads": threads, "time_limit": -1.0 if tl is None else float(tl),
            "fast_penalties": True, "repair": True, "time_schedule": tl is not None,
            "spf_scale": [float(v) for v in spf]}
    nf = target.size
    out = _core.opt_marxan(
        mm["cost"], status, mm["offset"], mm["spindex"], mm["amount"], target, np.zeros(nf),
        np.ones(nf), np.zeros(nf, dtype=np.int32), np.zeros(n), nb_start, nb, nb_cost, opts)
    sol = np.asarray(out["solutions"])
    pi, pj = mm["pi"], mm["pj"]
    best, best_obj = None, math.inf
    tol = 1e-9 * np.maximum(1.0, np.abs(target))
    for k in range(sol.shape[0]):
        x = ((sol[k] == 1) | (sol[k] == 2)).astype(np.float64)
        if nf and ((f @ x) < target - tol).any():
            continue
        obj = float(m.obj @ _pair_values(m, n, x, pi, pj))
        if obj < best_obj:
            best, best_obj = x, obj
    # the lower bound: the relaxation without the boundaries' cut terms
    lb_cost = mm["cost"].copy()
    neg = mm["bound"] < 0
    np.add.at(lb_cost, pi[neg], mm["bound"][neg])
    np.add.at(lb_cost, pj[neg], mm["bound"][neg])
    lower = OptimizationProblem(
        A=f.tocsr(), obj=lb_cost, lb=(status == 2).astype(np.float64), ub=(status != 3).astype(np.float64),
        rhs=target, sense=(">=",) * nf, vtype=("C",) * n, modelsense="min", row_ids=("target",) * nf,
        col_ids=("pu",) * n)
    remaining = math.inf if tl is None else max(1.0, float(tl) - (time.perf_counter() - t0))
    lr = _highs(lower, 0.0, remaining, threads, "choose", False, None, relaxation=True, n_units=n)
    bound = float(lr["bound"]) if lr.get("status") != "infeasible" else math.nan
    timed_out = bool(out.get("timed_out", False))
    if best is None:
        return {"status": "time_limit" if timed_out else "error: no run met every target", "has_solution": False,
                "x": None, "objective": math.nan, "bound": bound, "gap": math.nan, "nodes": 0,
                "seconds": time.perf_counter() - t0}
    if math.isfinite(bound):
        g = best_obj - bound
        gap = 0.0 if g <= 0 else g / max(abs(best_obj), 1e-10)
    else:
        gap = math.inf
    if gap <= 1e-9:
        st = "optimal"
    else:
        st = "time_limit" if timed_out else "heuristic"
    return {"status": st, "has_solution": True, "x": _pair_values(m, n, best, pi, pj), "objective": best_obj,
            "bound": bound, "gap": gap, "nodes": 0, "seconds": time.perf_counter() - t0,
            "runs": sol.shape[0]}


def cut_applies(m: OptimizationProblem, n: int | None) -> dict | None:
    """The model in Marxan's form if the cut engine solves it (a minimum set with pair terms, all of them
    boundaries >= 0: boundary penalties, symmetric connectivity rewards); else None."""
    if n is None or n <= 0 or m.ncol <= n or m.modelsense != "min":
        return None
    try:
        mm = marxan_model(m, n)
    except ValueError:
        return None
    if (mm["bound"] < 0).any():
        return None
    return mm


def closure_model(m: OptimizationProblem, n: int) -> dict | None:
    """The model as the closure engine's problem (opt/closure.hpp), or None: binary units, pair columns
    of boundary or connectivity terms that are penalties (each q x_i x_j with q <= 0: w = -q/2 on both
    directions, -w on both units' costs), and rows on the units alone with coefficients >= 0, '>='
    (targets) or '<=' (budgets). For min penalties and other budgeted minimisations with pair terms."""
    if n is None or n <= 0 or m.modelsense != "min":
        return None
    vt = np.asarray(m.vtype)
    if not (vt[:n] == "B").all():
        return None
    a = m.A.tocsr()
    a.sort_indices()
    nnz = np.diff(a.indptr)
    row = np.repeat(np.arange(a.shape[0]), nnz)
    col, val = a.indices, a.data
    sense = np.asarray(m.sense)
    pair_row = np.zeros(a.shape[0], dtype=bool)
    pair_row[row[col >= n]] = True
    pi = pj = np.zeros(0, dtype=np.int64)
    q = np.zeros(0)
    if m.ncol > n:
        rr = np.flatnonzero(pair_row)
        le = (sense[rr] == "<=") & (nnz[rr] == 2) & (m.rhs[rr] == 0)
        ge = (sense[rr] == ">=") & (nnz[rr] == 3) & (m.rhs[rr] == -1)
        if not (le | ge).all():
            return None
        on_pair = pair_row[row]
        if not np.where(col >= n, val == 1.0, val == -1.0)[on_pair].all():
            return None
        first = a.indptr[rr[le]]
        unit, pcol = col[first], col[first + 1]
        if (unit >= n).any() or (pcol < n).any():
            return None
        order = np.lexsort((unit, pcol))
        unit, pcol = unit[order], pcol[order]
        cols, counts = np.unique(pcol, return_counts=True)
        if (counts != 2).any() or cols.size != m.ncol - n or (unit[0::2] == unit[1::2]).any():
            return None
        pi, pj = unit[0::2], unit[1::2]
        q = m.obj[cols]
        if (q > 0).any() or (m.lb[cols] > 0).any() or (m.ub[cols] < 1).any():
            return None
    side = np.flatnonzero(~pair_row)
    s_sense = sense[side]
    if not np.isin(s_sense, (">=", "<=")).all():
        return None
    f = a[side][:, :n].tocsr()
    if f.nnz and (f.data < 0).any():
        return None
    rhs = m.rhs[side]
    keep = (s_sense == "<=") | (rhs > 0)  # '>=' rows with rhs <= 0 are always met
    f = f[np.flatnonzero(keep)]
    f.eliminate_zeros()
    by_unit = f.T.tocsr()
    by_unit.sort_indices()
    w = -q / 2
    cost = np.asarray(m.obj[:n], dtype=np.float64).copy()
    np.add.at(cost, pi, -w)
    np.add.at(cost, pj, -w)
    fixed = np.full(n, -1, dtype=np.int8)
    fixed[m.lb[:n] >= 0.5] = 1
    fixed[m.ub[:n] <= 0.5] = 0
    return {"cost": cost, "fixed": fixed, "pen_i": np.concatenate([pi, pj]).astype(np.int32),
            "pen_j": np.concatenate([pj, pi]).astype(np.int32), "pen_w": np.concatenate([w, w]),
            "rhs": np.asarray(rhs[keep], dtype=np.float64), "geq": (s_sense[keep] == ">=").astype(np.uint8),
            "col_start": by_unit.indptr.astype(np.int64), "col_row": by_unit.indices.astype(np.int32),
            "col_val": by_unit.data.astype(np.float64), "pi": pi, "pj": pj}


def solve_closure(m: OptimizationProblem, n: int, cm: dict, gap: float, time_limit: float, threads: int,
                  verbose: bool, start: np.ndarray | None = None) -> dict:
    """The closure engine on a model in its form (see :func:`closure_model`)."""
    from .. import _core

    empty = np.zeros(0, dtype=np.int32)
    out = _core.opt_solve_closure(
        cm["cost"], cm["fixed"], empty, empty, cm["pen_i"], cm["pen_j"], cm["pen_w"], cm["rhs"], cm["geq"],
        cm["col_start"], cm["col_row"], cm["col_val"], float(gap), float(time_limit), int(threads), bool(verbose),
        None if start is None else np.asarray(start, dtype=np.float64)[:n])
    if out["status"] in ("infeasible", "no_solution"):
        return {"status": out["status"].replace("no_solution", "time_limit"), "has_solution": False, "x": None,
                "objective": math.nan, "bound": float(out["bound"]), "gap": math.nan, "nodes": 0,
                "seconds": out["seconds"]}
    full = _pair_values(m, n, np.asarray(out["z"], dtype=np.float64), cm["pi"], cm["pj"])
    obj = float(m.obj @ full)
    bound = float(out["bound"])
    g = obj - bound
    gp = 0.0 if g <= 0 else g / max(abs(obj), 1e-10)
    return {"status": "optimal" if gp <= gap else out["status"], "has_solution": True, "x": full, "objective": obj,
            "bound": bound, "gap": gp, "nodes": 0, "seconds": out["seconds"], "evaluations": out["evaluations"]}


def solve_cut(m: OptimizationProblem, n: int, mm: dict, gap: float, time_limit: float, verbose: bool,
              start: np.ndarray | None = None) -> dict:
    """The cut engine (opt/cut.cpp) on a model in Marxan's form."""
    from .. import _core

    nb_start, nb, nb_cost = _neighbours(n, mm["pi"], mm["pj"], mm["bound"])
    init = np.zeros(0, dtype=np.int32) if start is None else (np.asarray(start)[:n] >= 0.5).astype(np.int32)
    out = _core.opt_solve_cut(
        mm["cost"], mm["status"], mm["offset"], mm["spindex"], mm["amount"], mm["target"], np.zeros(n), nb_start,
        nb, nb_cost, float(gap), -1.0 if not math.isfinite(time_limit) else float(time_limit), 1000, bool(verbose),
        init)
    if out["status"] == "infeasible":
        return {"status": "infeasible", "has_solution": False, "x": None, "objective": math.nan, "bound": math.nan,
                "gap": math.nan, "nodes": 0, "seconds": out["seconds"]}
    x = np.asarray(out["x"], dtype=np.float64)
    full = _pair_values(m, n, x, mm["pi"], mm["pj"])
    obj = float(m.obj @ full)
    bound = float(out["bound"])
    g = obj - bound
    gp = 0.0 if g <= 0 else g / max(abs(obj), 1e-10)
    return {"status": "optimal" if gp <= gap else out["status"], "has_solution": True, "x": full, "objective": obj,
            "bound": bound, "gap": gp, "nodes": 0, "seconds": out["seconds"], "evaluations": out["evaluations"]}
