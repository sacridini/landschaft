# SPDX-License-Identifier: GPL-3.0-or-later
"""Solving compiled models: HiGHS (built in) and Gurobi (gurobipy), and the solution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._save import Saveable
from ._milp import OptimizationProblem, Units

__all__ = ["Solution", "solve_model"]


def _is_covering(m: OptimizationProblem) -> bool:
    """A minimisation with only >= rows and finite bounds (prioritizr's minimum set and its kin)."""
    return (m.modelsense == "min" and all(s == ">=" for s in m.sense) and bool(np.isfinite(m.lb).all())
            and bool(np.isfinite(m.ub).all()))


def _method(m: OptimizationProblem, n_units: int | None) -> str:
    if _is_covering(m):
        return "covering"
    if n_units is not None and n_units > 0 and m.ncol - n_units <= 20000 and bool(
            np.isfinite(m.lb[:n_units]).all() and np.isfinite(m.ub[:n_units]).all()):
        return "wide"
    return "highs"


def _highs(m: OptimizationProblem, gap: float, time_limit: float, threads: int, presolve: str, verbose: bool,
           start: np.ndarray | None, relaxation: bool = False, n_units: int | None = None) -> dict:
    """HiGHS, behind landschaft's own engines: covering models (min set) and wide models with a budget
    (min shortfall, max utility, ...) get a sifted LP, rounding and a greedy step, and HiGHS's MIP only
    if the gap is still too wide and the model narrow enough for it to keep the time limit."""
    from .. import _core

    lo, hi = m.row_bounds()
    a = m.A
    return _core.opt_solve_highs(
        start=a.indptr.astype(np.int64), index=a.indices.astype(np.int64), value=a.data, obj=m.obj, lb=m.lb,
        ub=m.ub, row_lo=lo, row_hi=hi, integer=np.isin(np.asarray(m.vtype), ("B", "I")).astype(np.uint8),
        maximize=m.modelsense == "max", offset=0.0, gap=gap, time_limit=time_limit, threads=threads,
        presolve=presolve, verbose=verbose, relaxation=relaxation, initial=start,
        method=_method(m, n_units), n_units=-1 if n_units is None else int(n_units))


def _gurobi(m: OptimizationProblem, gap: float, time_limit: float, threads: int, verbose: bool,
            start: np.ndarray | None) -> dict:
    try:
        import gurobipy as gp
    except ImportError as e:
        raise ImportError("the Gurobi solver needs the gurobipy package (and a Gurobi licence)") from e
    vtype = {"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER}
    with gp.Env(params={"OutputFlag": int(verbose)}) as env, gp.Model(env=env) as g:
        x = g.addMVar(m.ncol, lb=m.lb, ub=np.where(np.isinf(m.ub), gp.GRB.INFINITY, m.ub),
                      vtype=[vtype.get(t, gp.GRB.CONTINUOUS) for t in m.vtype])
        g.addMConstr(m.A, x, np.asarray([s[0] for s in m.sense]), m.rhs)
        g.setObjective(m.obj @ x, gp.GRB.MAXIMIZE if m.modelsense == "max" else gp.GRB.MINIMIZE)
        g.Params.MIPGap = gap
        g.Params.Threads = threads
        if math.isfinite(time_limit):
            g.Params.TimeLimit = time_limit
        if start is not None:
            x.Start = start
        g.optimize()
        status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit", gp.GRB.INFEASIBLE: "infeasible",
                  gp.GRB.UNBOUNDED: "unbounded", gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded"}.get(
            g.Status, f"error: Gurobi status {g.Status}")
        ok = g.SolCount > 0
        mip = g.IsMIP == 1
        return {"status": status, "has_solution": ok, "x": x.X.copy() if ok else None,
                "objective": g.ObjVal if ok else math.nan,
                "bound": g.ObjBound if mip else (g.ObjVal if ok else math.nan),
                "gap": (g.MIPGap if mip else 0.0) if ok else math.nan,
                "nodes": int(g.NodeCount) if mip else 0, "seconds": g.Runtime}


def solve_model(m: OptimizationProblem, solver: str, options: dict, start: np.ndarray | None = None,
                n_units: int | None = None) -> dict:
    """Solve a compiled model on a solver; the result's status, solution and bounds."""
    from .. import _threads

    gap = float(options.get("gap", 0.1))
    tl = options.get("time_limit")
    time_limit = math.inf if tl is None else float(tl)
    threads = _threads.resolve(options.get("n_threads"))
    verbose = bool(options.get("verbose", False))
    if solver == "highs":
        r = _highs(m, gap, time_limit, threads, options.get("presolve", "choose"), verbose, start, n_units=n_units)
    elif solver == "gurobi":
        r = _gurobi(m, gap, time_limit, threads, verbose, start)
    elif solver == "anneal":
        from ._milp_anneal import solve_anneal

        if n_units is None:
            raise ValueError("the anneal solver needs the number of planning units")
        r = solve_anneal(m, n_units, options, start)
    else:
        raise ValueError(f"unknown solver {solver!r}")
    r["solver"] = solver
    return r


# ----------------------------------------------------------------------------- solution

@dataclass
class Solution(Saveable):
    """A solution of a :class:`~landschaft.prioritize.Problem`.

    Attributes
    ----------
    selection
        The selected share of every planning unit (1 / 0 with binary
        decisions): a raster on the planning units' grid (NaN outside the
        planning units) or a copy of their table with a ``solution`` column.
    values
        The same, one value per planning unit (in their order).
    coverage
        Every target's feature, sense, the feature's total amount, the target,
        the amount held and the shortfall, absolute and relative, and whether
        it is met (prioritizr's ``eval_target_coverage_summary()``); without
        targets (max utility, max cover), every feature's amount held.
    objective, bound, gap
        The objective of the solution, the solver's bound on the optimum and
        their relative gap.
    status
        ``"optimal"`` (within the gap asked for), ``"time_limit"`` (the gap
        asked for not reached within the time limit: with a limit, models too
        wide for HiGHS's MIP to keep it stop at the engine's own solution,
        before the limit), ``"infeasible"``, ...
    solver, seconds
        The solver used and the time it took.
    """

    selection: object = field(repr=False)
    values: np.ndarray = field(repr=False)
    coverage: pd.DataFrame = field(repr=False)
    objective: float = math.nan
    bound: float = math.nan
    gap: float = math.nan
    status: str = ""
    solver: str = ""
    seconds: float = 0.0

    @property
    def cost(self) -> float:
        """Total cost of the selection."""
        return float(self._cost @ self.values)

    _cost: np.ndarray = field(default=None, repr=False)

    def _save_items(self) -> dict:
        return {"selection": self.selection, "coverage": self.coverage, "objective": self.objective,
                "bound": self.bound, "gap": self.gap, "status": self.status, "solver": self.solver,
                "seconds": self.seconds, "cost": self.cost}


def _selection(u: Units, values: np.ndarray):
    if u.kind == "raster":
        out = np.full(u.template.size, np.nan)
        out[u.index] = values
        da = u.template.copy(data=out.reshape(u.template.shape))
        da.name = "solution"
        return da
    t = u.template.copy()
    col = np.full(len(t), np.nan)
    col[u.index] = values
    t["solution"] = col
    return t


def coverage_table(u: Units, values: np.ndarray, targets=None, m: OptimizationProblem | None = None) -> pd.DataFrame:
    """prioritizr's target coverage summary: one row per target (per feature, without targets)."""
    if targets is None or targets.sense[:1] == ("?",):  # objectives without targets
        held = u.rij @ values
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(u.totals != 0, held / u.totals, np.nan)
        return pd.DataFrame({"feature": list(u.feature_names), "total_amount": u.totals,
                             "absolute_held": held, "relative_held": rel})
    f = targets.feature
    # the target rows' amounts (their region's units only, for sub-regional targets)
    held = targets.matrix @ values if targets.matrix is not None else (u.rij @ values)[f]
    target = targets.value
    total = targets.total if targets.total is not None else u.totals[f]
    sense = np.asarray(targets.sense)
    shortfall = np.where(sense == ">=", np.maximum(target - held, 0.0),
                         np.where(sense == "<=", np.maximum(held - target, 0.0), np.abs(held - target)))
    tol = 1e-9 * np.maximum(1.0, np.abs(target))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(total != 0, held / total, np.nan)
        rel_target = np.where(total != 0, target / total, np.nan)
        rel_short = np.where(total != 0, shortfall / total, np.nan)
    out = pd.DataFrame({
        "feature": [u.feature_names[k] for k in f], "sense": sense, "total_amount": total,
        "absolute_target": target, "absolute_held": held, "absolute_shortfall": shortfall,
        "relative_target": rel_target, "relative_held": rel, "relative_shortfall": rel_short,
        "met": shortfall <= tol,
    })
    if targets.region is not None and (targets.region >= 0).any():
        out.insert(1, "region", list(targets.region_label))
    return out


def make_solution(u: Units, m: OptimizationProblem, r: dict, targets=None) -> Solution:
    if not r["has_solution"]:
        raise RuntimeError(f"no solution ({r['solver']}: {r['status']})")
    values = np.asarray(r["x"], dtype=np.float64)[: u.n]
    integer = np.isin(np.asarray(m.vtype[: u.n]), ("B", "I"))
    values = np.where(integer, np.round(values), values)
    return Solution(selection=_selection(u, values), values=values, coverage=coverage_table(u, values, targets, m),
                    objective=float(r["objective"]), bound=float(r["bound"]), gap=float(r["gap"]),
                    status=r["status"], solver=r["solver"], seconds=float(r["seconds"]), _cost=u.cost)
