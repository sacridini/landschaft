# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibrating trade-offs (prioritizr's "Calibrating trade-offs" tutorial, Hanson et al.): candidate
solutions over a trade-off parameter by the blended approach (a penalty's weight) or the hierarchical
one (the least penalty within cost thresholds), the penalty of Cohon et al. (1979) from the two ideal
solutions, the selection of a candidate by TOPSIS (Hwang & Yoon 1981) or the knee of the trade-off
curve, the Pareto front of cost and shortfall, and solving again and again as the landscape changes
(coupled with ``ls.simulate``). A step that fails raises: nothing is swallowed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._save import Saveable

__all__ = ["Calibration", "calibrate", "calibrate_budget", "calibrate_penalty", "cohon_penalty", "iterative_solve",
           "knee_point", "pareto_front", "protected_cells", "topsis"]

PENALTIES = ("boundary", "connectivity", "asym_connectivity", "linear")


@dataclass
class Calibration(Saveable):
    """Candidate solutions over a trade-off parameter.

    Attributes
    ----------
    table
        One row per candidate: ``value`` (the penalty, the cost threshold or the budget), ``cost``, the
        criterion traded off against it (``boundary``, ``connectivity``, ``shortfall``...), the
        objective, status and gap; ``topsis`` and ``knee`` mark the candidates those methods choose.
    solutions
        The candidates' :class:`Solution`\\ s, in the table's order.
    """

    table: pd.DataFrame = field(repr=False)
    solutions: list = field(repr=False)
    criterion: str = ""

    def __repr__(self) -> str:
        return f"Calibration({len(self.solutions)} candidates; cost vs {self.criterion})"

    def best(self, method: str = "topsis"):
        """The candidate the method chooses (``"topsis"`` with equal weights, or ``"knee"``)."""
        if method not in ("topsis", "knee"):
            raise ValueError("method must be 'topsis' or 'knee'")
        return self.solutions[int(np.flatnonzero(self.table[method].to_numpy())[0])]

    def _save_items(self) -> dict:
        return {"table": self.table}


# ----------------------------------------------------------------------------- choosing a candidate

def topsis(decision, weights=None, impacts=None) -> pd.DataFrame:
    """TOPSIS (as R's ``topsis`` package): the alternatives (rows of ``decision``) scored by their
    closeness to the ideal (``impacts``: "+" larger is better, "-" smaller is better, per column; equal
    ``weights`` by default). Returns ``score`` (0-1, larger is better) and ``rank`` (1 = best)."""
    d = np.asarray(decision, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 1:
        raise ValueError("decision must be a matrix: alternatives x criteria")
    nc = d.shape[1]
    w = np.ones(nc) if weights is None else np.asarray(weights, dtype=np.float64)
    imp = ["-"] * nc if impacts is None else list(impacts)
    if w.size != nc or len(imp) != nc or any(i not in ("+", "-") for i in imp):
        raise ValueError("one weight and one impact ('+' or '-') per criterion")
    w = w / w.sum()
    norm = np.sqrt((d ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d / norm * w
    plus = np.array([i == "+" for i in imp])
    best = np.where(plus, v.max(axis=0), v.min(axis=0))
    worst = np.where(plus, v.min(axis=0), v.max(axis=0))
    du = np.sqrt(((v - best) ** 2).sum(axis=1))
    dl = np.sqrt(((v - worst) ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = dl / (dl + du)
    rank = pd.Series(-score).rank(method="average").to_numpy()
    return pd.DataFrame({"alt.row": np.arange(1, d.shape[0] + 1), "score": score, "rank": rank})


def knee_point(x, y) -> int:
    """The knee (elbow) of a trade-off curve: the point farthest from the straight line between its
    ends, both axes scaled to 0-1 (the points in the order of x). Returns its position."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 1:
        raise ValueError("x and y need the same, non-zero length")
    if x.size < 3:
        return 0
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]

    def scale(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    xs, ys = scale(xs), scale(ys)
    p0, p1 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    line = p1 - p0
    length = np.hypot(*line)
    if length == 0:
        return int(order[0])
    dist = np.abs(line[0] * (ys - p0[1]) - line[1] * (xs - p0[0])) / length
    return int(order[int(np.argmax(dist))])


# ----------------------------------------------------------------------------- the criteria

def _with_penalty(p, penalty: str, value: float, data=None, edge_factor: float = 0.5):
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}")
    if penalty == "boundary":
        return p.add_boundary_penalties(value, edge_factor=edge_factor, data=data)
    if data is None:
        raise ValueError(f"{penalty} penalties need data")
    return getattr(p, f"add_{penalty}_penalties")(value, data)


def _criterion(p, sol, penalty: str, data=None, edge_factor: float = 0.5) -> float:
    from ._milp_eval import (_values, eval_asym_connectivity_summary, eval_boundary_summary,
                             eval_connectivity_summary)

    if penalty == "boundary":
        return float(eval_boundary_summary(p, sol, edge_factor=edge_factor, data=data)["boundary"].iloc[0])
    if penalty == "connectivity":
        return float(eval_connectivity_summary(p, sol, data)["connectivity"].iloc[0])
    if penalty == "asym_connectivity":
        return float(eval_asym_connectivity_summary(p, sol, data)["asym_connectivity"].iloc[0])
    s = _values(p, sol)
    v = _linear_values(p, data)
    return float(np.dot(v, s))


def _linear_values(p, data) -> np.ndarray:
    u = p.units
    if isinstance(data, str):
        return np.asarray(u.template[data], dtype=np.float64)[u.index]
    arr = data.values.ravel()[u.index] if hasattr(data, "values") and getattr(data, "ndim", 1) >= 2 else data
    return np.nan_to_num(np.asarray(arr, dtype=np.float64))


def _solve_all(problems: list, n_threads: int | None):
    """Solve the problems, several at once (one solver thread each); a failure raises."""
    from concurrent.futures import ThreadPoolExecutor

    from .. import _threads

    threads = _threads.resolve(n_threads)

    def one(q):
        solver, opts = q.solver
        q = q._with(solver=(solver, {**opts, "n_threads": 1}))
        return q.solve()

    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(problems)))) as ex:
        return list(ex.map(one, problems))


def _table(values, sols, criterion: str, crit_values) -> pd.DataFrame:
    t = pd.DataFrame({"value": np.asarray(values, dtype=np.float64), "cost": [s.cost for s in sols],
                      criterion: np.asarray(crit_values, dtype=np.float64),
                      "objective": [s.objective for s in sols], "status": [s.status for s in sols],
                      "gap": [s.gap for s in sols]})
    choice = topsis(t[["cost", criterion]].to_numpy(), impacts=["-", "-"])
    t["topsis_score"] = choice["score"].to_numpy()
    t["topsis"] = False
    t.loc[int(np.nanargmax(t["topsis_score"].to_numpy())), "topsis"] = True
    t["knee"] = False
    t.loc[knee_point(t[criterion].to_numpy(), t["cost"].to_numpy()), "knee"] = True
    return t


def _r_round(v: float, digits: int) -> float:
    from ._milp_spatial import r_round

    return float(r_round(np.array([v]), digits)[0])


# ----------------------------------------------------------------------------- the approaches

def cohon_penalty(p, penalty: str = "boundary", data=None, edge_factor: float = 0.5, gap: float = 0.0):
    """The penalty of Cohon et al. (1979): the trade-off rate between the solution with the least cost
    (``p`` as given, a min set problem without the penalty) and the one with the least penalty
    (min penalties within a budget above every unit's cost): |cost difference / criterion difference|,
    rounded to 5 decimals. Returns (penalty, a table of the two ideal solutions' cost and criterion)."""
    if p.objective is None or p.objective[0] != "min_set":
        raise ValueError("cohon_penalty needs a min set problem (without the penalty)")
    solver, opts = p.solver
    ideal_cost = p._with(solver=(solver, {**opts, "gap": gap}))
    budget = float(np.sum(p.units.cost)) + 1.0
    ideal_pen = _with_penalty(p, penalty, 1.0, data, edge_factor).add_min_penalties_objective(budget)
    ideal_pen = ideal_pen._with(solver=(solver, {**opts, "gap": gap}))
    s3, s4 = ideal_cost.solve(), ideal_pen.solve()
    m3 = (s3.cost, _criterion(p, s3, penalty, data, edge_factor))
    m4 = (s4.cost, _criterion(p, s4, penalty, data, edge_factor))
    if m3[1] == m4[1]:
        raise ValueError(f"the two ideal solutions have the same {penalty}: no trade-off to calibrate")
    value = _r_round(abs((m3[0] - m4[0]) / (m3[1] - m4[1])), 5)
    table = pd.DataFrame({"solution": ["least cost", f"least {penalty}"], "cost": [m3[0], m4[0]],
                          penalty: [m3[1], m4[1]]})
    return value, table


def calibrate_penalty(p, penalty: str = "boundary", values=None, n: int = 9, data=None, edge_factor: float = 0.5,
                      n_threads: int | None = None) -> Calibration:
    """The blended approach: the problem solved with a range of penalty weights (``values``; default
    ``n`` values evenly spaced in log scale over 1/100 to 100 times the Cohon et al. penalty), every
    candidate's cost and criterion (boundary length, connectivity, ...), and the candidates TOPSIS and the
    knee of the curve choose. ``p``: the problem without the penalty. Solves in parallel."""
    if values is None:
        c, _ = cohon_penalty(p, penalty, data, edge_factor)
        if c <= 0:
            raise ValueError("the Cohon penalty is 0: give the values")
        values = c * 10.0 ** np.linspace(-2, 2, int(n))
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 1 or (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("values must be finite penalties >= 0")
    problems = [_with_penalty(p, penalty, float(v), data, edge_factor) for v in values]
    sols = _solve_all(problems, n_threads)
    crit = [_criterion(p, s, penalty, data, edge_factor) for s in sols]
    return Calibration(_table(values, sols, penalty, crit), sols, penalty)


def calibrate_budget(p, penalty: str = "boundary", thresholds=None, inflation=None, data=None,
                     edge_factor: float = 0.5, n_threads: int | None = None) -> Calibration:
    """The hierarchical approach: the least cost solution first (``p``, a min set problem without the
    penalty, solved to optimality), then the least penalty within cost thresholds above its cost
    (``thresholds``, or the cost x (1 + ``inflation``); default inflation 1e-5 to 4 in 9 steps, the
    thresholds rounded up as in prioritizr's tutorial). Solves in parallel."""
    if p.objective is None or p.objective[0] != "min_set":
        raise ValueError("calibrate_budget needs a min set problem (without the penalty)")
    if thresholds is None:
        solver, opts = p.solver
        base = p._with(solver=(solver, {**opts, "gap": 0.0})).solve()
        inf = np.linspace(1e-5, 4, 9) if inflation is None else np.asarray(inflation, dtype=np.float64)
        thresholds = np.ceil(base.cost + base.cost * inf)
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    if (thresholds <= 0).any() or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be finite and > 0")
    pen = _with_penalty(p, penalty, 1.0, data, edge_factor)
    problems = [pen.add_min_penalties_objective(float(t)) for t in thresholds]
    sols = _solve_all(problems, n_threads)
    crit = [_criterion(p, s, penalty, data, edge_factor) for s in sols]
    return Calibration(_table(thresholds, sols, penalty, crit), sols, penalty)


def pareto_front(p, budgets=None, n: int = 10, n_threads: int | None = None) -> Calibration:
    """The trade-off between cost and representation: the least total shortfall of the targets (min
    shortfall objective: the sum of the shortfalls as shares of their targets) within budgets from a tenth of the least-cost solution's cost up to it
    (``n`` steps) or the ``budgets`` given. ``p``: a problem with targets (its objective is replaced).
    The table adds ``met`` (targets met) and ``efficient`` (no other candidate is cheaper and not
    worse, or better and not dearer)."""
    from ._milp_eval import eval_target_coverage_summary

    if p.targets is None and p.subregional is None:
        raise ValueError("pareto_front needs targets")
    if budgets is None:
        top = p.add_min_set_objective().solve().cost
        budgets = np.linspace(top / n, top, int(n))
    budgets = np.asarray(budgets, dtype=np.float64).ravel()
    problems = [p.add_min_shortfall_objective(float(b)) for b in budgets]
    sols = _solve_all(problems, n_threads)
    short, met = [], []
    for q, s in zip(problems, sols):
        cov = eval_target_coverage_summary(q, s)
        with np.errstate(divide="ignore", invalid="ignore"):  # min shortfall's: each as a share of its target
            rel = np.where(cov["absolute_target"] > 0, cov["absolute_shortfall"] / cov["absolute_target"], 0.0)
        short.append(float(np.sum(rel)))
        met.append(int(cov["met"].sum()))
    t = _table(budgets, sols, "shortfall", short)
    t.insert(3, "met", met)
    c, sh = t["cost"].to_numpy(), t["shortfall"].to_numpy()
    t["efficient"] = [not any((c[j] <= c[i] and sh[j] <= sh[i]) and (c[j] < c[i] or sh[j] < sh[i])
                              for j in range(len(t))) for i in range(len(t))]
    return Calibration(t, sols, "shortfall")


def calibrate(make, values, criterion, name: str = "criterion", n_threads: int | None = None) -> Calibration:
    """Any trade-off parameter: ``make(value)`` gives the problem for every value (a feature's weight,
    a target, a penalty of any kind...), ``criterion(problem, solution)`` the quantity traded off
    against the cost (smaller is better). The candidates' table as :func:`calibrate_penalty`'s (TOPSIS,
    knee), solved in parallel."""
    values = list(values)
    if not values:
        raise ValueError("no values")
    problems = [make(v) for v in values]
    sols = _solve_all(problems, n_threads)
    crit = [float(criterion(q, s)) for q, s in zip(problems, sols)]
    vals = np.asarray(values, dtype=np.float64) if all(np.isscalar(v) for v in values) else np.arange(len(values))
    return Calibration(_table(vals, sols, name, crit), sols, name)


def iterative_solve(p, update, steps: int, lock_previous: bool = False):
    """Solve, change the problem, solve again: ``update(step, solution, problem)`` gives the next
    problem (for instance from the landscape a ``ls.simulate`` step leaves, with the solution's cells
    protected by :func:`protected_cells` in the simulation's rules), or ``None`` to stop. With
    ``lock_previous``, every solution's units stay selected in the next problem (protected areas that
    accumulate). Returns the solutions and a table (step, cost, units, status)."""
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a whole number >= 1")
    sols, rows = [], []
    q = p
    for k in range(int(steps)):
        s = q.solve()
        sols.append(s)
        rows.append({"step": k, "cost": s.cost, "units": float(np.sum(s.values)), "objective": s.objective,
                     "status": s.status})
        if k == steps - 1:
            break
        nxt = update(k, s, q)
        if nxt is None:
            break
        if lock_previous:
            if nxt.units.n != s.values.size:
                raise ValueError("lock_previous needs the same planning units at every step")
            nxt = nxt.add_locked_in_constraints(nxt._mask(nxt.locked_in) | (s.values > 0.5))
        q = nxt
    return sols, pd.DataFrame(rows)


def protected_cells(solution):
    """The selected cells of a raster solution as a boolean raster (for ``ls.simulate``'s restrictions)."""
    sel = solution.selection
    if not hasattr(sel, "dims") or sel.ndim < 2:
        raise ValueError("protected_cells needs a solution on raster planning units")
    out = sel.fillna(0) > 0.5
    out.name = "protected"
    return out

