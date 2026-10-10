# SPDX-License-Identifier: GPL-3.0-or-later
"""Portfolios of solutions (prioritizr's add_*_portfolio) on any solver.

* cuts and top: the k best distinct solutions, one solve after another, each
  forbidding the earlier ones with a no-good cut on the planning units
  (sum over the selected of x - sum over the others of x <= selected - 1).
  prioritizr 8.0.6's cut has the coefficients 2 s instead of 2 s - 1 (its
  comment says "convert 1s to 1 and 0s to -1"): it forbids keeping half of
  the earlier solution, not the solution, so the next ones are far from
  optimal and the portfolio soon runs out (only the correct cut here);
* gap: solutions within ``pool_gap`` of the optimum (the cuts and a row
  bounding the objective), as many as found up to ``number_solutions``;
* shuffle: the model with its columns in random orders (seeded): optimal
  solutions that differ when there are ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._save import Saveable

__all__ = ["Portfolio", "solve_portfolio"]


@dataclass
class Portfolio(Saveable):
    """Several solutions of a problem (:meth:`Problem.add_cuts_portfolio` and the others).

    Attributes
    ----------
    solutions
        The solutions, best first (cuts, top, gap) or in the order found (shuffle).
    selection
        Every solution's selection: a raster with one band per solution, or the planning units' table
        with columns ``solution_1``, ``solution_2``...
    frequency
        The share of the solutions that select every planning unit (a raster or a table column, as
        ``selection``): a simple irreplaceability.
    summary
        Objective, cost, status and gap of every solution.
    """

    solutions: list = field(repr=False)
    selection: object = field(repr=False)
    frequency: object = field(repr=False)
    summary: pd.DataFrame = field(repr=False)

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, k):
        return self.solutions[k]

    def __iter__(self):
        return iter(self.solutions)

    def __repr__(self) -> str:
        obj = ", ".join(f"{s.objective:.6g}" for s in self.solutions)
        return f"Portfolio({len(self)} solutions; objectives {obj})"

    def _save_items(self) -> dict:
        return {"selection": self.selection, "frequency": self.frequency, "summary": self.summary}


def _with_rows(m, a_rows: sp.csr_matrix, sense: list[str], rhs: list[float], ids: list[str]):
    from ._milp import OptimizationProblem

    a = sp.vstack([m.A, a_rows]).tocsr()
    a.sort_indices()
    return OptimizationProblem(A=a, obj=m.obj, lb=m.lb, ub=m.ub, rhs=np.concatenate([m.rhs, rhs]),
                               sense=(*m.sense, *sense), vtype=m.vtype, modelsense=m.modelsense,
                               row_ids=(*m.row_ids, *ids), col_ids=m.col_ids, connected=m.connected, corridors=m.corridors)


def _cut(values: np.ndarray, n: int, ncol: int) -> tuple[sp.csr_matrix, float]:
    """The no-good cut of a 0/1 selection of the n planning units."""
    s = np.round(values[:n]).astype(np.float64)
    row = sp.csr_matrix((2.0 * s - 1.0, (np.zeros(n, dtype=np.int64), np.arange(n))), shape=(1, ncol))
    return row, float(s.sum() - 1.0)


def _permuted(m, perm: np.ndarray):
    """The model with its columns in the order perm (column k of the new model is perm[k])."""
    from ._milp import OptimizationProblem

    a = m.A[:, perm].tocsr()
    a.sort_indices()
    return OptimizationProblem(A=a, obj=m.obj[perm], lb=m.lb[perm], ub=m.ub[perm], rhs=m.rhs, sense=m.sense,
                               vtype=tuple(np.asarray(m.vtype)[perm]), modelsense=m.modelsense, row_ids=m.row_ids,
                               col_ids=tuple(np.asarray(m.col_ids)[perm]))


def _zone_portfolio_maps(u, sols, names, values):
    """With zones: every solution's zone map (a layer per solution, or a column of zone names per solution)
    and how often every unit goes to every zone (a layer per zone, or a column ``frequency_<zone>`` per zone)."""
    freq = values.mean(axis=0)  # units x zones
    if u.kind == "raster":
        import xarray as xr

        layers = [s.zones.rename(None) for s in sols]
        selection = xr.concat(layers, dim="band").assign_coords(band=names)
        selection.attrs = dict(sols[0].zones.attrs)
        ny, nx = u.template.shape
        out = np.full((u.nz, ny * nx), np.nan)
        out[:, u.index] = freq.T
        frequency = sols[0].selection.copy(data=out.reshape(u.nz, ny, nx))
        frequency.name = "frequency"
        return selection, frequency
    selection = u.template.copy()
    for nm, s in zip(names, sols):
        selection[nm] = s.zones.to_numpy()
    frequency = u.template.copy()
    for k, z in enumerate(u.zone_names):
        col = np.full(len(frequency), np.nan)
        col[u.index] = freq[:, k]
        frequency[f"frequency_{z}"] = col
    return selection, frequency


def solve_portfolio(problem, m, targets) -> Portfolio:
    from ._milp_solve import make_solution, solve_model

    kind, opts = problem.portfolio
    u = problem.units
    zoned = u.zoned
    n = u.cost.size  # the units' columns (with zones: of every unit and zone)
    zkw = {"n_units": None, "engines": False, "zone_units": u.n} if zoned else {"n_units": n}
    integer = np.isin(np.asarray(m.vtype[:n]), ("B", "I"))
    k_max = int(opts["number_solutions"])
    solver, sopts = problem.solver
    raw = []
    if kind in ("cuts", "top", "gap"):
        if not integer.all() or not (m.ub[:n] <= 1).all():
            raise ValueError(f"the {kind} portfolio needs binary decisions")
        if kind in ("top", "gap"):
            sopts = {**sopts, "gap": 0.0} if kind == "top" else sopts
        cur = m
        best = None
        for _ in range(k_max):
            r = solve_model(cur, solver, sopts, **zkw)
            if not r["has_solution"] or r["status"] == "infeasible":
                break
            raw.append(r)
            if kind == "gap":
                if best is None:
                    best = float(r["objective"])
                    g = float(opts["pool_gap"])
                    # Gurobi's PoolGap: within pool_gap of the optimum, relative to it
                    bound = best + g * abs(best) if m.modelsense == "min" else best - g * abs(best)
                    cur = _with_rows(cur, sp.csr_matrix(m.obj.reshape(1, -1)),
                                     ["<=" if m.modelsense == "min" else ">="], [bound], ["pool_gap"])
            row, rhs = _cut(np.asarray(r["x"], dtype=np.float64), n, m.ncol)
            cur = _with_rows(cur, row, ["<="], [rhs], ["cuts"])
    elif kind == "shuffle":
        rng = np.random.default_rng(opts["seed"])
        seen = set()
        tries = 0
        while len(raw) < k_max and tries < 4 * k_max:
            tries += 1
            # the units in a random order, then the other columns in a random order (the engines want the
            # units first)
            if zoned:  # the same order of the units in every zone (the engines want zone after zone)
                pu = rng.permutation(u.n)
                units = (np.arange(u.nz)[:, None] * u.n + pu[None, :]).ravel()
            else:
                units = rng.permutation(n)
            perm = np.concatenate([units, n + rng.permutation(m.ncol - n)])
            pm = _permuted(m, perm)
            r = solve_model(pm, solver, sopts, **zkw)
            if not r["has_solution"]:
                break
            x = np.empty(m.ncol)
            x[perm] = np.asarray(r["x"], dtype=np.float64)
            r = {**r, "x": x}
            key = tuple(np.round(x[:n], 6))
            if opts["remove_duplicates"] and key in seen:
                continue
            seen.add(key)
            raw.append(r)
    else:
        raise ValueError(f"unknown portfolio {kind!r}")
    if not raw:
        raise RuntimeError(f"no solution ({solver})")
    sols = [make_solution(u, m, r, targets) for r in raw]
    if kind in ("cuts", "top", "gap") and len(sols) < k_max:
        import warnings

        warnings.warn(f"the {kind} portfolio found {len(sols)} of {k_max} solutions", UserWarning, stacklevel=3)
    values = np.stack([s.values for s in sols])
    names = [f"solution_{k + 1}" for k in range(len(sols))]
    if zoned:
        selection, frequency = _zone_portfolio_maps(u, sols, names, values)
    elif u.kind == "raster":
        import xarray as xr

        layers = [s.selection.rename(None) for s in sols]
        selection = xr.concat(layers, dim="band").assign_coords(band=names)
        frequency = sols[0].selection.copy(data=np.where(np.isnan(sols[0].selection.values), np.nan,
                                                        selection.mean("band").values))
        frequency.name = "frequency"
    else:
        selection = u.template.copy()
        for nm, s in zip(names, sols):
            col = np.full(len(selection), np.nan)
            col[u.index] = s.values
            selection[nm] = col
        frequency = u.template.copy()
        col = np.full(len(frequency), np.nan)
        col[u.index] = values.mean(axis=0)
        frequency["frequency"] = col
    summary = pd.DataFrame({"solution": names, "objective": [s.objective for s in sols],
                            "cost": [s.cost for s in sols], "status": [s.status for s in sols],
                            "gap": [s.gap for s in sols]})
    return Portfolio(sols, selection, frequency, summary)


def check_portfolio(kind: str, number_solutions, **kw) -> tuple:
    k = int(number_solutions)
    if k != number_solutions or k < 1:
        raise ValueError("number_solutions must be a whole number >= 1")
    if kind == "gap" and not (kw["pool_gap"] >= 0 and math.isfinite(kw["pool_gap"])):
        raise ValueError("pool_gap must be >= 0")
    return kind, {"number_solutions": k, **kw}

