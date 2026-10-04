# SPDX-License-Identifier: GPL-3.0-or-later
"""Portfolios of multi-action solutions (an extension: prioriactions has none), as the problem's
portfolios (:mod:`._milp_portfolio`): cuts, top and gap forbid every earlier solution with a no-good
cut on the units and the actions (two solutions differ when they monitor or act differently anywhere);
shuffle solves the model with its units and actions in random orders. On the equivalent model (curve 1)
or the full one, on HiGHS's MIP or Gurobi: the no-good cuts have negative coefficients, which the closure
engine's side rows cannot hold."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._save import Saveable

__all__ = ["ActionsPortfolio"]


@dataclass
class ActionsPortfolio(Saveable):
    """Several solutions of an :class:`ActionsProblem` (:meth:`ActionsProblem.add_cuts_portfolio` and the
    others).

    Attributes
    ----------
    solutions
        The :class:`ActionsSolution`\\ s, best first (cuts, top, gap) or in the order found (shuffle).
    summary
        Objective, cost (monitoring and actions), status and gap of every solution.
    frequency
        The share of the solutions that monitor every unit and take every action: ``monitoring``
        (``id``, ``frequency``) and ``actions`` (``pu``, ``threat``, ``frequency``).
    """

    solutions: list = field(repr=False)
    summary: pd.DataFrame = field(repr=False)
    frequency: dict = field(repr=False)

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, k):
        return self.solutions[k]

    def __iter__(self):
        return iter(self.solutions)

    def __repr__(self) -> str:
        obj = ", ".join(f"{s.objective:.6g}" for s in self.solutions)
        return f"ActionsPortfolio({len(self)} solutions; objectives {obj})"

    def maps(self):
        """The frequencies as rasters (band ``monitoring``, then the threats), for problems from
        :func:`actions_raster_data`."""
        from ._actions_raster import solution_maps

        s0 = self.solutions[0]
        return solution_maps(s0.data, self.frequency["monitoring"]["frequency"].to_numpy(),
                             self.frequency["actions"]["frequency"].to_numpy())

    def _save_items(self) -> dict:
        maps = {"frequency_maps": self.maps()} if self.solutions[0].data.grid is not None else {}
        return {**maps, "summary": self.summary, "monitoring_frequency": self.frequency["monitoring"],
                "actions_frequency": self.frequency["actions"]}


def _with_cut(m, x: np.ndarray, nn: int):
    """m with the no-good cut of the units and actions of x: sum_sel z - sum_other z <= |sel| - 1."""
    from ._milp import OptimizationProblem

    s = np.round(x[:nn])
    row = sp.csr_matrix((2.0 * s - 1.0, (np.zeros(nn, dtype=np.int64), np.arange(nn))), shape=(1, m.ncol))
    a = sp.vstack([m.A, row]).tocsr()
    a.sort_indices()
    return OptimizationProblem(A=a, obj=m.obj, lb=m.lb, ub=m.ub, rhs=np.append(m.rhs, s.sum() - 1.0),
                               sense=(*m.sense, "<="), vtype=m.vtype, modelsense=m.modelsense,
                               row_ids=(*m.row_ids, "cut"), col_ids=m.col_ids)


def _with_objective_row(m, value: float, pool_gap: float):
    from ._milp import OptimizationProblem

    bound = value + pool_gap * abs(value) if m.modelsense == "min" else value - pool_gap * abs(value)
    a = sp.vstack([m.A, sp.csr_matrix(m.obj.reshape(1, -1))]).tocsr()
    a.sort_indices()
    return OptimizationProblem(A=a, obj=m.obj, lb=m.lb, ub=m.ub, rhs=np.append(m.rhs, bound),
                               sense=(*m.sense, "<=" if m.modelsense == "min" else ">="), vtype=m.vtype,
                               modelsense=m.modelsense, row_ids=(*m.row_ids, "pool_gap"), col_ids=m.col_ids)


def solve_actions_portfolio(p) -> ActionsPortfolio:
    from .. import _core
    from ._actions import _arrays, _optimization_problem, _pairs, _settings, compile_actions
    from ._milp_portfolio import _permuted
    from ._milp_solve import solve_model

    kind, opts = p.portfolio
    d = p.data
    n, na = len(d.pu), len(d.dist_threats)
    nn = n + na
    arrays = _arrays(d)
    s = _settings(p)
    c = _core.opt_actions_equivalent(arrays, s) if p.curve == 1 else None
    m = _optimization_problem(c, s["maximize"]) if c is not None else compile_actions(p, disaggregate=True,
                                                                                       curves=True)
    solver, sopts = p.solver
    sopts = {k: v for k, v in sopts.items() if k != "engine"}
    k_max = int(opts["number_solutions"])

    npairs = 0 if c is not None else _pairs(d).feature.size

    def full(x) -> np.ndarray:  # the units, the actions and the pairs' benefits
        x = np.asarray(x, dtype=np.float64)
        wx = np.round(x[:nn])
        if c is not None:
            return np.concatenate([wx, _core.opt_actions_local_benefit(arrays, wx)])
        return np.concatenate([wx, x[nn:nn + npairs]])

    raw = []
    if kind in ("cuts", "top", "gap"):
        if kind == "top":
            sopts = {**sopts, "gap": 0.0}
        cur = m
        for k in range(k_max):
            r = solve_model(cur, solver, sopts)
            if not r["has_solution"] or r["status"] == "infeasible":
                break
            raw.append(r)
            if kind == "gap" and k == 0:
                cur = _with_objective_row(cur, float(r["objective"]), float(opts["pool_gap"]))
            cur = _with_cut(cur, np.asarray(r["x"], dtype=np.float64), nn)
    elif kind == "shuffle":
        rng = np.random.default_rng(opts["seed"])
        seen = set()
        tries = 0
        while len(raw) < k_max and tries < 4 * k_max:
            tries += 1
            perm = np.concatenate([rng.permutation(nn), nn + rng.permutation(m.ncol - nn)])
            r = solve_model(_permuted(m, perm), solver, sopts)
            if not r["has_solution"]:
                break
            x = np.empty(m.ncol)
            x[perm] = np.asarray(r["x"], dtype=np.float64)
            key = tuple(np.round(x[:nn]).astype(np.int8))
            if opts["remove_duplicates"] and key in seen:
                continue
            seen.add(key)
            raw.append({**r, "x": x})
    else:
        raise ValueError(f"unknown portfolio {kind!r}")
    if not raw:
        raise RuntimeError(f"no solution ({solver})")
    if kind in ("cuts", "top", "gap") and len(raw) < k_max:
        import warnings

        warnings.warn(f"the {kind} portfolio found {len(raw)} of {k_max} solutions", UserWarning, stacklevel=3)
    sols = [p._solution(full(r["x"]), {**r, "solver": solver}) for r in raw]
    names = [f"solution_{k + 1}" for k in range(len(sols))]
    costs = [s.cost() for s in sols]
    summary = pd.DataFrame({"solution": names, "objective": [s.objective for s in sols],
                            "monitoring_cost": [float(c_.iloc[0, 0]) for c_ in costs],
                            "action_cost": [float(c_.iloc[0, 1:].sum()) for c_ in costs],
                            "status": [s.status for s in sols], "gap": [s.gap for s in sols]})
    units = np.mean([s.units for s in sols], axis=0)
    acts = np.mean([s.actions for s in sols], axis=0)
    frequency = {"monitoring": pd.DataFrame({"id": d.pu["id"].to_numpy(), "frequency": units}),
                 "actions": pd.DataFrame({"pu": d.dist_threats["pu"].to_numpy(),
                                          "threat": d.dist_threats["threat"].to_numpy(), "frequency": acts})}
    return ActionsPortfolio(sols, summary, frequency)

