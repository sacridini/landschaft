# SPDX-License-Identifier: GPL-3.0-or-later
"""A local search over the targets met, for the budgeted objectives that reward meeting targets (maximum
features, maximum phylogenetic diversity and endemism).

Such a model is a choice of a set S of targets: its value is the reward of the targets in S (and of the
branches of a phylogeny with a target of S below them), and S is feasible when the cheapest selection
meeting S fits the budget. The cheapest selection is a minimum set problem, which the covering engine
solves exactly and fast; the search moves from the wide engine's set by additions and swaps that raise the
value, the most valuable first, each accepted when the covering engine finds a selection within the budget.
A selection may meet targets outside S on the way: they count too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._milp import OptimizationProblem

__all__ = ["subset_model", "improve_subset"]

ORACLE_GAP = 0.005  # the covering engine's gap when it prices a set of targets
ORACLE_SECONDS = 20.0  # and its time for one set at most
MAX_SWAPS = 20000  # swaps weighed per step


@dataclass
class SubsetModel:
    n: int
    flags: np.ndarray  # the targets' columns (target met), in target order
    target_rows: np.ndarray
    budget_row: int
    cost: np.ndarray  # the units' coefficients in the budget row
    budget: float
    amounts: sp.csr_matrix  # targets x units
    target: np.ndarray  # the amount each target needs
    reward: np.ndarray  # each target's own reward (maximisation)
    branch_cols: np.ndarray  # columns rewarded when a target of theirs is met (phylogenies' branches)
    branch_reward: np.ndarray
    branch_targets: sp.csr_matrix  # branches x targets: 1 where a met target allows the branch


def subset_model(m: OptimizationProblem, n_units: int | None) -> SubsetModel | None:
    """The structure of a target-rewarding budgeted model, or None when the model is not one."""
    if n_units is None or m.modelsense != "max" or "spp_met" not in m.col_ids:
        return None
    n = int(n_units)
    cid = np.asarray(m.col_ids)
    rid = np.asarray(m.row_ids)
    sense = np.asarray(m.sense)
    flags = np.flatnonzero(cid == "spp_met")
    trows = np.flatnonzero(rid == "spp_target")
    brows = np.flatnonzero(rid == "budget")
    if flags.size != trows.size or brows.size != 1 or not (sense[trows] == ">=").all() or sense[brows[0]] != "<=":
        return None
    other = np.setdiff1d(np.arange(n, m.ncol), flags)
    known = np.concatenate([trows, brows, np.flatnonzero(rid == "branch_target")])
    if known.size != m.nrow or (other.size and not np.isin(cid[other], ("branch_met",)).all()):
        return None
    if not np.isin(np.asarray(m.vtype)[np.concatenate([flags, other])], ("B",)).all():
        return None
    a = m.A.tocsr()
    t = a[trows].tocsc()
    # a target's row: its amounts in the units, minus the target times its flag (and nothing else)
    fl = t[:, flags].toarray()
    if not np.allclose(fl, np.diag(np.diag(fl))) or (np.diag(fl) > 0).any() or t[:, other].nnz:
        return None
    if a[brows[0], n:].nnz:
        return None
    branch_targets = sp.csr_matrix((other.size, flags.size))
    if other.size:
        rows = np.flatnonzero(rid == "branch_target")
        br = a[rows].tocsc()
        bo = br[:, other].toarray()
        # each branch row: the targets below the branch (+1) minus the branch (-1)
        if not np.array_equal(bo, -np.eye(other.size)) or (sense[rows] != ">=").any() or (m.rhs[rows] != 0).any():
            return None
        if br[:, :n].nnz:
            return None
        bt = br[:, flags].tocsr()
        if (bt.data != 1.0).any():
            return None
        branch_targets = bt
    return SubsetModel(n=n, flags=flags, target_rows=trows, budget_row=int(brows[0]),
                       cost=a[brows[0], :n].toarray().ravel(), budget=float(m.rhs[brows[0]]),
                       amounts=a[trows][:, :n].tocsr(), target=-np.diag(fl).copy(), reward=m.obj[flags].copy(),
                       branch_cols=other, branch_reward=m.obj[other].copy(), branch_targets=branch_targets)


def _value(sm: SubsetModel, met: np.ndarray) -> float:
    """The reward of a set of targets met (a boolean per target)."""
    v = float(sm.reward[met].sum())
    if sm.branch_cols.size:
        on = (sm.branch_targets @ met.astype(np.float64)) > 0
        v += float(sm.branch_reward[on & (sm.branch_reward > 0)].sum())
    return v


def _complete(m: OptimizationProblem, sm: SubsetModel, xu: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Every column from the units' values: the targets they meet, the branches those allow; the
    objective, or None when a row is broken."""
    z = np.zeros(m.ncol)
    z[:sm.n] = xu
    held = sm.amounts @ xu
    met = held >= sm.target - 1e-9 * np.maximum(1.0, np.abs(sm.target))
    z[sm.flags] = met
    if sm.branch_cols.size:
        on = (sm.branch_targets @ met.astype(np.float64)) > 0
        z[sm.branch_cols] = on & (sm.branch_reward > 0)
    act = m.A @ z
    lo, hi = m.row_bounds()
    tol = 1e-9 * np.maximum(1.0, np.abs(m.rhs))
    if (act < lo - tol).any() or (act > hi + tol).any():
        return None
    return z, float(m.obj @ z)


def improve_subset(m: OptimizationProblem, sm: SubsetModel, x: np.ndarray, deadline: float, threads: int,
                   verbose: bool = False) -> tuple[np.ndarray, float] | None:
    """From a solution x, the local search; the best solution found and its objective (None if none
    better than x). ``deadline``: a time.perf_counter() value (math.inf: until no move helps)."""
    from ._milp_solve import _highs

    lb, ub = m.lb[:sm.n], m.ub[:sm.n]
    vt = tuple(m.vtype[:sm.n])
    nt = sm.flags.size

    def left() -> float:
        return deadline - time.perf_counter()

    def price(s: np.ndarray):
        """The cheapest selection meeting the targets s: its units' values, or None (over the budget)."""
        idx = np.flatnonzero(s)
        if idx.size == 0:
            xu = lb.copy()
        else:
            cm = OptimizationProblem(A=sm.amounts[idx], obj=sm.cost, lb=lb, ub=ub, rhs=sm.target[idx],
                                     sense=(">=",) * idx.size, vtype=vt, modelsense="min",
                                     row_ids=("spp_target",) * idx.size, col_ids=("pu",) * sm.n)
            r = _highs(cm, ORACLE_GAP, max(min(ORACLE_SECONDS, left()), 0.0), threads, "choose", False, None)
            if not r["has_solution"]:
                return None
            # over the budget within the gap, but the bound fits: the cheapest selection may fit, priced exactly
            room = sm.budget + 1e-9 * max(1.0, abs(sm.budget))
            if r["objective"] > room and r["bound"] <= room:
                r2 = _highs(cm, 0.0, max(min(ORACLE_SECONDS, left()), 0.0), threads, "choose", False, r["x"])
                if r2["has_solution"] and r2["objective"] < r["objective"]:
                    r = r2
            xu = np.asarray(r["x"], dtype=np.float64)
            xu = np.where(np.isin(np.asarray(vt), ("B", "I")), np.round(xu), xu)
        return xu if float(sm.cost @ xu) <= sm.budget + 1e-9 * max(1.0, abs(sm.budget)) else None

    best_z = np.asarray(x, dtype=np.float64)
    best = float(m.obj @ best_z)
    met = best_z[sm.flags] > 0.5
    value = _value(sm, met)
    improved_any = False
    calls = 0
    moved = True
    while moved and left() > 0:
        moved = False
        inside = np.flatnonzero(met)
        outside = np.flatnonzero(~met)
        moves = []
        for f in outside:  # additions
            s = met.copy()
            s[f] = True
            moves.append((_value(sm, s), 0, s))
        if inside.size * outside.size > MAX_SWAPS:  # many targets: swaps in of the best additions only
            keep = max(1, MAX_SWAPS // max(inside.size, 1))
            outside = outside[np.argsort([-mv[0] for mv in moves], kind="stable")[:keep]]
        for r_ in inside:  # swaps
            for f in outside:
                s = met.copy()
                s[r_] = False
                s[f] = True
                moves.append((_value(sm, s), 1, s))
        # the most valuable first, additions before swaps of the same value
        moves = [mv for mv in moves if mv[0] > value + 1e-9 * max(1.0, abs(value))]
        moves.sort(key=lambda mv: (-mv[0], mv[1]))
        for v, _, s in moves:
            if left() <= 0:
                break
            calls += 1
            xu = price(s)
            if xu is None:
                continue
            done = _complete(m, sm, xu)
            if done is None or done[1] <= best + 1e-12 * max(1.0, abs(best)):
                continue
            best_z, best = done
            met = best_z[sm.flags] > 0.5
            value = _value(sm, met)
            improved_any = moved = True
            if verbose:
                print(f"subset search: {int(met.sum())} of {nt} targets met, objective {best:.12g} "
                      f"({calls} sets priced)", flush=True)
            break
    return (best_z, best) if improved_any else None
