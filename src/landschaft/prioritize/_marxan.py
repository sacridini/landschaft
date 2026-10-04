# SPDX-License-Identifier: GPL-3.0-or-later
"""Marxan 4.0.6 (Ball, Possingham & Watts; MIT licence) in landschaft: simulated
annealing, iterative improvement and the greedy heuristics on Marxan's
objective, ported to C++ (``opt/marxan``) with the same runs, unit by unit, as
the Marxan executable built with gcc for the same seed (see
``src/cpp/opt/marxan.hpp`` for what is supported and the corrections)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._save import Saveable

__all__ = ["MarxanResult", "marxan"]


@dataclass
class MarxanResult(Saveable):
    """The runs of :func:`marxan`.

    Attributes
    ----------
    solutions
        Planning units (``planning_unit``: their ids) and every run's selection (``solution_1``, ...:
        1 selected, 0 not; locked units 1 or 0).
    best
        The best run's selection (``planning_unit``, ``solution``): the lowest score, the first run on
        ties.
    summary
        Every run's figures, as Marxan's ``_sum`` file: ``Run_Number``, ``Score``, ``Cost``,
        ``Planning_Units``, ``Connectivity`` (boundary length), ``Penalty``, ``Shortfall``,
        ``Missing_Values`` (features below ``misslevel`` of their target) and ``MPM`` (the lowest
        share of a target met).
    penalties
        Every feature's ``spid``, its ``target`` (after the proportion targets) and its ``penalty``
        (Marxan's greedy estimate of the cost of meeting its target alone).
    frequency
        ``planning_unit`` and ``frequency``: the share of the runs that select each unit (Marxan's
        summed solution over the runs).
    best_run
        The best run's number (from 1).
    """

    solutions: pd.DataFrame = field(repr=False)
    best: pd.DataFrame = field(repr=False)
    summary: pd.DataFrame = field(repr=False)
    penalties: pd.DataFrame = field(repr=False)
    frequency: pd.DataFrame = field(repr=False)
    best_run: int = 1
    unmet: int = 0

    def _save_items(self) -> dict:
        return {"solutions": self.solutions, "best": self.best, "summary": self.summary,
                "penalties": self.penalties, "frequency": self.frequency}


def _col(t: pd.DataFrame, name: str, default, dtype):
    if name in t.columns:
        return t[name].to_numpy(dtype=dtype)
    return np.full(len(t), default, dtype=dtype)


def marxan(pu: pd.DataFrame, spec: pd.DataFrame, puvspr: pd.DataFrame, bound: pd.DataFrame | None = None, *,
           blm: float = 0.0, num_reps: int = 10, num_iters: int = 1_000_000, start_temp: float = -1.0,
           cool_fac: float = 0.0, num_temp: int = 10_000, run_mode: int = 1, heur_type: int = 1,
           itimp_type: int = 0, prop: float = 0.0, misslevel: float = 1.0, cost_thresh: float = 0.0,
           thresh_pen1: float = 0.0, thresh_pen2: float = 0.0, connectivity_in: bool = False, seed: int = 1,
           n_threads: int | None = None) -> MarxanResult:
    """Run Marxan: ``num_reps`` independent runs of annealing and iterative improvement (or the
    greedy heuristics) minimising cost + ``blm`` x boundary + penalties for the features' shortfalls.

    Parameters
    ----------
    pu
        Planning units: ``id``, ``cost`` and optionally ``status`` (0 available, 1 in the initial
        reserve, 2 locked in, 3 locked out). Marxan's ``pu.dat``.
    spec
        Features: ``id``, and ``target`` (an amount), or ``prop`` (a share of the total), ``spf`` (the
        feature penalty factor, default 1) and optionally ``targetocc`` (occurrences). Marxan's
        ``spec.dat``.
    puvspr
        Amount of every feature in every planning unit: ``species``, ``pu``, ``amount`` (one row per
        pair). Marxan's ``puvspr.dat``.
    bound
        Boundaries: ``id1``, ``id2``, ``boundary``; a row with ``id1 == id2`` is a unit's boundary with
        the outside (its fixed boundary). Marxan's ``bound.dat``.
    blm, num_reps, num_iters, start_temp, cool_fac, num_temp, run_mode, heur_type, itimp_type, prop,
    misslevel, cost_thresh, thresh_pen1, thresh_pen2
        Marxan's ``input.dat`` options (BLM, NUMREPS, NUMITNS, STARTTEMP: < 0 for Connolly's (-1) or the
        adaptive (-2) schedule, COOLFAC, NUMTEMP, RUNMODE 0, 1, 2 or 6, HEURTYPE 0 or 1, ITIMPTYPE 0 or 3,
        PROP, MISSLEVEL, COSTTHRESH, THRESHPEN1, THRESHPEN2).
    connectivity_in
        Marxan's CONNECTIVITYIN: count the boundaries inside the reserve instead of at its edge.
    seed
        RANDSEED: the same seed gives the same runs on any number of threads.
    n_threads
        Runs in parallel (default: all cores but one).

    Returns
    -------
    MarxanResult
    """
    from .. import _core, _threads

    pu = pd.DataFrame(pu)
    spec = pd.DataFrame(spec)
    puvspr = pd.DataFrame(puvspr)
    for t, cols, what in ((pu, ("id", "cost"), "pu"), (spec, ("id",), "spec"),
                          (puvspr, ("species", "pu", "amount"), "puvspr")):
        missing = [c for c in cols if c not in t.columns]
        if missing:
            raise ValueError(f"{what} needs the columns {missing}")
    ids = pu["id"].to_numpy()
    pu_index = pd.Series(np.arange(len(pu)), index=ids)
    if not pu_index.index.is_unique:
        raise ValueError("planning-unit ids must be unique")
    sp_index = pd.Series(np.arange(len(spec)), index=spec["id"].to_numpy())
    if not sp_index.index.is_unique:
        raise ValueError("feature ids must be unique")
    r = pu_index.reindex(puvspr["pu"].to_numpy()).to_numpy()
    k = sp_index.reindex(puvspr["species"].to_numpy()).to_numpy()
    if np.isnan(r).any() or np.isnan(k).any():
        raise ValueError("puvspr refers to planning units or features that are not in pu or spec")
    r, k = r.astype(np.int64), k.astype(np.int64)
    pairs = r * len(spec) + k
    if np.unique(pairs).size != pairs.size:
        raise ValueError("puvspr has a feature twice in a planning unit")
    order = np.lexsort((k, r))  # units in order, features by index within a unit
    offset = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=len(pu)))]).astype(np.int64)
    # boundaries in Marxan's order: the file's, each pair added to both units unless already there
    fixed = np.zeros(len(pu))
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(len(pu))]
    if bound is not None and len(bound):
        b = pd.DataFrame(bound)
        i1 = pu_index.reindex(b["id1"].to_numpy()).to_numpy()
        i2 = pu_index.reindex(b["id2"].to_numpy()).to_numpy()
        if np.isnan(i1).any() or np.isnan(i2).any():
            raise ValueError("bound refers to planning units that are not in pu")
        for a, c, v in zip(i1.astype(np.int64), i2.astype(np.int64), b["boundary"].to_numpy(dtype=np.float64)):
            if a == c:
                fixed[a] += v
                continue
            if all(n != c for n, _ in nbrs[a]):
                nbrs[a].append((int(c), float(v)))
            if all(n != a for n, _ in nbrs[c]):
                nbrs[c].append((int(a), float(v)))
    nb_start = np.concatenate([[0], np.cumsum([len(x) for x in nbrs])]).astype(np.int64)
    nb = np.array([n for x in nbrs for n, _ in x], dtype=np.int32)
    nb_cost = np.array([v for x in nbrs for _, v in x], dtype=np.float64)
    opts = {"blm": float(blm), "prop": float(prop), "iterations": int(num_iters), "tinit": float(start_temp),
            "tcool": float(cool_fac), "titns": int(num_temp), "repeats": int(num_reps),
            "costthresh": float(cost_thresh), "tpf1": float(thresh_pen1), "tpf2": float(thresh_pen2),
            "misslevel": float(misslevel), "runmode": int(run_mode), "heurtype": int(heur_type),
            "itimptype": int(itimp_type), "connectivity_in": int(bool(connectivity_in)),
            "seed": int(seed) & 0xFFFFFFFF, "threads": _threads.resolve(n_threads)}
    out = _core.opt_marxan(
        pu["cost"].to_numpy(dtype=np.float64), _col(pu, "status", 0, np.int32), offset,
        k[order].astype(np.int32), puvspr["amount"].to_numpy(dtype=np.float64)[order],
        _col(spec, "target", 0.0, np.float64), _col(spec, "prop", 0.0, np.float64),
        _col(spec, "spf", -1.0, np.float64), _col(spec, "targetocc", 0, np.int32), fixed, nb_start, nb, nb_cost, opts)
    sol = np.asarray(out["solutions"])
    sel = ((sol == 1) | (sol == 2)).astype(np.int64)
    solutions = pd.DataFrame({"planning_unit": ids})
    for j in range(sel.shape[0]):
        solutions[f"solution_{j + 1}"] = sel[j]
    best = int(out["best"])
    summary = pd.DataFrame({
        "Run_Number": np.arange(1, sel.shape[0] + 1), "Score": out["incremental_score"], "Cost": out["incremental_cost"],
        "Planning_Units": out["pus"], "Connectivity": out["connectivity"], "Penalty": out["incremental_penalty"],
        "Shortfall": out["shortfall"], "Missing_Values": out["missing"], "MPM": out["mpm"],
        "Final_Score": out["score"]})
    return MarxanResult(
        solutions=solutions,
        best=pd.DataFrame({"planning_unit": ids, "solution": sel[best] if sel.size else np.zeros(len(ids), int)}),
        summary=summary,
        penalties=pd.DataFrame({"spid": spec["id"].to_numpy(), "target": out["target"], "penalty": out["penalty"]}),
        frequency=pd.DataFrame({"planning_unit": ids, "frequency": sel.mean(axis=0) if sel.size else 0.0}),
        best_run=best + 1, unmet=int(out["unmet"]))
