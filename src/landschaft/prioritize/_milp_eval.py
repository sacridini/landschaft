# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluating solutions (prioritizr 8.0.6's ``eval_*`` functions; GPL-3): summaries of a solution
(number of units, cost, boundary, connectivity, the features' representation, the targets' coverage)
and the importance of every selected planning unit (replacement cost, Ferrier et al. 2000's
irreplaceability, rarity-weighted richness, rank importance after Jung et al. 2021).

The summaries and the Ferrier and rarity scores are prioritizr's bit for bit (the same sums in the same
order: R's 80-bit sums, Matrix's, Armadillo's interleaved ones; R's pnorm in C++). The replacement and
rank importance re-solve the problem, the re-solves in parallel."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from ._zones import one_zone

__all__ = [
    "eval_asym_connectivity_summary",
    "eval_boundary_summary",
    "eval_connectivity_summary",
    "eval_cost_summary",
    "eval_extinction_risk",
    "eval_feature_representation_summary",
    "eval_ferrier_importance",
    "eval_landscape_metrics",
    "eval_mspa",
    "eval_n_summary",
    "eval_rank_importance",
    "eval_rare_richness_importance",
    "eval_replacement_importance",
    "eval_target_coverage_summary",
]


# ----------------------------------------------------------------------------- inputs and outputs

def _values(p, solution) -> np.ndarray:
    """The solution's value in every planning unit (prioritizr's planning_unit_solution_status), NaN as 0."""
    from ._milp_solve import Solution

    u = p.units
    if isinstance(solution, Solution):
        v = np.asarray(solution.values, dtype=np.float64)
    elif isinstance(solution, xr.DataArray) and solution.ndim >= 2:
        s = solution.squeeze("band", drop=True) if solution.ndim == 3 else solution
        if u.kind != "raster" or s.shape != u.template.shape:
            raise ValueError("a raster solution needs raster planning units on the same grid")
        v = np.asarray(s.values, dtype=np.float64).ravel()[u.index]
    elif isinstance(solution, pd.DataFrame):
        cols = [c for c in ("solution", "solution_1") if c in solution.columns]
        if not cols:
            num = [c for c in solution.columns if c != "geometry"]
            if len(num) != 1:
                raise ValueError("a table solution needs a 'solution' column")
            cols = num
        col = np.asarray(solution[cols[0]], dtype=np.float64)
        v = col[u.index] if col.size == len(u.template) and u.kind != "raster" else col
    else:
        v = np.asarray(solution, dtype=np.float64).ravel()
    if v.size != u.n:
        raise ValueError(f"the solution needs one value per planning unit ({u.n}), not {v.size}")
    return np.where(np.isfinite(v), v, 0.0)


def _per_unit(p, values: np.ndarray, names: list[str]):
    """Per-unit results like the solution's selection: a raster (a band per name, NaN outside the units)
    or the planning units' table with a column per name."""
    u = p.units
    values = np.atleast_2d(values)
    if u.kind == "raster":
        t = u.template
        out = np.full((len(names), t.size), np.nan)
        out[:, u.index] = values
        da = xr.DataArray(out.reshape((len(names), *t.shape)), dims=("band", *t.dims),
                          coords={"band": names, **{k: v for k, v in t.coords.items() if k in t.dims}})
        try:
            crs = t.rio.crs
        except Exception:  # noqa: BLE001 - no georeferencing
            crs = None
        if crs is not None:
            da = da.rio.write_crs(crs)
        return da.squeeze("band", drop=True) if len(names) == 1 else da
    tab = u.template.copy()
    for k, name in enumerate(names):
        col = np.full(len(tab), np.nan)
        col[u.index] = values[k]
        tab[name] = col
    return tab


def _r_sum(v: np.ndarray) -> float:
    from ._milp import _r_sum as rs

    return rs(np.asarray(v, dtype=np.float64))


def _csc(m) -> sp.csc_matrix:
    m = sp.csc_matrix(m)
    m.sort_indices()
    return m


def _unit_matrix(p, data) -> sp.csc_matrix:
    """A matrix over all units (raster cells, table rows) restricted to the planning units."""
    u = p.units
    m = sp.csr_matrix(data) if not isinstance(data, pd.DataFrame) else None
    if m is None:
        raise TypeError("data must be a matrix (dense or scipy sparse)")
    total = u.template.size if u.kind == "raster" else len(u.template)
    if m.shape == (u.n, u.n) and total != u.n:
        sub = m
    elif m.shape == (total, total):
        sub = m[u.index][:, u.index]
    else:
        raise ValueError(f"data must be {total} x {total} (every unit) or {u.n} x {u.n} (the planning units)")
    return _csc(sub)


def _rescale(x: np.ndarray, lo: float = 0.01, hi: float = 1.0) -> np.ndarray:
    """prioritizr's rescale(x, to = c(lo, hi)) of the values that are finite and not ~0."""
    out = np.array(x, dtype=np.float64)
    ind = np.isfinite(out) & (np.abs(out) > 1e-10)
    if not ind.any():
        return out
    v = out[ind]
    frm = (v.min(), v.max())
    if abs(frm[1] - frm[0]) < 1e-10:
        out[ind] = (lo + hi) / 2
    else:
        out[ind] = (v - frm[0]) / (frm[1] - frm[0]) * (hi - lo) + lo
    return out


# ----------------------------------------------------------------------------- summaries

def eval_n_summary(p, solution) -> pd.DataFrame:
    """The number of selected planning units (the sum of the solution's values).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> P.eval_n_summary(p, sol)
       summary      n
    0  overall  534.0
    """
    one_zone(p, "eval_n_summary")
    return pd.DataFrame({"summary": ["overall"], "n": [_r_sum(_values(p, solution))]})


def eval_cost_summary(p, solution) -> pd.DataFrame:
    """The solution's total cost.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> round(P.eval_cost_summary(p, sol)["cost"].item(), 1) == round(sol.cost, 1)
    True
    """
    one_zone(p, "eval_cost_summary")
    s = _values(p, solution)
    return pd.DataFrame({"summary": ["overall"], "cost": [_r_sum(p.units.cost * s)]})


def eval_boundary_summary(p, solution, edge_factor: float = 0.5, data=None) -> pd.DataFrame:
    """The solution's boundary length: the edges between selected and other units, plus
    ``edge_factor`` x the selected units' edges on the outside of the planning units.

    ``data``: the boundary matrix (every unit's shared lengths and its perimeter on the diagonal;
    default :func:`boundary_matrix` of the planning units).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> P.eval_boundary_summary(p, sol)
       summary  boundary
    0  overall  382500.0
    """
    one_zone(p, "eval_boundary_summary")
    from .. import _core
    from ._milp_matrices import boundary_matrix
    from ._milp_spatial import r_round

    if not (0 <= edge_factor <= 1):
        raise ValueError("edge_factor must be between 0 and 1")
    u = p.units
    if data is None:
        if u.kind == "table" and not hasattr(u.template, "geometry"):
            raise ValueError("table planning units need the boundary matrix (data)")
        bm = _csc(boundary_matrix(u.template))
        bm = _csc(bm[u.index][:, u.index]) if bm.shape[0] != u.n else bm
    else:
        bm = _unit_matrix(p, data)
    bm = bm.copy()
    bm.data = r_round(bm.data, 6)
    total = bm.diagonal()
    rows = bm.tocsr()
    rows.sort_indices()
    row_sums = np.array([_seq(rows.data[rows.indptr[i]:rows.indptr[i + 1]]) for i in range(u.n)])
    exposed = total - (row_sums - total)
    if (r_round(total - (row_sums - total), 6) < -1e-5).any():
        warnings.warn("the boundary matrix has unexpected values: a unit's perimeter (diagonal) below its shared "
                      "lengths (overlapping planning units, or a matrix without perimeters?)", UserWarning,
                      stacklevel=2)
    lower = sp.tril(bm, k=-1).tocsc()
    lower.eliminate_zeros()
    lower.sort_indices()
    v = _core.opt_eval_boundary(float(edge_factor), lower.indptr.astype(np.int64), lower.indices.astype(np.int64),
                                lower.data, exposed, total, _values(p, solution))
    return pd.DataFrame({"summary": ["overall"], "boundary": [v]})


def _seq(v: np.ndarray) -> float:
    """A sum in order, in double."""
    return float(np.cumsum(v, dtype=np.float64)[-1]) if v.size else 0.0


def _connectivity(p, solution, data, asym: bool) -> float:
    from .. import _core

    m = _unit_matrix(p, data)
    symmetric = (abs(m - m.T) > 0).nnz == 0
    if symmetric and asym:
        warnings.warn("data contains symmetric connectivity values; eval_connectivity_summary is recommended",
                      UserWarning, stacklevel=3)
    if not symmetric and not asym:
        warnings.warn("data contains asymmetric connectivity values; eval_asym_connectivity_summary is "
                      "recommended", UserWarning, stacklevel=3)
    return _core.opt_eval_connectivity(m.indptr.astype(np.int64), m.indices.astype(np.int64), m.data,
                                       _values(p, solution), asym)


def eval_connectivity_summary(p, solution, data) -> pd.DataFrame:
    """The connectivity among the selected units: the sum of ``data[i, j]`` x the solution's values of
    units i and j (a symmetric matrix over the units).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> cm = P.connectivity_matrix(cost, features[0])
    >>> round(P.eval_connectivity_summary(p, sol, cm)["connectivity"].item(), 2)
    1276348.0
    """
    one_zone(p, "eval_connectivity_summary")
    return pd.DataFrame({"summary": ["overall"], "connectivity": [_connectivity(p, solution, data, False)]})


def eval_asym_connectivity_summary(p, solution, data) -> pd.DataFrame:
    """As :func:`eval_connectivity_summary` for an asymmetric matrix (its diagonal counted once).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> import scipy.sparse as sp
    >>> cm = P.connectivity_matrix(cost, features[0])
    >>> asym = sp.triu(cm).tocsr()
    >>> round(P.eval_asym_connectivity_summary(p, sol, asym)["asym_connectivity"].item(), 2)
    638174.0
    """
    one_zone(p, "eval_asym_connectivity_summary")
    return pd.DataFrame({"summary": ["overall"], "asym_connectivity": [_connectivity(p, solution, data, True)]})


def _held(p, s: np.ndarray) -> np.ndarray:
    """Every feature's amount in the solution: rij %*% s by Matrix (each feature's units in order)."""
    rij = p.units.rij.tocsr()
    rij.sort_indices()
    out = np.zeros(rij.shape[0])
    for f in range(rij.shape[0]):
        lo, hi = rij.indptr[f], rij.indptr[f + 1]
        k = rij.indices[lo:hi]
        keep = s[k] != 0
        out[f] = _seq(rij.data[lo:hi][keep] * s[k][keep])
    return out


def eval_feature_representation_summary(p, solution) -> pd.DataFrame:
    """Every feature's total amount and the amount the solution holds, absolute and relative.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> rep = P.eval_feature_representation_summary(p, sol)
    >>> rep[["feature", "relative_held"]].round(3)
                   feature  relative_held
    0               jaguar          0.200
    1        lowland_tapir          0.201
    2          harpy_eagle          0.299
    3           maned_wolf          0.205
    4  golden_lion_tamarin          0.204
    5    neotropical_otter          0.201
    6     araucaria_forest          0.428
    7             mangrove          0.210
    """
    one_zone(p, "eval_feature_representation_summary")
    u = p.units
    held = _held(p, _values(p, solution))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = held / u.totals
    return pd.DataFrame({"summary": "overall", "feature": list(u.feature_names), "total_amount": u.totals,
                         "absolute_held": held, "relative_held": rel})


def eval_target_coverage_summary(p, solution, include_sense: bool = False) -> pd.DataFrame:
    """Every target's feature, whether it is met, the feature's total, and the target, the amount held
    and the shortfall, absolute and relative (shares of the total; 0 when the total is 0).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> cov = P.eval_target_coverage_summary(p, sol)
    >>> bool(cov["met"].all())
    True
    """
    one_zone(p, "eval_target_coverage_summary")
    from .. import _core
    from ._milp_compile import NO_TARGETS, compile_problem

    if p.objective is not None and p.objective[0] in NO_TARGETS or p.targets is None and p.subregional is None:
        raise ValueError("the problem has no targets: use eval_feature_representation_summary")
    _, t = compile_problem(p)
    s = _values(p, solution)
    u = p.units
    rij = u.rij.tocsr()
    rij.sort_indices()
    mat = t.matrix.tocsr() if t.matrix is not None else None
    held = np.empty(t.n)
    for k in range(t.n):  # Armadillo: the non-zero products of the feature's column, interleaved sums
        if mat is not None:
            lo, hi = mat.indptr[k], mat.indptr[k + 1]
            idx, val = mat.indices[lo:hi], mat.data[lo:hi]
        else:
            f = t.feature[k]
            lo, hi = rij.indptr[f], rij.indptr[f + 1]
            idx, val = rij.indices[lo:hi], rij.data[lo:hi]
        prod = val * s[idx]
        prod = np.ascontiguousarray(prod[prod != 0])
        held[k] = _core.opt_eval_accumulate(prod)
    total = t.total if t.total is not None else u.totals[t.feature]
    target = np.asarray(t.value, dtype=np.float64)
    sense = np.asarray(t.sense)
    short = np.abs(target - held)
    short = np.where((sense == ">=") & (held >= target), 0.0, short)
    short = np.where((sense == "=") & ((held - target) < 1e-10), 0.0, short)
    short = np.where((sense == "<=") & (held <= target), 0.0, short)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt, rh, rs = target / total, held / total, short / total
    rt, rh, rs = (np.where(np.isfinite(v), v, 0.0) for v in (rt, rh, rs))
    out = {"feature": [u.feature_names[f] for f in t.feature]}
    if include_sense:
        out["sense"] = list(sense)
    out.update({"met": short < 1e-10, "total_amount": total, "absolute_target": target, "absolute_held": held,
                "absolute_shortfall": short, "relative_target": rt, "relative_held": rh, "relative_shortfall": rs})
    df = pd.DataFrame(out)
    if t.region is not None and (t.region >= 0).any():
        df.insert(1, "region", list(t.region_label))
    return df


def eval_extinction_risk(p, solution, z=0.25) -> pd.DataFrame:
    """Every feature's extinction risk by the species-area relationship, 1 - (held / total)^z (z = 0.25
    by default, one value or one per feature), and their sum (the expected number of features lost).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> risk = P.eval_extinction_risk(p, sol)
    >>> round(float(risk["extinction_risk"].iloc[-1]), 3)
    2.423
    """
    one_zone(p, "eval_extinction_risk")
    rep = eval_feature_representation_summary(p, solution)
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), (len(rep),))
    rel = np.clip(np.nan_to_num(rep["relative_held"].to_numpy(dtype=np.float64)), 0.0, 1.0)
    risk = 1.0 - rel ** z
    out = pd.DataFrame({"feature": rep["feature"], "relative_held": rel, "z": z, "extinction_risk": risk})
    total = pd.DataFrame({"feature": ["overall"], "relative_held": [np.nan], "z": [np.nan],
                          "extinction_risk": [float(np.sum(risk))]})
    return pd.concat([out, total], ignore_index=True)


# ----------------------------------------------------------------------------- importance

def _selected(s: np.ndarray) -> np.ndarray:
    sel = np.flatnonzero(s > 1e-10)
    if sel.size == 0:
        raise ValueError("the solution selects no planning unit")
    return sel


def eval_ferrier_importance(p, solution):
    """Ferrier et al. (2000)'s irreplaceability of the selected units for every feature and in total
    (prioritizr's version: the problem needs one ">=" target per feature, amounts >= 0).

    Returns a raster with a band per feature and ``total`` (or the units' table with those columns): 0
    for the units not selected.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> fi = P.eval_ferrier_importance(p, sol)
    >>> [str(b) for b in fi.band.values][-2:]
    ['mangrove', 'total']
    >>> float(fi.sel(band="total").max()) > 0
    True
    """
    one_zone(p, "eval_ferrier_importance")
    from .. import _core
    from ._milp_compile import NO_TARGETS, compile_problem

    u = p.units
    if p.objective is not None and p.objective[0] in NO_TARGETS or p.targets is None:
        raise ValueError("eval_ferrier_importance needs a problem with targets (try eval_replacement_importance)")
    if "total" in u.feature_names:
        raise ValueError("no feature may be called 'total'")
    _, t = compile_problem(p)
    nf = len(u.feature_names)
    if t.n != nf or not np.array_equal(t.feature, np.arange(nf)) or any(x != ">=" for x in t.sense):
        raise ValueError("eval_ferrier_importance needs one '>=' target per feature")
    if (np.asarray(t.value) < 0).any() or (u.rij.data < 0).any():
        raise ValueError("eval_ferrier_importance needs targets and amounts >= 0")
    s = _values(p, solution)
    sel = _selected(s)
    rij = _csc(u.rij)
    mask = np.zeros(u.n, dtype=np.uint8)
    mask[sel] = 1
    st, ix, val = _core.opt_eval_ferrier(rij.indptr.astype(np.int64), rij.indices.astype(np.int64), rij.data, nf,
                                        np.asarray(t.value, dtype=np.float64), mask)
    score = sp.csc_matrix((val, ix, st), shape=(nf, u.n))
    out = np.zeros((nf + 1, u.n))
    out[:nf] = score.toarray()
    # colSums of the scores (Matrix: every unit's features in order)
    for j in sel:
        lo, hi = score.indptr[j], score.indptr[j + 1]
        out[nf, j] = _seq(score.data[lo:hi])
    return _per_unit(p, out, [*u.feature_names, "total"])


def eval_rare_richness_importance(p, solution, rescale: bool = True):
    """The rarity-weighted richness of the selected units (Albuquerque & Beier 2015): the sum over the
    features of the unit's amount / the feature's largest amount / its total, rescaled to 0.01-1.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> rwr = P.eval_rare_richness_importance(p, sol)
    >>> float(rwr.max())
    1.0
    """
    one_zone(p, "eval_rare_richness_importance")
    u = p.units
    s = _values(p, solution)
    sel = _selected(s)
    rij = u.rij.tocsc()
    dense = np.asarray(rij[:, sel].todense(), dtype=np.float64)
    full = u.rij.tocsr()
    m = np.array([max(0.0, full.data[full.indptr[f]:full.indptr[f + 1]].max()) if full.indptr[f + 1] > full.indptr[f]
                  else 0.0 for f in range(full.shape[0])])
    if (u.rij.data < 0).any():  # apply(rij, 1, max) over the dense rows
        m = np.asarray(full.max(axis=1).todense()).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dense / m[:, None]
        out[~np.isfinite(out)] = 0.0
        out = out / u.totals[:, None]
        out[~np.isfinite(out)] = 0.0
    v = np.array([_seq(out[:, j]) for j in range(out.shape[1])])  # Matrix's colSums: in order, in double
    if rescale:
        v = _rescale(v)
    res = np.zeros(u.n)
    res[sel] = v
    return _per_unit(p, res, ["rwr"])


def _solve_values(m, solver: str, opts: dict, n: int, start=None) -> dict:
    from ._milp_solve import solve_model

    return solve_model(m, solver, opts, start, n_units=n)


def eval_replacement_importance(p, solution, rescale: bool = True, n_threads: int | None = None):
    """The replacement cost of every selected unit (Cabeza & Moilanen 2006): how much the objective
    worsens when the unit is left out and the problem solved again with the other selected units kept
    (infinite when nothing can replace it), rescaled to 0.01-1. One solve per selected unit, in parallel
    (``n_threads``, default all cores but one).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> rc = P.eval_replacement_importance(p, sol)
    >>> float(rc.max())
    1.0
    """
    one_zone(p, "eval_replacement_importance")
    from concurrent.futures import ThreadPoolExecutor

    from .. import _threads
    from ._milp import OptimizationProblem
    from ._milp_compile import compile_problem

    u = p.units
    s = _values(p, solution)
    sel = _selected(s)
    m, _ = compile_problem(p)
    solver, opts = p.solver
    if solver not in ("highs", "gurobi"):
        raise ValueError("eval_replacement_importance needs the HiGHS or the Gurobi solver")
    lb, ub = m.lb.copy(), m.ub.copy()
    lb[sel] = 1.0
    ub[sel] = 1.0

    def model(lo, hi):
        return OptimizationProblem(A=m.A, obj=m.obj, lb=lo, ub=hi, rhs=m.rhs, sense=m.sense, vtype=m.vtype,
                                   modelsense=m.modelsense, row_ids=m.row_ids, col_ids=m.col_ids)

    threads = _threads.resolve(n_threads)
    base = _solve_values(model(lb, ub), solver, opts, u.n)
    if not base["has_solution"]:
        raise ValueError("the solution is not feasible for the problem")
    one = {**opts, "n_threads": 1}

    def alt(i):
        lo, hi = lb.copy(), ub.copy()
        lo[i] = hi[i] = 0.0
        r = _solve_values(model(lo, hi), solver, one, u.n)
        return float(r["objective"]) if r["has_solution"] else math.inf

    if solver == "highs" and not any(sp_[0] == "penalty" and sp_[1] in ("boundary", "connectivity", "asym_connectivity")
                                     for sp_ in p.spatial):
        # the arrays made once: every re-solve is then the C++ call alone (no GIL), in parallel
        from ._milp_solve import Prepared

        prep = Prepared(m, u.n)

        def alt(i):
            lo, hi = lb.copy(), ub.copy()
            lo[i] = hi[i] = 0.0
            r = prep.solve(lo, hi, one)
            return float(r["objective"]) if r["has_solution"] else math.inf

    with ThreadPoolExecutor(max_workers=threads) as ex:
        objs = np.array(list(ex.map(alt, sel)))
    out = objs - float(base["objective"])
    if m.modelsense == "max":
        out = out * -1
    if rescale:
        out = _rescale(out)
    res = np.zeros(u.n)
    res[sel] = out
    return _per_unit(p, res, ["rc"])


def _r_seq(lo: float, hi: float, n: int) -> np.ndarray:
    """R's seq(lo, hi, length.out = n)."""
    if n <= 2:
        return np.array([lo, hi])[:n]
    if lo == hi:
        return np.full(n, lo)
    n1 = n - 1
    return np.concatenate([[lo], lo + np.arange(1, n1) * ((hi - lo) / n1), [hi]])


def eval_rank_importance(p, solution, n: int | None = None, budgets=None, rescale: bool = True):
    """Rank importance (Jung et al. 2021, prioritizr's eval_rank_importance): the problem is solved
    with increasing budgets (the min set objective becomes min shortfall), only the solution's units
    allowed and each step keeping the units of the step before; a unit's rank is how early it enters
    (n = the first budget, ..., 1 = the last), rescaled to 0.01-1. ``n`` budgets evenly spaced between
    the locked-in units' cost and the solution's, or the ``budgets`` themselves.

    Returns the per-unit ranks (a raster or the units' table) and a table of every budget's objective,
    status and gap.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> ranks, steps = P.eval_rank_importance(p, sol, n=4)
    >>> steps[["budget", "status"]].round(1)
        budget   status
    0   5898.5  optimal
    1  11797.0  optimal
    2  17695.4  optimal
    3  23593.9  optimal
    """
    one_zone(p, "eval_rank_importance")
    from ._milp import OptimizationProblem
    from ._milp_compile import compile_problem

    u = p.units
    s = _values(p, solution)
    sel = _selected(s)
    if (n is None) == (budgets is None):
        raise ValueError("give either n or budgets")
    kind = p.objective[0] if p.objective else None
    if kind is None:
        raise ValueError("the problem has no objective")
    if budgets is None:
        if int(n) != n or n < 1:
            raise ValueError("n must be a whole number >= 1")
        total_cost = _r_sum(u.cost * s)
        locked = p._unit_bounds()[0]  # the least share of every unit (locked in, manual bounds)
        min_cost = _r_sum(u.cost * locked)
        budgets = _r_seq(min_cost, total_cost, int(n) + 1)[1:]
    budgets = np.asarray(budgets, dtype=np.float64).ravel()
    if kind == "min_set":
        q = p._with(objective=("min_shortfall", float(budgets[0])))
    elif len(p.objective) > 1 and isinstance(p.objective[1], float):
        q = p._with(objective=(kind, float(budgets[0]), *p.objective[2:]))
    else:
        raise ValueError(f"the {kind} objective has no budget")
    m, _ = compile_problem(q)
    rows = [k for k, r in enumerate(m.row_ids) if r == "budget"]
    if len(rows) != 1:
        raise ValueError("the compiled problem has no budget row")
    lb, ub, rhs = m.lb.copy(), m.ub.copy(), m.rhs.copy()
    out_idx = np.setdiff1d(np.arange(u.n), sel)
    lb[out_idx] = 0.0
    ub[out_idx] = 0.0
    solver, opts = q.solver
    ranks = np.zeros(sel.size)
    prev = None
    info = []
    for i, b in enumerate(budgets):
        rhs[rows[0]] = b
        if prev is not None:
            chosen = np.flatnonzero(prev[:u.n] > 1e-10)
            lb[chosen] = prev[chosen]
        mm = OptimizationProblem(A=m.A, obj=m.obj, lb=lb.copy(), ub=ub, rhs=rhs.copy(), sense=m.sense,
                                 vtype=m.vtype, modelsense=m.modelsense, row_ids=m.row_ids, col_ids=m.col_ids)
        r = _solve_values(mm, solver, opts, u.n, start=prev)
        if not r["has_solution"]:
            raise RuntimeError("no solution: the targets, budgets or constraints cannot be met")
        info.append({"budget": float(b), "objective": float(r["objective"]), "status": r["status"],
                     "gap": float(r["gap"])})
        prev = np.asarray(r["x"], dtype=np.float64)
        rank = len(budgets) - i
        ranks[(ranks < 1e-10) & (prev[sel] > 1e-10)] = rank
    if rescale:
        ranks = _rescale(ranks)
    res = np.zeros(u.n)
    res[sel] = ranks
    return _per_unit(p, res, ["rs"]), pd.DataFrame(info)


# ----------------------------------------------------------------------------- the solution as a landscape

def _selection_map(solution) -> xr.DataArray:
    """A raster solution (a Solution on raster units, or its selection) as 1 selected, 0 not, NaN outside."""
    from ._milp_solve import Solution

    sel = solution.selection if isinstance(solution, Solution) else solution
    if not isinstance(sel, xr.DataArray) or sel.ndim not in (2, 3):
        raise ValueError("the landscape of a solution needs raster planning units")
    if sel.ndim == 3:
        sel = sel.squeeze("band", drop=True)
    v = np.asarray(sel.values, dtype=np.float64)
    return sel.copy(data=np.where(np.isfinite(v), (v > 0.5).astype(np.float64), np.nan))


def eval_landscape_metrics(solution, level: str | None = "class", metric=None, **kw) -> pd.DataFrame:
    """Landscape metrics of a raster solution (:func:`landschaft.lsm.landscape_metrics` of the map of
    the selected cells: class 1 selected, 0 not, outside the planning units no data): e.g. the number,
    sizes and shapes of the selected patches. Class-level and patch-level rows of the selected class only.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> P.eval_landscape_metrics(sol, metric=["np", "ca"])[["metric", "value"]]
      metric    value
    0     ca  53400.0
    1     np     13.0
    """
    one_zone(solution, "eval_landscape_metrics")
    from .. import lsm

    m = _selection_map(solution)
    t = lsm.landscape_metrics(m, level=level, metric=metric, nodata=np.nan, **kw)
    if "class" in t.columns and level in ("class", "patch"):
        t = t[t["class"] == 1].reset_index(drop=True)
    return t


def eval_mspa(solution, **kw):
    """MSPA (:func:`landschaft.mspa`) of a raster solution: the selected cells as foreground, the other
    planning units as background, the cells outside them missing. Returns (the MSPA map, its class
    areas).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> sol = p.solve()
    >>> mspa, areas = P.eval_mspa(sol)
    >>> int(areas.loc["Core", "pixels"]) >= 0
    True
    """
    one_zone(solution, "eval_mspa")
    import landschaft as ls

    m = _selection_map(solution)
    v = m.values
    data = m.copy(data=np.where(np.isnan(v), 0, np.where(v > 0.5, 2, 1)).astype(np.uint8))
    res = ls.mspa(data, **kw)
    return res, ls.mspa_stats(res)
