# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluating solutions of problems with management zones (prioritizr 8.0.6's ``eval_*`` with zones):
the summaries have a row ``overall`` and one per zone, the same sums in the same order as prioritizr's
(R's 80-bit sums, Matrix's, Armadillo's), bit for bit; the replacement and rank importance re-solve the
problem for every selected unit and zone (in parallel)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr


def _r_sum(v) -> float:
    from ._milp import _r_sum as rs

    return rs(np.asarray(v, dtype=np.float64))


def zone_values(p, solution) -> np.ndarray:
    """The solution's value in every planning unit and zone (units x zones), NaN as 0: a Solution, a raster with
    a layer per zone, a table with a column ``solution_<zone>`` (or prioritizr's ``solution_1_<zone>``) per zone,
    a mapping zone -> values, or an array (units x zones, or every cell / row x zones)."""
    from collections.abc import Mapping

    from ._milp_solve import Solution
    from ._zones import zone_key

    u = p.units
    n, nz = u.n, u.nz
    total = u.template.size if u.kind == "raster" else len(u.template)
    if isinstance(solution, Solution):
        v = np.asarray(solution.values, dtype=np.float64)
    elif isinstance(solution, xr.DataArray):
        if solution.ndim != 3 or solution.shape[0] != nz or u.kind != "raster":
            raise ValueError(f"a raster solution with zones needs a layer per zone ({nz}) on the units' grid")
        v = np.asarray(solution.values, dtype=np.float64).reshape(nz, -1)[:, u.index].T
    elif isinstance(solution, pd.DataFrame):
        cols = []
        for z in u.zone_names:
            for name in (f"solution_{z}", f"solution_1_{z}"):
                if name in solution.columns:
                    cols.append(name)
                    break
            else:
                raise ValueError(f"the table solution needs a column solution_{z}")
        v = solution[cols].to_numpy(dtype=np.float64)
        if v.shape[0] == len(u.template) and u.kind != "raster":
            v = v[u.index]
    elif isinstance(solution, Mapping):
        v = np.zeros((n, nz))
        for z, vals in solution.items():
            a = np.asarray(vals, dtype=np.float64).ravel()
            v[:, zone_key(u, z, single=True)[0]] = a[u.index] if a.size == total and total != n else a
    else:
        v = np.asarray(solution, dtype=np.float64)
        if v.ndim == 2 and v.shape == (total, nz) and total != n:
            v = v[u.index]
    if v.shape != (n, nz):
        raise ValueError(f"the solution needs units x zones values ({n} x {nz}), not {v.shape}")
    return np.where(np.isfinite(v), v, 0.0)


def _summary(u, overall: float, per_zone, name: str) -> pd.DataFrame:
    return pd.DataFrame({"summary": ["overall", *u.zone_names], name: [overall, *per_zone]})


def n_summary(p, solution) -> pd.DataFrame:
    s = zone_values(p, solution)
    return _summary(p.units, _r_sum(s.T.ravel()), [_r_sum(s[:, z]) for z in range(p.units.nz)], "n")


def cost_summary(p, solution) -> pd.DataFrame:
    u = p.units
    s = zone_values(p, solution)
    c = u.cost.reshape(u.nz, u.n).T
    costs = []
    for z in range(u.nz):
        prod = c[:, z] * s[:, z]
        costs.append(_r_sum(prod[np.isfinite(prod)]))  # colSums(cost * status, na.rm = TRUE)
    return _summary(u, _r_sum(np.array(costs)), costs, "cost")


def _boundary_parts(p, data):
    """The boundary matrix of the units (rounded to 6 decimals), its lower triangle, the exposed and total
    boundary of every unit."""
    from ._milp_eval import _csc, _seq, _unit_matrix
    from ._milp_spatial import r_round
    from ._zones_spatial import boundary_matrix

    u = p.units
    if data is None:
        if u.kind == "table" and not hasattr(u.template, "geometry"):
            raise ValueError("table planning units need the boundary matrix (data)")
        bm = _csc(boundary_matrix(u))
    else:
        bm = _unit_matrix(p, data)
    bm = bm.copy()
    bm.data = r_round(bm.data, 6)
    total = bm.diagonal()
    rows = bm.tocsr()
    rows.sort_indices()
    row_sums = np.array([_seq(rows.data[rows.indptr[i]:rows.indptr[i + 1]]) for i in range(u.n)])
    exposed = total - (row_sums - total)
    lower = sp.tril(bm, k=-1).tocsc()
    lower.eliminate_zeros()
    lower.sort_indices()
    return lower, exposed, total


def boundary_summary(p, solution, edge_factor, zones, data) -> pd.DataFrame:
    from .. import _core
    from ._zones_spatial import per_zone, zone_matrix

    u = p.units
    s = zone_values(p, solution)
    ef = per_zone(u, edge_factor, "edge_factor")
    if ((ef < 0) | (ef > 1)).any():
        raise ValueError("edge_factor must be between 0 and 1")
    zm = zone_matrix(u, zones, "eval_boundary_summary")
    lower, exposed, total = _boundary_parts(p, data)
    args = (lower.indptr.astype(np.int64), lower.indices.astype(np.int64), lower.data, exposed, total)
    overall = _core.opt_eval_boundary_zones(ef, zm.ravel(), *args, np.ascontiguousarray(s.T.ravel()))
    per = [_core.opt_eval_boundary(float(ef[z]), *args, np.ascontiguousarray(s[:, z])) for z in range(u.nz)]
    return _summary(u, overall, per, "boundary")


def connectivity_summary(p, solution, data, zones, asym: bool) -> pd.DataFrame:
    import warnings

    from .. import _core
    from ._milp_eval import _unit_matrix
    from ._zones_spatial import zone_matrix

    u = p.units
    s = zone_values(p, solution)
    zm = zone_matrix(u, zones, "eval_connectivity_summary", symmetric=False)
    m = _unit_matrix(p, data)
    symmetric = (abs(m - m.T) > 0).nnz == 0
    if symmetric and asym:
        warnings.warn("data contains symmetric connectivity values; eval_connectivity_summary is recommended",
                      UserWarning, stacklevel=3)
    if not symmetric and not asym:
        warnings.warn("data contains asymmetric connectivity values; eval_asym_connectivity_summary is "
                      "recommended", UserWarning, stacklevel=3)
    args = (m.indptr.astype(np.int64), m.indices.astype(np.int64), m.data)
    overall = _core.opt_eval_connectivity_zones(*args, zm.ravel(), np.ascontiguousarray(s.T.ravel()), asym)
    per = [_core.opt_eval_connectivity(*args, np.ascontiguousarray(s[:, z]), asym) for z in range(u.nz)]
    return _summary(u, overall, per, "asym_connectivity" if asym else "connectivity")


def _held(m: sp.csr_matrix, s: np.ndarray) -> np.ndarray:
    """rij %*% s by Matrix: every feature's units in order, the solution's zeros left out."""
    out = np.zeros(m.shape[0])
    for f in range(m.shape[0]):
        lo, hi = m.indptr[f], m.indptr[f + 1]
        k = m.indices[lo:hi]
        keep = s[k] != 0
        v = m.data[lo:hi][keep] * s[k][keep]
        out[f] = float(np.cumsum(v)[-1]) if v.size else 0.0
    return out


def feature_representation_summary(p, solution) -> pd.DataFrame:
    u = p.units
    s = zone_values(p, solution)
    nf, nz = len(u.feature_names), u.nz
    held = np.stack([_held(sp.csr_matrix(u.zone_rij[z]), s[:, z]) for z in range(nz)], axis=1)  # nf x nz
    total = np.asarray(u.totals, dtype=np.float64)
    tot = np.concatenate([[_r_sum(total[f]) for f in range(nf)], total.T.ravel()])  # rowSums, then by zone
    hel = np.concatenate([[_r_sum(held[f]) for f in range(nf)], held.T.ravel()])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = hel / tot
    return pd.DataFrame({"summary": np.repeat(["overall", *u.zone_names], nf),
                         "feature": list(u.feature_names) * (nz + 1), "total_amount": tot, "absolute_held": hel,
                         "relative_held": rel})


def target_coverage_summary(p, solution, include_sense: bool, include_zone: bool) -> pd.DataFrame:
    from .. import _core
    from ._milp_compile import NO_TARGETS, compile_problem

    if (p.objective is not None and p.objective[0] in NO_TARGETS) or (
            p.targets is None and not p.zone_targets and p.subregional is None):
        raise ValueError("the problem has no targets: use eval_feature_representation_summary")
    u = p.units
    _, t = compile_problem(p)
    s = zone_values(p, solution)
    n = u.n
    rows = [sp.csr_matrix(u.zone_rij[z]) for z in range(u.nz)]
    unit_codes = None
    if t.region is not None and (t.region >= 0).any():
        from ._zones import resolve_zone_subregional

        _, unit_codes = resolve_zone_subregional(p)
    held = np.zeros(t.n)
    for k in range(t.n):  # every zone of the target in turn, Armadillo's accu of the non-zero products
        f = t.feature[k]
        for z in t.zone[k]:
            m = rows[z]
            lo, hi = m.indptr[f], m.indptr[f + 1]
            idx, val = m.indices[lo:hi], m.data[lo:hi]
            if unit_codes is not None and t.region[k] >= 0:
                keep = unit_codes[idx] == t.region[k]
                idx, val = idx[keep], val[keep]
            prod = val * s[idx, z]
            prod = np.ascontiguousarray(prod[prod != 0])
            held[k] += _core.opt_eval_accumulate(prod)
    del n
    total = np.asarray(t.total, dtype=np.float64)
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
    if include_zone:
        out["zone"] = list(t.zone_label)
    if include_sense:
        out["sense"] = list(sense)
    out.update({"met": short < 1e-10, "total_amount": total, "absolute_target": target, "absolute_held": held,
                "absolute_shortfall": short, "relative_target": rt, "relative_held": rh, "relative_shortfall": rs})
    df = pd.DataFrame(out)
    if t.region is not None and (t.region >= 0).any():
        df.insert(1, "region", list(t.region_label))
    return df


def per_unit_zones(p, values: np.ndarray, prefix: str):
    """Values per unit and zone (units x zones) like a solution's selection: a raster with a layer per zone
    or the units' table with a column ``<prefix>_<zone>`` per zone."""
    u = p.units
    closed = ~np.isfinite(u.cost.reshape(u.nz, u.n).T)
    v = np.where(closed, np.nan, values)
    if u.kind == "raster":
        t = u.template
        out = np.full((u.nz, t.size), np.nan)
        out[:, u.index] = v.T
        da = xr.DataArray(out.reshape(u.nz, *t.shape), dims=("zone", *t.dims),
                          coords={"zone": list(u.zone_names), **{k: c for k, c in t.coords.items() if k in t.dims}},
                          name=prefix)
        crs = t.rio.crs
        return da.rio.write_crs(crs) if crs is not None else da
    tab = u.template.copy()
    for k, z in enumerate(u.zone_names):
        col = np.full(len(tab), np.nan)
        col[u.index] = v[:, k]
        tab[f"{prefix}_{z}"] = col
    return tab


def replacement_importance(p, solution, rescale: bool, n_threads):
    """prioritizr's replacement importance with zones: every selected (unit, zone) column left out in turn
    (the unit may then go to another zone or none), the other selected columns kept, the problem solved again."""
    from concurrent.futures import ThreadPoolExecutor

    from .. import _threads
    from ._milp import OptimizationProblem
    from ._milp_compile import compile_problem
    from ._milp_eval import _rescale
    from ._milp_solve import solve_model

    u = p.units
    s = zone_values(p, solution)
    x = s.T.ravel()
    sel = np.flatnonzero(x > 1e-10)
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

    zkw = {"n_units": None, "engines": False, "zone_units": u.n}
    base = solve_model(model(lb, ub), solver, opts, **zkw)
    if not base["has_solution"]:
        raise ValueError("the solution is not feasible for the problem")
    one = {**opts, "n_threads": 1}

    def alt(j):
        lo, hi = lb.copy(), ub.copy()
        lo[j] = hi[j] = 0.0
        r = solve_model(model(lo, hi), solver, one, **zkw)
        return float(r["objective"]) if r["has_solution"] else math.inf

    with ThreadPoolExecutor(max_workers=_threads.resolve(n_threads)) as ex:
        objs = np.array(list(ex.map(alt, sel)))
    out = objs - float(base["objective"])
    if m.modelsense == "max":
        out = out * -1
    if rescale:
        out = _rescale(out)
    res = np.zeros(u.cost.size)
    res[sel] = out
    return per_unit_zones(p, res.reshape(u.nz, u.n).T, "rc")


def rank_importance(p, solution, n, budgets, rescale: bool, by_zone: bool):
    """Rank importance with zones: increasing budgets (one per zone with ``by_zone``: every zone's cost in
    equal steps from its locked units' cost to the solution's), only the solution's (unit, zone) columns
    allowed, every step keeping the columns of the step before; a column's rank is how early it enters."""
    from ._milp import OptimizationProblem
    from ._milp_compile import compile_problem
    from ._milp_eval import _r_seq, _rescale
    from ._milp_solve import solve_model

    u = p.units
    s = zone_values(p, solution)
    x = s.T.ravel()
    sel = np.flatnonzero(x > 1e-10)
    kind = p.objective[0] if p.objective else None
    if kind is None:
        raise ValueError("the problem has no objective")
    if (n is None) == (budgets is None):
        raise ValueError("give either n or budgets")
    cost = np.where(np.isfinite(u.cost), u.cost, 0.0)
    locked = p._unit_bounds()[0]
    nz = u.nz
    if budgets is None:
        if int(n) != n or n < 1:
            raise ValueError("n must be a whole number >= 1")
        if by_zone:
            cz = (cost * x).reshape(nz, u.n)
            lz = (cost * locked).reshape(nz, u.n)
            steps = np.stack([_r_seq(_r_sum(lz[z]), _r_sum(cz[z]), int(n) + 1)[1:] for z in range(nz)], axis=1)
        else:
            steps = _r_seq(_r_sum(cost * locked), _r_sum(cost * x), int(n) + 1)[1:, None]
    else:
        steps = np.asarray(budgets, dtype=np.float64)
        steps = steps[:, None] if steps.ndim == 1 else steps
        if steps.shape[1] not in (1, nz):
            raise ValueError(f"budgets: one per step, or steps x zones ({nz})")
    first = tuple(float(v) for v in steps[0]) if steps.shape[1] > 1 else float(steps[0, 0])
    if kind == "min_set":
        q = p._with(objective=("min_shortfall", first))
    elif len(p.objective) > 1 and isinstance(p.objective[1], (float, tuple)):
        q = p._with(objective=(kind, first, *p.objective[2:]))
    else:
        raise ValueError(f"the {kind} objective has no budget")
    m, _ = compile_problem(q)
    rows = [k for k, r in enumerate(m.row_ids) if r == "budget"]
    if len(rows) != steps.shape[1]:
        raise ValueError("the compiled problem's budget rows do not match the budgets")
    lb, ub, rhs = m.lb.copy(), m.ub.copy(), m.rhs.copy()
    out_idx = np.setdiff1d(np.arange(u.cost.size), sel)
    lb[out_idx] = 0.0
    ub[out_idx] = 0.0
    solver, opts = q.solver
    ranks = np.zeros(sel.size)
    prev = None
    info = []
    for i, b in enumerate(steps):
        rhs[rows] = b
        if prev is not None:
            chosen = np.flatnonzero(prev[:u.cost.size] > 1e-10)
            lb[chosen] = prev[chosen]
        mm = OptimizationProblem(A=m.A, obj=m.obj, lb=lb.copy(), ub=ub, rhs=rhs.copy(), sense=m.sense, vtype=m.vtype,
                                 modelsense=m.modelsense, row_ids=m.row_ids, col_ids=m.col_ids)
        r = solve_model(mm, solver, opts, prev, n_units=None, engines=False, zone_units=u.n)
        if not r["has_solution"]:
            raise RuntimeError("no solution: the targets, budgets or constraints cannot be met")
        row = {"budget": float(np.sum(b))} if steps.shape[1] == 1 else {
            f"budget_{z}": float(b[k]) for k, z in enumerate(u.zone_names)}
        info.append({**row, "objective": float(r["objective"]), "status": r["status"], "gap": float(r["gap"])})
        prev = np.asarray(r["x"], dtype=np.float64)
        ranks[(ranks < 1e-10) & (prev[sel] > 1e-10)] = len(steps) - i
    if rescale:
        ranks = _rescale(ranks)
    res = np.zeros(u.cost.size)
    res[sel] = ranks
    return per_unit_zones(p, res.reshape(nz, u.n).T, "rs"), pd.DataFrame(info)
