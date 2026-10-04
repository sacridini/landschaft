# SPDX-License-Identifier: GPL-3.0-or-later
"""The two paradigms side by side: a Zonation ranking cut at the cost of an optimization solution, and
how much of every feature each keeps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .._save import Saveable

__all__ = ["PrioritizationReport", "ZonationComparison", "coarsen", "compare_zonation", "refine", "report",
           "sdm_features"]


@dataclass
class ZonationComparison(Saveable):
    """A Zonation ranking and an optimization solution with the same budget.

    Attributes
    ----------
    table
        Every feature's share held by the solution (``milp``) and by Zonation's best cells up to the same
        cost (``zonation``), with its target and whether each meets it (with targets).
    summary
        Cost, units and boundary of each.
    selection
        The two selections (band ``milp`` and ``zonation``: 1 selected, 0 not, NaN outside the units).
    """

    table: pd.DataFrame = field(repr=False)
    summary: pd.DataFrame = field(repr=False)
    selection: xr.DataArray = field(repr=False)
    solution: object = field(repr=False)
    zonation: object = field(repr=False)

    def __repr__(self) -> str:
        m, z = self.table["milp"].mean(), self.table["zonation"].mean()
        return f"ZonationComparison(mean share held: milp {m:.3f}, zonation {z:.3f})"

    def _save_items(self) -> dict:
        return {"table": self.table, "summary": self.summary, "selection": self.selection}


def _zonation_selection(p, rank_values: np.ndarray, budget: float) -> np.ndarray:
    """The units from the highest rank down while their cost fits the budget (locked-in units first,
    locked-out never)."""
    u = p.units
    li, lo = p._mask(p.locked_in), p._mask(p.locked_out)
    r = np.where(np.isfinite(rank_values), rank_values, -np.inf)
    order = np.lexsort((np.arange(u.n), -r))  # highest rank first, then the unit order
    order = np.concatenate([order[li[order]], order[~li[order] & ~lo[order]]])
    take = np.zeros(u.n)
    spent = 0.0
    for i in order:
        c = u.cost[i]
        if li[i] or spent + c <= budget * (1 + 1e-12):
            take[i] = 1.0
            spent += c
    return take


def compare_zonation(p, solution=None, zonation=None, rule: str = "CAZ2", n_threads: int | None = None,
                     **zonation_kw) -> ZonationComparison:
    """Compare an optimization solution of ``p`` (solved if not given) with a Zonation ranking of the
    same features and costs (run with ``rule`` and ``zonation_kw`` if not given: a
    :class:`ZonationResult` or a rank raster) cut at the solution's cost. Raster planning units."""
    from ._milp_eval import _values, eval_boundary_summary, eval_feature_representation_summary
    from ._zonation import ZonationResult, zonation as run_zonation

    u = p.units
    if u.kind != "raster":
        raise ValueError("compare_zonation needs raster planning units")
    sol = p.solve() if solution is None else solution
    s = _values(p, sol)
    budget = float(np.dot(u.cost, s))
    if zonation is None:
        zonation = run_zonation(u.features, cost=u.template, rule=rule, n_threads=n_threads, **zonation_kw)
    rank = zonation.rank if isinstance(zonation, ZonationResult) else zonation
    rank = rank.squeeze("band", drop=True) if isinstance(rank, xr.DataArray) and rank.ndim == 3 else rank
    rv = np.asarray(rank.values if isinstance(rank, xr.DataArray) else rank, dtype=np.float64).ravel()
    if rv.size != u.template.size:
        raise ValueError("the rank raster must be on the planning units' grid")
    z = _zonation_selection(p, rv[u.index], budget)
    rm = eval_feature_representation_summary(p, s)
    rz = eval_feature_representation_summary(p, z)
    table = pd.DataFrame({"feature": rm["feature"], "milp": rm["relative_held"], "zonation": rz["relative_held"]})
    if p.targets is not None:
        from ._milp_compile import NO_TARGETS

        if p.objective is None or p.objective[0] not in NO_TARGETS:
            from ._milp_eval import eval_target_coverage_summary

            cm, cz = eval_target_coverage_summary(p, s), eval_target_coverage_summary(p, z)
            if len(cm) == len(table):
                table["target"] = cm["relative_target"].to_numpy()
                table["milp_met"] = cm["met"].to_numpy()
                table["zonation_met"] = cz["met"].to_numpy()
    rows = []
    for name, v in (("milp", s), ("zonation", z)):
        rows.append({"method": name, "cost": float(np.dot(u.cost, v)), "units": float(v.sum()),
                     "boundary": float(eval_boundary_summary(p, v)["boundary"].iloc[0]),
                     "mean_share_held": float(np.nanmean(rm["relative_held"] if name == "milp" else rz["relative_held"]))})
    t = u.template
    out = np.full((2, t.size), np.nan)
    out[0, u.index] = s
    out[1, u.index] = z
    sel = xr.DataArray(out.reshape((2, *t.shape)), dims=("band", *t.dims),
                       coords={"band": ["milp", "zonation"], **{k: v for k, v in t.coords.items() if k in t.dims}})
    try:
        if t.rio.crs is not None:
            sel = sel.rio.write_crs(t.rio.crs)
    except Exception:  # noqa: BLE001 - no georeferencing
        pass
    sel.name = "selection"
    return ZonationComparison(table, pd.DataFrame(rows), sel, sol, zonation)


@dataclass
class PrioritizationReport(Saveable):
    """A solution and its evaluation in one object: ``ls.plot(report)`` draws it, ``ls.plot(report).save(
    "report.html")`` writes it as a page that works offline, ``ls.save(report, folder)`` its tables and maps."""

    solution: object = field(repr=False)
    tables: dict = field(repr=False)
    maps: dict = field(repr=False)
    comparison: ZonationComparison | None = field(default=None, repr=False)
    title: str = "Prioritization"

    def _save_items(self) -> dict:
        out = {k.replace(" ", "_"): v for k, v in self.tables.items()}
        out.update({k.replace(" ", "_"): v for k, v in self.maps.items()})
        out["selection"] = self.solution.selection
        if self.comparison is not None:
            out["zonation_comparison"] = self.comparison.table
        return out


def report(p, solution=None, *, importance: bool = True, zonation: bool = False, extinction_z=None,
           title: str = "Prioritization", n_threads: int | None = None) -> PrioritizationReport:
    """Solve ``p`` (unless a solution is given) and evaluate the solution: the number of units, cost,
    boundary (raster and polygon units), the features' representation and the targets' coverage; with
    ``importance`` the Ferrier irreplaceability of the selected units (problems with one target per
    feature) or else their rarity-weighted richness; with ``extinction_z`` the species-area extinction
    risk; with ``zonation`` the comparison with a Zonation ranking at the same cost (raster units)."""
    from ._milp_compile import NO_TARGETS
    from ._milp_eval import (eval_boundary_summary, eval_cost_summary, eval_extinction_risk,
                             eval_feature_representation_summary, eval_ferrier_importance, eval_n_summary,
                             eval_rare_richness_importance, eval_target_coverage_summary)

    sol = p.solve() if solution is None else solution
    u = p.units
    summary = {"units": float(eval_n_summary(p, sol)["n"].iloc[0]),
               "cost": float(eval_cost_summary(p, sol)["cost"].iloc[0])}
    if u.kind == "raster" or hasattr(u.template, "geometry"):  # tables without geometry have no boundary
        summary["boundary"] = float(eval_boundary_summary(p, sol)["boundary"].iloc[0])
    tables = {"summary": pd.DataFrame([summary]),
              "representation": eval_feature_representation_summary(p, sol)}
    has_targets = (p.targets is not None or p.subregional is not None) and (
        p.objective is None or p.objective[0] not in NO_TARGETS)
    if has_targets:
        tables["targets"] = eval_target_coverage_summary(p, sol)
    if extinction_z is not None:
        tables["extinction risk"] = eval_extinction_risk(p, sol, extinction_z)
    maps = {}
    if importance and np.nansum(np.asarray(sol.values)) > 0:
        try:
            imp = eval_ferrier_importance(p, sol)
            maps["irreplaceability"] = imp.sel(band="total") if isinstance(imp, xr.DataArray) else imp
        except ValueError:
            imp = eval_rare_richness_importance(p, sol)
            maps["rarity-weighted richness"] = imp
    comp = compare_zonation(p, sol, n_threads=n_threads) if zonation else None
    return PrioritizationReport(sol, tables, maps, comp, title)


def sdm_features(results, env, *, threshold=None, ensemble: str = "weighted") -> xr.DataArray:
    """Features from species distribution models (``ls.sdm``): every result's prediction on ``env``
    (its ensemble when it has several models, else its model), as a stack for :func:`problem`.

    Parameters
    ----------
    results
        ``{name: SDMResult}`` or a list of them (named ``species_1``, ...).
    env
        The predictors (as for :func:`landschaft.sdm.predict`).
    threshold
        ``None``: the suitability (0-1) is the amount; a number, or ``"max_tss"`` (every model's
        maximum-TSS threshold, their mean for an ensemble): presence (1) or absence (0)."""
    from .. import sdm

    items = list(results.items()) if isinstance(results, dict) else [
        (f"species_{k + 1}", r) for k, r in enumerate(results)]
    if not items:
        raise ValueError("no results")
    layers, names = [], []
    for name, res in items:
        pred = sdm.predict(res, env, ensemble=ensemble, sd=False)
        bands = [str(b) for b in pred["band"].values]
        band = "ensemble" if "ensemble" in bands else bands[0]
        v = pred.sel(band=band).drop_vars("band")
        if threshold is not None:
            if isinstance(threshold, str):
                if threshold != "max_tss":
                    raise ValueError("threshold must be a number or 'max_tss'")
                ts = res.thresholds
                t = float(np.mean([ts[m] for m in ts])) if band == "ensemble" else float(ts[band])
            else:
                t = float(threshold)
            v = xr.where(np.isfinite(v), (v >= t).astype(np.float64), np.nan)
        layers.append(v)
        names.append(str(name))
    out = xr.concat(layers, dim="band").assign_coords(band=names)
    try:
        crs = layers[0].rio.crs
        if crs is not None:
            out = out.rio.write_crs(crs)
    except Exception:  # noqa: BLE001 - no georeferencing
        pass
    out.name = "features"
    return out


def _block_sum(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Sums over blocks of factor x factor cells of the last two dimensions (y, x), NaN where a block has
    no value; the blocks' coordinates are the means of their cells' (the edges padded)."""
    v = np.asarray(da.values, dtype=np.float64)
    ny, nx = v.shape[-2:]
    by, bx = -(-ny // factor), -(-nx // factor)
    pad = [(0, 0)] * (v.ndim - 2) + [(0, by * factor - ny), (0, bx * factor - nx)]
    vp = np.pad(v, pad, constant_values=np.nan)
    blocks = vp.reshape(*v.shape[:-2], by, factor, bx, factor)
    with np.errstate(invalid="ignore"):
        total = np.nansum(blocks, axis=(-3, -1))
        count = np.sum(np.isfinite(blocks), axis=(-3, -1))
    total = np.where(count > 0, total, np.nan)
    ydim, xdim = da.dims[-2:]

    def centres(c):
        c = np.asarray(c, dtype=np.float64)
        step = c[1] - c[0] if c.size > 1 else 1.0
        full = np.concatenate([c, c[-1] + step * np.arange(1, -(-c.size // factor) * factor - c.size + 1)])
        return full.reshape(-1, factor).mean(axis=1)

    coords = {d: da.coords[d].values for d in da.dims[:-2] if d in da.coords}
    coords[ydim] = centres(da[ydim].values)
    coords[xdim] = centres(da[xdim].values)
    out = xr.DataArray(total, dims=da.dims, coords=coords, name=da.name)
    try:
        if da.rio.crs is not None:
            out = out.rio.write_crs(da.rio.crs)
    except Exception:  # noqa: BLE001 - no georeferencing
        pass
    return out


def coarsen(cost: xr.DataArray, features, factor: int):
    """A problem's rasters on blocks of ``factor`` x ``factor`` cells (for continental problems: solve
    the coarse problem, then :func:`refine` the selection): every block's cost and amounts are the sums
    of its cells' with a cost (the fine planning units; a block without any has none), so the refined
    selection holds what the coarse one does. Returns (cost, features)."""
    from ._milp import _layers

    if int(factor) != factor or factor < 1:
        raise ValueError("factor must be a whole number >= 1")
    factor = int(factor)
    c = cost.squeeze("band", drop=True) if cost.ndim == 3 else cost
    layers = _layers(features)
    stack = xr.concat([f.drop_vars([v for v in f.coords if v not in f.dims], errors="ignore") for _, f in layers],
                      dim="band").assign_coords(band=[n for n, _ in layers])
    stack = stack.where(np.isfinite(np.asarray(c.values, dtype=np.float64)))  # the planning units' amounts only
    return _block_sum(c, factor), _block_sum(stack, factor)


def refine(selection, like: xr.DataArray) -> xr.DataArray:
    """A selection on coarse blocks (a :class:`Solution` of a :func:`coarsen`-ed problem, or its selection
    raster) back on the fine grid of ``like`` (the fine cost raster): every cell takes its block's value,
    NaN where ``like`` has none."""
    from ._milp_solve import Solution

    sel = selection.selection if isinstance(selection, Solution) else selection
    sel = sel.squeeze("band", drop=True) if sel.ndim == 3 else sel
    fine = like.squeeze("band", drop=True) if like.ndim == 3 else like
    fy, fx = fine.shape
    cy, cx = sel.shape
    factor_y, factor_x = -(-fy // cy), -(-fx // cx)
    if cy * factor_y < fy or cx * factor_x < fx or (cy - 1) * factor_y >= fy or (cx - 1) * factor_x >= fx:
        raise ValueError("the selection is not a coarsened version of this grid")
    v = np.kron(np.asarray(sel.values, dtype=np.float64), np.ones((factor_y, factor_x)))[:fy, :fx]
    v = np.where(np.isfinite(np.asarray(fine.values, dtype=np.float64)), v, np.nan)
    out = fine.copy(data=v)
    out.name = "solution"
    return out
