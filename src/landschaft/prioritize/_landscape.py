# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape identification and comparison over any priority surface (prioripy's
``eval_landscape_identification`` / ``eval_landscape_comparison``, after Zonation's LSI and LSC).

The top fraction of a surface follows Zonation's rule: the cells are ranked by position (the lowest value
first; ties: the earlier cell in row order first, as a stable sort), the rank of the i-th of N cells is
float32(i / N), and the top fraction ``f`` is the cells ranked at or above 1 - f (compared in double).
On a Zonation rank this gives the ranking's own ranks, so the areas are those of
:meth:`ZonationResult.area_statistics_rank_range` and ``area_statistics_separate_areas``, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .._save import Saveable

__all__ = ["LandscapeAreas", "LandscapeComparison", "eval_landscape_comparison", "eval_landscape_identification",
           "position_rank"]


def _surface(x) -> tuple[np.ndarray, xr.DataArray | None]:
    """The surface's values (2-D, NaN outside) and its raster (None for an array)."""
    from ._milp_portfolio import Portfolio
    from ._milp_solve import Solution
    from ._zonation import ZonationResult

    if isinstance(x, ZonationResult):
        x = x.rank
    elif isinstance(x, Portfolio):
        x = x.frequency
    elif isinstance(x, Solution):
        x = x.selection
    if isinstance(x, xr.DataArray):
        if x.ndim == 3:
            if x.sizes["band"] == 1:
                x = x.isel(band=0, drop=True)
            elif "total" in [str(b) for b in x["band"].values]:  # eval_ferrier_importance: its total
                x = x.sel(band="total", drop=True)
            else:
                raise ValueError("the surface has several bands: select one (surface.sel(band=...))")
        if x.ndim != 2:
            raise ValueError("the surface must be a raster (2-D)")
        return np.asarray(x.values, dtype=np.float64), x
    if isinstance(x, pd.DataFrame):
        raise TypeError("landscape identification needs a raster surface (raster planning units)")
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("the surface must be a raster or a 2-D array")
    return v, None


def position_rank(values) -> np.ndarray:
    """Zonation's ranks of a surface by position: the i-th lowest of the N finite cells gets
    float32(i * (1 / N)); ties go to the earlier cell in row order first; NaN stays NaN.

    On a Zonation rank map this returns the rank map itself, bit for bit.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> ls.prioritize.position_rank(np.array([[3.0, 1.0], [np.nan, 1.0]])).tolist()
    [[0.6666666865348816, 0.0], [nan, 0.3333333432674408]]
    """
    v = np.asarray(values, dtype=np.float64)
    flat = v.ravel()
    ok = np.flatnonzero(np.isfinite(flat))
    n = ok.size
    out = np.full(flat.size, np.nan, dtype=np.float32)
    if n:
        order = ok[np.argsort(flat[ok], kind="stable")]
        ndr = 1.0 / float(n)
        out[order] = (np.arange(n, dtype=np.float64) * ndr).astype(np.float32)
    return out.reshape(v.shape)


def _top(rank: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("a top fraction must be within (0, 1]")
    r = rank.astype(np.float64)
    return np.isfinite(r) & (r >= 1.0 - float(fraction)) & (r <= 1.0)


def _like(template: xr.DataArray | None, values: np.ndarray, name: str):
    if template is None:
        return values
    da = template.copy(data=values)
    da.name = name
    return da


@dataclass
class LandscapeAreas(Saveable):
    """The areas of the top fraction of a priority surface (:func:`eval_landscape_identification`).

    Attributes
    ----------
    areas
        The area of every cell (0: not in an area, NaN outside the surface): a raster like the surface
        (an array for an array).
    table
        Per area: ``area`` (its number), ``cells``, ``mean_value`` (the surface's mean), ``cost`` (with a
        cost) and, per feature, the amount in the area (``<feature>``) and its share of the feature's
        total (``<feature>_share``).
    top
        The cells of the top fraction (1, 0, NaN outside), before areas under ``min_size`` were dropped.
    """

    areas: object = field(repr=False)
    table: pd.DataFrame = field(repr=False)
    top: object = field(repr=False)
    top_fraction: float = 0.1

    def __repr__(self) -> str:
        return (f"LandscapeAreas(top {self.top_fraction:g}: {len(self.table)} areas, "
                f"{int(self.table['cells'].sum()) if len(self.table) else 0} cells)")

    def _save_items(self) -> dict:
        return {"areas": self.areas, "table": self.table, "top": self.top, "top_fraction": self.top_fraction}


def _feature_stack(features, problem, shape):
    """(names, (k, cells) amounts over the whole grid, totals, cost per cell or None)."""
    if problem is not None:
        u = problem.units
        if u.kind != "raster" or u.template.shape != shape:
            raise ValueError("the problem's planning units must be raster cells on the surface's grid")
        amounts = np.zeros((len(u.feature_names), int(np.prod(shape))))
        rij = u.rij.tocoo()
        amounts[rij.row, u.index[rij.col]] = rij.data
        cost = np.full(int(np.prod(shape)), np.nan)
        cost[u.index] = u.cost
        return list(u.feature_names), amounts, np.asarray(u.totals, dtype=np.float64), cost
    if features is None:
        return [], np.zeros((0, int(np.prod(shape)))), np.zeros(0), None
    if isinstance(features, xr.DataArray):
        f = features if features.ndim == 3 else features.expand_dims(band=["feature"])
        names = [str(b) for b in f["band"].values]
        v = np.asarray(f.values, dtype=np.float64)
    else:
        v = np.asarray(features, dtype=np.float64)
        if v.ndim == 2:
            v = v[None]
        names = [f"f{k + 1}" for k in range(v.shape[0])]
    if v.shape[1:] != shape:
        raise ValueError("the features must be on the surface's grid")
    v = np.nan_to_num(v.reshape(v.shape[0], -1), nan=0.0)
    totals = np.array([float(np.cumsum(row, dtype=np.float64)[-1]) if row.size else 0.0 for row in v])
    return names, v, totals, None


def eval_landscape_identification(surface, top_fraction: float = 0.1, *, min_size: int = 1,
                                  merge_distance: int = 0, connectivity: int = 8, problem=None, features=None,
                                  cost=None) -> LandscapeAreas:
    """Landscape identification: the separate areas of the top fraction of a priority surface, and what
    each holds (Zonation's area statistics of separate areas, over any surface).

    Parameters
    ----------
    surface
        The priority surface: a raster (a ranking, an importance, a selection frequency...), a
        :class:`ZonationResult` (its rank), a :class:`Portfolio` (its selection frequency), a
        :class:`Solution` (its selection) or a 2-D array; a raster of several bands needs one chosen,
        except an :func:`eval_ferrier_importance` result (its ``total``).
    top_fraction
        The share of the surface's cells taken from the top, by Zonation's rule (see Notes).
    min_size
        Areas of fewer cells are dropped.
    merge_distance
        Areas this many cells apart (dilations of the top cells) are merged into one; the area keeps
        only its own top cells (prioripy's ``merge_distance``).
    connectivity
        8 (Zonation's) or 4.
    problem
        A problem on raster planning units of the surface's grid: the features' amounts and the cost
        in every area, the shares of the features' totals (those relative targets use).
    features, cost
        Without a problem: feature rasters (bands) on the surface's grid and a cost raster.

    Returns
    -------
    LandscapeAreas

    Notes
    -----
    The cells are ranked by position as Zonation does (:func:`position_rank`; ties: the earlier cell in
    row order ranks lower) and the top fraction ``f`` is the cells ranked at or above ``1 - f``: on 600
    cells, ``f = 0.1`` takes 59 cells (rank 540/600 is just below 0.9 in single precision), as
    :meth:`ZonationResult.area_statistics_rank_range` does. Areas are numbered by their first cell in
    row order; the numbers of dropped areas are left out (as Zonation's).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    >>> ferrier = P.eval_ferrier_importance(p, p.add_highs_solver(gap=0.01).solve())
    >>> lsi = P.eval_landscape_identification(ferrier, 0.1, min_size=5, problem=p)
    >>> len(lsi.table) > 0, bool((lsi.table["cells"] >= 5).all())
    (True, True)
    """
    from scipy import ndimage

    values, template = _surface(surface)
    if int(min_size) != min_size or min_size < 1:
        raise ValueError("min_size must be a whole number >= 1")
    if int(merge_distance) != merge_distance or merge_distance < 0:
        raise ValueError("merge_distance must be a whole number >= 0")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rank = position_rank(values)
    top = _top(rank, top_fraction)
    structure = np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    work = ndimage.binary_dilation(top, structure=structure, iterations=int(merge_distance)) \
        if merge_distance else top
    labels, n = ndimage.label(work, structure=structure)
    labels[~top] = 0
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= int(min_size)
    keep[0] = False
    labels = np.where(keep[labels], labels, 0).astype(np.int32)

    names, amounts, totals, unit_cost = _feature_stack(features, problem, values.shape)
    if cost is not None and unit_cost is None:
        c = cost.values if isinstance(cost, xr.DataArray) else np.asarray(cost)
        unit_cost = np.asarray(c, dtype=np.float64).squeeze().ravel()
        if unit_cost.size != values.size:
            raise ValueError("the cost must be on the surface's grid")
    flat = labels.ravel()
    ids = np.flatnonzero(keep)
    row = np.searchsorted(ids, flat)
    sel = flat != 0
    r = row[sel]
    na = ids.size
    cells = np.bincount(r, minlength=na)
    mean = np.bincount(r, weights=values.ravel()[sel], minlength=na)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cells > 0, mean / cells, np.nan)
    table = pd.DataFrame({"area": ids.astype(np.int64), "cells": cells.astype(np.int64), "mean_value": mean})
    if unit_cost is not None:
        table["cost"] = np.bincount(r, weights=np.nan_to_num(unit_cost[sel], nan=0.0), minlength=na)
    for k, name in enumerate(names):
        held = np.bincount(r, weights=amounts[k, sel], minlength=na)
        table[name] = held
        with np.errstate(invalid="ignore", divide="ignore"):
            table[f"{name}_share"] = np.where(totals[k] != 0, held / totals[k], np.nan)
    outside = ~np.isfinite(values)
    area_map = np.where(outside, np.nan, labels.astype(np.float64))
    top_map = np.where(outside, np.nan, top.astype(np.float64))
    return LandscapeAreas(_like(template, area_map, "area"), table, _like(template, top_map, "top"),
                          float(top_fraction))


@dataclass
class LandscapeComparison(Saveable):
    """Two priority surfaces compared at several top fractions (:func:`eval_landscape_comparison`).

    Attributes
    ----------
    table
        Per fraction: the cells in the top of ``a``, of ``b``, of both, of only one and of neither, the
        Jaccard index (both / either) and the share of ``a``'s top in ``b``'s.
    maps
        Per fraction (band), Zonation's solution comparison codes: 1 in both tops, 2 only in ``a``'s,
        3 only in ``b``'s, 4 in neither, NaN outside.
    """

    table: pd.DataFrame = field(repr=False)
    maps: object = field(repr=False)

    def __repr__(self) -> str:
        j = ", ".join(f"{f:g}: {v:.3f}" for f, v in zip(self.table["fraction"], self.table["jaccard"]))
        return f"LandscapeComparison(Jaccard {j})"

    def _save_items(self) -> dict:
        return {"table": self.table, "maps": self.maps}


def eval_landscape_comparison(a, b, fractions=(0.05, 0.1, 0.2, 0.3, 0.5)) -> LandscapeComparison:
    """Landscape comparison: how much the top fractions of two priority surfaces overlap (Zonation's
    solution comparison at several fractions, prioripy's ``eval_landscape_comparison``).

    Parameters
    ----------
    a, b
        Two surfaces on the same grid, of any kind :func:`eval_landscape_identification` takes. Cells
        outside either surface are left out of both.
    fractions
        The top fractions compared (Zonation's rule: see :func:`eval_landscape_identification`).

    Returns
    -------
    LandscapeComparison

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> features = ls.load_raster(P.example_data("features"))
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> caz = P.zonation(features, rule="CAZ1")
    >>> abf = P.zonation(features, rule="ABF")
    >>> cmp = P.eval_landscape_comparison(caz, abf, fractions=[0.1, 0.3])
    >>> cmp.table[["fraction", "both"]].shape
    (2, 2)
    """
    va, ta = _surface(a)
    vb, tb = _surface(b)
    if va.shape != vb.shape:
        raise ValueError("the two surfaces must be on the same grid")
    common = np.isfinite(va) & np.isfinite(vb)
    ra = position_rank(np.where(common, va, np.nan))
    rb = position_rank(np.where(common, vb, np.nan))
    fr = np.asarray(fractions, dtype=np.float64).ravel()
    rows, maps = [], []
    for f in fr:
        ia, ib = _top(ra, f), _top(rb, f)
        both, only_a, only_b = int((ia & ib).sum()), int((ia & ~ib).sum()), int((~ia & ib).sum())
        neither = int((common & ~ia & ~ib).sum())
        either = both + only_a + only_b
        rows.append({"fraction": float(f), "cells_a": both + only_a, "cells_b": both + only_b, "both": both,
                     "only_a": only_a, "only_b": only_b, "neither": neither,
                     "jaccard": both / either if either else np.nan,
                     "share_of_a_in_b": both / (both + only_a) if both + only_a else np.nan})
        code = np.full(va.shape, np.nan)
        code[common] = 4.0
        code[ia & ~ib] = 2.0
        code[~ia & ib] = 3.0
        code[ia & ib] = 1.0
        maps.append(code)
    stack = np.stack(maps)
    template = ta if ta is not None else tb
    if template is not None:
        da = xr.DataArray(stack, dims=("band", *template.dims),
                          coords={"band": [f"{f:g}" for f in fr], **{k: v for k, v in template.coords.items()
                                                                      if k in template.dims}})
        try:
            crs = template.rio.crs
        except Exception:  # noqa: BLE001 - no georeferencing
            crs = None
        if crs is not None:
            da = da.rio.write_crs(crs)
        stack = da
    return LandscapeComparison(pd.DataFrame(rows), stack)
