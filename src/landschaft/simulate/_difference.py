# SPDX-License-Identifier: GPL-3.0-or-later
"""Components of the difference between two maps (Pontius & Millones 2011; Pontius & Santacruz 2014).

The port of the R package diffeR 0.0-8 (Pontius & Santacruz): quantity,
exchange and shift differences of a cross-tabulation matrix, overall and
per category; the same at coarser resolutions (the composite method of
Pontius & Cheuk 2006); the mean absolute deviation of two continuous maps
by component and resolution. The sums follow R (80-bit ``sum``) and terra
(``global``, ``aggregate``: sequential double sums, in the order of the
cells), so the numbers are diffeR's.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._change import _classes, _rsum, crosstab
from ._maps import read_map

__all__ = ["composite_matrix", "difference_components", "difference_resolutions", "mad", "memberships",
           "overall_difference", "sample_to_population"]


# ----------------------------------------------------------------------------- matrix components

def _matrix(ctmatrix) -> tuple[np.ndarray, list]:
    if isinstance(ctmatrix, pd.DataFrame):
        if ctmatrix.shape[0] != ctmatrix.shape[1]:
            raise ValueError("the cross-tabulation matrix must be square")
        return ctmatrix.to_numpy(dtype=np.float64), list(ctmatrix.columns)
    m = np.asarray(ctmatrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("the cross-tabulation matrix must be square")
    return m, list(range(1, m.shape[0] + 1))


def _row_sums(m):
    return np.array([_rsum(r) for r in m])


def _col_sums(m):
    return np.array([_rsum(c) for c in m.T])


def _exchange(m):
    ex = 2 * np.minimum(m, m.T)
    lt = np.tril(ex, -1)
    return _row_sums(lt) + _col_sums(lt)


def _components(m):
    rows, cols, diag = _row_sums(m), _col_sums(m), np.diag(m)
    omission = cols - diag
    commission = rows - diag
    agreement = rows - commission
    quantity = np.abs(rows - cols)
    exchange = _exchange(m)
    shift = (rows + cols - 2 * diag) - quantity - exchange
    return omission, agreement, commission, quantity, exchange, shift


def _r_round(x, digits):
    from ..sdm._evaluate import _r_round as rr

    return np.array([rr(float(v), digits) for v in np.ravel(x)]).reshape(np.shape(x))


def difference_components(ctmatrix, *, analysis: str = "error", digits: int | None = None) -> pd.DataFrame:
    """Agreement and the components of difference of each category and overall (diffeR's ``diffTablej``).

    Parameters
    ----------
    ctmatrix
        A square cross-tabulation (a DataFrame from :func:`crosstab`, or an
        array): rows the comparison map (or the earlier map), columns the
        reference map (or the later map); counts, areas or percentages.
    analysis : {"error", "change"}
        Names of the first three columns: ``Omission``, ``Agreement``,
        ``Comission`` (diffeR's spelling), or ``Gain``, ``Persistence``,
        ``Loss`` for change between two dates.
    digits
        Round to this many decimals as R does (diffeR rounds to 0 by
        default; here nothing is rounded unless asked).

    Returns
    -------
    pandas.DataFrame
        One row per category and an ``Overall`` row (the column sums, with
        quantity, exchange and shift halved, as each difference counts in two
        categories): ``Category``, omission (gain), agreement (persistence),
        commission (loss), ``Quantity``, ``Exchange``, ``Shift``.

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    >>> d = ls.simulate.difference_components(ls.simulate.crosstab(lu[0], lu[2]), analysis="change")
    >>> d.columns.tolist()
    ['Category', 'Gain', 'Persistence', 'Loss', 'Quantity', 'Exchange', 'Shift']
    """
    if analysis not in ("error", "change"):
        raise ValueError("analysis must be 'error' or 'change'")
    m, names = _matrix(ctmatrix)
    comp = np.column_stack(_components(m))
    overall = np.array([_rsum(c) for c in comp.T])
    overall[3:] = overall[3:] / 2
    table = np.vstack([comp, overall])
    if digits is not None:
        table = _r_round(table, digits)
    cols = (["Gain", "Persistence", "Loss"] if analysis == "change" else ["Omission", "Agreement", "Comission"]) \
        + ["Quantity", "Exchange", "Shift"]
    out = pd.DataFrame(table, columns=cols)
    out.insert(0, "Category", [*names, "Overall"])
    return out


def overall_difference(ctmatrix) -> dict:
    """The overall difference and its components (diffeR's ``overallDiff``, ``overallQtyD``...).

    Returns
    -------
    dict
        ``quantity``, ``exchange``, ``shift``, ``allocation`` (exchange plus
        shift) and ``overall`` (all the off-diagonal entries), in the units
        of the matrix.

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    >>> d = ls.simulate.overall_difference(ls.simulate.crosstab(lu[0], lu[1]))
    >>> d["overall"] == d["quantity"] + d["exchange"] + d["shift"]
    True
    """
    m, _ = _matrix(ctmatrix)
    rows, cols = _row_sums(m), _col_sums(m)
    overall = _rsum(m.T.ravel()) - _rsum(np.diag(m))
    quantity = _rsum(np.abs(rows - cols)) / 2
    *_, e, s = _components(m)
    return {"quantity": quantity, "exchange": _rsum(e) / 2, "shift": _rsum(s) / 2,
            "allocation": overall - quantity, "overall": overall}


def sample_to_population(ctmatrix, population) -> pd.DataFrame:
    """A sample's cross-tabulation scaled to the population of each row category (diffeR's ``sample2pop``).

    Each row ``j`` becomes ``population[j] * row / sum(row)``. diffeR takes
    the population of the row's position instead of the row's category
    (a wrong result when the population table is in another order); here the
    category decides.

    Parameters
    ----------
    ctmatrix : pandas.DataFrame
        Sample counts, rows and columns named by category.
    population
        The population of each row category: a two-column table (category,
        size), a ``dict`` or a ``Series``.

    Returns
    -------
    pandas.DataFrame
    """
    if not isinstance(ctmatrix, pd.DataFrame):
        raise TypeError("ctmatrix must be a DataFrame with categories as row and column names")
    if isinstance(population, pd.DataFrame):
        if population.shape[1] != 2:
            raise ValueError("population must have two columns")
        pop = dict(zip(population.iloc[:, 0], population.iloc[:, 1]))
    else:
        pop = dict(population)
    if len(pop) != ctmatrix.shape[0]:
        raise ValueError("population and ctmatrix must have the same number of rows")
    m = ctmatrix.to_numpy(dtype=np.float64).copy()
    names = list(ctmatrix.columns)
    for cat, size in pop.items():
        if cat not in names:
            raise ValueError(f"category {cat!r} of the population is not in ctmatrix")
        j = names.index(cat)
        m[j] = float(size) * m[j] / _rsum(m[j])
    return pd.DataFrame(m, index=ctmatrix.index, columns=ctmatrix.columns)


# ----------------------------------------------------------------------------- terra helpers

def _terra_sum(v) -> float:
    """terra's global(sum, na.rm = TRUE) of one block: a double sum in cell order, NaN skipped."""
    a = np.ravel(v)
    a = a[~np.isnan(a)]
    return float(np.cumsum(a)[-1]) if a.size else math.nan


def _blocks(values: np.ndarray, factor: int) -> np.ndarray:
    """The cells of each factor x factor block, in terra's order: (out_rows, out_cols, factor²), NaN padded."""
    ny, nx = values.shape
    by, bx = -(-ny // factor), -(-nx // factor)
    pad = np.full((by * factor, bx * factor), np.nan)
    pad[:ny, :nx] = values
    return pad.reshape(by, factor, bx, factor).transpose(0, 2, 1, 3).reshape(by, bx, factor * factor)


def _agg_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """terra's aggregate(fun = "mean") without na.rm: NaN when a block has a missing (or padded) cell."""
    b = _blocks(values, factor)
    s = np.cumsum(b, axis=2)[..., -1]
    return s / (factor * factor)


def _agg_sum(values: np.ndarray, factor: int, na_rm: bool) -> np.ndarray:
    """terra's aggregate(fun = "sum"): a double sum in cell order (NaN skipped with ``na_rm``)."""
    b = _blocks(values, factor)
    if not na_rm:
        return np.cumsum(b, axis=2)[..., -1]
    ok = ~np.isnan(b)
    s = np.cumsum(np.where(ok, b, 0.0), axis=2)[..., -1]
    return np.where(ok.any(axis=2), s, np.nan)


def memberships(map, factor: int = 2, *, nodata=None) -> dict:
    """The share of each category in each block of factor x factor cells (diffeR's ``memberships``).

    ``terra::segregate`` then ``aggregate(fun = mean)``: blocks with a
    missing cell, and the partial blocks at the right and bottom edges, are
    NaN.

    Returns
    -------
    dict
        ``{category: 2-D array}``, the categories in increasing order.
    """
    m = read_map(map, nodata)
    v = m.as_float()
    return {c: _agg_mean(np.where(np.isnan(v), np.nan, (v == c).astype(np.float64)), factor)
            for c in _classes([m])}


def _composite(comp: np.ndarray, ref: np.ndarray, classes: np.ndarray, factor: int) -> np.ndarray:
    comp_ag = [_agg_mean(np.where(np.isnan(comp), np.nan, (comp == c) * 1.0), factor) for c in classes]
    ref_ag = [_agg_mean(np.where(np.isnan(ref), np.nan, (ref == c) * 1.0), factor) for c in classes]
    weights = _agg_sum(np.where(np.isnan(comp), np.nan, 1.0), factor, na_rm=False)
    mins = [np.minimum(a, b) for a, b in zip(comp_ag, ref_ag)]  # NaN when either is
    sum_diag = np.zeros_like(weights)
    for mi in mins:
        sum_diag = sum_diag + mi
    total = _terra_sum(weights)
    k = classes.size
    out = np.empty((k, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(k):
            for j in range(k):
                a = weights * ((comp_ag[i] - mins[i]) * (ref_ag[j] - mins[j])) / (1 - sum_diag)
                out[i, j] = _terra_sum(a) / total
        for i in range(k):
            out[i, i] = _terra_sum(mins[i] * weights) / total
    return out


def composite_matrix(comp, ref, factor: int, *, nodata=None) -> pd.DataFrame:
    """Cross-tabulation at a coarser resolution by the composite method (diffeR's ``composite``).

    Each block of factor x factor cells is a mix of categories; the agreement
    of a category in a block is the smaller of its two shares, and the rest
    is distributed among the pairs of categories as Pontius & Cheuk (2006)
    propose. Blocks with a missing cell or cut by the edge are left out.

    Parameters
    ----------
    comp, ref
        The comparison and the reference map. The categories are those of
        either map, matched by value (diffeR counts a missing value as a
        category, which leaves a row and a column of NA, and names the rows
        in the order the values first appear).
    factor
        Cells per side of a block.

    Returns
    -------
    pandas.DataFrame
        Proportions of the domain (rows ``comp``, columns ``ref``).
    """
    a, b = read_map(comp, nodata), read_map(ref, nodata)
    ca = _classes([a, b])
    names = [int(c) if float(c).is_integer() else float(c) for c in ca]
    m = _composite(a.as_float(), b.as_float(), ca, int(factor))
    return pd.DataFrame(m, index=pd.Index(names, name="comp"), columns=pd.Index(names, name="ref"))


def _log_base(x: float, base: float) -> float:
    """R's log(x, base): log2 / log10 for those bases, else log(x) / log(base)."""
    if base == 2:
        return math.log2(x)
    if base == 10:
        return math.log10(x)
    return math.log(x) / math.log(base)


def _factors(nrow: int, ncol: int, fact: int, cap: bool = True) -> list[int]:
    """1, fact, fact², ... up to the larger side (differenceMR caps the last at that side, MAD does not)."""
    mdim = max(ncol, nrow)
    lg = _log_base(mdim, fact)
    maxp = math.floor(lg)
    if lg - round(lg) != 0:
        maxp += 1
    out = [1, *(int(fact) ** p for p in range(1, maxp + 1))]
    if cap and out[-1] > mdim:
        out[-1] = mdim
    return out


def _resolution(m) -> float:
    t = m.template
    if t is not None:
        try:
            return abs(float(t.rio.resolution()[0]))
        except Exception:
            pass
    return 1.0


def difference_resolutions(comp, ref, *, factor: int = 2, percent: bool = True, nodata=None,
                           population=None) -> pd.DataFrame:
    """Quantity, exchange and shift differences at increasingly coarse resolutions (diffeR's ``differenceMR``).

    At the original resolution the cross-tabulation of the two maps; at
    ``factor``, ``factor²``... cells per side (up to the larger side of the
    map) the composite cross-tabulation (:func:`composite_matrix`). Shift
    shrinks as the resolution coarsens; quantity does not change.

    Parameters
    ----------
    comp, ref
        The comparison and the reference map (same grid). diffeR gives NaN at
        the coarser resolutions when a map has missing cells (see
        :func:`composite_matrix`); here they are left out of the blocks.
    factor
        Growth of the block side from one resolution to the next.
    percent
        Differences in % of the domain (default) or as proportions.
    population
        Population of each category (see :func:`sample_to_population`) to
        scale the original-resolution matrix.

    Returns
    -------
    pandas.DataFrame
        ``Multiples``, ``Resolution`` (block side in map units),
        ``Quantity``, ``Exchange``, ``Shift``, ``Overall``.
    """
    a, b = read_map(comp, nodata), read_map(ref, nodata)
    ct = crosstab(a, b)
    if population is not None:
        ct = sample_to_population(ct.astype(np.float64), population)
    m = ct.to_numpy(dtype=np.float64)
    m = m / _rsum(m.T.ravel()) * 100

    def row(mat):
        o = overall_difference(mat)
        return [o["quantity"], o["exchange"], o["shift"], o["overall"]]

    facts = _factors(*a.shape, factor)
    res = _resolution(a)
    rows = [row(m)]
    va, vb = a.as_float(), b.as_float()
    classes = _classes([a])
    for f in facts[1:]:
        rows.append(row(_composite(va, vb, classes, f) * 100))
    out = pd.DataFrame(rows, columns=["Quantity", "Exchange", "Shift", "Overall"])
    if not percent:
        out = out / 100
    out.insert(0, "Resolution", [f * res for f in facts])
    out.insert(0, "Multiples", facts)
    return out


# ----------------------------------------------------------------------------- continuous maps

def mad(grid1, grid2, *, strata=None, multiple: bool = True, nodata=None) -> pd.DataFrame:
    """Mean absolute deviation between two continuous maps and its components (diffeR's ``MAD``).

    The deviation ``grid1 - grid2`` is split into quantity (the mean
    deviation), strata (deviations that cancel within each stratum) and
    element (the rest, which coarser resolutions remove); ``multiple``
    repeats it at blocks of 2, 4, 8... cells (sums of the deviations).

    Parameters
    ----------
    grid1, grid2
        Continuous maps on the same grid (NaN = missing).
    strata
        A map of strata (integers; 0 and missing are outside every stratum);
        default: one stratum over the cells of ``grid1``.
    multiple
        All resolutions (default) or only the original one.

    Returns
    -------
    pandas.DataFrame
        ``Multiples`` (with ``multiple``), ``Resolution``, ``Quantity``,
        ``Strata``, ``Element``, ``Total``.
    """
    g1, g2 = read_map(grid1, nodata), read_map(grid2, nodata)
    v1, v2 = g1.as_float(), g2.as_float()
    if strata is None:
        s = np.where(np.isnan(v1), np.nan, 1.0)
    else:
        s = read_map(strata).as_float()
        s[s == 0] = np.nan
    dev = v1 - v2
    values = np.unique(s[~np.isnan(s)])
    layers = [np.where(np.isnan(s), np.nan, (s == c) * 1.0) for c in values]
    nozeros = _terra_sum(np.where(np.isnan(s), np.nan, 1.0))
    made = [dev * layer for layer in layers]

    def components(maps):
        sums = np.array([_terra_sum(x) for x in maps])
        q = abs(_rsum(sums / nozeros))
        st = _rsum(np.abs(sums)) / nozeros - q
        tot = _rsum(np.array([_terra_sum(np.abs(x)) for x in maps])) / nozeros
        return q, st, tot

    facts = _factors(*g1.shape, 2, cap=False)
    res = _resolution(g1)
    q, st, tot = components(made)
    rows = [[facts[0] * res, q, st, tot - q - st, tot]]
    if multiple:
        for f in facts[1:]:
            q, st, tot = components([_agg_sum(x, f, na_rm=True) for x in made])
            rows.append([f * res, q, st, tot - q - st, tot])
    out = pd.DataFrame(rows, columns=["Resolution", "Quantity", "Strata", "Element", "Total"])
    if multiple:
        out.insert(0, "Multiples", facts)
    return out

