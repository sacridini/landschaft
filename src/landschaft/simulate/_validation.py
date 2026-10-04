# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of land change simulations: three maps at several resolutions (Pontius et al. 2011).

A reference map at the start (``t0``), the reference at the end (``t1``)
and the simulation of the end (``sim``) are compared at several
resolutions: each class's share of every block of ``f x f`` cells, the
three-dimensional table of agreement of Pontius, Peethambaram & Castella
(2011), the budget of agreement and disagreement (hits, misses, wrong
hits, false alarms, correct rejections) and the figure of merit. Ported
from lulcc's ``ThreeMapComparison``, ``AgreementBudget`` and
``FigureOfMerit`` (the same numbers; the block sums of the ``raster``
package, R's 80-bit sums).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import threaded
from ._change import _class_value, _rsum
from ._maps import read_map

__all__ = ["ThreeMapComparison", "compare_maps", "error_budget", "futures_validation", "three_map_comparison"]

_BUDGET = ["misses", "hits", "wrong_hits", "false_alarms", "correct_rejections"]


def _sum_na(v) -> float:
    """R's ``sum(v, na.rm = TRUE)``."""
    v = np.asarray(v, dtype=np.float64).ravel()
    return _rsum(v[~np.isnan(v)])


def _aggregate_sum(v: np.ndarray, f: int) -> np.ndarray:
    """raster::aggregate(fun = sum, na.rm = TRUE, expand = TRUE): blocks from the top-left, NA if all NA."""
    ny, nx = v.shape
    fy, fx = min(f, ny), min(f, nx)
    by, bx = -(-ny // fy), -(-nx // fx)
    pad = np.full((by * fy, bx * fx), np.nan)
    pad[:ny, :nx] = v
    blocks = pad.reshape(by, fy, bx, fx).transpose(0, 2, 1, 3).reshape(by, bx, fy * fx)
    valid = ~np.isnan(blocks)
    out = np.where(valid, blocks, 0.0).sum(axis=2)  # 0/1 counts: exact in any order
    out[~valid.any(axis=2)] = np.nan
    return out


class ThreeMapComparison:
    """Three maps compared at several resolutions (lulcc's ``ThreeMapComparison``).

    Attributes
    ----------
    factors : list of int
        The resolutions (block sides, in cells).
    categories : list
        The classes.
    tables : dict
        ``{factor: array}``: lulcc's table, ``(n + 1)^2`` rows (reference
        ``t0`` class ``i`` within reference ``t1`` class ``j``, row
        ``i + j (n + 1)``, 0-based; index ``n`` the totals) by ``n + 1``
        columns (simulated class, then the total), as shares of the area.
    """

    def __init__(self, tables: dict, factors: list, categories: list):
        self.tables = tables
        self.factors = factors
        self.categories = categories

    def table(self, factor=1) -> pd.DataFrame:
        """The three-dimensional table as a DataFrame: rows (reference t0, reference t1), columns simulated."""
        n = len(self.categories)
        names = [*self.categories, "total"]
        idx = pd.MultiIndex.from_tuples([(names[i], names[j]) for j in range(n + 1) for i in range(n + 1)],
                                        names=["reference_t0", "reference_t1"])
        t = pd.DataFrame(self.tables[factor], index=idx, columns=pd.Index(names, name="simulated_t1"))
        return t.sort_index(level=["reference_t0", "reference_t1"], sort_remaining=False,
                            key=lambda s: s.map({c: k for k, c in enumerate(names)}))

    # -- lulcc's AgreementBudget ------------------------------------------------------------------------

    def _budget(self, tab, kind, from_ix, to_ix):
        n = len(self.categories)
        N = n + 1

        def T(i, j, k):  # lulcc's tab[i + (j - 1) * (n + 1), k], 1-based
            return tab[(i - 1) + (j - 1) * N, k - 1]

        a = np.zeros(n)
        for j in to_ix:
            asub = np.zeros(n)
            for i in from_ix:
                if i != j:
                    asub[i - 1] = T(i, j, i)
            a[j - 1] = _sum_na(asub)
        a = _rsum(a)
        b = np.zeros(n)
        for j in to_ix:
            bsub = np.zeros(n)
            for i in from_ix:
                if i != j:
                    bsub[i - 1] = T(i, j, j)
            b[j - 1] = _rsum(bsub)
        b = _rsum(b)
        c = np.zeros(n)
        for j in to_ix:
            csub = np.zeros(n)
            for i in from_ix:
                csubsub = np.zeros(n)
                for k in range(1, n + 1):
                    if i != j and i != k and j != k:
                        csubsub[k - 1] = T(i, j, k)
                csub[i - 1] = _sum_na(csubsub)
            c[j - 1] = _sum_na(csub)
        c = _rsum(c)
        if len(to_ix) == 1:
            csub = np.zeros(n)
            (f,), (t,) = from_ix, to_ix
            for j in range(1, n + 1):
                if j != f and j != t:
                    csub[j - 1] = T(f, j, t)
            c = c + _rsum(csub)
        d = np.zeros(n)
        for j in from_ix:
            dsub = np.zeros(n)
            for k in to_ix:
                if k != j:
                    dsub[k - 1] = T(j, j, k)
            d[j - 1] = _rsum(dsub)
        d = _rsum(d)
        if kind == "transition":
            return [a, b, c, d]
        e = np.zeros(n)
        for i in from_ix:
            e[i - 1] = T(i, i, i)
        return [a, b, c, d, _rsum(e)]

    def agreement_budget(self) -> pd.DataFrame:
        """Sources of agreement and disagreement (lulcc's ``AgreementBudget``), as shares of the area.

        ``misses`` (change simulated as persistence), ``hits``,
        ``wrong_hits`` (change simulated as change to another class),
        ``false_alarms`` (persistence simulated as change) and
        ``correct_rejections`` (persistence simulated correctly; not for a
        transition): a row per level (``overall``, a class, a transition)
        and resolution.
        """
        n = len(self.categories)
        rows = []
        everything = list(range(1, n + 1))
        for f in self.factors:
            tab = self.tables[f]
            rows.append({"level": "overall", "from": None, "to": None, "factor": f,
                         **dict(zip(_BUDGET, self._budget(tab, "overall", everything, everything)))})
            for i in everything:
                rows.append({"level": "category", "from": self.categories[i - 1], "to": None, "factor": f,
                             **dict(zip(_BUDGET, self._budget(tab, "category", [i], everything)))})
            for i in everything:
                for j in everything:
                    rows.append({"level": "transition", "from": self.categories[i - 1],
                                 "to": self.categories[j - 1], "factor": f,
                                 **dict(zip(_BUDGET, self._budget(tab, "transition", [i], [j])))})
        return pd.DataFrame(rows, columns=["level", "from", "to", "factor", *_BUDGET])

    # -- lulcc's FigureOfMerit ---------------------------------------------------------------------------

    def figure_of_merit(self) -> dict:
        """The figure of merit (Pontius et al. 2011, equations 9-11) at each resolution.

        Returns
        -------
        dict
            ``overall``: a Series by factor; ``category``: a DataFrame
            (factor x class: change from the class); ``transition``:
            ``{factor: DataFrame}`` (rows: from, columns: to).
        """
        n = len(self.categories)
        N = n + 1
        overall, category, transition = {}, {}, {}
        with np.errstate(divide="ignore", invalid="ignore"):  # 0 / 0 is NaN, as in R
            return self._fom(n, N, overall, category, transition)

    def _fom(self, n, N, overall, category, transition):
        for f in self.factors:
            tab = self.tables[f]

            def T(r, k, tab=tab):  # 1-based
                return tab[r - 1, k - 1]

            a = 0.0
            b = 0.0
            for j in range(1, n + 1):
                a = _sum_na([a, T(j * N, j)])
                b = _sum_na([b, T(j + (j - 1) * N, j)])
            overall[f] = (a - b) / (1 - b)
            eq10 = np.zeros(n)
            for i in range(1, n + 1):
                a = 0.0
                b = 0.0
                for j in range(1, n + 1):
                    a = _sum_na([a, T(i + (j - 1) * N, j)])
                    b = _sum_na([b, T(i + (j - 1) * N, N)])
                c = T(i + (i - 1) * N, i)
                eq10[i - 1] = (a - c) / (b - c)
            category[f] = eq10
            eq11 = np.zeros((n, n))
            for j in range(1, n + 1):
                for i in range(1, n + 1):
                    a = T(i + (j - 1) * N, j)
                    b = T(i + (j - 1) * N, N)
                    c = 0.0
                    for k in range(1, n + 1):
                        c = _sum_na([c, T(i + (k - 1) * N, j)])
                    eq11[i - 1, j - 1] = a / (b + c - a)
            transition[f] = pd.DataFrame(eq11, index=pd.Index(self.categories, name="from"),
                                         columns=pd.Index(self.categories, name="to"))
        return {"overall": pd.Series(overall, name="figure_of_merit").rename_axis("factor"),
                "category": pd.DataFrame(category, index=self.categories).T.rename_axis("factor"),
                "transition": transition}

    def __repr__(self) -> str:
        return f"ThreeMapComparison({len(self.categories)} classes, factors {self.factors})"


@threaded
def three_map_comparison(t0, t1, simulated, factors=(1,), *, categories=None, nodata=None) -> ThreeMapComparison:
    """Compare the reference maps at the start and the end with the simulated end at several resolutions.

    Parameters
    ----------
    t0, t1
        The reference maps at the start and at the end.
    simulated
        The simulated map of the end.
    factors
        Block sides (cells) of the resolutions (1: the cells themselves).
    categories
        The classes (default: those found in the three maps).

    Returns
    -------
    ThreeMapComparison
        With ``table``, ``agreement_budget`` and ``figure_of_merit``.

    Notes
    -----
    lulcc takes the cells with a value in ``t0`` and, where ``t1`` or the
    simulation lack one there, misaligns their blocks; here the cells
    without a value in any of the three maps are left out of all three.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> t0 = np.array([[1, 1, 1, 1], [1, 1, 2, 2]])
    >>> t1 = np.array([[2, 2, 1, 1], [1, 1, 2, 2]])
    >>> sim = np.array([[2, 1, 2, 1], [1, 1, 2, 2]])
    >>> tmc = ls.simulate.three_map_comparison(t0, t1, sim, factors=[1, 2])
    >>> tmc.figure_of_merit()["overall"].round(3).tolist()
    [0.333, 0.333]
    """
    maps = [read_map(m, nodata).as_float() for m in (t0, t1, simulated)]
    if len({m.shape for m in maps}) > 1:
        raise ValueError("the three maps differ in shape")
    miss = np.isnan(maps[0]) | np.isnan(maps[1]) | np.isnan(maps[2])
    x, x1, y1 = (np.where(miss, np.nan, m) for m in maps)
    if categories is None:
        categories = sorted({_class_value(c) for m in (x, x1, y1) for c in np.unique(m[~np.isnan(m)])})
    else:
        categories = [_class_value(c) for c in categories]
    n = len(categories)
    if n < 2:
        raise ValueError("the comparison needs at least two classes")
    factors = [int(f) for f in factors]
    if any(f < 1 for f in factors):
        raise ValueError("the factors must be at least 1")
    ones = np.where(miss, np.nan, 1.0)
    tables = {}
    for f in factors:
        weight = _aggregate_sum(ones, f) if f > 1 else ones
        wt = weight.ravel()
        wt = wt[~np.isnan(wt)]
        counts = []
        for m in (x, x1, y1):
            per = []
            for c in categories:
                b = np.where(np.isnan(m), np.nan, (m == c).astype(np.float64))
                if f > 1:
                    b = _aggregate_sum(b, f)
                v = b.ravel()
                per.append(v[~np.isnan(v)])
            counts.append(np.ascontiguousarray(np.stack(per)))
        tab = np.asarray(_core.sim_tmc_table(counts[0], counts[1], counts[2], np.ascontiguousarray(wt)))
        N = n + 1
        for j in range(n):                                                    # totals
            for k in range(n):
                tab[j + N * n, k] = _sum_na(tab[[j + N * m for m in range(n)], k])
        for j in range(N):
            for k in range(n):
                tab[(j + 1) * N - 1, k] = _sum_na(tab[[i + j * N for i in range(n)], k])
        for j in range(N):
            for i in range(N):
                tab[i + j * N, N - 1] = _sum_na(tab[i + j * N, :n])
        tables[f] = tab
    return ThreeMapComparison(tables, factors, categories)


# ============================================================================= MOLUSCE


def _molusce_crosstab(x: np.ndarray, y: np.ndarray, expand: bool):
    """MOLUSCE's CrossTable of the cells with a value in both maps: (table, n)."""
    gx, gy = np.unique(x), np.unique(y)
    if expand:
        gx = gy = np.unique(np.concatenate([gx, gy]))
    t = np.zeros((gx.size, gy.size), dtype=int)
    np.add.at(t, (np.searchsorted(gx, x), np.searchsorted(gy, y)), 1)
    return t, x.size


def compare_maps(reference, simulated, *, nodata=None) -> dict:
    """MOLUSCE's comparison of two maps (``DependenceCoef``): kappa, agreement and dependence.

    Returns
    -------
    dict
        ``kappa``, ``kappa_location``, ``kappa_histogram`` and ``correctness``
        (% of cells of the same class) from the cross table over the classes
        of both maps (MOLUSCE's validation); ``cramer`` (Cramer's V) and
        ``jiu`` (joint information uncertainty) from the cross table of the
        classes each map has; ``correlation`` (Pearson, of the class values).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> a = np.array([[1, 1, 2, 2], [1, 3, 3, 2]])
    >>> b = np.array([[1, 1, 2, 3], [1, 3, 3, 2]])
    >>> round(ls.simulate.compare_maps(a, b)["kappa"], 4)
    0.814
    """
    import math

    a, b = read_map(reference, nodata).as_float().ravel(), read_map(simulated, nodata).as_float().ravel()
    if a.size != b.size:
        raise ValueError("the maps differ in size")
    ok = ~(np.isnan(a) | np.isnan(b))
    x, y = a[ok], b[ok]
    if not x.size:
        raise ValueError("no cell has a value in both maps")
    out = {"correlation": float(np.corrcoef(x, y)[0][1])}
    t, n = _molusce_crosstab(x, y, True)
    rows = t.shape[0]
    p = 1.0 * t / n
    pa = 0
    for i in range(rows):
        pa = pa + p[i, i]
    prows = 1.0 * t.sum(axis=1) / n
    pcols = 1.0 * t.sum(axis=0) / n
    pexpect = sum(prows * pcols)
    pmax = sum(np.min([prows, pcols], axis=0))
    out["kappa"] = float((pa - pexpect) / (1 - pexpect))
    out["kappa_location"] = float((pa - pexpect) / (pmax - pexpect))
    out["kappa_histogram"] = float((pmax - pexpect) / (1 - pexpect))
    s = 0.0
    for i in range(rows):
        s = s + t[i][i]
    out["correctness"] = 100.0 * s / n
    t, n = _molusce_crosstab(x, y, False)
    r, c = t.shape
    srows = np.tile(np.reshape(t.sum(axis=1), (r, 1)), (1, c))
    scols = np.tile(t.sum(axis=0), (r, 1))
    expected = 1.0 * srows * scols / n
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = np.sum(np.where(expected == 0, 0.0, np.square(t - expected) / expected))
    out["cramer"] = math.sqrt(x2 / (n * min(c - 1, r - 1))) if min(c, r) > 1 else float("nan")
    p = 1.0 * t / n
    pr, pc = 1.0 * t.sum(axis=1) / n, 1.0 * t.sum(axis=0) / n
    pr, pc = pr[pr != 0], pc[pc != 0]
    hx, hy = -np.sum(pr * np.log(pr)), -np.sum(pc * np.log(pc))
    pt = p[p != 0]
    hxy = -np.sum(pt * np.log(pt))
    out["jiu"] = float(2.0 * ((hx + hy - hxy) / (hx + hy))) if hx + hy else float("nan")
    return out


def _block_sums(v: np.ndarray, s: int) -> np.ndarray:
    """``np.sum`` of every s x s block from the top-left (the edges cut off), as in MOLUSCE's loops."""
    ny, nx = v.shape[0] // s, v.shape[1] // s
    blk = np.ascontiguousarray(v[:ny * s, :nx * s].reshape(ny, s, nx, s).transpose(0, 2, 1, 3)).reshape(ny, nx, s * s)
    return blk.sum(axis=2)


def error_budget(reference, simulated, *, iterations: int = 3, scale: int = 2, nodata=None) -> pd.DataFrame:
    """MOLUSCE's error budget (Pontius 2002): agreement for levels of information, at coarser resolutions.

    The share of agreement between the maps when the simulation is given
    no / medium / perfect information of quantity and of location
    (``no_no``, ``no_medium``, ``medium_medium``, ``medium_perfect``,
    ``perfect_perfect``), at the map's resolution and ``iterations - 1``
    times coarser, each time merging ``scale x scale`` cells (the edges
    left over dropped, as MOLUSCE does).

    Returns
    -------
    pandas.DataFrame
        A row per resolution (``cells``: the block side in cells).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> a = np.repeat(np.array([[1, 1, 2, 2], [1, 3, 3, 2]]), 2, axis=0)
    >>> b = np.repeat(np.array([[1, 1, 2, 3], [1, 3, 3, 2]]), 2, axis=0)
    >>> ls.simulate.error_budget(a, b, iterations=2)["medium_medium"].round(3).tolist()
    [0.875, 0.875]
    """
    r = read_map(reference, nodata).as_float()
    s = read_map(simulated, nodata).as_float()
    if r.shape != s.shape:
        raise ValueError("the maps differ in shape")
    if scale < 2:
        raise ValueError("scale must be at least 2")
    cats = sorted({_class_value(c) for c in np.unique(r[~np.isnan(r)])})
    extra = {_class_value(c) for c in np.unique(s[~np.isnan(s)])} - set(cats)
    if extra:
        raise ValueError(f"the simulation has classes the reference lacks: {sorted(extra)}")
    mask = np.isnan(r) | np.isnan(s)
    w = np.ones(r.shape) - mask
    rf, sf = np.where(mask, 0, np.nan_to_num(r)), np.where(mask, 0, np.nan_to_num(s))
    rj = {c: 1.0 * (rf == c) for c in cats}
    sj = {c: 1.0 * (sf == c) for c in cats}

    def weighted(v):
        return np.sum(w * v) / np.sum(w)

    rows = []
    side = 1
    for _ in range(int(iterations)):
        size = len(cats)
        arr = np.zeros(w.shape)
        for c in cats:
            arr = arr + np.minimum(rj[c], 1.0 / size)
        no_no = np.sum(w * arr) / np.sum(w)
        arr = np.zeros(w.shape)
        for c in cats:
            arr = arr + np.minimum(rj[c], weighted(sj[c]))
        no_med = np.sum(w * arr) / np.sum(w)
        arr = np.zeros(w.shape)
        for c in cats:
            arr = arr + np.minimum(rj[c], sj[c])
        med_med = np.sum(w * arr) / np.sum(w)
        med_per = 0
        for c in cats:
            med_per = med_per + np.minimum(weighted(rj[c]), weighted(sj[c]))
        rows.append({"cells": side, "no_no": float(no_no), "no_medium": float(no_med),
                     "medium_medium": float(med_med), "medium_perfect": float(med_per), "perfect_perfect": 1.0})
        if w.shape[0] < scale or w.shape[1] < scale:  # MOLUSCE: nothing more to merge
            continue
        sum_w = 1.0 * _block_sums(w, scale)
        zero = sum_w == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rj = {c: np.where(zero, 0.0, 1.0 * _block_sums(rj[c] * w, scale) / sum_w) for c in cats}
            sj = {c: np.where(zero, 0.0, 1.0 * _block_sums(sj[c] * w, scale) / sum_w) for c in cats}
        w = 1.0 * sum_w / (scale * scale)
        side *= scale
    return pd.DataFrame(rows)


# ============================================================================= FUTURES


def futures_validation(reference, simulated, original=None, *, nodata=None) -> dict:
    """FUTURES' validation (``r.futures.validation``): disagreement, kappa, kappa simulation, change metrics.

    Parameters
    ----------
    reference, simulated
        The observed and the simulated maps (integer classes; FUTURES uses
        0 undeveloped, 1 developed).
    original
        The map at the start: adds kappa simulation (van Vliet et al. 2011)
        and, for 0/1 maps, the change metrics of Pontius et al. (2008).

    Returns
    -------
    dict
        ``classes``; per class ``quantity`` and ``allocation``
        disagreement (Pontius & Millones 2011) and their totals
        ``total_quantity``, ``total_allocation``; ``kappa``; with the
        original: ``kappasimulation`` and (0/1 maps) ``misses``, ``hits``,
        ``false_alarms``, ``null_successes``, ``initially_developed``
        (shares of the cells), ``figure_of_merit``, ``producer`` and
        ``user`` accuracy. The cells without a value in any map are left
        out, as ``r.stats -n`` does.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> orig = np.array([[0, 0, 1], [0, 0, 1]])
    >>> ref = np.array([[1, 0, 1], [1, 0, 1]])
    >>> sim = np.array([[1, 1, 1], [0, 0, 1]])
    >>> round(ls.simulate.futures_validation(ref, sim, orig)["figure_of_merit"], 4)
    0.3333
    """
    maps = [read_map(m, nodata).as_float() for m in ([original] if original is not None else []) +
            [reference, simulated]]
    if len({m.shape for m in maps}) > 1:
        raise ValueError("the maps differ in shape")
    ok = ~np.any([np.isnan(m) for m in maps], axis=0)
    cols = np.stack([m[ok] for m in maps], axis=1)
    if cols.size and (np.abs(cols - np.round(cols)) > 0).any():
        raise ValueError("the maps must hold integer classes")
    combos, counts = (np.unique(cols.astype(np.int64), axis=0, return_counts=True) if cols.size
                      else (np.zeros((0, len(maps)), dtype=np.int64), np.zeros(0, dtype=np.int64)))
    cats = sorted({int(c) for c in np.unique(combos)})
    res = {"classes": cats}
    keys = ["misses", "hits", "false_alarms", "null_successes", "figure_of_merit", "producer", "user",
            "initially_developed"]
    if not cats:
        res.update({k: None for k in ("quantity", "total_quantity", "allocation", "total_allocation", "kappa")})
        if original is not None:
            res.update({"kappasimulation": None, **{k: None for k in keys}})
        return res
    n = len(cats)
    index = {c: k for k, c in enumerate(cats)}
    ref_sim, orig_sim, orig_ref = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    binary = set(cats) == {0, 1}
    hits = false_alarms = misses = null_successes = initially_developed = 0
    for line, cnt in zip(combos.tolist(), counts.tolist()):
        if original is not None:
            o, r, s = (index[v] for v in line)
            ref_sim[s, r] += cnt
            orig_sim[o, s] += cnt
            orig_ref[o, r] += cnt
            if binary:
                if o == 0:
                    if r == 0 and s == 1:
                        false_alarms += cnt
                    elif r == 1 and s == 0:
                        misses += cnt
                    elif r == 1 and s == 1:
                        hits += cnt
                    elif r == 0 and s == 0:
                        null_successes += cnt
                else:
                    initially_developed += cnt
        else:
            r, s = (index[v] for v in line)
            ref_sim[s, r] += cnt
    if original is not None and binary:
        total = misses + hits + false_alarms + null_successes + initially_developed
        res["misses"] = misses / total if total else None
        res["hits"] = hits / total if total else None
        res["false_alarms"] = false_alarms / total if total else None
        res["null_successes"] = null_successes / total if total else None
        res["figure_of_merit"] = hits / (misses + hits + false_alarms) if misses + hits + false_alarms else 0
        res["producer"] = hits / (misses + hits) if misses + hits else 0
        res["user"] = hits / (hits + false_alarms) if hits + false_alarms else 0
        res["initially_developed"] = initially_developed / total if total else None
    elif original is not None:
        res.update({k: None for k in keys})
    quantity = np.abs(ref_sim.sum(axis=0) - ref_sim.sum(axis=1)) / ref_sim.sum()
    res["quantity"] = quantity
    res["total_quantity"] = quantity.sum() / 2
    allocation = 2 * np.minimum(ref_sim.sum(axis=0) - ref_sim.diagonal(),
                                ref_sim.sum(axis=1) - ref_sim.diagonal()) / ref_sim.sum()
    res["allocation"] = allocation
    res["total_allocation"] = allocation.sum() / 2
    p_0 = ref_sim.diagonal().sum() / ref_sim.sum()
    p_e = np.multiply(ref_sim.sum(axis=0) / ref_sim.sum(), ref_sim.sum(axis=1) / ref_sim.sum()).sum()
    with np.errstate(all="raise"):
        try:
            res["kappa"] = (p_0 - p_e) / (1 - p_e)
        except FloatingPointError:
            res["kappa"] = None
    if original is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            a = orig_sim / orig_sim.sum(axis=1)[:, None]
            b = orig_ref / orig_ref.sum(axis=1)[:, None]
        c = np.multiply(a, b)
        d = orig_sim.sum(axis=1) / orig_sim.sum()
        e = np.multiply(c.sum(axis=1), d)
        p_e = np.nansum(e)
        with np.errstate(all="raise"):
            try:
                res["kappasimulation"] = (p_0 - p_e) / (1 - p_e)
            except FloatingPointError:
                res["kappasimulation"] = None
    return res
