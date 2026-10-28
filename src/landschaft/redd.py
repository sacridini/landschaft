# SPDX-License-Identifier: GPL-3.0-or-later
"""Verra's VT0007: the allocation of a jurisdiction's unplanned deforestation (UDef-A).

The steps of VT0007's Unplanned Deforestation Allocation as Clark Labs'
UDef-ARP implements them (GPL-3.0), the same numbers: the negligible risk
threshold of the distance to the forest's edge (:func:`nrt`), the vulnerability
in classes of geometric widths, the benchmark from that distance
(:func:`benchmark`) or an alternative map (:func:`alternative`), the relative
frequency of deforestation per class and subdivision and the density it gives
(:func:`fit`), that density in another period scaled to the activity data,
never above a cell (:func:`allocate`), and the predicted against the actual
deforestation in the cells of an assessment grid (:func:`assess`). As
Terraflux's ``redd.*`` operators.

Typical use::

    import landschaft as ls

    forest = ls.load_raster("forest_2015.tif")          # 1 forest
    cal = ls.load_raster("deforestation_2015_2020.tif") # 1 deforested in the calibration period
    juris = ls.load_raster("jurisdiction.tif")          # non-zero inside
    d = ls.redd.edge_distance(forest)
    t = ls.redd.nrt(d, cal, mask=juris)
    classes = ls.redd.benchmark(d, t, mask=juris)
    f = ls.redd.fit(classes, cal, subdivisions=ls.load_raster("states.tif"))
    a = ls.redd.allocate(classes_2020, f, expected=12_000)   # hectares of the activity data
    a.density                                             # hectares expected in each cell
    ls.redd.assess(a_cnf.density, cnf, grid_area=10_000, mask=juris).metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ._save import Saveable

__all__ = ["Allocation", "Assessment", "Fit", "allocate", "alternative", "assess", "benchmark", "edge_distance",
           "fit", "nrt"]


# -- maps -------------------------------------------------------------------------------------------------------------

def _read(x, what: str, nodata=None):
    """(values float64 with NaN, template DataArray) of a map on a grid."""
    from .simulate._maps import read_map

    try:
        m = read_map(x, nodata)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: {e}") from None
    if m.template is None:
        raise ValueError(f"{what}: a map with its grid (a DataArray from ls.load_raster)")
    return np.asarray(m.as_float(), dtype=np.float64), m.template


def _transform(template) -> tuple[float, ...]:
    from .io import _transform as transform_of

    t = transform_of(template)
    return (t.c, t.a, t.b, t.f, t.d, t.e)


def _zeroed(v: np.ndarray) -> np.ndarray:
    """A map's values with missing cells as 0 (UDef-ARP's binary maps)."""
    return np.where(np.isnan(v), 0.0, v)


def _same(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{what} and the other maps differ in size: {a.shape} and {b.shape}")


def _mask(mask, like: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.ones_like(like)
    m, _ = _read(mask, "mask")
    _same(m, like, "mask")
    return _zeroed(m)


def _cell_ha(t) -> float:
    return t[1] * abs(t[5]) / 10_000.0


def _out(values: np.ndarray, template, name: str) -> xr.DataArray:
    out = template.copy(data=values).rename(name)
    out.attrs = {}
    out.encoding = {}
    try:
        out = out.rio.write_nodata(np.nan)
    except Exception:  # noqa: BLE001, S110  (no rioxarray georeference)
        pass
    return out


# -- the distance to the edge and the negligible risk threshold -------------------------------------------------------

def edge_distance(forest, *, nodata=None) -> xr.DataArray:
    """The distance of every forest cell to the nearest cell without forest, in map units (VT0007's input).

    The Euclidean distance between cell centres (GDAL's proximity), 0
    outside the forest: the map :func:`nrt` and :func:`benchmark` take.

    Parameters
    ----------
    forest
        The forest (non-zero) at the start of the period.
    nodata
        Value marking missing cells (no forest).

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> f = xr.DataArray(np.pad(np.ones((5, 5)), 1), dims=("y", "x"),
    ...                  coords={"y": 30.0 * np.arange(7)[::-1], "x": 30.0 * np.arange(7)})
    >>> ls.redd.edge_distance(f).values[3].tolist()
    [0.0, 30.0, 60.0, 90.0, 60.0, 30.0, 0.0]
    """
    from scipy import ndimage

    v, tpl = _read(forest, "forest", nodata)
    t = _transform(tpl)
    f = _zeroed(v) != 0
    d = ndimage.distance_transform_edt(f, sampling=(abs(t[5]), abs(t[1]))) * f
    return _out(d.astype(np.float64), tpl, "edge_distance")


def _arange(start: float, stop: float, step: float) -> np.ndarray:
    """numpy's arange of floats, as UDef-ARP calls it."""
    return np.arange(start, stop, step)


def nrt(distance, deforestation, *, mask=None) -> float:
    """VT0007's negligible risk threshold (UDef-ARP's ``nrt_calculation``).

    The distance to the forest's edge below which 99.5 % of a period's
    deforestation happened: the middle of the first bin (a cell wide) where
    the cumulative share of the deforested cells' distances reaches 0.995, in
    whole map units.

    Parameters
    ----------
    distance
        The distance of the forest's cells to the edge, in map units
        (:func:`edge_distance`).
    deforestation
        The deforestation of the period (1).
    mask
        The jurisdiction (non-zero; default all the map).

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> c = {"y": 30.0 * np.arange(9)[::-1], "x": 30.0 * np.arange(9)}
    >>> f = xr.DataArray(np.pad(np.ones((7, 7)), 1), dims=("y", "x"), coords=c)
    >>> d = ls.redd.edge_distance(f)
    >>> ls.redd.nrt(d, (d > 0) & (d <= 60))
    45.0
    """
    d, tpl = _read(distance, "distance")
    dv, _ = _read(deforestation, "deforestation")
    _same(dv, d, "deforestation")
    m = _mask(mask, d)
    cell = _transform(tpl)[1]
    v = (_zeroed(d) * m * _zeroed(dv)).ravel()
    v = np.sort(v[v != 0.0])
    if v.size == 0:
        raise ValueError("no deforested cell of the jurisdiction has a distance")
    edges = _arange(v[0], v[-1] + cell, cell)
    if edges.size < 2:
        raise ValueError("no deforested cell of the jurisdiction has a distance")
    hist, _ = np.histogram(v, bins=edges)
    hist = hist.astype(np.float64)
    total = hist.sum()
    acc, index = 0.0, 0
    for i, h in enumerate(hist):
        acc += h / total
        if acc >= 0.995:
            index = i
            break
    return float(math.trunc((edges[index] + edges[index + 1]) / 2.0))


# -- the vulnerability classes ----------------------------------------------------------------------------------------

def benchmark(distance, nrt: float, *, mask=None, classes: int = 29) -> xr.DataArray:
    """VT0007's benchmark vulnerability map (UDef-ARP's ``geometric_classification``).

    The forest's distance to the edge in ``classes`` classes of geometric
    widths from the cell size to the negligible risk threshold: 2 the nearest
    the edge, ``classes + 1`` the farthest below the threshold, 1 beyond it,
    0 outside the forest and the jurisdiction.

    Parameters
    ----------
    distance
        The distance of the forest's cells to the edge, in map units.
    nrt
        The negligible risk threshold (:func:`nrt`).
    mask
        The jurisdiction (non-zero; default all the map).
    classes
        The classes of geometric widths (default 29).

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> c = {"y": 30.0 * np.arange(9)[::-1], "x": 30.0 * np.arange(9)}
    >>> d = ls.redd.edge_distance(xr.DataArray(np.pad(np.ones((7, 7)), 1), dims=("y", "x"), coords=c))
    >>> ls.redd.benchmark(d, 75, classes=3).values[4].tolist()
    [0.0, 4.0, 2.0, 1.0, 1.0, 1.0, 2.0, 4.0, 0.0]
    """
    if int(classes) < 1:
        raise ValueError("classes: 1 or more")
    d, tpl = _read(distance, "distance")
    m = _mask(mask, d)
    n = int(classes)
    cell = _transform(tpl)[1]
    ll, ul = math.trunc(cell), math.trunc(float(nrt))
    if ul <= ll:
        raise ValueError("the threshold must be above the cell size")
    ll, ul = float(ll), float(ul)
    r = math.pow(ll / ul, 1.0 / n)
    risk = [[ul * math.pow(r, i), ul * math.pow(r, i + 1)] for i in range(n)]
    risk[n - 1][1] = ll
    risk[0][0] = ul
    a = np.where(np.isnan(d), 0.0, d * m)
    a[a >= ul] = 1.0
    for i, (lower, upper) in enumerate(risk):
        a[(lower > a) & (a >= upper)] = i + 2
    return _out(np.trunc(a), tpl, "vulnerability")


def alternative(vulnerability, forest, *, mask=None, classes: int = 29) -> xr.DataArray:
    """VT0007's alternative vulnerability map (UDef-ARP's ``geometric_classification_alternative``).

    A map of vulnerability (a transition potential, a probability), rescaled
    to 1-2 by its largest value, in ``classes`` classes of geometric widths (1
    the lowest), inside the jurisdiction's forest; 0 elsewhere. As UDef-ARP, a
    float32 map is rescaled in float32 (its largest values, 2 exactly, stay 2).

    Parameters
    ----------
    vulnerability
        The vulnerability of every cell (higher: more vulnerable).
    forest
        The forest (non-zero).
    mask
        The jurisdiction (non-zero; default all the map).
    classes
        The classes of geometric widths (default 29).

    Returns
    -------
    xarray.DataArray
    """
    if int(classes) < 1:
        raise ValueError("classes: 1 or more")
    v, tpl = _read(vulnerability, "vulnerability")
    fv, _ = _read(forest, "forest")
    _same(fv, v, "forest")
    m = _mask(mask, v) * _zeroed(fv)
    n = int(classes)
    finite = v[~np.isnan(v)]
    vmax = float(finite.max()) if finite.size else -math.inf
    single = bool(np.all(np.isnan(v) | (v.astype(np.float32).astype(np.float64) == v)))
    r = math.pow(2.0, 1.0 / n)
    risk = [[1.0 + (2.0 - math.pow(r, i)), 1.0 + (2.0 - math.pow(r, i + 1))] for i in reversed(range(n))]
    risk[0][1] = 1.0
    x = _zeroed(v)
    if single:
        f32 = np.float32
        scaled = f32(1.0) + (x.astype(f32) * f32(1.0)) / f32(vmax)
        a = (scaled.astype(f32) * m.astype(f32)).astype(np.float64)
    else:
        a = (1.0 + x * 1.0 / vmax) * m
    for i, (upper, lower) in enumerate(risk):
        a[(upper > a) & (a >= lower)] = i + 1
    return _out(np.trunc(a), tpl, "vulnerability")


# -- the relative frequency and the density ---------------------------------------------------------------------------

def _regions(classes: np.ndarray, subdivisions: np.ndarray | None) -> np.ndarray:
    """The modelling region of every cell: class x 1000 + subdivision (0 outside the classes), as int16."""
    c = np.where(np.isnan(classes), 0, classes).astype(np.int64)
    s = np.zeros_like(c) if subdivisions is None else np.where(np.isnan(subdivisions), 0, subdivisions).astype(np.int64)
    r = (c * 1000 + s).astype(np.int16).astype(np.int64)
    return np.where(c > 0, r, 0)


def _subdivisions(subdivisions, like: np.ndarray) -> np.ndarray | None:
    if subdivisions is None:
        return None
    s, _ = _read(subdivisions, "subdivisions")
    _same(s, like, "subdivisions")
    return s


@dataclass
class Fit(Saveable):
    """The fit of a vulnerability map to a period's deforestation (:func:`fit`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per modelling region (increasing): ``id`` (class x 1000 +
        subdivision), ``class``, ``subdivision``, ``deforested`` and ``cells``
        (its cells), ``frequency`` (their ratio).
    density : xarray.DataArray
        The fitted density of deforestation, hectares per cell (float32's
        values, as UDef-ARP writes it).
    """

    table: pd.DataFrame
    density: xr.DataArray

    def __repr__(self) -> str:
        return f"Fit({len(self.table)} modelling regions, {int(self.table['deforested'].sum())} deforested cells)"


def fit(classes, deforestation, *, subdivisions=None) -> Fit:
    """VT0007's fit of a vulnerability map to a period's deforestation (UDef-ARP's fitting).

    The relative frequency of deforestation in every modelling region (a
    vulnerability class in a subdivision of the jurisdiction) and the density
    it gives every cell, in hectares.

    Parameters
    ----------
    classes
        The vulnerability classes (:func:`benchmark` or :func:`alternative`).
    deforestation
        The deforestation of the period (1).
    subdivisions
        The subdivisions of the jurisdiction (ids 0-999; default one).

    Returns
    -------
    Fit
    """
    c, tpl = _read(classes, "classes")
    dv, _ = _read(deforestation, "deforestation")
    _same(dv, c, "deforestation")
    regions = _regions(c, _subdivisions(subdivisions, c))
    d = _zeroed(dv)
    inside = regions != 0
    ids, inv = np.unique(regions[inside], return_inverse=True)
    cells = np.bincount(inv, minlength=len(ids)).astype(np.float64)
    hit = (d[inside] != 0.0) & ((regions[inside] * d[inside]) != 0.0)
    deforested = np.bincount(inv, weights=hit.astype(np.float64), minlength=len(ids))
    frequency = deforested / cells
    table = pd.DataFrame({"id": ids, "class": ids // 1000, "subdivision": ids % 1000, "deforested": deforested,
                          "cells": cells, "frequency": frequency})
    ha = _cell_ha(_transform(tpl))
    density = np.zeros(regions.shape)
    density[inside] = frequency[inv]
    density = (density * ha).astype(np.float32).astype(np.float64)
    return Fit(table, _out(density, tpl, "density"))


# -- the allocation ---------------------------------------------------------------------------------------------------

def _kahan(values) -> float:
    """pandas' groupby sum (Kahan's compensated summation)."""
    total, comp = 0.0, 0.0
    for v in values:
        y = v - comp
        t = total + y
        comp = t - total - y
        if math.isnan(comp):
            comp = 0.0
        total = t
    return total


@dataclass
class Allocation(Saveable):
    """VT0007's allocation of the expected deforestation (:func:`allocate`).

    Attributes
    ----------
    density : xarray.DataArray
        The expected deforestation of each cell, hectares (float32's values, as
        UDef-ARP writes it).
    adjustment_ratio : float
        The last adjustment ratio (1.00001 or less).
    iterations : int
        The adjustments made.
    expected_ha, allocated_ha : float
        The activity data and the hectares allocated.
    filled : int
        The modelling regions the fit lacked (their class's average).
    """

    density: xr.DataArray
    adjustment_ratio: float
    iterations: int
    expected_ha: float
    allocated_ha: float
    filled: int

    def __repr__(self) -> str:
        return (f"Allocation({self.allocated_ha:.1f} of {self.expected_ha:.1f} ha, ratio "
                f"{self.adjustment_ratio:.6f} after {self.iterations} iterations)")


def allocate(classes, fit: Fit | pd.DataFrame, *, expected: float | None = None, deforestation=None,
             subdivisions=None, max_iterations: int = 50) -> Allocation:
    """VT0007's allocation of the expected deforestation (UDef-ARP's prediction).

    The fitted relative frequency of every modelling region (a region the fit
    lacks takes its class's average over the subdivisions, weighted by their
    cells) gives each cell a density; the densities are scaled to the activity
    data (or a period's observed deforestation, to confirm the model), never
    above a cell's area, again until the adjustment ratio is 1.00001 or less.

    Parameters
    ----------
    classes
        The vulnerability classes of the period.
    fit
        The :class:`Fit` (or its table: ``id``, ``cells``, ``frequency``).
    expected
        The expected deforestation, hectares.
    deforestation
        Or the period's observed deforestation (1).
    subdivisions
        The subdivisions, as in the fit.
    max_iterations
        The adjustments at most (default 50).

    Returns
    -------
    Allocation
    """
    table = fit.table if isinstance(fit, Fit) else fit
    for col in ("id", "cells", "frequency"):
        if col not in table.columns:
            raise ValueError(f"the fit's table has no column {col!r}")
    fit_ids = table["id"].to_numpy(np.int64)
    fit_cells = table["cells"].to_numpy(np.float64)
    fit_freq = table["frequency"].to_numpy(np.float64)
    c, tpl = _read(classes, "classes")
    regions = _regions(c, _subdivisions(subdivisions, c))
    inside = regions != 0
    present, inv = np.unique(regions[inside], return_inverse=True)
    freqs = np.empty(len(present))
    filled = 0
    for k, rid in enumerate(present):
        hit = np.flatnonzero(fit_ids == rid)
        if hit.size:
            freqs[k] = fit_freq[hit[0]]
            continue
        filled += 1
        same = np.flatnonzero(fit_ids // 1000 == rid // 1000)
        if same.size == 0:
            raise ValueError(f"the vulnerability class {rid // 1000} has no cells in the fit")
        freqs[k] = _kahan(fit_cells[same] * fit_freq[same]) / _kahan(fit_cells[same])
    ha = _cell_ha(_transform(tpl))
    density = np.zeros(regions.shape)
    density[inside] = freqs[inv] * ha
    density = density.ravel()
    if (expected is None) == (deforestation is None):
        raise ValueError("give the expected deforestation (hectares) or a period's deforestation, one of them")
    if expected is not None:
        target = float(expected)
    else:
        dv, _ = _read(deforestation, "deforestation")
        _same(dv, c, "deforestation")
        target = float(np.sum(_zeroed(dv).ravel() * ha))
    ar = target / np.sum(density)
    it = 0
    while ar > 1.00001 and it <= max_iterations:
        density = np.minimum(ar * density, ha)
        ar = target / np.sum(density)
        it += 1
    if it > max_iterations:
        raise ValueError(f"the adjustment ratio is still {ar} after {max_iterations} iterations: the cells cannot "
                         "take the expected deforestation (raise max_iterations, or lower the expectation)")
    # the last ratio applied once more, as UDef-ARP writes the map (float32)
    out = np.minimum(ar * density, ha).astype(np.float32).astype(np.float64)
    return Allocation(_out(out.reshape(regions.shape), tpl, "density"), float(ar), it, target, float(np.sum(out)),
                      filled)


# -- the assessment ---------------------------------------------------------------------------------------------------

def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Theil and Sen's slope (pairs of different x) and intercept (scipy's theilslopes)."""
    slopes = [(y[i] - y[j]) / (x[i] - x[j]) for i in range(len(x)) for j in range(i) if x[i] != x[j]]
    slope = float(np.median(slopes)) if slopes else math.nan
    return slope, float(np.median(y)) - slope * float(np.median(x))


@dataclass
class Assessment(Saveable):
    """VT0007's assessment of an allocation (:func:`assess`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per assessment cell: ``cell``, ``x``, ``y`` (its centre),
        ``actual_ha``, ``predicted_ha``, ``residual_ha``.
    metrics : dict
        ``samples``, ``mae``, ``mae_percent``, ``medae``, ``medae_percent``
        (the absolute errors, also as shares of an assessment cell),
        ``iou_percent`` (the intersection over the union), ``agreement``,
        ``difference``, ``r2``, ``ols_slope``, ``ols_intercept``,
        ``theil_sen_slope``, ``theil_sen_intercept``.
    residuals : xarray.DataArray
        Each assessment cell's residual (predicted - actual, ha) on its cells.
    """

    table: pd.DataFrame
    metrics: dict
    residuals: xr.DataArray

    def __repr__(self) -> str:
        m = self.metrics
        return f"Assessment({int(m['samples'])} cells, MAE {m['mae']:.3f} ha, R² {m['r2']:.3f})"


def assess(density, deforestation, *, grid_area: float, mask=None) -> Assessment:
    """VT0007's assessment of an allocation (UDef-ARP's model evaluation).

    The predicted and the actual deforestation in the square cells of an
    assessment grid wholly inside the jurisdiction (centred on the grid's
    nodes from the map's corner, as UDef-ARP's Thiessen polygons), and their
    agreement: the mean and median absolute errors, the intersection over
    union, R², the ordinary least squares and Theil-Sen lines.

    Parameters
    ----------
    density
        The predicted deforestation of each cell, hectares (:func:`allocate`).
    deforestation
        The actual deforestation (1).
    grid_area
        The area of an assessment cell, hectares.
    mask
        The jurisdiction (non-zero; default all the map).

    Returns
    -------
    Assessment
    """
    p, tpl = _read(density, "density")
    dv, _ = _read(deforestation, "deforestation")
    _same(dv, p, "deforestation")
    m = _mask(mask, p)
    t = _transform(tpl)
    px = math.trunc(t[1])
    size = math.floor(math.trunc(math.sqrt(float(grid_area) * 10_000.0)) / px) if px > 0 else 0
    if not size >= 1:
        raise ValueError("grid_area: at least a cell's area, hectares")
    ny, nx = p.shape
    ha = _cell_ha(t)
    half = size / 2.0
    inside = m != 0.0
    squares = []
    for j in range(-size, ny + size + 1, size):
        for k in range(-size, nx + size + 1, size):
            x0, x1, y0, y1 = k - half, k + half, j - half, j + half
            c0, c1 = int(max(math.floor(x0), 0.0)), int(min(math.ceil(x1), nx))
            r0, r1 = int(max(math.floor(y0), 0.0)), int(min(math.ceil(y1), ny))
            if c1 <= c0 or r1 <= r0:
                continue
            sub = inside[r0:r1, c0:c1]
            if not sub.any():
                continue
            cc = np.arange(c0, c1)
            rr = np.arange(r0, r1)
            ox = np.maximum(np.minimum(x1, cc + 1.0) - np.maximum(x0, cc.astype(float)), 0.0)
            oy = np.maximum(np.minimum(y1, rr + 1.0) - np.maximum(y0, rr.astype(float)), 0.0)
            # the area in the cells' order of a row-by-row sum
            area = 0.0
            for ri in range(len(rr)):
                for ci in range(len(cc)):
                    if sub[ri, ci]:
                        area += ox[ci] * oy[ri]
            if area <= 0.0:
                continue
            mx, my = cc + 0.5, rr + 0.5
            hold = sub & ((my >= y0) & (my < y1))[:, None] & ((mx >= x0) & (mx < x1))[None, :]
            ri, ci = np.nonzero(hold)
            squares.append((k, j, area, (rr[ri] * nx + cc[ci])))
    max_area = max((s[2] for s in squares), default=0.0)
    squares = [s for s in squares if s[2] / max_area > 0.999]
    if not squares:
        raise ValueError("no assessment cell fits in the jurisdiction: a smaller grid_area")
    pf, df = p.ravel(), dv.ravel()
    residuals = np.full(p.size, np.nan)
    rows, xs, ys = [], [], []
    for n, (k, j, _, cells) in enumerate(squares):
        dd = df[cells]
        actual = float(np.count_nonzero((dd != 0.0) & ~np.isnan(dd))) * ha
        predicted = 0.0
        for v in pf[cells]:
            predicted += 0.0 if math.isnan(v) else v
        rows.append((n + 1, t[0] + k * t[1], t[3] + j * t[5], actual, predicted, predicted - actual))
        residuals[cells] = predicted - actual
        xs.append(actual)
        ys.append(predicted)
    x, y = np.array(xs), np.array(ys)
    count = float(len(x))
    diff = np.abs(x - y)
    medae = float(np.median(diff))
    difference = sum(diff.tolist(), 0.0)
    mae = difference / count
    agreement = sum(np.minimum(x, y).tolist(), 0.0)
    union = sum(np.maximum(x, y).tolist(), 0.0)
    iou = 0.0 if union == 0.0 else agreement / union * 100.0
    mean_x, mean_y = sum(xs, 0.0) / count, sum(ys, 0.0) / count
    sxy = sum(((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys)), 0.0)
    sxx = sum(((a - mean_x) * (a - mean_x) for a in xs), 0.0)
    syy = sum(((b - mean_y) * (b - mean_y) for b in ys), 0.0)
    slope = sxy / sxx if sxx else math.nan
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else math.nan
    ts, ti = _theil_sen(x, y)
    area = math.trunc(float(grid_area))
    metrics = {"samples": int(count), "mae": mae, "mae_percent": mae / area * 100.0, "medae": medae,
               "medae_percent": medae / area * 100.0, "iou_percent": iou, "agreement": agreement,
               "difference": difference, "r2": r * r, "ols_slope": slope, "ols_intercept": mean_y - slope * mean_x,
               "theil_sen_slope": ts, "theil_sen_intercept": ti}
    table = pd.DataFrame(rows, columns=["cell", "x", "y", "actual_ha", "predicted_ha", "residual_ha"])
    return Assessment(table, metrics, _out(residuals.reshape(p.shape), tpl, "residual_ha"))
