# SPDX-License-Identifier: GPL-3.0-or-later
"""The accuracy and the area of a map from a stratified random sample.

The good practices of Olofsson et al. (2014, *Remote Sensing of Environment*
148, 42-57): :func:`sample_size` gives the size of the sample and its share of
each stratum, :func:`sample` draws the points, and :func:`assess` takes them
back labelled with their reference class (from imagery or the field): the
error matrix in shares of the area, the user's, producer's and overall
accuracies and the area of each class estimated from the sample, with standard
errors and confidence intervals, as R's mapaccuracy computes them (``olofsson``
when the strata are the map's classes, ``stehman2014`` for strata of their
own). The same as Terraflux's ``accuracy.*`` operators.

Typical use::

    import landschaft as ls

    lulc = ls.load_raster("map_2020.tif")
    design = ls.accuracy.sample_size(lulc, user_accuracy={3: 0.7, 4: 0.6}, standard_error=0.01, minimum=75)
    points = ls.accuracy.sample(lulc, design, seed=1)
    ls.write_vector(points, "to_label.gpkg")           # fill the column ``reference``
    acc = ls.accuracy.assess(lulc, ls.load_vector("labelled.gpkg"))
    acc.table[["class", "area_ha", "area_ha_ci", "user_accuracy", "producer_accuracy"]]
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._save import Saveable

__all__ = ["Accuracy", "SampleSize", "allocate", "assess", "qnorm", "sample", "sample_size"]

ALLOCATIONS = ("minimum", "proportional", "equal", "neyman")


# -- the map and its strata ------------------------------------------------------------------------------------------

def _map(x, nodata=None, what: str = "the map"):
    """(values as float64 with NaN, template DataArray or None)."""
    from .simulate._maps import read_map

    try:
        m = read_map(x, nodata)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: {e}") from None
    return m.as_float(), m.template


def _class_cells(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The classes of a map (increasing) and the cells of each."""
    v = values[~np.isnan(values)]
    classes, cells = np.unique(v, return_counts=True)
    return classes.astype(np.float64), cells.astype(np.int64)


def _label(c: float):
    return int(c) if float(c).is_integer() else float(c)


def _per_class(v, classes: np.ndarray, default: float, what: str) -> np.ndarray:
    """A number per class: one for all, or ``{class: value}`` (``default`` for the others)."""
    if v is None:
        return np.full(len(classes), float(default))
    if isinstance(v, pd.Series):
        v = v.to_dict()
    if isinstance(v, dict):
        out = np.full(len(classes), float(default))
        for k, x in v.items():
            hit = np.flatnonzero(classes == float(k))
            if hit.size == 0:
                raise ValueError(f"{what}: the map has no class {k}")
            out[hit[0]] = float(x)
        return out
    return np.full(len(classes), float(v))


# -- the size of the sample ------------------------------------------------------------------------------------------

def _apportion(n: int, weights, caps) -> list[int]:
    """``n`` shared in proportion to ``weights`` without going over ``caps``, in whole numbers adding up to ``n``
    (the largest remainders take what is left, the first on ties)."""
    k = len(weights)
    target = [0.0] * k
    capped = [False] * k
    while True:
        fixed = sum(caps[i] for i in range(k) if capped[i])
        left = float(max(n - fixed, 0))
        w = 0.0
        for i in range(k):
            if not capped[i]:
                w += weights[i]
        again = False
        for i in range(k):
            if capped[i]:
                target[i] = float(caps[i])
            else:
                target[i] = left * weights[i] / w if w > 0.0 else 0.0
                if target[i] > caps[i]:
                    capped[i] = True
                    again = True
        if not again:
            break
    out = [math.floor(t) for t in target]
    left = max(n - sum(out), 0)
    order = sorted(range(k), key=lambda i: (-(target[i] - math.floor(target[i])), i))
    while left > 0:
        before = left
        for i in order:
            if left > 0 and out[i] < caps[i]:
                out[i] += 1
                left -= 1
        if left == before:
            break
    return out


def allocate(n: int, weights, sd, cells, allocation: str = "minimum", minimum: int = 50) -> tuple[list[int], bool]:
    """The sample of each stratum, and whether the minima asked for more than ``n``.

    ``allocation``: ``"minimum"`` (at least ``minimum`` in every stratum, or all
    its cells, the rest in proportion to the areas: the good practice when some
    classes are rare, such as change), ``"proportional"`` (to the strata's
    shares ``weights``), ``"equal"`` or ``"neyman"`` (to the shares times the
    standard deviations ``sd``). No stratum gets more than its ``cells``.

    Examples
    --------
    >>> import landschaft as ls
    >>> w = [0.02, 0.015, 0.32, 0.645]                       # Olofsson et al. (2014), section 5
    >>> sd = [(u * (1 - u)) ** 0.5 for u in (0.7, 0.6, 0.9, 0.95)]
    >>> ls.accuracy.allocate(641, w, sd, [200_000, 150_000, 3_200_000, 6_450_000], minimum=75)
    ([75, 75, 163, 328], False)
    """
    weights = [float(w) for w in weights]
    sd = [float(s) for s in sd]
    cells = [int(c) for c in cells]
    k = len(weights)
    if allocation == "proportional":
        return _apportion(n, weights, cells), False
    if allocation == "equal":
        return _apportion(n, [1.0] * k, cells), False
    if allocation == "neyman":
        return _apportion(n, [w * s for w, s in zip(weights, sd)], cells), False
    if allocation != "minimum":
        raise ValueError(f"allocation: {allocation!r} is not one of {ALLOCATIONS}")
    floor = [min(c, int(minimum)) for c in cells]
    fixed = [False] * k
    while True:
        given = sum(floor[i] for i in range(k) if fixed[i])
        left = float(max(n - given, 0))
        w = 0.0
        for i in range(k):
            if not fixed[i]:
                w += weights[i]
        again = False
        for i in range(k):
            if not fixed[i] and (w <= 0.0 or left * weights[i] / w < floor[i]):
                fixed[i] = True
                again = True
        if not again:
            break
    given = sum(floor[i] for i in range(k) if fixed[i])
    rest = [0.0 if fixed[i] else weights[i] for i in range(k)]
    caps = [0 if fixed[i] else cells[i] for i in range(k)]
    shared = _apportion(max(n - given, 0), rest, caps)
    return [floor[i] if fixed[i] else shared[i] for i in range(k)], given > n


@dataclass
class SampleSize(Saveable):
    """The size of a stratified sample and its share of each stratum (:func:`sample_size`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per stratum: ``stratum``, ``cells``, ``weight`` (its share of the
        map), ``user_accuracy`` (expected), ``sd`` and ``n`` (its sample).
    n : int
        The size of the sample (the sum of ``n``).
    """

    table: pd.DataFrame
    n: int

    def __repr__(self) -> str:
        return f"SampleSize(n={self.n}, strata={len(self.table)})"


def sample_size(map, *, user_accuracy=0.9, standard_error: float = 0.01, n: int | None = None,
                allocation: str = "minimum", minimum: int = 50, nodata=None) -> SampleSize:
    """The size of a stratified random sample for the accuracy and the area of a map, and its share of each stratum.

    Olofsson et al. (2014), equation 13: ``n = (Σ W_i S_i / S(Ô))²``, ``W_i`` a
    stratum's share of the map, ``S_i = √(U_i (1 − U_i))`` from the user's
    accuracy expected of it, ``S(Ô)`` the standard error wanted of the overall
    accuracy. Shared with a minimum in every stratum (the good practice when
    some classes are rare, such as change; the rest in proportion to the
    areas), in proportion, equally or by Neyman's allocation.

    Parameters
    ----------
    map
        The strata: the classified map, or a map of strata (a 2-D DataArray or
        array; NaN and ``nodata`` are missing).
    user_accuracy
        The user's accuracy expected of each stratum: a number for all (default
        0.9), or ``{class: accuracy}`` (the others 0.9); lower for the classes
        hard to map, such as change (Olofsson's 0.6 to 0.7).
    standard_error
        The standard error wanted of the overall accuracy (default 0.01).
    n
        The size of the sample, instead of the formula's.
    allocation
        ``"minimum"`` (default), ``"proportional"``, ``"equal"`` or ``"neyman"``.
    minimum
        The sample of every stratum at least, or all its cells (``"minimum"``;
        default 50; Olofsson suggests 50 to 100).
    nodata
        Value marking missing cells.

    Returns
    -------
    SampleSize

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> m = np.repeat([1, 2, 3, 4], [40, 30, 640, 1290]).reshape(1, -1)
    >>> s = ls.accuracy.sample_size(m, user_accuracy={1: 0.7, 2: 0.6, 3: 0.9, 4: 0.95}, minimum=30)
    >>> s.n, s.table["n"].tolist()
    (641, [30, 30, 193, 388])
    """
    values, _ = _map(map, nodata)
    classes, cells = _class_cells(values)
    if classes.size == 0:
        raise ValueError("the map has no cells with a value")
    total = int(cells.sum())
    weights = cells / total
    ua = _per_class(user_accuracy, classes, 0.9, "user_accuracy")
    bad = ua[(ua < 0) | (ua > 1) | np.isnan(ua)]
    if bad.size:
        raise ValueError(f"user_accuracy: {bad[0]} is not between 0 and 1")
    sd = np.sqrt(ua * (1.0 - ua))
    if n is None:
        if not standard_error > 0:
            raise ValueError("standard_error must be above 0")
        s = 0.0
        for w, d in zip(weights, sd):
            s += w * d
        n = math.ceil((s / standard_error) * (s / standard_error))
    n = min(int(n), total)
    sizes, over = allocate(n, weights, sd, cells, allocation, minimum)
    if over:
        warnings.warn(f"the minimum of every stratum asks for {sum(sizes)} points, more than the {n} of the sample",
                      stacklevel=2)
    table = pd.DataFrame({"stratum": [_label(c) for c in classes], "cells": cells, "weight": weights,
                          "user_accuracy": ua, "sd": sd, "n": np.array(sizes, dtype=np.int64)})
    return SampleSize(table, int(sum(sizes)))


# -- the sample ------------------------------------------------------------------------------------------------------

def _transform_of(template, shape):
    from affine import Affine

    if template is None:
        return Affine(1, 0, 0, 0, -1, shape[0]), None
    from .io import _transform

    try:
        crs = template.rio.crs
    except Exception:  # noqa: BLE001  (no rioxarray georeference)
        crs = None
    return _transform(template), crs


def sample(map, sizes=None, *, n=None, seed: int = 1, nodata=None):
    """A stratified random sample of a map's cells for its accuracy and area (Olofsson et al. 2014).

    In every stratum its share of cells drawn without replacement, as numpy's
    ``default_rng(seed).choice`` of the stratum's cells, stratum after stratum
    (increasing); the points at the cells' centres, by stratum and cell. Give
    each point its reference class (the column ``reference``, from imagery or
    the field) and take them to :func:`assess`.

    Parameters
    ----------
    map
        The strata: the classified map, or a map of strata.
    sizes
        The sample of each stratum: a :class:`SampleSize` (or its table: columns
        ``stratum`` and ``n``).
    n
        The sample of each stratum instead of ``sizes``: a number for all, or
        ``{stratum: n}``.
    seed
        The seed of numpy's generator (default 1).
    nodata
        Value marking missing cells.

    Returns
    -------
    geopandas.GeoDataFrame
        The points (in the map's CRS): ``id``, ``stratum``, ``row``, ``col``,
        ``x``, ``y`` and ``reference`` (empty, to be filled).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> m = np.repeat([1, 2], [90, 10]).reshape(10, 10)
    >>> pts = ls.accuracy.sample(m, n={1: 5, 2: 3}, seed=7)
    >>> pts["stratum"].tolist(), len(pts)
    ([1, 1, 1, 1, 1, 2, 2, 2], 8)
    """
    import geopandas as gpd

    values, template = _map(map, nodata)
    classes, cells = _class_cells(values)
    if sizes is not None and n is not None:
        raise ValueError("the sizes or n, not both")
    if sizes is not None:
        t = sizes.table if isinstance(sizes, SampleSize) else sizes
        if not isinstance(t, pd.DataFrame) or "stratum" not in t.columns or "n" not in t.columns:
            raise ValueError("sizes: a SampleSize or a table with the columns stratum and n")
        want = np.zeros(len(classes))
        for s, k in zip(t["stratum"], t["n"]):
            hit = np.flatnonzero(classes == float(s))
            if hit.size == 0:
                raise ValueError(f"sizes: the map has no stratum {s}")
            want[hit[0]] = float(k)
    elif n is not None:
        want = _per_class(n, classes, 0.0, "n")
    else:
        raise ValueError("give the sizes (ls.accuracy.sample_size) or n")
    g = np.random.default_rng(seed)
    flat = values.ravel()
    drawn, strata = [], []
    for k, c in enumerate(classes):
        m = want[k]
        if m < 0 or m != int(m):
            raise ValueError(f"stratum {_label(c)}: {m} is not a number of points")
        m = int(m)
        if m > cells[k]:
            raise ValueError(f"stratum {_label(c)}: {m} points, but it has {cells[k]} cells")
        if m == 0:
            warnings.warn(f"stratum {_label(c)}: no points", stacklevel=2)
            continue
        idx = np.flatnonzero(flat == c)
        drawn.append(np.sort(idx[g.choice(idx.size, size=m, replace=False)]))
        strata += [_label(c)] * m
    cells_drawn = np.concatenate(drawn) if drawn else np.empty(0, dtype=np.int64)
    width = values.shape[1]
    row, col = cells_drawn // width, cells_drawn % width
    tr, crs = _transform_of(template, values.shape)
    x = tr.c + (col + 0.5) * tr.a + (row + 0.5) * tr.b
    y = tr.f + (col + 0.5) * tr.d + (row + 0.5) * tr.e
    df = pd.DataFrame({"id": np.arange(1, len(cells_drawn) + 1), "stratum": strata, "row": row, "col": col,
                       "x": x, "y": y, "reference": np.full(len(cells_drawn), np.nan)})
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=crs)


# -- the estimates ---------------------------------------------------------------------------------------------------

@dataclass
class _Estimates:
    classes: np.ndarray
    overall: float
    overall_se: float
    user: np.ndarray
    user_se: np.ndarray
    producer: np.ndarray
    producer_se: np.ndarray
    area: np.ndarray
    area_se: np.ndarray
    matrix: np.ndarray
    total: float


def _sum_na(values) -> float:
    s = 0.0
    for v in values:
        if not math.isnan(v):
            s += v
    return s


def _finite(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def _div(a: float, b: float) -> float:
    """IEEE division (R's, Rust's): x / 0 is ±Inf or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _sq(x: float) -> float:
    return x * x


def _sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else math.nan


def _olofsson(m, r, classes, cells) -> _Estimates:
    """mapaccuracy's ``olofsson(r, m, Nh)``: the strata are the map's classes, ``cells`` the cells of each of
    ``classes``; classes of the reference that are not in the map come after them, without cells."""
    cls = [float(c) for c in classes]
    nhc = [float(c) for c in cells]
    for v in m:
        if v not in cls:
            raise ValueError(f"the map's class {_label(v)} of a sample has no cells in the map")
    extra = len(cls)
    for v in r:
        if v not in cls:
            cls.append(float(v))
            nhc.append(0.0)
    q = len(cls)
    pos = {c: i for i, c in enumerate(cls)}
    count = [[0.0] * q for _ in range(q)]
    for a, b in zip(m, r):
        count[pos[a]][pos[b]] += 1.0
    total = 0.0
    for v in nhc:
        total += v
    nh = [sum(row) for row in count]
    wh = [v / total for v in nhc]
    props = [[0.0 if (i >= extra and j >= extra) else _div(count[i][j], nh[i]) * wh[i] for j in range(q)]
             for i in range(q)]

    def row(i):
        return _sum_na(props[i])

    def col(j):
        return _sum_na(props[i][j] for i in range(q))

    overall = _sum_na(props[j][j] for j in range(q))
    user = [_div(props[j][j], row(j)) for j in range(q)]
    producer = [_div(props[j][j], col(j)) for j in range(q)]
    area = [col(j) for j in range(q)]
    var_oa = 0.0
    for j in range(q):
        var_oa += _finite(_div(_sq(wh[j]) * user[j] * (1.0 - user[j]), nh[j] - 1.0))
    user_se = []
    for j in range(q):
        if math.isnan(user[j]):
            user_se.append(math.nan)
            continue
        v = _div(user[j] * (1.0 - user[j]), nh[j] - 1.0)
        user_se.append(0.0 if math.isnan(v) else _sqrt(v))
    n_j = [_sum_na(_div(nhc[i], nh[i]) * count[i][j] for i in range(q)) for j in range(q)]
    producer_se = []
    for j in range(q):
        t1 = _finite(_div(_sq(nhc[j]) * _sq(1.0 - producer[j]) * user[j] * (1.0 - user[j]), nh[j] - 1.0))
        t2 = _finite(_sum_na(_div(_div(_sq(nhc[i]) * count[i][j], nh[i]) * (1.0 - _div(count[i][j], nh[i])),
                                  nh[i] - 1.0) for i in range(q) if i != j))
        producer_se.append(_sqrt(_div(1.0, _sq(n_j[j])) * (t1 + _sq(producer[j]) * t2)))
    area_se = []
    for j in range(q):
        s = _sum_na(_sq(wh[i]) * _div(_div(count[i][j], nh[i]) * (1.0 - _div(count[i][j], nh[i])), nh[i] - 1.0)
                    for i in range(q))
        area_se.append(_sqrt(s))
    matrix = np.array([[math.nan if count[i][j] == 0.0 else props[i][j] for j in range(q)] for i in range(q)])
    return _Estimates(np.array(cls), overall, math.sqrt(var_oa), np.array(user), np.array(user_se),
                      np.array(producer), np.array(producer_se), np.array(area), np.array(area_se), matrix, total)


def _moments(y, x):
    """The means and the sample variances and covariance (n - 1); NaN variances of fewer than two values as 0."""
    n = float(len(y))
    sy = 0.0
    for v in y:
        sy += v
    sx = 0.0
    for v in x:
        sx += v
    my, mx = sy / n, sx / n
    if len(y) < 2:
        return my, mx, 0.0, 0.0, 0.0
    vy = vx = c = 0.0
    for v in y:
        vy += _sq(v - my)
    for v in x:
        vx += _sq(v - mx)
    for a, b in zip(y, x):
        c += (a - my) * (b - mx)
    return my, mx, vy / (n - 1.0), vx / (n - 1.0), c / (n - 1.0)


def _stehman(s, m, r, strata_classes, cells) -> _Estimates:
    """mapaccuracy's ``stehman2014(s, r, m, Nh_strata)``: strata of their own; the strata without a sample unit
    are left out of the total. The classes are those of the map and of the reference, increasing."""
    sc = [float(c) for c in strata_classes]
    for v in s:
        if v not in sc:
            raise ValueError(f"the stratum {_label(v)} of a sample has no cells")
    kept = [h for h in range(len(sc)) if sc[h] in set(s)]
    units = [[u for u in range(len(s)) if s[u] == sc[h]] for h in kept]
    big_n = [float(cells[h]) for h in kept]
    small_n = [float(len(u)) for u in units]
    total = 0.0
    for v in big_n:
        total += v
    order = sorted(set(m) | set(r))
    where = {sc[h]: k for k, h in enumerate(kept)}
    incl = [small_n[where[s[u]]] / big_n[where[s[u]]] for u in range(len(s))]
    nu = len(s)

    def eq2(y):
        acc = 0.0
        for u in range(nu):
            acc += y(u) / incl[u]
        return acc / total

    def eq25(var):
        acc = 0.0
        for h in range(len(kept)):
            acc += _sq(big_n[h]) * (1.0 - small_n[h] / big_n[h]) * var[h] / small_n[h]
        return acc * (1.0 / _sq(total))

    def eq28(rr, x, vy, vx, cov):
        acc = 0.0
        for h in range(len(kept)):
            acc += _sq(big_n[h]) * (1.0 - small_n[h] / big_n[h]) * (vy[h] + _sq(rr) * vx[h] - 2.0 * rr * cov[h]) \
                / small_n[h]
        return acc * _div(1.0, _sq(x))

    def by_stratum(y, x):
        out = ([], [], [], [], [])
        for u in units:
            res = _moments([y(i) for i in u], [x(i) for i in u])
            for k in range(5):
                out[k].append(res[k])
        return out

    def ind(b):
        return 1.0 if b else 0.0

    def hit(u):
        return ind(m[u] == r[u])

    overall = eq2(hit)
    overall_se = _sqrt(eq25(by_stratum(hit, hit)[2]))
    area, area_se, user, user_se, producer, producer_se = [], [], [], [], [], []

    def ratio(y, x):
        my, mx, vy, vx, cov = by_stratum(y, x)
        num = den = 0.0
        for h in range(len(kept)):
            num += my[h] * big_n[h]
        for h in range(len(kept)):
            den += mx[h] * big_n[h]
        rr = _div(num, den)
        return rr, _sqrt(eq28(rr, den, vy, vx, cov))

    for c in order:
        def is_r(u, c=c):
            return ind(r[u] == c)

        area.append(eq2(is_r))
        area_se.append(_sqrt(eq25(by_stratum(is_r, is_r)[2])))
        a, b = ratio(lambda u, c=c: ind(m[u] == c and m[u] == r[u]), lambda u, c=c: ind(m[u] == c))
        user.append(a)
        user_se.append(b)
        a, b = ratio(lambda u, c=c: ind(r[u] == c and r[u] == m[u]), lambda u, c=c: ind(r[u] == c))
        producer.append(a)
        producer_se.append(b)
    matrix = np.array([[eq2(lambda u, i=i, j=j: ind(m[u] == i and r[u] == j)) for j in order] for i in order])
    matrix[matrix == 0.0] = np.nan
    return _Estimates(np.array(order, dtype=np.float64), overall, overall_se, np.array(user), np.array(user_se),
                      np.array(producer), np.array(producer_se), np.array(area), np.array(area_se), matrix, total)


def qnorm(p: float) -> float:
    """The standard normal's quantile (Wichura's AS 241, as R's ``qnorm``).

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.accuracy.qnorm(0.975), 12)
    1.95996398454
    """
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                         + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                     + 133.14166789178437745) * r + 3.387132872796366608) \
            / (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r
                   + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                + 42.313330701600911252) * r + 1.0)
    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        val = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                   + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                + 4.6303378461565452959) * r + 1.42343711074968357734) \
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                   + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5.0
        val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                   + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                + 5.4637849111641143699) * r + 6.6579046435011037772) \
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                   + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                + 0.59983220655588793769) * r + 1.0)
    return -val if q < 0.0 else val


@dataclass
class Accuracy(Saveable):
    """The accuracy and the area of a map's classes from a stratified sample (:func:`assess`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per class: ``class``, ``map_cells``, ``samples``,
        ``user_accuracy``, ``user_se``, ``user_ci``, ``producer_accuracy``,
        ``producer_se``, ``producer_ci``, ``area_share``, ``area_share_se``,
        ``area_cells``, ``area_cells_ci``, ``area_ha``, ``area_ha_ci`` (the
        ``_ci`` the half-widths of the confidence intervals).
    overall : pandas.DataFrame
        ``overall_accuracy``, ``overall_se``, ``overall_ci``, ``samples``,
        ``cells``, ``confidence``.
    matrix : pandas.DataFrame
        The error matrix in shares of the area: a row per class of the map, a
        column per reference class (NaN where no sample fell).
    counts : pandas.DataFrame
        The error matrix in sample units.
    estimator : str
        ``"olofsson"`` or ``"stehman"``.
    """

    table: pd.DataFrame
    overall: pd.DataFrame
    matrix: pd.DataFrame
    counts: pd.DataFrame
    estimator: str

    def __repr__(self) -> str:
        o = self.overall.iloc[0]
        return (f"Accuracy({self.estimator}: overall {o['overall_accuracy']:.4f} +/- {o['overall_ci']:.4f}, "
                f"{int(o['samples'])} samples)")


def _units(values, strata_values, template, points, samples, column):
    """The sample units: (map class, reference class, stratum) of each, from points over the map or a table."""
    if points is not None and samples is not None:
        raise ValueError("the points or the samples (a table), not both")
    if points is not None:
        tr, crs = _transform_of(template, values.shape)
        if tr.b != 0.0 or tr.d != 0.0:
            raise ValueError("the map's grid is rotated")
        if crs is not None and getattr(points, "crs", None) is not None and points.crs != crs:
            points = points.to_crs(crs)
        if column not in points.columns:
            raise ValueError(f"the points have no column {column!r} (they have {list(points.columns)})")
        geom = points.geometry
        if not geom.is_empty.all() and not (geom[~geom.is_empty].geom_type == "Point").all():
            raise ValueError("the sample's features must be points")
        ref = pd.to_numeric(points[column], errors="coerce").to_numpy(np.float64)
        ok = ~geom.is_empty.to_numpy()
        px = np.where(ok, geom.x.where(ok, 0.0), np.nan) if len(geom) else np.empty(0)
        py = np.where(ok, geom.y.where(ok, 0.0), np.nan) if len(geom) else np.empty(0)
        unlabelled = int((ok & np.isnan(ref)).sum())
        keep = ok & ~np.isnan(ref)
        c = np.floor((px - tr.c) / tr.a)
        rr = np.floor((py - tr.f) / tr.e)
        ny, nx = values.shape
        inside = keep & (c >= 0) & (rr >= 0) & (c < nx) & (rr < ny)
        m = np.full(len(ref), np.nan)
        s = np.full(len(ref), np.nan)
        ri, ci = rr[inside].astype(np.int64), c[inside].astype(np.int64)
        m[inside] = values[ri, ci]
        s[inside] = (values if strata_values is None else strata_values)[ri, ci]
        good = inside & ~np.isnan(m) & ~np.isnan(s)
        outside = int((keep & ~good).sum())
        if unlabelled:
            warnings.warn(f"{unlabelled} points without a reference class left out", stacklevel=3)
        if outside:
            warnings.warn(f"{outside} points outside the map or on its missing cells left out", stacklevel=3)
        return m[good], ref[good], s[good]
    if samples is not None:
        if not isinstance(samples, pd.DataFrame):
            raise TypeError(f"samples: a table, not a {type(samples).__name__}")
        for name in ("map", column) + (("stratum",) if strata_values is not None else ()):
            if name not in samples.columns:
                raise ValueError(f"the samples have no column {name!r}")
        m = pd.to_numeric(samples["map"], errors="coerce").to_numpy(np.float64)
        r = pd.to_numeric(samples[column], errors="coerce").to_numpy(np.float64)
        s = (pd.to_numeric(samples["stratum"], errors="coerce").to_numpy(np.float64) if strata_values is not None
             else m.copy())
        good = ~(np.isnan(m) | np.isnan(r) | np.isnan(s))
        return m[good], r[good], s[good]
    raise ValueError("give the sample: points labelled with their reference class, or a table (samples=)")


def assess(map, points=None, *, samples=None, strata=None, reference: str = "reference",
           estimator: str = "auto", confidence: float = 0.95, cell_area: float | None = None,
           nodata=None) -> Accuracy:
    """The accuracy and the area of a map's classes from a stratified random sample labelled with the reference.

    Olofsson et al. (2014): the error matrix in shares of the area, the
    user's, producer's and overall accuracies, and each class's area estimated
    from the sample instead of counted on the map, with standard errors and
    confidence intervals; as R's mapaccuracy (to 1e-12): ``olofsson`` when the
    strata are the map's classes, ``stehman2014`` for strata of their own (a
    map of strata), with the finite population correction.

    Parameters
    ----------
    map
        The classified map: its class at each point and its cells by class.
    points
        The sample's points (a GeoDataFrame, e.g. :func:`sample`'s) with their
        reference class in the column ``reference``.
    samples
        The sample as a table instead: columns ``map`` and ``reference`` (and
        ``stratum``, with ``strata``).
    strata
        The strata of the sample when they are not the map's classes (e.g. an
        older map, or a buffer around change).
    reference
        The column of the reference class (default ``"reference"``).
    estimator
        ``"olofsson"`` (the strata are the map's classes), ``"stehman"``
        (strata of their own, the finite population correction); ``"auto"``
        (default): stehman with ``strata``, else olofsson.
    confidence
        The level of the confidence intervals (default 0.95).
    cell_area
        A cell's area in m² (default: the product of the resolutions; 1 for an
        array without georeference).
    nodata
        Value marking missing cells.

    Returns
    -------
    Accuracy

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> m = np.repeat([1, 2], [900, 100]).reshape(20, 50)
    >>> s = pd.DataFrame({"map": [1] * 50 + [2] * 50, "reference": [1] * 48 + [2] * 2 + [2] * 45 + [1] * 5})
    >>> acc = ls.accuracy.assess(m, samples=s, cell_area=900)
    >>> acc.table[["class", "user_accuracy", "producer_accuracy", "area_ha"]].round(3)
       class  user_accuracy  producer_accuracy  area_ha
    0      1           0.96              0.989    78.66
    1      2           0.90              0.714    11.34
    >>> round(float(acc.overall["overall_accuracy"].iloc[0]), 3)
    0.954
    """
    values, template = _map(map, nodata)
    strata_values = None
    if strata is not None:
        strata_values, _ = _map(strata, nodata, "the strata")
        if strata_values.shape != values.shape:
            raise ValueError("the strata and the map differ in size")
    if estimator == "auto":
        use_stehman = strata is not None
    elif estimator == "olofsson":
        if strata is not None:
            raise ValueError("olofsson takes the map's classes as strata: leave the strata out, or use stehman")
        use_stehman = False
    elif estimator == "stehman":
        use_stehman = True
    else:
        raise ValueError(f"estimator: {estimator!r} is not auto, olofsson or stehman")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z = qnorm(0.5 + confidence / 2.0)
    if cell_area is None:
        from .simulate._maps import Map, cell_area_of

        cell_area = cell_area_of(Map(values, None, template), None)
    m, r, s = _units(values, strata_values, template, points, samples, reference)
    if m.size == 0:
        raise ValueError("no sample unit has a reference class")
    m, r, s = m.tolist(), r.tolist(), s.tolist()
    classes, cells = _class_cells(values)
    if use_stehman:
        if strata_values is not None:
            sc, scells = _class_cells(strata_values)
        else:
            sc, scells = classes, cells
        for c in sc:
            if s.count(float(c)) == 1:
                warnings.warn(f"the stratum {_label(c)} has a single sample unit", stacklevel=2)
        e = _stehman(s, m, r, sc, scells)
    else:
        e = _olofsson(m, r, classes, cells)
    idx = np.argsort(e.classes, kind="stable")
    rows = []
    for j in idx:
        c = e.classes[j]
        hit = np.flatnonzero(classes == c)
        map_cells = float(cells[hit[0]]) if hit.size else 0.0
        a, a_ci = e.area[j] * e.total, z * e.area_se[j] * e.total
        rows.append((_label(c), map_cells, float(sum(1 for v in m if v == c)), e.user[j], e.user_se[j],
                     z * e.user_se[j], e.producer[j], e.producer_se[j], z * e.producer_se[j], e.area[j], e.area_se[j],
                     a, a_ci, a * cell_area / 10000.0, a_ci * cell_area / 10000.0))
    table = pd.DataFrame(rows, columns=["class", "map_cells", "samples", "user_accuracy", "user_se", "user_ci",
                                        "producer_accuracy", "producer_se", "producer_ci", "area_share",
                                        "area_share_se", "area_cells", "area_cells_ci", "area_ha", "area_ha_ci"])
    overall = pd.DataFrame([{"overall_accuracy": e.overall, "overall_se": e.overall_se,
                             "overall_ci": z * e.overall_se, "samples": float(len(m)), "cells": e.total,
                             "confidence": confidence}])
    labels = [_label(e.classes[j]) for j in idx]
    matrix = pd.DataFrame(e.matrix[np.ix_(idx, idx)], index=pd.Index(labels, name="map"), columns=labels)
    cnt = np.zeros((len(idx), len(idx)))
    pos = {float(e.classes[j]): k for k, j in enumerate(idx)}
    for a, b in zip(m, r):
        cnt[pos[a], pos[b]] += 1
    counts = pd.DataFrame(cnt.astype(np.int64), index=pd.Index(labels, name="map"), columns=labels)
    return Accuracy(table, overall, matrix, counts, "stehman" if use_stehman else "olofsson")
