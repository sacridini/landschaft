# SPDX-License-Identifier: GPL-3.0-or-later
"""Route overlap between destinations (gdistance's pathInc) and barrier / area transitions."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .. import _core
from ._points import as_cells
from ._transition import Transition, read_grid_values

__all__ = ["path_inc", "barriers", "areas", "overlap", "nonoverlap"]

_FUNCTIONS = {"overlap": 0, "nonoverlap": 1}


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """gdistance's ``overlap``: the flow two routes share on each connection, ``min(a, b)``."""
    return np.minimum(a, b)


def nonoverlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """gdistance's ``nonoverlap``: ``max(max(a, b) * (1 - min(a, b)) - min(a, b), 0)``."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.maximum(hi * (1 - lo) - lo, 0)


def _matr_index(i: int, n: int) -> tuple[int, int]:
    """gdistance's .matrIndex: the i-th pair (1-based) of n categories, (larger, smaller)."""
    cc = np.cumsum(np.arange(n - 1, 0, -1))
    second = int(np.flatnonzero(cc - i >= 0)[0]) + 1
    first = -int(np.concatenate([[0], cc])[second - 1]) + i + second
    return first, second


def _pair_layer(values: np.ndarray, grid, directions: int, a_cells: np.ndarray, b_cells: np.ndarray,
                both: bool) -> np.ndarray:
    """Asymmetric layout with 1 on the connections from a cell of `a_cells` to a neighbour in `b_cells`
    (and back, with `both`)."""
    nb = _core.cost_neighbours(grid.core, directions, False)
    n = nb.shape[1]
    in_a = np.zeros(n, bool)
    in_b = np.zeros(n, bool)
    in_a[a_cells] = True
    in_b[b_cells] = True
    cell = np.broadcast_to(np.arange(n), nb.shape)
    ok = nb >= 0
    target = np.clip(nb, 0, None)
    hit = ok & ((in_a[cell] & in_b[target]) | (both & in_b[cell] & in_a[target]))
    return hit.astype(np.float64)


def barriers(x, directions: int = 8, *, breaks: Sequence[float] | None = None, symmetric: bool = True,
             extent=None, crs=None, band=None, nodata=None) -> list[Transition]:
    """Barrier transitions: gdistance's ``transition(x, "barriers", directions)``.

    Each layer marks (with 1, as resistances) the connections that cross a
    boundary, for use as weights of :func:`path_inc`.

    Parameters
    ----------
    x
        A raster.
    directions : {4, 8, 16}
    breaks
        Continuous values: one layer per break ``b``, the connections between
        a cell below ``b`` and one above it. Without breaks the values are
        categories: one layer per pair of categories (in gdistance's order:
        (2, 1), (3, 1), ..., (3, 2), ...), the connections between cells of the
        two; with ``symmetric=False`` two layers per pair, one per direction.
    symmetric
        See ``breaks`` (continuous barriers are always symmetric, as in
        gdistance).

    Returns
    -------
    list of Transition

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.array([[1.0, 1.0, 5.0], [1.0, 5.0, 5.0]])
    >>> layer = ls.cost.barriers(r, 4, breaks=[3])[0]
    >>> layer.n_connections, layer.matrix_values          # 3 boundary pairs, both ways
    (6, 'resistance')
    """
    values, grid, template = read_grid_values(x, extent=extent, crs=crs, band=band, nodata=nodata)
    flat = values.ravel()
    layers = []
    if breaks is not None:
        lo = _core.cost_builtin(values, grid.core, directions, False, "min")
        hi = _core.cost_builtin(values, grid.core, directions, False, "max")
        for b in breaks:
            hit = ((lo < b) & (hi > b)).astype(np.float64)
            layers.append(Transition(hit, grid, directions, False, "resistance", template))
        return layers
    cats = np.unique(flat[np.isfinite(flat)])
    n = cats.size
    for i in range(1, (n * n - n) // 2 + 1):
        j1, j2 = _matr_index(i, n)
        c1 = np.flatnonzero(flat == cats[j1 - 1])
        c2 = np.flatnonzero(flat == cats[j2 - 1])
        if symmetric:
            layers.append(Transition(_pair_layer(values, grid, directions, c1, c2, True), grid, directions,
                                     False, "resistance", template))
        else:
            layers.append(Transition(_pair_layer(values, grid, directions, c1, c2, False), grid, directions,
                                     False, "resistance", template))
            layers.append(Transition(_pair_layer(values, grid, directions, c2, c1, False), grid, directions,
                                     False, "resistance", template))
    return layers


def areas(x, directions: int = 8, *, extent=None, crs=None, band=None, nodata=None) -> list[Transition]:
    """Area transitions: gdistance's ``transition(x, "areas", directions)``.

    One layer per category k (1, 2, ... n; as gdistance, category k is the
    value k): 1 between two cells of the category, 0.5 between one of it and
    another cell, for use as weights of :func:`path_inc`.

    Returns
    -------
    list of Transition

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.array([[1.0, 1.0, 2.0]])
    >>> [t.to_sparse().toarray()[0, 1:].tolist() for t in ls.cost.areas(r, 4)]
    [[1.0, 0.0], [0.0, 0.0]]
    """
    values, grid, template = read_grid_values(x, extent=extent, crs=crs, band=band, nodata=nodata)
    flat = values.ravel()
    n = np.unique(flat[np.isfinite(flat)]).size
    layers = []
    nb = _core.cost_neighbours(grid.core, directions, True)
    cell = np.broadcast_to(np.arange(flat.size), nb.shape)
    ok = nb >= 0
    ok[ok] = np.isfinite(flat[cell[ok]]) & np.isfinite(flat[nb[ok]])
    for k in range(1, n + 1):
        v = np.zeros(nb.shape)
        v[ok] = ((flat[cell[ok]] == k).astype(float) + (flat[nb[ok]] == k)) / 2  # sum(v == i) / 2
        layers.append(Transition(v, grid, directions, True, "conductance", template))
    return layers


def _weights(x: Transition, weight, frm: np.ndarray, to: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R: the weight of every connection (layers x connections), Inf set to 0."""
    def one(w: Transition) -> np.ndarray:
        if not w.grid.same(x.grid):
            raise ValueError("the weight is on another grid")
        m = w.to_sparse().tocsr()
        v = np.asarray(m[np.array(frm), np.array(to)]).ravel()  # own copies: scipy's indexing needs them writeable
        if w.matrix_values == "conductance":
            with np.errstate(divide="ignore"):
                v = 1.0 / v
        v[np.isinf(v)] = 0.0
        return v

    if weight is None:
        with np.errstate(divide="ignore"):
            r = 1.0 / t
        r[np.isinf(r)] = 0.0
        return r[None, :]
    if isinstance(weight, Transition):
        return one(weight)[None, :]
    return np.stack([one(w) for w in weight])


def path_inc(x: Transition, origin, from_coords, to_coords=None, *, theta: float | None = None, weight=None,
             functions: Sequence[str | Callable] = ("overlap", "nonoverlap"), gdistance_compat: bool = True) -> dict:
    """Overlap and non-overlap of the routes from a common origin to each location: gdistance's ``pathInc()``.

    For each location, the flow of a random walk (or, with ``theta``, of the
    randomised shortest paths) from ``origin`` to it on every connection;
    then, for each pair of locations, the sum over the connections of
    ``function(flow_1, flow_2) * weight``. With ``overlap`` it measures how
    much the two routes share, with ``nonoverlap`` how much they differ: the
    "path of dispersal from a common origin" distances of van Etten and
    Hijmans (2010).

    Parameters
    ----------
    x
        A :class:`Transition` of conductances (symmetric without ``theta``).
    origin
        The common origin (a point).
    from_coords, to_coords
        Locations (points). Without ``to_coords``, the pairs among
        ``from_coords``.
    theta
        None: random walks; else randomised shortest paths (0 to 20).
    weight
        None: the cost of each connection (the reciprocal of the transition's
        value); a :class:`Transition` (a conductance is turned into its
        reciprocal); or a list of them (e.g. :func:`barriers`, :func:`areas`),
        one result per layer.
    functions
        ``"overlap"``, ``"nonoverlap"`` or vectorised callables ``f(a, b)`` of
        two arrays of flows (one value per connection).
    gdistance_compat
        Random walks: keep gdistance's potential of the grounded cell (see
        :func:`passage`).

    Returns
    -------
    dict
        Function name (or ``function<k>`` for callables) -> ``(n, n)`` matrix
        (``(n_from, n_to)`` with ``to_coords``; ``(layers, n, n)`` for a list
        of weights), NaN for locations off the graph. Among one set of
        locations the matrix is symmetric with 0 on the diagonal, as
        gdistance's ``dist``.

    Notes
    -----
    Two gdistance behaviours are not reproduced: with ``to`` its result is
    all missing (``.finishFlowFromTo`` copies an empty matrix), and its default
    weights are read from the wrong connections when some cells have no
    connection (the full matrix indexed with the positions of the reduced
    one); landschaft gives what the code intends in both cases.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((5, 5)), "mean", 8)
    >>> out = ls.cost.path_inc(tr, (2.5, 2.5), [(0.5, 0.5), (4.5, 0.5), (0.5, 4.5)])
    >>> sorted(out), out["overlap"].shape
    (['nonoverlap', 'overlap'], (3, 3))
    >>> bool(np.allclose(out["overlap"], out["overlap"].T))
    True
    """
    if not isinstance(x, Transition):
        raise TypeError("x must be a Transition (see ls.cost.transition)")
    if theta is None:
        if not x.is_symmetric():
            raise ValueError("path_inc without theta needs a symmetric transition matrix")
        th = float("nan")
    else:
        th = float(theta)
        if not 0 <= th <= 20:
            raise ValueError("theta value out of range (between 0 and 20)")
    oc = as_cells(origin, x.grid, "origins")
    if oc.size != 1 or oc[0] < 0:
        raise ValueError("path_inc needs one origin inside the grid")
    frm, to, t = _core.cost_graph_edges(x.values, x.grid.core, x.directions, x.symmetric)
    r = np.ascontiguousarray(_weights(x, weight, frm, to, t))
    fc = as_cells(from_coords, x.grid, "locations")
    tc = None if to_coords is None else as_cells(to_coords, x.grid, "locations")

    def flows(cells: np.ndarray) -> np.ndarray:
        uniq = np.unique(cells[cells >= 0])
        f = _core.cost_path_flows(x.values, x.grid.core, x.directions, x.symmetric, int(oc[0]), uniq.tolist(), th,
                                  bool(gdistance_compat))
        return uniq, np.ascontiguousarray(f.reshape(uniq.size, t.size))

    ua, fa = flows(fc)
    ub, fb = (ua, fa) if tc is None else flows(tc)
    names, codes, custom = [], [], []
    for k, fn in enumerate(functions):
        if isinstance(fn, str):
            if fn not in _FUNCTIONS:
                raise ValueError(f"unknown function {fn!r}: 'overlap', 'nonoverlap' or a callable")
            names.append(fn)
            codes.append(_FUNCTIONS[fn])
        else:
            names.append(getattr(fn, "__name__", f"function{k + 1}"))
            if names[-1] == "<lambda>":
                names[-1] = f"function{k + 1}"
            custom.append((len(names) - 1, fn))
    upper = tc is None
    res = np.full((len(names), r.shape[0], ua.size, ub.size), np.nan)
    if codes:
        builtin = [k for k, fn in enumerate(functions) if isinstance(fn, str)]
        res[builtin] = _core.cost_path_overlap(fa, fb, r, codes, upper)
    for k, fn in custom:
        for p in range(ua.size):
            for q in range(p if upper else 0, ub.size):
                prod = np.asarray(fn(fa[p], fb[q]), dtype=np.float64)
                res[k, :, p, q] = (prod[None, :] * r).sum(axis=1)
    # locations off the graph (their flows are NaN): no result
    off_a = np.isnan(fa).any(axis=1) if fa.size else np.zeros(ua.size, bool)
    off_b = np.isnan(fb).any(axis=1) if fb.size else np.zeros(ub.size, bool)
    res[:, :, off_a, :] = np.nan
    res[:, :, :, off_b] = np.nan
    out = {}
    ia = np.where(fc >= 0, np.searchsorted(ua, fc), -1)
    ib = ia if tc is None else np.where(tc >= 0, np.searchsorted(ub, tc), -1)
    ia[ia >= 0] = np.where(off_a[ia[ia >= 0]], -1, ia[ia >= 0])
    ib[ib >= 0] = np.where(off_b[ib[ib >= 0]], -1, ib[ib >= 0])
    for k, name in enumerate(names):
        m = np.full((r.shape[0], fc.size, (fc if tc is None else tc).size), np.nan)
        for li in range(r.shape[0]):
            v = res[k, li]
            if upper:
                v = np.triu(v) + np.triu(v, 1).T
            ok_a, ok_b = ia >= 0, ib >= 0
            m[li][np.ix_(ok_a, ok_b)] = v[np.ix_(ia[ok_a], ib[ok_b])]
            if upper:
                d = np.arange(fc.size)[ok_a]
                m[li][d, d] = 0.0
        out[name] = m[0] if (weight is None or isinstance(weight, Transition)) else m
    return out
