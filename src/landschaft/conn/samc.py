# SPDX-License-Identifier: GPL-3.0-or-later
"""Settlement probabilities of a spatial absorbing Markov chain, as Makurhini's MK_dPC_SAMC (samc package)."""

from __future__ import annotations

import numpy as np

from .. import _core
from .nodes import _vector_frame, cells_of, read_resistance

SETTLE = ("sum", "mean")


def boundary_vertices(geometry, n: int | None = 10, fraction: float | None = None) -> np.ndarray:
    """Vertices of a polygon picked as Makurhini's ``SR_vertices`` does (systematic).

    All coordinates of the geometry in order (every ring and part); when the
    first equals the last, the last is dropped and repeated points removed.
    ``n`` of them (or ``round(fraction × count)``) at ``round(seq(1, count,
    length.out = n))``, all when there are no more.
    """
    import shapely

    xy = shapely.get_coordinates(geometry)
    if len(xy) > 1 and np.array_equal(xy[0], xy[-1]):
        xy = xy[:-1]
        _, first = np.unique(xy, axis=0, return_index=True)
        xy = xy[np.sort(first)]
    count = len(xy)
    if fraction is not None:
        n = int(np.round(count * fraction))
    if n is None:
        raise ValueError("give the number of vertices or their fraction")
    if n >= count:
        return xy
    if n > 2:
        by = (count - 1) / (n - 1)
        seq = np.concatenate([[1.0], 1 + np.arange(1, n - 1) * by, [float(count)]])
    else:
        seq = np.array([1.0, float(count)])[:n]
    idx = np.unique(np.round(seq)).astype(np.int64) - 1  # R's round: half to even, as numpy's
    return xy[idx]


def _absorption(absorption, nodes_geoms, res, dispersal_distance):
    if absorption is not None:
        a = read_resistance(absorption, positive=False)
        if a.values.shape != res.values.shape:
            raise ValueError("the absorption raster must have the resistance's shape")
        return a.values
    if dispersal_distance is None:
        raise ValueError("give the dispersal distance (or an absorption raster)")
    from rasterio import features
    from rasterio.transform import from_bounds

    ny, nx = res.values.shape
    xmin, xmax, ymin, ymax = res.extent
    inside = features.rasterize(((g, 1) for g in nodes_geoms), out_shape=(ny, nx), fill=0, all_touched=False,
                                transform=from_bounds(xmin, ymin, xmax, ymax, nx, ny)).astype(bool)
    # Fletcher et al. (2023): absorption per step from the mean dispersal distance in cells.
    rate = np.exp(-1.247 - 1.788 * np.log(dispersal_distance / ((xmax - xmin) / nx)))
    return np.where(inside, 0.0, rate)


def settlement_probabilities(nodes, resistance, *, dispersal_distance: float | None = None, absorption=None,
                             directions: int = 8, cost_function: str = "mean", vertices: int | None = 10,
                             vertex_fraction: float | None = None, settle: str = "sum", id: str | None = None):
    """Probabilities of settling in one patch from another under a spatial absorbing Markov chain.

    Makurhini's ``MK_dPC_SAMC`` (``samc_settlement``, samc's random walk):
    the cells with a resistance are the transient states; from a cell the walk
    steps to a neighbour (``directions`` 4 or 8) with probability
    (1 − absorption) × w / Σw, w = 1 / the mean resistance of both cells
    (``cost_function="mean"``) or 1 / the neighbour's (``"makurhini"``), over
    √2 for diagonal steps, and is absorbed (settles) with the cell's
    absorption. The probability of settling in cell d when starting in cell o
    is F[o, d] × absorption[d], F = (I − Q)⁻¹. Between two patches it is the
    ``settle`` (``"sum"``, Makurhini's default, or ``"mean"``) of those
    values from the boundary vertices of the later patch to those of the
    earlier one (:func:`boundary_vertices`; vertices on NA cells are left out).

    Parameters
    ----------
    nodes
        Polygons (GeoDataFrame).
    resistance
        Resistance raster (path, DataArray or array; NA cells are not states).
    dispersal_distance
        Mean dispersal distance (map units) for the default absorption: 0 in
        the patches (cells whose centre is inside), exp(−1.247 − 1.788 ×
        log(distance / cell size)) elsewhere (Fletcher et al. 2023).
    absorption
        Absorption raster (0-1, the resistance's grid) instead.
    directions
        4 or 8 neighbours.
    cost_function
        ``"mean"`` (1 / the mean resistance) or ``"makurhini"`` (1 / the
        neighbour's resistance).
    vertices, vertex_fraction
        Number of boundary vertices per patch (default 10), or their fraction.
    settle
        ``"sum"`` or ``"mean"`` of the probabilities between the vertices.
    id
        Column with the node ids (default 1..n).

    Returns
    -------
    ids, numpy.ndarray
        The node ids and the n × n matrix of probabilities (1 on the diagonal).
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    if settle not in SETTLE:
        raise ValueError(f"settle must be one of {SETTLE}, got {settle!r}")
    if directions not in (4, 8):
        raise ValueError("directions must be 4 or 8")
    frame, ids = _vector_frame(nodes, id)
    geoms = frame.geometry.values
    n = len(geoms)
    res = read_resistance(resistance)
    absorb = np.ascontiguousarray(_absorption(absorption, geoms, res, dispersal_distance), dtype=np.float64)
    finite = np.isfinite(absorb) & np.isfinite(res.values)
    if np.any((absorb[finite] < 0) | (absorb[finite] > 1)):
        raise ValueError("absorption values must be between 0 and 1")
    if not np.any(absorb[finite] > 0):
        raise ValueError("at least one cell must absorb")
    t = _core.conn_samc_transitions(res.values, absorb, directions, cost_function)
    cells = t["cell"]
    m = cells.size
    # The states of every patch's vertices (first occurrence order, no repeats).
    state_of = np.full(res.values.size, -1, dtype=np.int64)
    state_of[cells] = np.arange(m)
    own = []
    for g in geoms:
        xy = boundary_vertices(g, vertices, vertex_fraction)
        c = cells_of(xy[:, 0], xy[:, 1], res.values.shape, res.extent)
        s = np.where(c >= 0, state_of[np.clip(c, 0, None)], -1)
        s = s[s >= 0]
        _, first = np.unique(s, return_index=True)
        own.append(s[np.sort(first)])
    P = np.eye(n)
    targets = np.unique(np.concatenate(own)) if n else np.zeros(0, np.int64)
    if targets.size == 0:
        return np.asarray(ids), P
    A = (sparse.identity(m, format="csr") - sparse.csr_matrix((t["q"], (t["from"], t["to"])), shape=(m, m))).tocsc()
    lu = splu(A)
    # F[o, d] for o, d among the targets, from the columns F[:, d].
    F = np.empty((targets.size, targets.size))
    block = max(1, min(64, int(2e7 // max(1, m))))
    for s in range(0, targets.size, block):
        cols = targets[s:s + block]
        rhs = np.zeros((m, cols.size))
        rhs[cols, np.arange(cols.size)] = 1.0
        F[:, s:s + cols.size] = lu.solve(rhs)[targets]
    M = F * absorb.ravel()[cells][targets][None, :]
    where = [np.searchsorted(targets, o) for o in own]
    aggregate = np.sum if settle == "sum" else np.mean
    for y in range(n):
        if where[y].size == 0:
            continue
        for x in range(y):
            if where[x].size == 0:
                continue
            # Makurhini keeps the values from the later patch to the earlier one.
            P[x, y] = P[y, x] = aggregate(M[np.ix_(where[y], where[x])])
    return np.asarray(ids), P
