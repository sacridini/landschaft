# SPDX-License-Identifier: GPL-3.0-or-later
"""gdistance's Mahalanobis transition of a multi-layer raster."""

from __future__ import annotations

import numpy as np

from .. import _core

# raster::adjacent's direction order (row, column offsets), which sets the
# order of the pairs and so the bits of their mean.
_ADJACENT_ORDER = {
    4: [(0, -1), (0, 1), (-1, 0), (1, 0)],
    8: [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0)],
    16: [(-1, -2), (1, -2), (-1, 2), (1, 2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
         (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 0), (1, 0)],
}


def adjacent_pairs(grid, cells: np.ndarray, targets: np.ndarray, directions: int) -> tuple[np.ndarray, np.ndarray]:
    """raster::adjacent(pairs = TRUE, target = targets) in its order."""
    ny, nx = grid.shape
    r, c = cells // nx, cells % nx
    is_target = np.zeros(ny * nx, dtype=bool)
    is_target[targets] = True
    froms, tos = [], []
    for dr, dc in _ADJACENT_ORDER[directions]:
        r2, c2 = r + dr, c + dc
        if grid.wrap:
            c2 = c2 % nx
        ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        to = np.where(ok, r2 * nx + c2, 0)
        ok &= is_target[to]
        froms.append(cells[ok])
        tos.append(to[ok])
    return np.concatenate(froms), np.concatenate(tos)


def mahal_transition(values: np.ndarray, grid, directions: int, template, *, x87_sqrt: bool = True):
    """gdistance's ``transition(brick, "mahal", directions)``.

    For every pair of neighbouring cells with values in all layers, the
    Mahalanobis distance d between their layer values (with the inverse of the
    layers' covariance matrix); the connection is ``mean(d) / (d + mean(d))``.
    Computed as R does (``cov``, ``solve``, ``%*%``, ``mean``), bit for bit.
    """
    from ._rlinalg import quadratic_forms, r_solve
    from ._transition import Transition

    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("the Mahalanobis transition needs a raster with two or more layers")
    k = values.shape[0]
    flat = np.ascontiguousarray(values.reshape(k, -1).T)
    complete = ~np.isnan(flat).any(axis=1)  # na.omit
    data_cells = np.flatnonzero(complete)
    a, b = adjacent_pairs(grid, data_cells, data_cells, directions)
    diff = flat[a] - flat[b]
    cov_inv = r_solve(_core.cost_r_cov(np.ascontiguousarray(flat[data_cells])))
    q = quadratic_forms(diff, cov_inv)
    d = _core.cost_rmath("sqrt", q) if x87_sqrt else np.sqrt(q)
    m = float(_core.cost_r_mean(d))
    w = m / (d + m)
    # both orientations of a pair carry the same bits: keep a < b, into the symmetric layout
    keep = a < b
    lo, hi, w = a[keep], b[keep], w[keep]
    nb = _core.cost_neighbours(grid.core, directions, True)
    n = nb.shape[1]
    cell = np.broadcast_to(np.arange(n, dtype=np.int64), nb.shape)
    ok = nb >= 0
    pos = np.flatnonzero(ok.ravel())
    keys = (np.minimum(cell[ok], nb[ok]) * n + np.maximum(cell[ok], nb[ok]))
    order = np.argsort(keys, kind="stable")
    where = np.searchsorted(keys[order], lo * n + hi)
    out = np.zeros(nb.shape)
    out.ravel()[pos[order[where]]] = w
    return Transition(out, grid, directions, True, "conductance", template)
