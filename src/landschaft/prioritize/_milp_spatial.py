# SPDX-License-Identifier: GPL-3.0-or-later
"""Penalties and constraints of :class:`~landschaft.prioritize.Problem` on top of
its objective, as prioritizr 8.0.6 adds them (``rcpp_apply_*_penalties``,
``rcpp_apply_*_constraints``): in the order they were added, the penalties
first, then the constraints. Every one appends columns and rows to the model
with the same values and in the same order as prioritizr.

Unit data (matrices, values per unit) refer to every unit, planning unit or not
(raster: every cell; tables and polygons: every row) and are cut to the
planning units, as prioritizr's ``planning_unit_indices()``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["Builder", "apply_constraint", "apply_penalty", "boundary_matrix"]


class Builder:
    """The model being compiled: triplets and vectors that penalties and constraints extend."""

    def __init__(self, nrow, ncol, rows, cols, vals, obj, lb, ub, vtype, sense, rhs, row_ids, col_ids, modelsense,
                 n_units):
        self.nrow, self.ncol = nrow, ncol
        self.rows, self.cols, self.vals = list(rows), list(cols), list(vals)
        self.obj, self.lb, self.ub = obj, lb, ub
        self.vtype, self.sense, self.rhs = list(vtype), list(sense), list(rhs)
        self.row_ids, self.col_ids = list(row_ids), list(col_ids)
        self.modelsense = modelsense
        self.n = n_units

    def add_columns(self, obj, lb, ub, vtype, col_id) -> int:
        """New columns; their first index."""
        first = self.ncol
        k = len(obj)
        self.obj = np.concatenate([self.obj, np.asarray(obj, dtype=np.float64)])
        self.lb = np.concatenate([self.lb, np.broadcast_to(np.asarray(lb, dtype=np.float64), (k,))])
        self.ub = np.concatenate([self.ub, np.broadcast_to(np.asarray(ub, dtype=np.float64), (k,))])
        self.vtype += [vtype] * k
        self.col_ids += [col_id] * k
        self.ncol += k
        return first

    def add_rows(self, r, c, v, sense, rhs, row_id) -> None:
        """Rows numbered from 0 (r), appended after the current ones."""
        k = len(sense)
        self.rows.append(np.asarray(r, dtype=np.int64) + self.nrow)
        self.cols.append(np.asarray(c, dtype=np.int64))
        self.vals.append(np.asarray(v, dtype=np.float64))
        self.sense += list(sense)
        self.rhs.append(np.asarray(rhs, dtype=np.float64))
        self.row_ids += list(row_id) if not isinstance(row_id, str) else [row_id] * k
        self.nrow += k

    def matrix(self) -> sp.csr_matrix:
        r, c, v = np.concatenate(self.rows), np.concatenate(self.cols), np.concatenate(self.vals)
        keep = v != 0.0  # prioritizr stores the zeros it pushes: no entries here
        a = sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(self.nrow, self.ncol))
        a.sort_indices()
        return a


# ----------------------------------------------------------------------------- unit data

def unit_values(u, data, what: str) -> np.ndarray:
    """One value per unit (every cell or row), NaN as 0: a raster on the units' grid, a column name, or
    values (raster units: one per cell, or the grid's shape)."""
    import xarray as xr

    if isinstance(data, str):
        if u.kind == "raster":
            raise TypeError(f"{what}: a column name needs table or polygon planning units")
        v = u.template[data].to_numpy(dtype=np.float64)
    elif isinstance(data, xr.DataArray):
        if u.kind == "raster":
            v = np.asarray(data.squeeze().values, dtype=np.float64)
            if v.shape != u.template.shape:
                raise ValueError(f"{what}: the raster must be on the planning units' grid")
            v = v.ravel()
        elif hasattr(u.template, "geometry"):  # polygons: summed over every polygon (exactextractr)
            from ._milp import exact_sums

            geoms = u.template
            crs = data.rio.crs
            if geoms.crs is not None and crs is not None and geoms.crs != crs:
                from .._vector import to_crs

                geoms = to_crs(geoms, crs)
            v = exact_sums(data, geoms.geometry.values)[:, 0]
        else:
            raise TypeError(f"{what}: a raster needs raster or polygon planning units")
    else:
        v = np.asarray(data, dtype=np.float64).ravel()
    total = u.template.size if u.kind == "raster" else len(u.template)
    if v.size != total:
        raise ValueError(f"{what}: one value per unit ({total}), got {v.size}")
    return np.nan_to_num(v, nan=0.0)


def unit_matrix(u, data, what: str) -> sp.csc_matrix:
    """A units x units matrix (every unit) cut to the planning units (CSC, zeros dropped)."""
    m = sp.csc_matrix(data, dtype=np.float64)
    total = u.template.size if u.kind == "raster" else len(u.template)
    if m.shape != (total, total):
        raise ValueError(f"{what}: a {total} x {total} matrix (every unit), got {m.shape[0]} x {m.shape[1]}")
    m = m[u.index][:, u.index].tocsc()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def r_round(x: np.ndarray, digits: int) -> np.ndarray:
    """R's round(x, digits) (R >= 4.0): of the two decimal candidates around x, the closer one to x
    (not x * 10^d rounded, as numpy does), the even one on a tie."""
    x = np.asarray(x, dtype=np.float64)
    sgn = np.where(x < 0, -1.0, 1.0)
    a = np.abs(x)
    p10 = 10.0 ** digits
    x10 = p10 * a
    i10 = np.floor(x10)
    xd = i10 / p10
    xu = np.ceil(x10) / p10
    du = xu - a
    dd = a - xd
    up = (du < dd) | ((du == dd) & (np.fmod(i10, 2.0) == 1))
    return sgn * np.where(up, xu, xd)


def _r_round(m: sp.csc_matrix, digits: int) -> sp.csc_matrix:
    m = m.copy()
    m.data = r_round(m.data, digits)
    m.eliminate_zeros()
    return m


# ----------------------------------------------------------------------------- boundary

def _raster_boundary(u) -> sp.csc_matrix:
    """prioritizr's boundary_matrix of the cost raster, cut to the planning units: neighbours above and
    below share xres, left and right yres (terra's res: the extent over the cells), and the diagonal
    holds every cell's perimeter, 2 (xres + yres)."""
    from ..io import _transform

    t = _transform(u.template)
    ny, nx = u.template.shape
    xmin, xmax = t.c, t.c + nx * t.a
    ymax, ymin = t.f, t.f + ny * t.e
    xres, yres = (xmax - xmin) / nx, (ymax - ymin) / ny
    unit = np.full(ny * nx, -1, dtype=np.int64)
    unit[u.index] = np.arange(u.n)
    grid = unit.reshape(ny, nx)
    pairs = []
    for a, b, length in ((grid[:-1, :], grid[1:, :], xres), (grid[:, :-1], grid[:, 1:], yres)):
        a, b = a.ravel(), b.ravel()
        ok = (a >= 0) & (b >= 0)
        pairs.append((a[ok], b[ok], np.full(int(ok.sum()), length)))
    i = np.concatenate([p[0] for p in pairs] + [p[1] for p in pairs] + [np.arange(u.n)])
    j = np.concatenate([p[1] for p in pairs] + [p[0] for p in pairs] + [np.arange(u.n)])
    x = np.concatenate([p[2] for p in pairs] * 2 + [np.full(u.n, (xres + yres) * 2)])
    m = sp.csc_matrix((x, (i, j)), shape=(u.n, u.n))
    m.sort_indices()
    return m


def _polygon_boundary(u) -> sp.csc_matrix:
    """prioritizr's boundary_matrix of polygons, cut to the planning units: the length of the paths
    two polygons share (terra's sharedPaths) and every polygon's perimeter on the diagonal, rounded
    to 8 decimals."""
    import shapely

    geoms = np.asarray(u.template.geometry.values, dtype=object)
    n = geoms.size
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    bounds = shapely.boundary(geoms)
    shared = shapely.intersection(bounds[left], bounds[right])
    length = np.round(shapely.length(shared), 8)
    perim = np.round(shapely.length(geoms), 8)
    i = np.concatenate([left, right, np.arange(n)])
    j = np.concatenate([right, left, np.arange(n)])
    x = np.concatenate([length, length, perim])
    m = sp.csc_matrix((x, (i, j)), shape=(n, n))
    m = m[u.index][:, u.index].tocsc()
    m.sort_indices()
    return m


def boundary_matrix(u) -> sp.csc_matrix:
    """The planning units' boundary matrix (prioritizr's boundary_matrix of the planning-unit data)."""
    if u.kind == "raster":
        return _raster_boundary(u)
    if hasattr(u.template, "geometry"):
        return _polygon_boundary(u)
    raise ValueError("boundary penalties of tables need data: a boundary matrix of every unit")


def _row_sums(m: sp.csc_matrix) -> np.ndarray:
    """Matrix::rowSums: every row's sum in double, over the columns in order."""
    c = m.tocsr()
    c.sort_indices()
    out = np.zeros(c.shape[0])
    for k in range(c.shape[0]):
        d = c.data[c.indptr[k]:c.indptr[k + 1]]
        out[k] = float(np.cumsum(d)[-1]) if d.size else 0.0
    return out


def _lower_pairs(m: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The strictly lower triangle's entries in armadillo's order: by column, then row."""
    low = sp.tril(m, k=-1).tocsc()
    low.eliminate_zeros()
    low.sort_indices()
    col = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr))
    return low.indices.astype(np.int64), col.astype(np.int64), low.data


def _pair_columns(b: Builder, pi, pj, coef, id_prefix: str) -> None:
    """One column per pair (b = x_i x_j, linearised) and its rows, as rcpp_apply_*_penalties."""
    if not len(pi):
        return
    first = b.add_columns(coef, 0.0, 1.0, b.vtype[0], id_prefix)
    var = first + np.arange(len(pi))
    third = (coef > 0) if b.modelsense == "min" else (coef < 0)
    r, c, v, sense, rhs, ids = [], [], [], [], [], []
    row = 0
    for k in range(len(pi)):
        r += [row, row]
        c += [var[k], pi[k]]
        v += [1.0, -1.0]
        sense.append("<=")
        rhs.append(0.0)
        ids.append(id_prefix + "1")
        row += 1
        r += [row, row]
        c += [var[k], pj[k]]
        v += [1.0, -1.0]
        sense.append("<=")
        rhs.append(0.0)
        ids.append(id_prefix + "2")
        row += 1
        if third[k]:
            r += [row, row, row]
            c += [var[k], pi[k], pj[k]]
            v += [1.0, -1.0, -1.0]
            sense.append(">=")
            rhs.append(-1.0)
            ids.append(id_prefix + "3")
            row += 1
    b.add_rows(r, c, v, sense, rhs, ids)


def _apply_boundary(b: Builder, u, penalty: float, edge_factor: float, data) -> None:
    if data is None:
        bm = boundary_matrix(u)
    else:
        bm = _r_round(unit_matrix(u, data, "boundary data"), 6)
        if abs((bm - bm.T)).max() > 0 if bm.nnz else False:
            raise ValueError("the boundary data must be symmetric")
    if not abs(penalty) > 1e-50:
        return
    total = bm.diagonal()
    exposed = total - (_row_sums(bm) - total)
    p = -penalty if b.modelsense == "max" else penalty
    unit_pen = p * 1.0 * ((total - exposed) + (exposed * edge_factor))
    b.obj[:b.n] += unit_pen
    pi, pj, val = _lower_pairs(bm)
    cv = val * 1.0 * p
    keep = np.abs(cv) > 1.0e-15
    _pair_columns(b, pi[keep], pj[keep], cv[keep] * -2.0, "b")


# ----------------------------------------------------------------------------- connectivity

def _apply_connectivity(b: Builder, u, penalty: float, data) -> None:
    m = unit_matrix(u, data, "connectivity data")
    if m.nnz and abs(m - m.T).max() > 0:
        raise ValueError("the connectivity data must be symmetric (use add_asym_connectivity_penalties)")
    mult = -1.0 if b.modelsense == "min" else 1.0
    low = sp.tril(m).tocsc()
    low.sort_indices()
    col = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr))
    row = low.indices
    cv = low.data * (penalty * mult)
    keep = np.abs(cv) > 1.0e-15
    row, col, cv = row[keep], col[keep], cv[keep]
    diag = row == col
    unit_pen = np.zeros(b.n)
    for k in np.flatnonzero(diag):  # in armadillo's order (one per unit)
        unit_pen[row[k]] += cv[k]
    b.obj[:b.n] += unit_pen
    off = ~diag
    _pair_columns(b, row[off], col[off], cv[off], "c")


def _apply_asym_connectivity(b: Builder, u, penalty: float, data) -> None:
    m = unit_matrix(u, data, "connectivity data")
    mult1, mult2 = (1.0, -1.0) if b.modelsense == "min" else (-1.0, 1.0)
    col = np.repeat(np.arange(m.shape[1]), np.diff(m.indptr))
    row = m.indices
    cv = m.data * penalty
    keep = np.abs(cv) > 1.0e-15
    row, col, cv = row[keep], col[keep], cv[keep]
    unit_pen = np.zeros(b.n)
    pi, pj, pb = [], [], []
    for k in range(row.size):  # armadillo's order: by column, then row
        if row[k] == col[k]:
            unit_pen[row[k]] += mult2 * cv[k]
        else:
            pi.append(row[k])
            pj.append(col[k])
            pb.append(mult2 * cv[k])
            unit_pen[row[k]] += mult1 * cv[k]
    b.obj[:b.n] += unit_pen
    _pair_columns(b, np.array(pi, dtype=np.int64), np.array(pj, dtype=np.int64), np.array(pb), "ac")


# ----------------------------------------------------------------------------- linear

def _apply_linear_penalty(b: Builder, u, penalty: float, data) -> None:
    if not abs(penalty) > 1e-50:
        return
    v = unit_values(u, data, "linear penalty data")[u.index]
    p = -penalty if b.modelsense == "max" else penalty
    cv = v * p
    unit_pen = np.where(np.abs(cv) > 1.0e-15, cv, 0.0)
    b.obj[:b.n] += np.where(v != 0, unit_pen, 0.0)


def _apply_linear_constraint(b: Builder, u, threshold: float, sense: str, data) -> None:
    v = unit_values(u, data, "linear constraint data")[u.index]
    nz = np.flatnonzero(v)
    b.add_rows(np.zeros(nz.size, dtype=np.int64), nz, v[nz], [sense], [threshold], "lc")


# ----------------------------------------------------------------------------- adjacency

def adjacency_matrix(u) -> sp.csc_matrix:
    """prioritizr's adjacency_matrix of the planning-unit data, cut to the planning units: raster cells
    with a cost that share an edge (4 directions); polygons that intersect (touching at a corner
    included), 1 for each pair."""
    if u.kind == "raster":
        ny, nx = u.template.shape
        unit = np.full(ny * nx, -1, dtype=np.int64)
        unit[u.index] = np.arange(u.n)
        grid = unit.reshape(ny, nx)
        i, j = [], []
        for a, b in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
            a, b = a.ravel(), b.ravel()
            ok = (a >= 0) & (b >= 0)
            i += [a[ok], b[ok]]
            j += [b[ok], a[ok]]
        i, j = np.concatenate(i), np.concatenate(j)
        m = sp.csc_matrix((np.ones(i.size), (i, j)), shape=(u.n, u.n))
    elif hasattr(u.template, "geometry"):
        import shapely

        geoms = np.asarray(u.template.geometry.values, dtype=object)
        left, right = shapely.STRtree(geoms).query(geoms, predicate="intersects")
        keep = left != right
        n = geoms.size
        m = sp.csc_matrix((np.ones(int(keep.sum())), (left[keep], right[keep])), shape=(n, n))
        m.data[:] = 1.0
        m = m[u.index][:, u.index].tocsc()
    else:
        raise ValueError("constraints on neighbouring units of tables need data: an adjacency matrix of every unit")
    m.sum_duplicates()
    m.data[:] = 1.0
    m.eliminate_zeros()
    m.sort_indices()
    return m


def _adjacency(u, data) -> sp.csc_matrix:
    if data is None:
        return adjacency_matrix(u)
    return unit_matrix(u, data, "adjacency data")


def _apply_contiguity(b: Builder, u, data) -> None:
    """rcpp_apply_contiguity_constraints (one zone): a column per adjacent pair (lower triangle, by
    column), y <= x_i for every pair, sum of the pairs of a unit j <= x_j, and pairs = selected - 1."""
    d = _adjacency(u, data)
    low = sp.tril(d, k=-1).tocsc()  # forceSymmetric(uplo = "L") then tril: the lower triangle as given
    low.eliminate_zeros()
    low.sort_indices()
    pj = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr))
    pi = low.indices.astype(np.int64)
    keep = low.data > 1.0e-15
    pi, pj = pi[keep], pj[keep]
    k = pi.size
    first = b.add_columns(np.zeros(k), 0.0, 1.0, b.vtype[0], "c")
    var = first + np.arange(k)
    # c1: y_k - x_i <= 0
    r1 = np.repeat(np.arange(k), 2)
    c1 = np.column_stack([var, pi]).ravel()
    v1 = np.tile([1.0, -1.0], k)
    # c2: one row per run of pairs with the same j: -x_j + sum y <= 0
    new = np.ones(k, dtype=bool)
    new[1:] = pj[1:] != pj[:-1]
    group = np.cumsum(new) - 1
    ng = int(new.sum()) if k else 0
    r2 = np.concatenate([np.arange(ng), group]) + k
    c2 = np.concatenate([pj[new], var])
    v2 = np.concatenate([-np.ones(ng), np.ones(k)])
    # c3: -sum x + sum y = -1
    r3 = np.full(b.n + k, k + ng)
    c3 = np.concatenate([np.arange(b.n), var])
    v3 = np.concatenate([-np.ones(b.n), np.ones(k)])
    b.add_rows(np.concatenate([r1, r2, r3]), np.concatenate([c1, c2, c3]), np.concatenate([v1, v2, v3]),
               ["<="] * (k + ng) + ["="], [0.0] * (k + ng) + [-1.0], ["c1"] * k + ["c2"] * ng + ["c3"])


def _apply_neighbors(b: Builder, u, k: int, clamp: bool, data) -> None:
    """rcpp_apply_neighbor_constraints (one zone): every selected unit has at least k selected neighbours
    (k clamped to its number of neighbours): sum of the neighbours - k x_i >= 0."""
    d = _adjacency(u, data)
    kk = np.full(b.n, float(k))
    if clamp:
        rs = np.asarray(d.sum(axis=1)).ravel()
        kk = np.minimum(kk, rs)
    if not kk.max(initial=0.0) > 0:
        return
    col = np.repeat(np.arange(d.shape[1]), np.diff(d.indptr))
    row = d.indices.astype(np.int64)
    keep = (d.data > 1.0e-15) & (row != col)
    r = np.concatenate([np.arange(b.n), row[keep]])
    c = np.concatenate([np.arange(b.n), col[keep]])
    v = np.concatenate([kk * -1.0, np.ones(int(keep.sum()))])
    b.add_rows(r, c, v, [">="] * b.n, [0.0] * b.n, "n")


# ----------------------------------------------------------------------------- dispatch

def apply_penalty(b: Builder, u, spec) -> None:
    kind = spec[0]
    if kind == "boundary":
        _apply_boundary(b, u, *spec[1:])
    elif kind == "connectivity":
        _apply_connectivity(b, u, *spec[1:])
    elif kind == "asym_connectivity":
        _apply_asym_connectivity(b, u, *spec[1:])
    elif kind == "linear":
        _apply_linear_penalty(b, u, *spec[1:])
    else:
        raise ValueError(f"unknown penalty {kind!r}")


def apply_constraint(b: Builder, u, spec) -> None:
    kind = spec[0]
    if kind == "linear":
        _apply_linear_constraint(b, u, *spec[1:])
    elif kind == "contiguity":
        _apply_contiguity(b, u, *spec[1:])
    elif kind == "neighbors":
        _apply_neighbors(b, u, *spec[1:])
    else:
        raise ValueError(f"unknown constraint {kind!r}")
