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

from dataclasses import dataclass

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
        self.n_features = 0
        self.connected: list = []  # Connected node sets (contiguity), for the solvers' cuts
        self.corridors: list = []  # Corridor networks, for the solvers' cuts

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

def _cut(u, m: sp.csc_matrix) -> sp.csc_matrix:
    m = m[u.index][:, u.index].tocsc()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def boundary_matrix(u) -> sp.csc_matrix:
    """The planning units' boundary matrix (prioritizr's boundary_matrix of the planning-unit data)."""
    from . import _milp_matrices as mx

    if u.kind == "raster" or hasattr(u.template, "geometry"):
        return _cut(u, mx.boundary_matrix(u.template))
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
    _connectivity_terms(b, m, penalty)


def neighbor_matrix(m: sp.csc_matrix) -> sp.csc_matrix:
    """The adjacency matrix of :meth:`~landschaft.prioritize.Problem.add_neighbor_penalties` (cut to the
    planning units), checked as prioritizr checks it: binary and symmetric."""
    if m.nnz and not np.isin(m.data, (0.0, 1.0)).all():
        raise ValueError("the neighbour data must be binary (0 or 1)")
    if m.nnz and abs(m - m.T).max() > 0:
        raise ValueError("the neighbour data must be symmetric")
    return m


def _apply_neighbor(b: Builder, u, penalty: float, data) -> None:
    """prioritizr's add_neighbor_penalties: the connectivity penalties of the adjacency matrix."""
    _connectivity_terms(b, neighbor_matrix(_adjacency(u, data)), penalty)


def _connectivity_terms(b: Builder, m: sp.csc_matrix, penalty: float) -> None:
    """rcpp_apply_connectivity_penalties on a symmetric matrix of the planning units."""
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
    """prioritizr's adjacency_matrix of the planning-unit data, cut to the planning units."""
    from . import _milp_matrices as mx

    if u.kind == "raster" or hasattr(u.template, "geometry"):
        return _cut(u, mx.adjacency_matrix(u.template))
    raise ValueError("constraints on neighbouring units of tables need data: an adjacency matrix of every unit")


def _adjacency(u, data) -> sp.csc_matrix:
    if data is None:
        return adjacency_matrix(u)
    return unit_matrix(u, data, "adjacency data")


@dataclass(frozen=True)
class Connected:
    """One set of node columns that must form one patch, as compiled by :func:`_add_connected`: the
    columns of its roots, arcs (both directions of every edge) and flows, and the arcs' local tail
    and head (positions in ``nodes``). The solvers separate cuts on it."""

    nodes: np.ndarray
    root: int
    arc: int
    tail: np.ndarray
    head: np.ndarray


@dataclass(frozen=True)
class Corridor:
    """A corridor network as compiled by :func:`_apply_corridors`: its source and other terminals and
    the edges (each listed once) between the units; the solvers separate node cuts on it
    (every set of units that separates a terminal from the source holds a selected unit)."""

    source: int
    sinks: np.ndarray
    tail: np.ndarray
    head: np.ndarray


def _edges(d: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    """The edges (i < j) of an adjacency matrix (either triangle; values > 1e-15)."""
    s = sp.triu(d + d.T, k=1).tocoo()
    keep = s.data > 1.0e-15
    i, j = s.row[keep].astype(np.int64), s.col[keep].astype(np.int64)
    o = np.lexsort((j, i))
    return i[o], j[o]


def _add_connected(b: Builder, nodes: np.ndarray, d: sp.csc_matrix, tag: str) -> None:
    """The selected ``nodes`` columns form one patch (or none is selected), exactly: a spanning
    arborescence of the selected nodes with a flow that rules out cycles. Columns: one root per node
    (at most one root), one arc per direction of every edge (a selected node has exactly one incoming
    arc unless it is the root; an arc leaves a selected node; not both directions), and a flow per
    arc (at most n - 1 where the arc is used; every non-root selected node keeps at least 1).

    prioritizr's formulation (one arc per edge, from the higher index to the lower) is a restriction:
    it misses the patches in which a unit has no selected neighbour of a higher index (three units in
    a line, the middle one first, have no feasible patch), so its optimum can be worse than the best
    patch, or a feasible problem infeasible."""
    n = nodes.size
    i, j = _edges(d)
    tail = np.concatenate([i, j])
    head = np.concatenate([j, i])
    na, ne = tail.size, i.size
    vt = b.vtype[int(nodes[0])] if n else "B"
    root = b.add_columns(np.zeros(n), 0.0, 1.0, vt, f"{tag}_root")
    arc = b.add_columns(np.zeros(na), 0.0, 1.0, vt, f"{tag}_arc")
    flow = b.add_columns(np.zeros(na), 0.0, max(n - 1.0, 0.0), "C", f"{tag}_flow")
    ar, aa = np.arange(n), np.arange(na)
    rows, cols, vals, sense, rhs, ids = [], [], [], [], [], []
    nrow = 0

    def block(r, c, v, s, h, k, rid):
        nonlocal nrow
        rows.append(np.asarray(r, dtype=np.int64) + nrow)
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(np.asarray(v, dtype=np.float64))
        sense.extend([s] * k)
        rhs.extend([h] * k)
        ids.extend([f"{tag}_{rid}"] * k)
        nrow += k

    block(np.zeros(n), root + ar, np.ones(n), "<=", 1.0, 1, "root")
    # incoming arcs + root - node = 0
    block(np.concatenate([head, ar, ar]), np.concatenate([arc + aa, root + ar, nodes]),
          np.concatenate([np.ones(na), np.ones(n), -np.ones(n)]), "=", 0.0, n, "in")
    # arc - tail <= 0
    block(np.repeat(aa, 2), np.column_stack([arc + aa, nodes[tail]]).ravel(), np.tile([1.0, -1.0], na),
          "<=", 0.0, na, "tail")
    # both directions <= 1
    block(np.tile(np.arange(ne), 2), arc + aa, np.ones(na), "<=", 1.0, ne, "pair")
    # flow - (n - 1) arc <= 0
    block(np.repeat(aa, 2), np.column_stack([flow + aa, arc + aa]).ravel(), np.tile([1.0, -(n - 1.0)], na),
          "<=", 0.0, na, "cap")
    # flow in - flow out - node + n root >= 0
    block(np.concatenate([head, tail, ar, ar]), np.concatenate([flow + aa, flow + aa, nodes, root + ar]),
          np.concatenate([np.ones(na), -np.ones(na), -np.ones(n), np.full(n, float(n))]), ">=", 0.0, n, "flow")
    b.add_rows(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), sense, rhs, ids)
    b.connected.append(Connected(np.asarray(nodes, dtype=np.int64), root, arc, tail, head))


def _apply_contiguity(b: Builder, u, data) -> None:
    """The selected units form one patch: :func:`_add_connected` on the units' columns (an exact
    formulation, not prioritizr's)."""
    _add_connected(b, np.arange(b.n), _adjacency(u, data), "c")


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


def _apply_feature_contiguity(b: Builder, u, data) -> None:
    """Every feature's units (its y columns of the expanded formulation, the units where it counts) form
    one patch: :func:`_add_connected` for every feature (exact, not prioritizr's formulation); ``data``:
    one adjacency matrix for every feature, a list of one per feature, or None (the default adjacency)."""
    nf = b.n_features
    if isinstance(data, (list, tuple)):
        if len(data) != nf:
            raise ValueError(f"feature contiguity data: one matrix per feature ({nf})")
        mats = [_adjacency(u, d) for d in data]
    else:
        m = _adjacency(u, data)
        mats = [m] * nf
    n = b.n
    for f in range(nf):
        _add_connected(b, n + f * n + np.arange(n), mats[f], f"fc{f + 1}")


def _corridor_edges(u, data, resistance):
    """The corridor's edges (i > j, by column) and their resistance (None: no penalty)."""
    if resistance is not None and hasattr(resistance, "to_sparse"):  # an ls.cost Transition
        if u.kind != "raster":
            raise TypeError("a Transition as resistance needs raster planning units")
        if not resistance.symmetric:
            raise ValueError("corridors need a symmetric Transition")
        t = resistance.to_sparse().tocsc()
        if t.shape != (u.template.size, u.template.size):
            raise ValueError("the Transition must be on the planning units' grid")
        t = t[u.index][:, u.index]
        low = sp.tril(t, k=-1).tocsc()
        low.eliminate_zeros()
        low.sort_indices()
        pj = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr))
        res = 1.0 / low.data if resistance.matrix_values == "conductance" else low.data.copy()
        return low.indices.astype(np.int64), pj.astype(np.int64), res
    low = sp.tril(_adjacency(u, data), k=-1).tocsc()
    low.eliminate_zeros()
    low.sort_indices()
    pi = low.indices.astype(np.int64)
    pj = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr)).astype(np.int64)
    if resistance is None:
        return pi, pj, None
    r = unit_values(u, resistance, "corridor resistance")[u.index]
    return pi, pj, (r[pi] + r[pj]) / 2.0


def _apply_corridors(b: Builder, u, terminals, resistance, penalty: float, data) -> None:
    """Corridors (an extension, after prioripy's): every network's terminal units are selected and
    joined through selected units. A single-commodity flow on the edges between units, both ways: the
    first terminal sends one unit of flow to each of the K others, flows <= K x at both ends, and with
    a resistance, ``penalty`` x resistance x flow added to the objective (subtracted when it is
    maximised)."""
    nets = terminals if isinstance(terminals, (list, tuple)) else [terminals]
    pi, pj, res = _corridor_edges(u, data, resistance)
    ne = pi.size
    for net in nets:
        mask = unit_values(u, net, "corridor terminals")[u.index] > 0
        term = np.flatnonzero(mask)
        if term.size < 2:
            raise ValueError("a corridor network needs at least two terminal planning units")
        k = float(term.size - 1)
        sign = 1.0 if b.modelsense == "min" else -1.0
        cost = np.zeros(ne) if res is None else sign * penalty * res
        first = b.add_columns(np.concatenate([cost, cost]), 0.0, k, "C", "flow")
        fwd = first + np.arange(ne)  # pi -> pj
        bwd = first + ne + np.arange(ne)  # pj -> pi
        # capacities: f - K x <= 0 at both ends, both directions
        rows = np.repeat(np.arange(4 * ne), 2)
        f = np.concatenate([fwd, fwd, bwd, bwd])
        x = np.concatenate([pi, pj, pi, pj])
        cols = np.column_stack([f, x]).ravel()
        vals = np.tile([1.0, -k], 4 * ne)
        # balance: out - in = K at the source, -1 at the other terminals, 0 elsewhere
        br = 4 * ne + np.concatenate([pi, pj, pj, pi])
        bc = np.concatenate([fwd, bwd, fwd, bwd])
        bv = np.concatenate([np.ones(ne), np.ones(ne), -np.ones(ne), -np.ones(ne)])
        rhs = np.zeros(b.n)
        rhs[term[0]] = k
        rhs[term[1:]] = -1.0
        b.add_rows(np.concatenate([rows, br]), np.concatenate([cols, bc]), np.concatenate([vals, bv]),
                   ["<="] * (4 * ne) + ["="] * b.n, np.concatenate([np.zeros(4 * ne), rhs]),
                   ["corridor_capacity"] * (4 * ne) + ["corridor_flow"] * b.n)
        b.lb[term] = 1.0
        b.corridors.append(Corridor(int(term[0]), term[1:].astype(np.int64), pi.astype(np.int64),
                                    pj.astype(np.int64)))


# ----------------------------------------------------------------------------- dispatch

def apply_penalty(b: Builder, u, spec) -> None:
    kind = spec[0]
    if kind == "boundary":
        _apply_boundary(b, u, *spec[1:])
    elif kind == "connectivity":
        _apply_connectivity(b, u, *spec[1:])
    elif kind == "neighbor":
        _apply_neighbor(b, u, *spec[1:])
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
    elif kind == "feature_contiguity":
        _apply_feature_contiguity(b, u, *spec[1:])
    elif kind == "corridors":
        _apply_corridors(b, u, *spec[1:])
    else:
        raise ValueError(f"unknown constraint {kind!r}")
