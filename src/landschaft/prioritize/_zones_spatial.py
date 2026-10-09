# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial penalties and constraints with management zones (prioritizr 8.0.6's, bit for bit).

A matrix of zones says how much a pair of neighbouring (or connected) units counts when they are in
two given zones: the diagonal for units in the same zone, the other cells for units in different zones
(prioritizr's ``zones`` argument; by default the identity: only units of the same zone count). The
columns of the model are the units of every zone, zone after zone (column ``z * n + i``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

from ._milp_spatial import (
    Builder,
    _add_connected,
    _cut,
    _pair_columns,
    _r_round,
    _row_sums,
    unit_matrix,
    unit_values,
)


def _footprint(u):
    """The planning units of a raster problem with zones as one layer (1 on a unit, NaN elsewhere): the
    cells with a cost in any zone."""
    v = np.full(u.template.size, np.nan)
    v[u.index] = 1.0
    return u.template.copy(data=v.reshape(u.template.shape))


def boundary_matrix(u) -> sp.csc_matrix:
    """The boundary matrix of the units (prioritizr's boundary_matrix of the zones' cost layers: the cells
    with a cost in any zone)."""
    from . import _milp_matrices as mx
    from ._milp_spatial import boundary_matrix as one_zone_boundary

    if u.kind == "raster":
        return _cut(u, mx.boundary_matrix(_footprint(u)))
    return one_zone_boundary(u)


def _adjacency(u, data) -> sp.csc_matrix:
    """The adjacency of the units: the cells with a cost in any zone (prioritizr's adjacency_matrix of
    several cost layers takes only the cells with a cost in every zone, so a unit missing from one zone
    loses its neighbours, a bug), or ``data``."""
    from . import _milp_matrices as mx
    from ._milp_spatial import _adjacency as one_zone_adjacency

    if data is None and u.kind == "raster":
        return _cut(u, mx.adjacency_matrix(_footprint(u)))
    return one_zone_adjacency(u, data)


def zone_matrix(u, zones, what: str, binary: bool = False, symmetric: bool = True) -> np.ndarray:
    """A zones x zones matrix: None or ``"same"`` (the identity), ``"any"`` (every pair counts fully), a
    matrix in the zones' order, or a DataFrame / mapping of mappings by zone name."""
    nz = u.nz
    if zones is None or (isinstance(zones, str) and zones == "same"):
        return np.eye(nz)
    if isinstance(zones, str):
        if zones != "any":
            raise ValueError(f"{what}: zones must be 'same', 'any' or a zones x zones matrix")
        return np.ones((nz, nz))
    import pandas as pd

    if isinstance(zones, Mapping):
        zones = pd.DataFrame(zones).T
    if isinstance(zones, pd.DataFrame):
        names = list(u.zone_names)
        if set(map(str, zones.index)) != set(names) or set(map(str, zones.columns)) != set(names):
            raise ValueError(f"{what}: the zones matrix must name every zone ({names})")
        zones = zones.rename(index=str, columns=str).loc[names, names].to_numpy(dtype=np.float64)
    z = np.asarray(zones, dtype=np.float64)
    if z.shape != (nz, nz):
        raise ValueError(f"{what}: the zones matrix must be {nz} x {nz}")
    if not np.isfinite(z).all() or z.min() < -1 or z.max() > 1:
        raise ValueError(f"{what}: the zones matrix must hold values from -1 to 1")
    if symmetric and not np.array_equal(z, z.T):
        raise ValueError(f"{what}: the zones matrix must be symmetric")
    if binary and not np.isin(z, (0.0, 1.0)).all():
        raise ValueError(f"{what}: the zones matrix must hold 0 and 1")
    return z


def per_zone(u, value, what: str) -> np.ndarray:
    """One number per zone: a number, a list in the zones' order or a mapping zone -> number."""
    from ._zones import zone_key

    if isinstance(value, Mapping):
        out = np.full(u.nz, np.nan)
        for z, v in value.items():
            out[zone_key(u, z, single=True)[0]] = float(v)
        if np.isnan(out).any():
            raise ValueError(f"{what}: give a value for every zone ({list(u.zone_names)})")
        return out
    a = np.asarray(value, dtype=np.float64).ravel()
    if a.size == 1:
        return np.repeat(a, u.nz)
    if a.size != u.nz:
        raise ValueError(f"{what}: one value or one per zone ({u.nz})")
    return a


def zone_data(u, data, what: str) -> np.ndarray:
    """Values of every unit (cell or row) in every zone, NaN as 0 (zones x units): one dataset for every
    zone, or a mapping zone -> dataset (zones not given: 0); a dataset is what one-zone problems take."""
    from ._zones import zone_key

    total = u.template.size if u.kind == "raster" else len(u.template)
    out = np.zeros((u.nz, total))
    if isinstance(data, Mapping):
        for z, d in data.items():
            out[zone_key(u, z, single=True)[0]] = unit_values(u, d, what)
        return out
    import xarray as xr

    if isinstance(data, xr.DataArray) and data.ndim == 3 and data.shape[0] == u.nz and u.kind == "raster":
        for z in range(u.nz):  # a layer per zone
            out[z] = unit_values(u, data[z], what)
        return out
    out[:] = unit_values(u, data, what)[None, :]
    return out


# ----------------------------------------------------------------------------- penalties

def _boundary(b: Builder, u, penalty: float, edge_factor, data, zones) -> None:
    """rcpp_apply_boundary_penalties with zones."""
    if data is None:
        bm = boundary_matrix(u)
    else:
        bm = _r_round(unit_matrix(u, data, "boundary data"), 6)
        if abs(bm - bm.T).max() > 0 if bm.nnz else False:
            raise ValueError("the boundary data must be symmetric")
    zm = zone_matrix(u, zones, "boundary penalties")
    ef = per_zone(u, edge_factor, "edge_factor")
    if ((ef < 0) | (ef > 1)).any():
        raise ValueError("edge_factor must be from 0 to 1")
    if not abs(penalty) > 1e-50:
        return
    n, nz = u.n, u.nz
    total = bm.diagonal()
    exposed = total - (_row_sums(bm) - total)
    p = -penalty if b.modelsense == "max" else penalty
    low = sp.tril(bm, k=-1).tocsc()
    low.eliminate_zeros()
    low.sort_indices()
    li = low.indices.astype(np.int64)
    lj = np.repeat(np.arange(low.shape[1]), np.diff(low.indptr)).astype(np.int64)
    pen = np.zeros(nz * n)
    pi, pj, pb = [], [], []
    for z1 in range(nz):
        for z2 in range(z1, nz):
            if z1 == z2:
                pen[z1 * n:(z1 + 1) * n] += p * zm[z1, z2] * ((total - exposed) + (exposed * ef[z1]))
            cv = low.data * zm[z1, z2] * p
            keep = np.abs(cv) > 1.0e-15
            i, j, v = li[keep], lj[keep], cv[keep] * -2.0
            if z1 == z2:
                pi.append(z1 * n + i)
                pj.append(z1 * n + j)
                pb.append(v)
            else:  # both ways: (z1 i, z2 j) then (z2 i, z1 j), entry by entry
                pi.append(np.column_stack([z1 * n + i, z2 * n + i]).ravel())
                pj.append(np.column_stack([z2 * n + j, z1 * n + j]).ravel())
                pb.append(np.repeat(v, 2))
    b.obj[:nz * n] += pen
    if pi:
        _pair_columns(b, np.concatenate(pi), np.concatenate(pj), np.concatenate(pb), "b")


def _zone_pairs(u, m: sp.csc_matrix, zm: np.ndarray, scale: float, lower: bool, diag_same_only: bool):
    """The entries of every zone pair's matrix (m x zm[z1, z2], lower triangle with its diagonal when
    ``lower``) times ``scale``, in armadillo's order (zone pair by zone pair, then by column), with
    the diagonal of the pairs of different zones left out: (z1, z2, row, col, value)."""
    out = []
    for z1 in range(u.nz):
        for z2 in range(u.nz):
            mm = m * zm[z1, z2]
            if lower:
                mm = sp.tril(mm).tocsc()
            mm.sort_indices()
            col = np.repeat(np.arange(mm.shape[1]), np.diff(mm.indptr)).astype(np.int64)
            row = mm.indices.astype(np.int64)
            v = mm.data * scale
            keep = np.abs(v) > 1.0e-15
            if diag_same_only and z1 != z2:
                keep &= row != col
            out.append((z1, z2, row[keep], col[keep], v[keep]))
    return out


def _connectivity(b: Builder, u, penalty: float, data, zones) -> None:
    """rcpp_apply_connectivity_penalties with zones (symmetric data)."""
    m = unit_matrix(u, data, "connectivity data")
    if m.nnz and abs(m - m.T).max() > 0:
        raise ValueError("the connectivity data must be symmetric (use add_asym_connectivity_penalties)")
    zm = zone_matrix(u, zones, "connectivity penalties", symmetric=False)
    mult = -1.0 if b.modelsense == "min" else 1.0
    n = u.n
    pen = np.zeros(u.nz * n)
    pi, pj, pb = [], [], []
    for z1, z2, row, col, v in _zone_pairs(u, m, zm, penalty * mult, True, True):
        unit = (row == col) & (z1 == z2)
        np.add.at(pen, z1 * n + row[unit], v[unit])
        pi.append(z1 * n + row[~unit])
        pj.append(z2 * n + col[~unit])
        pb.append(v[~unit])
    b.obj[:u.nz * n] += pen
    if pi:
        _pair_columns(b, np.concatenate(pi), np.concatenate(pj), np.concatenate(pb), "c")


def _asym_connectivity(b: Builder, u, penalty: float, data, zones) -> None:
    """rcpp_apply_asym_connectivity_penalties with zones."""
    m = unit_matrix(u, data, "connectivity data")
    zm = zone_matrix(u, zones, "asymmetric connectivity penalties", symmetric=False)
    mult1, mult2 = (1.0, -1.0) if b.modelsense == "min" else (-1.0, 1.0)
    n = u.n
    pen = np.zeros(u.nz * n)
    pi, pj, pb = [], [], []
    for z1, z2, row, col, v in _zone_pairs(u, m, zm, penalty, False, True):
        unit = (row == col) & (z1 == z2)
        # entry by entry (the unit penalties add up in armadillo's order)
        for k in range(row.size):
            if unit[k]:
                pen[z1 * n + row[k]] += mult2 * v[k]
            else:
                pen[z1 * n + row[k]] += mult1 * v[k]
        pi.append(z1 * n + row[~unit])
        pj.append(z2 * n + col[~unit])
        pb.append(mult2 * v[~unit])
    b.obj[:u.nz * n] += pen
    if pi:
        _pair_columns(b, np.concatenate(pi), np.concatenate(pj), np.concatenate(pb), "ac")


def _linear_penalty(b: Builder, u, penalty, data) -> None:
    """rcpp_apply_linear_penalties with zones: a penalty per zone (zones with 0 add nothing; prioritizr
    then leaves out the penalties of every zone, a bug)."""
    p = per_zone(u, penalty, "linear penalties")
    if not np.abs(p).max() > 1e-50:
        return
    if b.modelsense == "max":
        p = -p
    d = zone_data(u, data, "linear penalty data")[:, u.index]
    n = u.n
    for z in range(u.nz):
        cv = d[z] * p[z]
        b.obj[z * n:(z + 1) * n] += np.where((d[z] != 0) & (np.abs(cv) > 1.0e-15), cv, 0.0)


def _linear_constraint(b: Builder, u, threshold: float, sense: str, data) -> None:
    """rcpp_apply_linear_constraints with zones: the sum over every zone of the units' values there."""
    d = zone_data(u, data, "linear constraint data")[:, u.index].ravel()  # zone after zone
    nz_ = np.flatnonzero(d)
    b.add_rows(np.zeros(nz_.size, dtype=np.int64), nz_, d[nz_], [sense], [threshold], "lc")


# ----------------------------------------------------------------------------- constraints

def _contiguity(b: Builder, u, data, zones) -> None:
    """The units of every cluster of connected zones (the components of the zones matrix's graph; zones
    with 0 on the diagonal free) form one patch: :func:`~._milp_spatial._add_connected` on the cluster's
    units (exact, not prioritizr's formulation); with several zones in a cluster, on columns that hold the
    sum of the unit's columns in them."""
    zm = zone_matrix(u, zones, "contiguity constraints", binary=True)
    d = _adjacency(u, data)
    n, nz = u.n, u.nz
    from scipy.sparse.csgraph import connected_components

    _, label = connected_components(sp.csr_matrix(zm - np.diag(np.diag(zm))), directed=False)
    for c in np.unique(label):
        members = [z for z in range(nz) if label[z] == c and zm[z, z] > 0]
        if not members:
            continue
        if len(members) == 1:
            nodes = members[0] * n + np.arange(n)
        else:  # a unit is in the cluster when it is in any of its zones: y_i = sum_z x_zi
            first = b.add_columns(np.zeros(n), 0.0, 1.0, b.vtype[0], "cluster")
            nodes = first + np.arange(n)
            k = len(members)
            r = np.repeat(np.arange(n), k + 1)
            cols = np.column_stack([nodes, *[z * n + np.arange(n) for z in members]]).ravel()
            vals = np.tile([1.0, *([-1.0] * k)], n)
            b.add_rows(r, cols, vals, ["="] * n, [0.0] * n, "cluster")
        _add_connected(b, nodes, d, f"c{c + 1}")


def _neighbors(b: Builder, u, k, clamp: bool, data, zones) -> None:
    """rcpp_apply_neighbor_constraints with zones: a unit in zone z1 has at least k[z1] neighbours in
    the zones z2 with zones[z1, z2] = 1."""
    zm = zone_matrix(u, zones, "neighbor constraints", binary=True)
    kk = per_zone(u, k, "k")
    if (kk < 0).any() or (kk != np.round(kk)).any():
        raise ValueError("k must be whole numbers >= 0")
    d = _adjacency(u, data)
    n, nz = u.n, u.nz
    kmat = np.repeat(kk[None, :], n, axis=0)
    if clamp:
        for z1 in range(nz):
            mx = None
            for z2 in range(nz):
                mm = d * zm[z1, z2]
                mx = mm if mx is None else mx.maximum(mm)
            rs = np.asarray(mx.sum(axis=1)).ravel()
            kmat[:, z1] = np.minimum(kk[z1], rs)
    if not kmat.max(initial=0.0) > 0:
        return
    rows = [np.arange(nz * n)]
    cols = [np.arange(nz * n)]
    vals = [(kmat.T.ravel()) * -1.0]
    for z1 in range(nz):
        for z2 in range(nz):
            mm = (d * zm[z1, z2]).tocsc()
            mm.sort_indices()
            col = np.repeat(np.arange(mm.shape[1]), np.diff(mm.indptr)).astype(np.int64)
            row = mm.indices.astype(np.int64)
            keep = (mm.data > 1.0e-15) & (row != col)
            rows.append(z1 * n + row[keep])
            cols.append(z2 * n + col[keep])
            vals.append(np.ones(int(keep.sum())))
    b.add_rows(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), [">="] * (nz * n),
               [0.0] * (nz * n), "n")


def apply_zone_penalty(b: Builder, u, spec) -> None:
    kind = spec[0]
    if kind == "boundary":
        _boundary(b, u, *spec[1:])
    elif kind == "connectivity":
        _connectivity(b, u, *spec[1:])
    elif kind == "asym_connectivity":
        _asym_connectivity(b, u, *spec[1:])
    elif kind == "linear":
        _linear_penalty(b, u, *spec[1:])
    else:
        raise ValueError(f"unknown penalty {kind!r}")


def apply_zone_constraint(b: Builder, u, spec) -> None:
    kind = spec[0]
    if kind == "linear":
        _linear_constraint(b, u, *spec[1:])
    elif kind == "contiguity":
        _contiguity(b, u, *spec[1:])
    elif kind == "neighbors":
        _neighbors(b, u, *spec[1:])
    else:
        raise ValueError(f"{kind.replace('_', ' ')} constraints are not supported with zones")
