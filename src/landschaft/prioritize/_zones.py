# SPDX-License-Identifier: GPL-3.0-or-later
"""Management zones (prioritizr 8.0.6's zones): planning units that go to at most one of several zones.

Every zone has its own cost and its own amount of every feature. The easy form
gives the features once with a *contribution* per zone (a number, one number per
feature, or a raster / column of factors: the share of the feature that counts
in the zone); the general form gives every zone its own features (a mapping
zone -> features), as prioritizr's ``zones()``.

The planning units are the cells (rows) with a cost in any zone; a NaN cost in
a zone keeps the unit out of it. The compiled model has one column per unit and
zone, zone after zone (column ``z * n + i``), as prioritizr's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from ._milp import (
    BLOCK_BYTES,
    Units,
    _layers,
    _r_sum,
    _rij,
    _sequential_sum,
    exact_sums,
)


def wants_zones(x, cost_column, contribution, zones) -> bool:
    """Whether problem() was given zones: several costs (a mapping, a multi-band raster, several cost
    columns), contributions or a zone table."""
    if contribution is not None or zones is not None:
        return True
    if isinstance(x, Mapping):
        return True
    if isinstance(x, xr.DataArray) and x.ndim == 3 and x.shape[0] > 1:
        return True
    return isinstance(cost_column, Mapping) or (
        isinstance(cost_column, Sequence) and not isinstance(cost_column, str) and len(cost_column) > 1)


def _names(names) -> tuple[str, ...]:
    out = tuple(str(z) for z in names)
    if len(out) < 1:
        raise ValueError("no zones")
    if len(set(out)) != len(out):
        raise ValueError(f"zone names must be unique: {list(out)}")
    return out


def _raster_costs(x) -> tuple[tuple[str, ...], list[xr.DataArray]]:
    if isinstance(x, Mapping):
        names = _names(x.keys())
        layers = []
        for z, c in x.items():
            if not isinstance(c, xr.DataArray):
                raise TypeError(f"the cost of zone {z!r} must be a raster")
            if c.ndim == 3 and c.shape[0] == 1:
                c = c[0]
            if c.ndim != 2:
                raise ValueError(f"the cost of zone {z!r} must have one band")
            layers.append(c)
        return names, layers
    band = x.dims[0]
    names = _names(x[band].values if band in x.coords else [f"zone_{k + 1}" for k in range(x.shape[0])])
    return names, [x[k] for k in range(x.shape[0])]


def _same_grid(a: xr.DataArray, b: xr.DataArray, what: str) -> None:
    if a.shape[-2:] != b.shape[-2:]:
        raise ValueError(f"{what} ({b.shape[-2:]}) must be on the planning units' grid ({a.shape[-2:]})")
    for dim in ("y", "x"):
        if dim in a.coords and dim in b.coords and not np.allclose(a[dim].values, b[dim].values):
            raise ValueError(f"{what} must be on the planning units' grid")


def _contributions(contribution, names: tuple[str, ...], nf: int, kind: str):
    """Every zone's contribution: ("scalar", per-feature factors) or ("map", raster / column / values)."""
    if contribution is None:
        return [("scalar", np.ones(nf)) for _ in names]
    if not isinstance(contribution, Mapping):
        raise TypeError("contribution must be a mapping zone -> number, per-feature numbers, raster or column")
    unknown = sorted(set(map(str, contribution)) - set(names))
    if unknown:
        raise ValueError(f"contribution: unknown zones {unknown} (zones: {list(names)})")
    given = {str(k): v for k, v in contribution.items()}
    out = []
    for z in names:
        c = given.get(z, 1.0)
        if isinstance(c, (xr.DataArray, str)):
            if isinstance(c, str) and kind == "raster":
                raise TypeError(f"contribution of zone {z!r}: a column name needs table planning units")
            if isinstance(c, xr.DataArray) and kind == "table":
                raise TypeError(f"contribution of zone {z!r}: a raster needs raster or polygon planning units")
            out.append(("map", c))
            continue
        a = np.asarray(c, dtype=np.float64).ravel()
        if a.size == 1:
            a = np.repeat(a, nf)
        if a.size != nf:
            if kind == "table":  # values per row of the table
                out.append(("map", np.asarray(c, dtype=np.float64).ravel()))
                continue
            raise ValueError(f"contribution of zone {z!r}: one number, one per feature ({nf}), or a raster")
        if not np.isfinite(a).all():
            raise ValueError(f"contribution of zone {z!r} must be finite")
        out.append(("scalar", a))
    return out


def _zone_matrix(blocks: list[sp.csr_matrix], n: int) -> sp.csr_matrix:
    """The zones' features x units matrices side by side: columns z * n + i."""
    m = sp.hstack(blocks, format="csr") if len(blocks) > 1 else sp.csr_matrix(blocks[0])
    m.sort_indices()
    if m.shape[1] != n * len(blocks):
        raise ValueError("internal: zone blocks of different widths")
    return m


def _units(kind, names, zcost, blocks, totals, feature_names, template, index, features=None, base_totals=None,
           all_blocks=None, r_sums=False, source=()) -> Units:
    n = zcost.shape[1]
    return Units(kind, zcost.ravel().copy(), _zone_matrix(blocks, n), totals, tuple(feature_names), template, index,
                 features, None if all_blocks is None else tuple(all_blocks), r_sums, zone_names=names,
                 zone_rij=tuple(blocks), base_totals=base_totals, zone_source=source)


# ----------------------------------------------------------------------------- rasters

def raster_zone_units(x, features, contribution, n_threads: int | None = None) -> Units:
    from .. import _core, _threads

    names, costs = _raster_costs(x)
    template = costs[0]
    for z, c in zip(names, costs):
        _same_grid(template, c, f"the cost of zone {z!r}")
    nz = len(names)
    ny, nx = template.shape
    c = np.stack([np.asarray(v.values, dtype=np.float64).ravel() for v in costs])
    cells = np.flatnonzero(np.isfinite(c).any(axis=0))
    if cells.size == 0:
        raise ValueError("no cell has a cost in any zone")
    unit = np.full(c.shape[1], -1, dtype=np.int64)
    unit[cells] = np.arange(cells.size)
    threads = _threads.resolve(n_threads)
    general = isinstance(features, Mapping)
    if general:
        if contribution is not None:
            raise ValueError("give either the features of every zone (a mapping) or contributions, not both")
        missing = sorted(set(names) - set(map(str, features)))
        extra = sorted(set(map(str, features)) - set(names))
        if missing or extra:
            raise ValueError(f"the features' zones {sorted(map(str, features))} must be the costs' zones {list(names)}")
        given = {str(k): v for k, v in features.items()}
        per_zone = [_layers(given[z]) for z in names]
        nf = len(per_zone[0])
        if any(len(p) != nf for p in per_zone):
            raise ValueError("every zone must have the same number of features")
        layers = [f for p in per_zone for _, f in p]
        feature_names = [n for n, _ in per_zone[0]]
        contrib = None
    else:
        named = _layers(features)
        nf = len(named)
        layers = [f for _, f in named]
        feature_names = [n for n, _ in named]
        contrib = _contributions(contribution, names, nf, "raster")
        for z, (k, v) in zip(names, contrib):
            if k == "map":
                v = v[0] if v.ndim == 3 and v.shape[0] == 1 else v
                if v.ndim != 2:
                    raise ValueError(f"the contribution raster of zone {z!r} must have one band")
                _same_grid(template, v, f"the contribution raster of zone {z!r}")
    if nf == 0:
        raise ValueError("no features")
    for f in layers:
        _same_grid(template, f, "the features")
    acc = _core.OptFeatureMatrix(nz * nf, nf)
    base = None if general else _core.OptFeatureMatrix(nf)  # the base layers' totals (targets summed over zones)
    width = len(layers) + (0 if general else sum(k == "map" for k, _ in contrib))
    step = max(1, BLOCK_BYTES // (8 * nx * max(1, width)))
    lazy = [k for k, f in enumerate(layers) if f.chunks is not None]
    scalar = np.ones(nz * nf) if general else np.concatenate([v if k == "scalar" else np.ones(nf) for k, v in contrib])
    has_scale = None if general else np.array([k == "map" for k, _ in contrib], dtype=np.uint8)
    maps = None if general else [(z, v[0] if v.ndim == 3 else v) for z, (k, v) in enumerate(contrib) if k == "map"]
    for y0 in range(0, ny, step):
        y1 = min(ny, y0 + step)
        rows = [f.data[y0:y1] for f in layers]
        if lazy:
            import dask

            for k, v in zip(lazy, dask.compute(*[rows[k] for k in lazy])):
                rows[k] = v
        block = np.empty((len(layers), (y1 - y0) * nx), dtype=np.float64)
        for k, v in enumerate(rows):
            v = np.ma.filled(v.astype(np.float64), np.nan) if np.ma.isMaskedArray(v) else v
            block[k] = np.asarray(v, dtype=np.float64).ravel()
        u = unit[y0 * nx:y1 * nx]
        if general:  # zone z's layers are rows z * nf .. (z + 1) * nf
            acc.add(block, u, threads)
            continue
        scale = None
        if maps:
            scale = np.ones((nz, block.shape[1]))
            for z, v in maps:
                d = v.data[y0:y1]
                if hasattr(d, "compute"):
                    d = d.compute()
                scale[z] = np.asarray(d, dtype=np.float64).ravel()
        acc.add_scaled(block, u, scale, scalar, has_scale, threads)
        base.add(block, np.full(block.shape[1], -1, dtype=np.int64), threads)
    tot, start, index, value = acc.finish()
    full = sp.csr_matrix((value, index, start), shape=(nz * nf, cells.size))
    full.has_sorted_indices = True
    blocks = [sp.csr_matrix(full[z * nf:(z + 1) * nf]) for z in range(nz)]
    totals = np.asarray(tot).reshape(nz, nf).T.copy()
    base_totals = None if general else np.asarray(base.finish()[0])
    return _units("raster", names, c[:, cells], blocks, totals, feature_names, template, cells,
                  features=None if general else features, base_totals=base_totals,
                  source=("raster", tuple(layers), None if general else tuple(contrib), nf))


# ----------------------------------------------------------------------------- tables and polygons

def _cost_columns(x: pd.DataFrame, cost_column, features) -> tuple[tuple[str, ...], list[str]]:
    if isinstance(cost_column, Mapping):
        names = _names(cost_column.keys())
        cols = [str(v) for v in cost_column.values()]
    elif isinstance(cost_column, str):
        cols = [cost_column]
        names = _names(features.keys()) if isinstance(features, Mapping) else (cost_column,)
    else:
        cols = [str(v) for v in cost_column]
        names = _names(features.keys()) if isinstance(features, Mapping) else _names(cols)
    if len(cols) != len(names):
        raise ValueError(f"one cost column per zone ({len(names)}), got {len(cols)}")
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise ValueError(f"no cost columns {missing}")
    return names, cols


def _zone_order(features: Mapping, names: tuple[str, ...]) -> list:
    given = {str(k): v for k, v in features.items()}
    if set(given) != set(names):
        raise ValueError(f"the features' zones {sorted(given)} must be the costs' zones {list(names)}")
    return [given[z] for z in names]


def table_zone_units(x: pd.DataFrame, features, cost_column, contribution) -> Units:
    names, cols = _cost_columns(x, cost_column, features)
    c = np.stack([x[col].to_numpy(dtype=np.float64) for col in cols])
    rows = np.flatnonzero(np.isfinite(c).any(axis=0))
    if rows.size == 0:
        raise ValueError("no row has a cost in any zone")
    base_totals = None
    if isinstance(features, Mapping):
        if contribution is not None:
            raise ValueError("give either the features of every zone (a mapping) or contributions, not both")
        per_zone = [[f] if isinstance(f, str) else list(f) for f in _zone_order(features, names)]
        nf = len(per_zone[0])
        if any(len(p) != nf for p in per_zone):
            raise ValueError("every zone must have the same number of feature columns")
        feature_names = per_zone[0]
        amounts = []
        for p in per_zone:
            missing = [n for n in p if n not in x.columns]
            if missing:
                raise ValueError(f"no feature columns {missing}")
            amounts.append(x[p].to_numpy(dtype=np.float64).T)
    else:
        feature_names = [features] if isinstance(features, str) else list(features)
        missing = [n for n in feature_names if n not in x.columns]
        if missing:
            raise ValueError(f"no feature columns {missing}")
        nf = len(feature_names)
        base = x[feature_names].to_numpy(dtype=np.float64).T
        base_totals = np.array([_r_sum(np.nan_to_num(a, nan=0.0)) for a in base])
        amounts = []
        for k, v in _contributions(contribution, names, nf, "table"):
            if k == "scalar":
                amounts.append(base * v[:, None])
            else:
                f = x[v].to_numpy(dtype=np.float64) if isinstance(v, str) else np.asarray(v, dtype=np.float64)
                if f.size != len(x):
                    raise ValueError("a contribution needs one value per row of the table")
                amounts.append(base * f[None, :])
    totals = np.array([[_r_sum(np.nan_to_num(a[f], nan=0.0)) for a in amounts] for f in range(nf)])  # colSums
    blocks = [_rij(a[:, rows]) for a in amounts]
    all_blocks = [_rij(a) for a in amounts]
    return _units("table", names, c[:, rows], blocks, totals, feature_names, x, rows, base_totals=base_totals,
                  all_blocks=all_blocks, r_sums=True, source=("table", None if base_totals is None else _rij(base)))


def polygon_zone_units(x, features, cost_column, contribution) -> Units:
    from ._milp import _stack

    names, cols = _cost_columns(x, cost_column, features)
    nz = len(names)
    c = np.stack([x[col].to_numpy(dtype=np.float64) for col in cols])
    rows = np.flatnonzero(np.isfinite(c).any(axis=0))
    if rows.size == 0:
        raise ValueError("no row has a cost in any zone")
    base_totals = None
    if isinstance(features, Mapping):
        if contribution is not None:
            raise ValueError("give either the features of every zone (a mapping) or contributions, not both")
        stacks = [_stack(f) for f in _zone_order(features, names)]
        nf = stacks[0].shape[0]
        if any(s.shape[0] != nf for s in stacks):
            raise ValueError("every zone must have the same number of features")
        ref = stacks[0]
    else:
        ref = _stack(features)
        nf = ref.shape[0]
        stacks = []
        for z, (k, v) in zip(names, _contributions(contribution, names, nf, "raster")):
            if k == "scalar":
                stacks.append(ref * xr.DataArray(v, dims=(ref.dims[0],)))
            else:
                v = v[0] if v.ndim == 3 else v
                _same_grid(ref, v, f"the contribution raster of zone {z!r}")
                stacks.append(ref * np.asarray(v.values, dtype=np.float64)[None])
    feature_names = [str(b) for b in ref[ref.dims[0]].values] if ref.dims[0] in ref.coords else [
        f"feature_{k + 1}" for k in range(nf)]
    geoms = x
    crs = ref.rio.crs
    if x.crs is not None and crs is not None and x.crs != crs:
        from .._vector import to_crs

        geoms = to_crs(x, crs)
    layers = [ref] if base_totals is None and not isinstance(features, Mapping) else []
    all_blocks, totals = [], np.zeros((nf, nz))
    for z, s in enumerate(stacks):
        s = s.rio.write_crs(crs) if crs is not None else s
        s = s.rio.write_transform(ref.rio.transform())
        a = exact_sums(s, geoms.geometry.values).T
        a[np.abs(a) < 1e-10] = 0.0  # prioritizr's fast_extract
        full = _rij(a)
        totals[:, z] = [_sequential_sum(full.getrow(i).data) for i in range(nf)]  # Matrix's rowSums
        all_blocks.append(full)
    base_full = None
    if layers:
        a = exact_sums(ref, geoms.geometry.values).T
        a[np.abs(a) < 1e-10] = 0.0
        base_full = _rij(a)
        base_totals = np.array([_sequential_sum(base_full.getrow(i).data) for i in range(nf)])
    blocks = []
    for full in all_blocks:
        m = full[:, rows].tocsr()
        m.sort_indices()
        blocks.append(m)
    return _units("table", names, c[:, rows], blocks, totals, feature_names, x, rows,
                  features=None if isinstance(features, Mapping) else features, base_totals=base_totals,
                  all_blocks=all_blocks, source=("table", base_full))


def rij_zone_units(x: pd.DataFrame, spec: pd.DataFrame, rij: pd.DataFrame, cost_column, zones) -> Units:
    """Marxan-style tables with a zone column (prioritizr's ``problem(x, features, rij, zones = ...)``)."""
    if not isinstance(spec, pd.DataFrame) or not {"id", "name"} <= set(spec.columns):
        raise ValueError("with rij, features must be a table with columns id and name")
    if not {"pu", "species", "amount"} <= set(rij.columns):
        raise ValueError("rij must have the columns pu, species and amount")
    if "id" not in x.columns:
        raise ValueError("with rij, the planning-unit table needs an id column")
    if zones is None:
        if "zone" in rij.columns and rij["zone"].nunique() > 1:
            raise ValueError("rij has several zones: give the zones table (id, name)")
        zones = pd.DataFrame({"id": [rij["zone"].iloc[0] if "zone" in rij.columns else 1],
                              "name": [cost_column if isinstance(cost_column, str) else "1"]})
    zones = pd.DataFrame(zones)
    if not {"id", "name"} <= set(zones.columns):
        raise ValueError("the zones table needs the columns id and name")
    names = _names(zones["name"])
    if isinstance(cost_column, Mapping):
        if set(map(str, cost_column)) != set(names):
            raise ValueError(f"the cost columns' zones {sorted(map(str, cost_column))} must be the zones {list(names)}")
        cols = [str(cost_column[z]) for z in names]
    else:
        cols = [cost_column] if isinstance(cost_column, str) else [str(v) for v in cost_column]
    if len(cols) != len(names):
        raise ValueError(f"one cost column per zone ({len(names)}), got {len(cols)}")
    nz, nf = len(names), len(spec)
    c = np.stack([x[col].to_numpy(dtype=np.float64) for col in cols])
    rows = np.flatnonzero(np.isfinite(c).any(axis=0))
    unit_of = pd.Series(np.arange(len(x)), index=x["id"].to_numpy())
    feat_of = pd.Series(np.arange(nf), index=spec["id"].to_numpy())
    zone_of = pd.Series(np.arange(nz), index=zones["id"].to_numpy())
    r = unit_of.reindex(rij["pu"].to_numpy()).to_numpy()
    k = feat_of.reindex(rij["species"].to_numpy()).to_numpy()
    zc = zone_of.reindex(rij["zone"].to_numpy()).to_numpy() if "zone" in rij.columns else np.zeros(len(rij))
    if np.isnan(r).any() or np.isnan(k).any() or np.isnan(zc).any():
        raise ValueError("rij refers to planning units, features or zones that are not in the tables")
    r, k, zc = r.astype(np.int64), k.astype(np.int64), zc.astype(np.int64)
    a = rij["amount"].to_numpy(dtype=np.float64)
    # the totals of every feature and zone: the amounts added in the table's order (Matrix's sparseMatrix)
    totals = np.zeros((nf, nz))
    np.add.at(totals, (k, zc), a)
    pos = np.full(len(x), -1, dtype=np.int64)
    pos[rows] = np.arange(rows.size)
    blocks, all_blocks = [], []
    for z in range(nz):
        sel = zc == z
        full = sp.csr_matrix((a[sel], (k[sel], r[sel])), shape=(nf, len(x)))
        full.sum_duplicates()
        all_blocks.append(full)
        keep = sel & (pos[r] >= 0)
        m = sp.csr_matrix((a[keep], (k[keep], pos[r[keep]])), shape=(nf, rows.size))
        m.sum_duplicates()
        m.eliminate_zeros()
        m.sort_indices()
        blocks.append(m)
    return _units("table", names, c[:, rows], blocks, totals, [str(v) for v in spec["name"]], x, rows,
                  all_blocks=all_blocks, source=("table", None))


def zone_units(x, features, *, cost_column=None, rij=None, contribution=None, zones=None) -> Units:
    if isinstance(x, (Mapping, xr.DataArray)):
        if features is None:
            raise ValueError("a raster problem needs the features' rasters")
        if zones is not None or rij is not None:
            raise ValueError("zones= and rij= are for tables of planning units")
        return raster_zone_units(x, features, contribution)
    if isinstance(x, pd.DataFrame):
        if cost_column is None:
            raise ValueError("a table of planning units needs its cost columns (one per zone)")
        if rij is not None:
            if contribution is not None:
                raise ValueError("with rij, the amounts of every zone are in its zone column (no contributions)")
            return rij_zone_units(x, features, rij, cost_column, zones)
        first = next(iter(features.values())) if isinstance(features, Mapping) else features
        rasters = isinstance(first, (xr.DataArray, xr.Dataset)) or (
            isinstance(first, list) and bool(first) and isinstance(first[0], xr.DataArray))
        if rasters:
            if not hasattr(x, "geometry"):
                raise TypeError("raster features need raster or polygon planning units")
            return polygon_zone_units(x, features, cost_column, contribution)
        return table_zone_units(x, features, cost_column, contribution)
    raise TypeError("x must be rasters of the zones' costs (a mapping or a multi-band raster) or a table")


# ----------------------------------------------------------------------------- zones in the problem's methods

def zone_key(u: Units, zone, single: bool = False) -> tuple[int, ...]:
    """Zone names (one, or a list) as zone indices, in the order given (duplicates dropped)."""
    if not u.zoned:
        raise ValueError("zones are given but the problem has none")
    items = [zone] if isinstance(zone, (str, int, np.integer)) else list(zone)
    index = {z: k for k, z in enumerate(u.zone_names)}
    out: list[int] = []
    for z in items:
        if isinstance(z, (int, np.integer)) and not isinstance(z, bool) and str(z) not in index:
            k = int(z)
            if not 0 <= k < u.nz:
                raise ValueError(f"zone index {k} out of range (0-{u.nz - 1})")
        else:
            if str(z) not in index:
                raise ValueError(f"unknown zone {z!r} (zones: {list(u.zone_names)})")
            k = index[str(z)]
        if k not in out:
            out.append(k)
    if not out:
        raise ValueError("no zone given")
    if single and len(out) != 1:
        raise ValueError("one zone expected")
    return tuple(out)


def zone_mask(p, locked) -> np.ndarray:
    """Locked units of every zone as a mask over the model's unit columns (zone after zone): a mapping zone
    -> units (anything a one-zone problem takes), a raster with a band per zone, or a list of one column
    name per zone. Only where the unit has a cost in the zone (as prioritizr's rasters); a unit locked
    into two zones is an error."""
    from ._milp import unit_mask

    u = p.units
    n, nz = u.n, u.nz
    out = np.zeros(nz * n, dtype=bool)
    if locked is None:
        return out
    if isinstance(locked, Mapping):
        items = [(zone_key(u, z, single=True)[0], v) for z, v in locked.items()]
    elif isinstance(locked, xr.DataArray) and locked.ndim == 3:
        if locked.shape[0] != nz:
            raise ValueError(f"a raster of locked units needs one band per zone ({nz})")
        items = [(z, locked[z]) for z in range(nz)]
    elif isinstance(locked, Sequence) and not isinstance(locked, str) and len(locked) == nz and all(
            isinstance(v, str) for v in locked):
        items = list(enumerate(locked))
    else:
        raise TypeError("with zones, give the locked units of every zone: a mapping zone -> units")
    for z, v in items:
        out[z * n:(z + 1) * n] |= unit_mask(u, v)
    out &= np.isfinite(u.cost)
    twice = out.reshape(nz, n).sum(axis=0) > 1
    if twice.any() and locked is p.locked_in:
        raise ValueError(f"{int(twice.sum())} planning units are locked into more than one zone")
    return out


def _zone_label(u: Units, zones: tuple[int, ...]) -> str:
    return " + ".join(u.zone_names[z] for z in zones)


def resolve_zone_targets(p):
    """The target rows of a problem with zones: the targets added without a zone (summed over every zone,
    a matrix per zone, or a table), then the targets of given zones, in the order added."""
    from ._milp_compile import Targets

    u = p.units
    nf, nz = len(u.feature_names), u.nz
    index = {n: k for k, n in enumerate(u.feature_names)}
    every = tuple(range(nz))
    feat, sense, value, total, zone = [], [], [], [], []

    def denominator(f: int, zs: tuple[int, ...]) -> float:
        if len(zs) == 1:
            return float(u.totals[f, zs[0]])
        if u.base_totals is not None:  # contributions: shares of the base features' totals
            return float(u.base_totals[f])
        return _r_sum(np.array([u.totals[f, z] for z in zs]))  # prioritizr: R's sum() of the zones' totals

    def add(f: int, zs: tuple[int, ...], kind: str, t: float, s: str = ">=") -> None:
        d = denominator(f, zs)
        feat.append(f)
        zone.append(zs)
        sense.append(s)
        total.append(d)
        value.append(d * t if kind == "relative" else t)

    spec = p.targets
    if spec is not None:
        kind = spec[0]
        if kind in ("relative", "absolute"):
            a = np.asarray(spec[1], dtype=np.float64)
            if a.ndim == 2:  # prioritizr's matrix: every zone's own targets (feature first, zone after zone)
                for z in range(nz):
                    for f in range(nf):
                        add(f, (z,), kind, a[f, z])
            else:
                for f in range(nf):
                    add(f, every, kind, a[f])
        elif kind == "manual":
            t: pd.DataFrame = spec[1]
            missing = sorted(set(map(str, t["feature"])) - set(index))
            if missing:
                raise ValueError(f"unknown features in the targets: {missing}")
            kinds = t["type"].astype(str).to_numpy()
            if not np.isin(kinds, ("relative", "absolute")).all():
                raise ValueError("target types must be 'relative' or 'absolute'")
            senses = list(t["sense"].astype(str)) if "sense" in t.columns else [">="] * len(t)
            if not set(senses) <= {">=", "<=", "="}:
                raise ValueError("target senses must be '>=', '<=' or '='")
            zones = list(t["zone"]) if "zone" in t.columns else [every] * len(t)
            for f, k, v, s, zs in zip(t["feature"], kinds, t["target"].to_numpy(dtype=np.float64), senses, zones):
                add(index[str(f)], zs, k, float(v), s)
        else:
            raise ValueError(f"{kind} targets are not supported with zones")
    for zs, kind, a in p.zone_targets:
        for f in range(nf):
            add(f, zs, kind, a[f])
    if not feat:
        raise ValueError("no targets")
    return Targets(np.array(feat, dtype=np.int64), tuple(sense), np.array(value), np.array(total),
                   zone=tuple(zone), zone_label=tuple(_zone_label(u, zs) for zs in zone))


def fake_zone_targets(u: Units):
    """The targets prioritizr's compile() makes for objectives without targets: every feature in every zone
    (feature first, zone after zone)."""
    from ._milp_compile import Targets

    nf, nz = len(u.feature_names), u.nz
    zones = tuple((z,) for z in range(nz) for _ in range(nf))
    return Targets(np.tile(np.arange(nf), nz), ("?",) * (nf * nz), np.zeros(nf * nz), u.totals.T.ravel().copy(),
                   zone=zones, zone_label=tuple(_zone_label(u, zs) for zs in zones))


def zone_target_matrix(u: Units, t) -> sp.csr_matrix:
    """The targets' rows over the unit columns: every target's feature in each of its zones."""
    n, nz = u.n, u.nz
    rows, cols, vals = [], [], []
    member = np.zeros((t.n, nz), dtype=bool)
    for i, zs in enumerate(t.zone):
        member[i, list(zs)] = True
    for z in range(nz):
        sel = np.flatnonzero(member[:, z])
        if not sel.size:
            continue
        m = u.zone_rij[z][t.feature[sel]].tocoo()
        rows.append(sel[m.row])
        cols.append(m.col.astype(np.int64) + z * n)
        vals.append(m.data)
    if not rows:
        return sp.csr_matrix((t.n, nz * n))
    a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(t.n, nz * n))
    a.sort_indices()
    return a


def category_layer(stack) -> xr.DataArray:
    """The zone of every cell of a stack of zone layers (prioritizr's ``category_layer()``): the number
    (1, 2, ...) of the layer with the largest value above 0, 0 where none is, NaN where every layer is
    NaN. ``stack``: a raster with a layer per zone (a solution's ``selection``) or a mapping zone -> raster.

    Examples
    --------
    >>> import landschaft as ls
    >>> cost = ls.load_raster(ls.prioritize.example_data("cost"))
    >>> features = ls.load_raster(ls.prioritize.example_data("features"))
    >>> p = (ls.prioritize.problem({"protect": cost, "use": cost * 0.4}, features, contribution={"use": 0.5})
    ...        .add_min_set_objective().add_relative_targets(0.3))
    >>> sol = p.solve()
    >>> zones = ls.prioritize.category_layer(sol.selection)
    >>> zones.attrs["zones"]
    {1: 'protect', 2: 'use'}
    >>> bool((zones == sol.zones).sum() == (sol.zones >= 0).sum())
    True
    """
    if isinstance(stack, Mapping):
        names = [str(k) for k in stack]
        layers = [v[0] if v.ndim == 3 else v for v in stack.values()]
        stack = xr.concat(layers, dim="zone").assign_coords(zone=names)
    if stack.ndim != 3:
        raise ValueError("a raster with a layer per zone, or a mapping zone -> raster")
    dim = stack.dims[0]
    v = np.asarray(stack.values, dtype=np.float64)
    empty = np.isnan(v).all(axis=0)
    w = np.where(np.isnan(v), -np.inf, v)
    code = np.where(w.max(axis=0) > 0, w.argmax(axis=0) + 1, 0).astype(np.float64)
    code[empty] = np.nan
    out = stack[0].copy(data=code).drop_vars(dim, errors="ignore")
    out.name = "zone"
    names = [str(z) for z in stack[dim].values] if dim in stack.coords else [str(k + 1) for k in range(v.shape[0])]
    out.attrs = {**out.attrs, "zones": {k + 1: z for k, z in enumerate(names)}}
    return out


def binary_stack(categories: xr.DataArray, zones=None) -> dict:
    """One 0 / 1 raster per zone from a raster of zone numbers (prioritizr's ``binary_stack()``): a mapping
    zone -> raster, ready for :meth:`Problem.add_locked_in_constraints` (for example an existing zoning
    plan). ``zones``: the zones' names in the order of their numbers (default: those of a solution's
    ``zones`` map, else ``"1"``, ``"2"``, ...); cells with 0 or NaN are in no zone.

    Examples
    --------
    >>> import landschaft as ls
    >>> cost = ls.load_raster(ls.prioritize.example_data("cost"))
    >>> features = ls.load_raster(ls.prioritize.example_data("features"))
    >>> protected = ls.load_raster(ls.prioritize.example_data("protected"))
    >>> p = (ls.prioritize.problem({"protect": cost, "use": cost * 0.4}, features, contribution={"use": 0.5})
    ...        .add_min_set_objective().add_relative_targets(0.3))
    >>> existing = protected.fillna(0)              # the protected areas: zone 1 (protect)
    >>> locked = ls.prioritize.binary_stack(existing, zones=p.zones)
    >>> sorted(locked)
    ['protect', 'use']
    >>> sol = p.add_locked_in_constraints(locked).solve()
    >>> bool((sol.zones.values[existing.values == 1] == 1).all())
    True
    """
    c = categories[0] if categories.ndim == 3 and categories.shape[0] == 1 else categories
    if c.ndim != 2:
        raise ValueError("the zone numbers must be a raster with one band")
    v = np.asarray(c.values, dtype=np.float64)
    ok = np.isfinite(v)
    if zones is None:
        known = c.attrs.get("zones")
        k = int(np.nanmax(np.where(ok, v, 0))) if ok.any() else 0
        zones = [known[i] for i in sorted(known)] if known else [str(i) for i in range(1, k + 1)]
    zones = [str(z) for z in zones]
    vals = np.unique(v[ok & (v != 0)])
    if vals.size and (vals.min() < 1 or vals.max() > len(zones) or (vals != np.round(vals)).any()):
        raise ValueError(f"zone numbers must be whole numbers from 1 to {len(zones)} (0: no zone)")
    out = {}
    for k, z in enumerate(zones):
        layer = c.copy(data=np.where(ok, (v == k + 1).astype(np.float64), np.nan))
        layer.name = z
        out[z] = layer
    return out


def zone_cut_model(m, n: int, nz: int) -> dict | None:
    """A compiled minimum set with zones and pair terms (boundary penalties, connectivity rewards) in the
    form of the zones cut engine (opt/zones_cut.hpp): unit-zone columns with their costs, the pairs as
    w [x_a != x_b] (each pair column q x_a x_b with q < 0 becomes w = -q / 2 and q / 2 on both columns),
    the targets by column. None when the model has another form (pair terms that are not submodular,
    other rows or columns, continuous units)."""
    ncol = n * nz
    if m.modelsense != "min" or m.ncol <= ncol:
        return None
    vt = np.asarray(m.vtype)
    if not (vt[:ncol] == "B").all():
        return None
    ids = np.asarray(m.row_ids)
    a = m.A.tocsr()
    a.sort_indices()
    nnz = np.diff(a.indptr)
    row = np.repeat(np.arange(a.shape[0]), nnz)
    col, val = a.indices, a.data
    target_rows = np.flatnonzero(ids == "spp_target")
    gub_rows = np.flatnonzero(ids == "pu_zone")
    pair_rows = np.setdiff1d(np.arange(a.shape[0]), np.concatenate([target_rows, gub_rows]))
    sense = np.asarray(m.sense)
    if (sense[target_rows] != ">=").any() or gub_rows.size != n:
        return None
    # the target rows on unit columns only, amounts >= 0
    tmask = np.isin(row, target_rows)
    if (col[tmask] >= ncol).any() or (val[tmask] < 0).any():
        return None
    # pair rows: pair - unit <= 0 (two entries); every pair column in exactly two of them
    pr = pair_rows
    if not ((sense[pr] == "<=") & (nnz[pr] == 2) & (m.rhs[pr] == 0)).all():
        return None
    first = a.indptr[pr]
    c0, c1 = col[first], col[first + 1]
    v0, v1 = val[first], val[first + 1]
    unit, pcol = np.where(c0 < ncol, c0, c1), np.where(c0 < ncol, c1, c0)
    uval, pval = np.where(c0 < ncol, v0, v1), np.where(c0 < ncol, v1, v0)
    if (unit >= ncol).any() or (pcol < ncol).any() or (uval != -1.0).any() or (pval != 1.0).any():
        return None
    order = np.lexsort((unit, pcol))
    unit, pcol = unit[order], pcol[order]
    cols, counts = np.unique(pcol, return_counts=True)
    if (counts != 2).any() or cols.size != m.ncol - ncol:
        return None
    pa, pb = unit[0::2], unit[1::2]
    q = m.obj[cols]
    if (q >= 0).any() or (pa == pb).any():
        return None
    cost = m.obj[:ncol].copy()
    np.add.at(cost, pa, q / 2.0)
    np.add.at(cost, pb, q / 2.0)
    t = a[target_rows][:, :ncol].tocsc()
    t.sort_indices()
    return {"n": n, "nz": nz, "cost": cost, "lb": m.lb[:ncol].copy(), "ub": m.ub[:ncol].copy(),
            "mandatory": bool((sense[gub_rows] == "=").all()), "t_start": t.indptr.astype(np.int64),
            "t_index": t.indices.astype(np.int64), "t_value": t.data.astype(np.float64),
            "target": m.rhs[target_rows].astype(np.float64), "pi": pa.astype(np.int32), "pj": pb.astype(np.int32),
            "pw": (-q / 2.0).astype(np.float64), "pair_cols": cols, "offset": 0.0}


def solve_zone_cut(m, zm: dict, gap: float, time_limit: float, verbose: bool, start, threads: int,
                   presolve: str) -> dict:
    """The zones cut engine, then HiGHS's MIP from its solution when the gap is not reached and the model
    is small enough; the result as solve_model gives it (over every column of the model)."""
    import math
    import time

    from .. import _core
    from ._milp_solve import CUT_HIGHS_COLUMNS, _highs

    ncol = zm["n"] * zm["nz"]
    t0 = time.perf_counter()
    r = _core.opt_zone_cut(zm["n"], zm["nz"], zm["cost"], zm["lb"], zm["ub"], zm["mandatory"], zm["t_start"],
                           zm["t_index"], zm["t_value"], zm["target"], zm["pi"], zm["pj"], zm["pw"], gap,
                           time_limit, verbose, None if start is None else np.asarray(start, dtype=np.float64)[:ncol])

    def full(xu):
        x = np.zeros(m.ncol)
        x[:ncol] = xu
        x[zm["pair_cols"]] = xu[zm["pi"]] * xu[zm["pj"]]
        return x

    out = {"status": r["status"], "has_solution": r["has_solution"], "x": None, "objective": math.nan,
           "bound": r["bound"], "gap": r["gap"], "nodes": 0, "seconds": r["seconds"]}
    if r["has_solution"]:
        out["x"] = full(np.asarray(r["x"]))
        out["objective"] = float(m.obj @ out["x"])
    if r["status"] in ("optimal", "infeasible") or m.ncol > CUT_HIGHS_COLUMNS:
        if out["status"] not in ("optimal", "infeasible", "time_limit"):
            out["status"] = "heuristic"
        return out
    left = time_limit - (time.perf_counter() - t0)
    if left <= 0:
        out["status"] = "time_limit"
        return out
    h = _highs(m, gap, left, threads, presolve, verbose, out["x"])
    res = dict(h)
    if not h["has_solution"] or (out["has_solution"] and h["objective"] > out["objective"]):
        res.update({k: out[k] for k in ("has_solution", "x", "objective")})
    bound = max(out["bound"], h["bound"]) if math.isfinite(h.get("bound", math.nan)) else out["bound"]
    res["bound"] = bound
    if res["has_solution"]:
        g = res["objective"] - bound
        res["gap"] = 0.0 if g <= 0 else g / max(abs(res["objective"]), 1e-10)
        res["status"] = "optimal" if res["gap"] <= gap else h["status"]
    res["seconds"] = time.perf_counter() - t0
    return res


def one_zone(obj, what: str) -> None:
    """Refuse problems and solutions with zones in the tools that do not support them yet."""
    units = getattr(obj, "units", None)
    if (units is not None and getattr(units, "zoned", False)) or getattr(obj, "zone_names", ()):
        raise ValueError(f"{what} does not support zones yet")


def zone_region_totals(u: Units, codes: np.ndarray, nreg: int, features: np.ndarray):
    """Every feature's total in every region (over all units, with or without a cost), in every zone
    (features x regions x zones) and in the base features (features x regions; None with every zone's own
    features), added in unit order as the problem's totals are."""
    nf, nz = len(u.feature_names), u.nz
    out = np.zeros((nf, nreg, nz))
    need = np.unique(features)
    kind = u.zone_source[0]
    if kind == "table":
        base_full = u.zone_source[1]
        base = None if base_full is None else np.zeros((nf, nreg))

        def add(a, f, target):
            lo, hi = a.indptr[f], a.indptr[f + 1]
            c = codes[a.indices[lo:hi]]
            v = a.data[lo:hi]
            keep = (c >= 0) & np.isfinite(v)
            if u.r_sums:  # R's colSums of the column cut by the region
                for r in range(nreg):
                    target[r] = _r_sum(v[keep & (c == r)])
            else:
                np.add.at(target, c[keep], v[keep])

        for f in need:
            for z in range(nz):
                col = np.zeros(nreg)
                add(u.all_rij[z], f, col)
                out[f, :, z] = col
            if base is not None:
                add(base_full, f, base[f])
        return out, base
    _, layers, contrib, nf = u.zone_source
    base = None if contrib is None else np.zeros((nf, nreg))
    ny, nx = u.template.shape
    step = max(1, BLOCK_BYTES // (8 * nx * 2))

    def block(da, y0, y1):
        v = da.data[y0:y1]
        if hasattr(v, "compute"):
            v = v.compute()
        return np.asarray(v, dtype=np.float64).ravel()

    for f in need:
        for y0 in range(0, ny, step):
            y1 = min(ny, y0 + step)
            c = codes[y0 * nx:y1 * nx]
            if contrib is None:  # every zone's own features: zone z's layers z * nf ...
                for z in range(nz):
                    v = block(layers[z * nf + f], y0, y1)
                    keep = (c >= 0) & np.isfinite(v)
                    np.add.at(out[f, :, z], c[keep], v[keep])
                continue
            b = block(layers[f], y0, y1)
            keep = (c >= 0) & np.isfinite(b)
            np.add.at(base[f], c[keep], b[keep])
            for z, (k, a) in enumerate(contrib):
                v = b * a[f] if k == "scalar" else b * block(a[0] if a.ndim == 3 else a, y0, y1)
                keep = (c >= 0) & np.isfinite(v)
                np.add.at(out[f, :, z], c[keep], v[keep])
    return out, base


def resolve_zone_subregional(p):
    """Sub-regional targets of a problem with zones (an extension): a ``zone`` column (a zone or a list;
    default every zone) and relative targets as shares of the feature's total in the region (in one zone,
    its total in that zone; summed over zones, the base features' or the sum of the zones' totals)."""
    from ._milp_compile import Targets, region_codes

    u = p.units
    regions, t = p.subregional
    codes, ids = region_codes(u, regions)
    numeric = all(isinstance(r, (int, float, np.integer, np.floating)) for r in ids)
    index = {(float(r) if numeric else str(r)): k for k, r in enumerate(ids)}
    findex = {n: k for k, n in enumerate(u.feature_names)}
    missing = sorted(set(map(str, t["feature"])) - set(findex))
    if missing:
        raise ValueError(f"unknown features in the targets: {missing}")
    want = [float(r) if numeric else str(r) for r in t["region"]]
    unknown = sorted({str(r) for r in want if r not in index})
    if unknown:
        raise ValueError(f"unknown regions in the targets: {unknown}")
    region = np.array([index[r] for r in want], dtype=np.int64)
    feature = np.array([findex[str(f)] for f in t["feature"]], dtype=np.int64)
    target = t["target"].to_numpy(dtype=np.float64)
    kinds = t["type"].astype(str).to_numpy() if "type" in t.columns else np.full(len(t), "relative")
    if not np.isin(kinds, ("relative", "absolute")).all():
        raise ValueError("target types must be 'relative' or 'absolute'")
    sense = tuple(t["sense"].astype(str)) if "sense" in t.columns else (">=",) * len(t)
    if not set(sense) <= {">=", "<=", "="}:
        raise ValueError("target senses must be '>=', '<=' or '='")
    every = tuple(range(u.nz))
    zones = [zone_key(u, z) for z in t["zone"]] if "zone" in t.columns else [every] * len(t)
    ztot, btot = zone_region_totals(u, codes, len(ids), feature)
    total = np.empty(len(t))
    for i, (f, r, zs) in enumerate(zip(feature, region, zones)):
        if len(zs) == 1:
            total[i] = ztot[f, r, zs[0]]
        elif btot is not None:
            total[i] = btot[f, r]
        else:
            total[i] = _r_sum(np.array([ztot[f, r, z] for z in zs]))
    value = np.where(kinds == "relative", total * target, target)
    return Targets(feature, sense, value, total, region, tuple(ids[k] for k in region), zone=tuple(zones),
                   zone_label=tuple(_zone_label(u, zs) for zs in zones)), codes[u.index]


def sum_zone_targets(u: Units):
    """Every feature over all its zones (the extinction risk with zones): its total as targets summed over
    the zones take it."""
    from ._milp_compile import Targets

    nf, nz = len(u.feature_names), u.nz
    every = tuple(range(nz))
    if u.base_totals is not None:
        total = np.asarray(u.base_totals, dtype=np.float64).copy()
    else:
        total = np.array([_r_sum(np.asarray(u.totals[f], dtype=np.float64)) for f in range(nf)])
    return Targets(np.arange(nf), ("?",) * nf, np.zeros(nf), total, zone=(every,) * nf,
                   zone_label=(_zone_label(u, every),) * nf)


def positive_abundances(u: Units, t) -> np.ndarray:
    """Every target's positive amount in the units that have a cost in its zones (prioritizr's
    feature_positive_abundances_in_planning_units: Matrix's rowSums of each zone, in unit order; then
    the zones in turn)."""
    n = u.n
    out = np.zeros(t.n)
    for i, (f, zs) in enumerate(zip(t.feature, t.zone)):
        acc = 0.0
        for k, z in enumerate(zs):
            m = u.zone_rij[z]
            lo, hi = m.indptr[f], m.indptr[f + 1]
            keep = np.isfinite(u.cost[z * n + m.indices[lo:hi]])
            v = np.maximum(m.data[lo:hi][keep], 0.0)
            s = float(np.cumsum(v)[-1]) if v.size else 0.0
            acc = s if k == 0 else acc + s
        out[i] = acc
    return out


# what the zones support so far (prioritizr's zones, phase by phase)
ZONE_OBJECTIVES = ("min_set", "min_shortfall", "min_largest_shortfall", "max_utility", "max_features",
                   "max_cover", "max_phylo_div", "max_phylo_end", "min_extinction_risk", "min_penalties")


def check_zone_problem(p) -> None:
    kind = p.objective[0]
    if kind not in ZONE_OBJECTIVES:
        raise ValueError(f"the {kind} objective is not supported with zones yet (supported: {list(ZONE_OBJECTIVES)})")
    for spec in p.spatial:
        if spec[1] in ("feature_contiguity", "corridors"):
            raise ValueError(f"{spec[1].replace('_', ' ')} constraints are not supported with zones")
    budget = p.objective[1] if len(p.objective) > 1 and kind != "min_set" else None
    if budget is not None and np.size(budget) not in (1, p.units.nz):
        raise ValueError(f"one budget, or one per zone ({p.units.nz})")
