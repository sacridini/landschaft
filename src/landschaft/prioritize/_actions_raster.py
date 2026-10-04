# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-action problems from rasters (an extension of prioriactions, after prioripy's raster input):
the planning units are the cells with a monitoring cost, one raster per feature (its amount in every
cell) and per threat (its intensity: > 0 where it occurs), the actions' costs as numbers or rasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ._actions import ActionsData, actions_data
from ._milp import _raster_units

__all__ = ["actions_raster_data"]


def _per_layer(v, names: list[str], what: str) -> list:
    """A value per layer from a scalar, a sequence (in the layers' order) or a mapping by name."""
    if isinstance(v, Mapping):
        missing = [n for n in names if n not in v]
        if missing:
            raise ValueError(f"{what}: no value for {missing}")
        return [v[n] for n in names]
    if isinstance(v, xr.DataArray) and v.ndim == 3:
        if v.shape[0] != len(names):
            raise ValueError(f"{what}: {v.shape[0]} bands for {len(names)} layers")
        return [v[k] for k in range(v.shape[0])]
    if isinstance(v, Sequence) and not isinstance(v, str):
        if len(v) != len(names):
            raise ValueError(f"{what}: {len(v)} values for {len(names)} layers")
        return list(v)
    if isinstance(v, np.ndarray) and v.ndim == 1:
        if v.size != len(names):
            raise ValueError(f"{what}: {v.size} values for {len(names)} layers")
        return list(v)
    return [v] * len(names)


def _cells(raster, cost: xr.DataArray, cells: np.ndarray, what: str) -> np.ndarray:
    """A raster's values in the planning units (or a number repeated)."""
    if np.isscalar(raster):
        return np.full(cells.size, float(raster))
    r = raster.squeeze("band", drop=True) if isinstance(raster, xr.DataArray) and raster.ndim == 3 else raster
    if not isinstance(r, xr.DataArray) or r.shape != cost.shape:
        raise ValueError(f"{what} must be a number or a raster on the monitoring cost's grid")
    return np.asarray(r.values, dtype=np.float64).ravel()[cells]


def actions_raster_data(monitoring_cost: xr.DataArray, features, threats, action_cost=1.0, *,
                        target_recovery=0.0, target_conservation=0.0, relative: bool = True,
                        sensitivity: pd.DataFrame | None = None, blm_actions=0.0, locked_in=None,
                        locked_out=None, boundary: bool = False, n_threads: int | None = None) -> ActionsData:
    """The inputs of a multi-action problem from rasters (see :func:`actions_data` for the tables).

    Parameters
    ----------
    monitoring_cost
        The cost of monitoring every cell: the planning units are the cells with a cost (not NaN), in
        the order of the cells (ids 1, 2, ...).
    features
        The features' amounts: a stack (band, y, x), a Dataset or a list of rasters on the same grid
        (the layers' names are the features' names; NaN = 0).
    threats
        The threats' intensity, the same way: a threat occurs in a cell with a value > 0, and its
        action there removes it.
    action_cost
        The cost of acting against every threat: a number, a raster, or one of them per threat (a
        sequence in the threats' order, a mapping by name or a stack); it needs a value wherever the
        threat occurs.
    target_recovery, target_conservation
        The features' targets: a number, a sequence in the features' order or a mapping by name. With
        ``relative`` (the default) shares of the amount the feature has where a threat harms it
        (recovery) or where none does (conservation); otherwise absolute amounts of benefit.
    sensitivity
        Which features each threat harms (``feature``, ``threat`` by name or 1-based position, and
        optionally ``delta1`` to ``delta4``); default every feature to every threat.
    blm_actions
        The weight of the boundary of every threat's actions (a number, sequence or mapping).
    locked_in, locked_out
        Rasters of the cells that must (or must not) be monitored (non-zero, not NaN).
    boundary
        Give the problem the cells' shared edges as boundary data (for ``blm`` and ``blm_actions``).

    Returns
    -------
    ActionsData
        With ``grid`` set: the solution's :meth:`~ActionsSolution.maps`.
    """
    if monitoring_cost.ndim == 3 and monitoring_cost.sizes.get("band", 1) == 1:
        monitoring_cost = monitoring_cost.squeeze("band", drop=True)
    fu = _raster_units(monitoring_cost, features, n_threads)
    tu = _raster_units(monitoring_cost, threats, n_threads)
    cells = fu.index
    n = cells.size
    if n == 0:
        raise ValueError("no planning units: the monitoring cost has no cell with a value")
    fnames, tnames = list(fu.feature_names), list(tu.feature_names)
    if len(set(fnames)) != len(fnames) or len(set(tnames)) != len(tnames):
        raise ValueError("the features and the threats need distinct layer names")
    ids = np.arange(1.0, n + 1)

    status = np.zeros(n)
    for mask, code, what in ((locked_in, 2.0, "locked_in"), (locked_out, 3.0, "locked_out")):
        if mask is not None:
            v = _cells(mask, monitoring_cost, cells, what)
            status[np.isfinite(v) & (v != 0)] = code
    pu = pd.DataFrame({"id": ids, "monitoring_cost": fu.cost, "status": status})

    f = fu.rij.tocoo()
    dist_features = pd.DataFrame({"pu": f.col + 1.0, "feature": f.row + 1.0, "amount": f.data})

    t = tu.rij.tocoo()
    if (t.data < 0).any():
        raise ValueError("threat intensities must be >= 0")
    costs = _per_layer(action_cost, tnames, "action_cost")
    acost = np.empty(t.data.size)
    for k, c in enumerate(costs):
        rows = t.row == k
        v = _cells(c, monitoring_cost, cells, f"action_cost of {tnames[k]}")[t.col[rows]]
        if not np.isfinite(v).all():
            raise ValueError(f"action_cost of {tnames[k]} has no value in cells where the threat occurs")
        acost[rows] = v
    dist_threats = pd.DataFrame({"pu": t.col + 1.0, "threat": t.row + 1.0, "amount": t.data, "action_cost": acost,
                                 "status": np.zeros(t.data.size)})
    dist_threats = dist_threats.sort_values(["pu", "threat"], kind="stable").reset_index(drop=True)
    threat_table = pd.DataFrame({"id": np.arange(1.0, len(tnames) + 1), "name": tnames,
                                 "blm_actions": np.asarray(_per_layer(blm_actions, tnames, "blm_actions"), float)})

    sens = None
    if sensitivity is not None:
        sens = pd.DataFrame(sensitivity).copy()
        for col, names in (("feature", fnames), ("threat", tnames)):
            if col not in sens:
                raise ValueError(f"sensitivity needs the column {col!r}")
            pos = {name: k + 1.0 for k, name in enumerate(names)}
            sens[col] = [pos[v] if v in pos else float(v) for v in sens[col]]

    bnd = None
    if boundary:
        from ._milp_matrices import boundary_matrix

        m = boundary_matrix(monitoring_cost).tocoo()
        # the matrix is over every cell with data: keep the planning units, each pair once
        unit = np.full(monitoring_cost.size, -1, dtype=np.int64)
        unit[cells] = np.arange(n)
        a, b = unit[m.row], unit[m.col]
        keep = (a >= 0) & (b >= 0) & (a <= b)
        bnd = pd.DataFrame({"id1": a[keep] + 1.0, "id2": b[keep] + 1.0, "boundary": m.data[keep]})

    feats = pd.DataFrame({"id": np.arange(1.0, len(fnames) + 1), "name": fnames, "target_recovery": 0.0,
                          "target_conservation": 0.0})
    d = actions_data(pu, feats, dist_features, threat_table, dist_threats, sens, bnd)
    tr = np.asarray(_per_layer(target_recovery, fnames, "target_recovery"), dtype=np.float64)
    tc = np.asarray(_per_layer(target_conservation, fnames, "target_conservation"), dtype=np.float64)
    if relative:
        if ((tr < 0) | (tr > 1) | (tc < 0) | (tc > 1)).any():
            raise ValueError("relative targets must be between 0 and 1")
        from ._actions import _pairs

        p = _pairs(d)
        held = np.zeros((2, len(fnames)))
        np.add.at(held, (p.threatened.astype(np.int64), p.feature), p.amount)
        # features dropped by actions_data (no amount) are not in d.features
        pos = d.features["id"].to_numpy(dtype=np.int64) - 1
        tr, tc = tr[pos] * held[1], tc[pos] * held[0]
    else:
        pos = d.features["id"].to_numpy(dtype=np.int64) - 1
        tr, tc = tr[pos], tc[pos]
    from ._milp_spatial import r_round

    features_table = d.features.copy()
    features_table["target_recovery"] = r_round(tr, 3)
    features_table["target_conservation"] = r_round(tc, 3)
    from dataclasses import replace

    return replace(d, features=features_table, grid=(monitoring_cost, cells))


def solution_maps(data: ActionsData, units: np.ndarray, actions: np.ndarray) -> xr.DataArray:
    """The monitored cells and every threat's actions as a stack (band: ``monitoring``, then the threats'
    names); NaN outside the planning units."""
    if data.grid is None:
        raise ValueError("maps need a problem built from rasters (actions_raster_data)")
    template, cells = data.grid
    n = cells.size
    names = ["monitoring", *[str(v) for v in data.threats["name"]]]
    out = np.full((len(names), template.size), np.nan)
    out[:, cells] = 0.0
    out[0, cells] = units
    unit = data.dist_threats["internal_pu"].to_numpy(dtype=np.int64)
    threat = data.dist_threats["internal_threat"].to_numpy(dtype=np.int64)
    flat = out[1:]
    flat[threat, cells[unit]] = actions
    if n != len(data.pu):
        raise ValueError("the planning units do not match the grid")
    da = xr.DataArray(out.reshape((len(names), *template.shape)), dims=("band", *template.dims),
                      coords={"band": names, **{k: v for k, v in template.coords.items() if k in template.dims}})
    try:
        crs = template.rio.crs
    except Exception:  # noqa: BLE001 - no georeferencing
        crs = None
    if crs is not None:
        da = da.rio.write_crs(crs)
    da.name = "actions"
    return da
