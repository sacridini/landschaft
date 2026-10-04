# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_optimize.py and the MILP prioritization tests.

Every input is built in Python (numpy's generator, fixed seeds): rasters as
DataArrays, planning-unit polygons as a GeoDataFrame, tables as DataFrames.
A case is R code that sees the inputs it names as variables and gives a
named list; ``py`` (optional) builds the same problem with
``ls.prioritize.problem``. ``tools/optimize_reference.R`` adds helpers:

- ``reference(p)``: a prioritizr problem's compiled model (``model``) and
  its optimum on Gurobi with gap 0 on one thread (``objective``,
  ``status``, ``solution`` in planning-unit order);
- ``actions_reference(m)``: the same for a prioriactions problem
  (``solution`` is its whole variable vector);
- ``pa_data(...)``: ``prioriactions::inputData`` of the ``pa_*`` tables.

A compiled model is stored as ``model/<field>``: the constraint matrix as
0-based triplets (``A_i``, ``A_j``, ``A_x``, in the reference's order),
``obj``, ``lb``, ``ub``, ``rhs``, ``sense``, ``vtype``, ``modelsense`` and,
for prioritizr, ``row_ids`` and ``col_ids``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import pairwise

import numpy as np

GROUPS: list[str] = ["core", "objectives", "spatial", "matrices", "portfolio", "actions", "exact"]

NY, NX = 12, 15
RES = 100.0


@dataclass
class Case:
    group: str
    inputs: tuple[str, ...]
    r: str
    notes: str = ""
    py: Callable | None = None


# ----------------------------------------------------------------------------- inputs

def grid(values: np.ndarray, res: float = RES, crs: str = "EPSG:32723", names=None):
    """A raster DataArray of the values (2-D, or bands first), top-left at (0, rows * res)."""
    import rioxarray  # noqa: F401  (the .rio accessor)
    import xarray as xr

    v = np.asarray(values, dtype=np.float64)
    ny, nx = v.shape[-2:]
    coords = {"y": (np.arange(ny)[::-1] + 0.5) * res, "x": (np.arange(nx) + 0.5) * res}
    if v.ndim == 3:
        coords["band"] = list(names) if names is not None else [f"f{k + 1}" for k in range(v.shape[0])]
        da = xr.DataArray(v, dims=("band", "y", "x"), coords=coords)
    else:
        da = xr.DataArray(v, dims=("y", "x"), coords=coords)
    return da.rio.write_crs(crs)


def smooth_field(seed: int, shape=(NY, NX), width: int = 2) -> np.ndarray:
    """Uniform noise smoothed by a (2 width + 1)-cell moving average: values in (0, 1), autocorrelated."""
    rng = np.random.default_rng(seed)
    v = rng.random((shape[0] + 2 * width, shape[1] + 2 * width))
    k = np.ones(2 * width + 1) / (2 * width + 1)
    v = np.apply_along_axis(np.convolve, 1, v, k, "valid")
    v = np.apply_along_axis(np.convolve, 0, v, k, "valid")
    return v


@cache
def _cost() -> np.ndarray:
    """Cost of the 12 x 15 cells: 10 to 100, no data in the top-left corner and two holes."""
    c = 10.0 + 90.0 * smooth_field(11)
    c[:3, :4] = np.nan
    c[6, 7] = np.nan
    c[9, 2] = np.nan
    return c


@cache
def _features() -> np.ndarray:
    """Five features: species-like (zero below a quantile), one rare, one everywhere, a few NaN cells."""
    out = []
    for k, share in enumerate((0.5, 0.3, 0.15, 1.0, 0.4)):
        f = smooth_field(100 + k)
        cut = np.quantile(f, 1 - share) if share < 1 else 0.0
        out.append(np.where(f > cut, f - cut, 0.0) * (k + 1))
    f = np.stack(out)
    f[1, 10, 12] = np.nan  # NaN in a planning unit: no amount
    f[4, 0, 0] = np.nan
    return f


def _units() -> np.ndarray:
    """Cell indices (row-major) of the planning units: the cells with a cost."""
    return np.flatnonzero(np.isfinite(_cost()).ravel())


def pu_cost():
    return grid(_cost())


def features():
    return grid(_features())


def locked_in():
    """Cells to select (1) or not constrained (0); NaN outside the planning units."""
    m = np.zeros((NY, NX))
    m[4:6, 10:12] = 1.0
    m[11, 14] = 1.0
    return grid(np.where(np.isfinite(_cost()), m, np.nan))


def locked_out():
    m = np.zeros((NY, NX))
    m[8:10, 5:8] = 1.0
    m[0, 14] = 1.0
    return grid(np.where(np.isfinite(_cost()), m, np.nan))


def pu_table():
    """The raster's planning units as a table: id, cost, f1..f5 amounts (NaN = 0), locks."""
    import pandas as pd

    cells = _units()
    f = np.nan_to_num(_features().reshape(5, -1)[:, cells])
    li = locked_in().values.ravel()[cells] == 1
    lo = locked_out().values.ravel()[cells] == 1
    data = {"id": np.arange(1, cells.size + 1, dtype=np.float64), "cost": _cost().ravel()[cells]}
    data.update({f"f{k + 1}": f[k] for k in range(5)})
    data["locked_in"] = li
    data["locked_out"] = lo
    return pd.DataFrame(data)


def spec_table():
    import pandas as pd

    return pd.DataFrame({"id": np.arange(1.0, 6.0), "name": [f"f{k}" for k in range(1, 6)]})


def rij_table():
    """Marxan-style amounts: one row per planning unit and feature with a non-zero amount."""
    import pandas as pd

    t = pu_table()
    rows = [(pu, k + 1.0, a) for k in range(5) for pu, a in zip(t["id"], t[f"f{k + 1}"]) if a > 0]
    pu, sp, amount = map(np.asarray, zip(*rows))
    return pd.DataFrame({"pu": pu, "species": sp, "amount": amount})


def _polygons(xbreaks, ybreaks):
    import geopandas as gpd
    from shapely.geometry import box

    rng = np.random.default_rng(7)
    boxes = [box(x0, y0, x1, y1) for y0, y1 in pairwise(ybreaks) for x0, x1 in pairwise(xbreaks)]
    cost = np.round(rng.uniform(50, 500, len(boxes)), 2)
    return gpd.GeoDataFrame({"cost": cost}, geometry=boxes, crs="EPSG:32723")


def pu_polygons():
    """4 x 5 rectangles of whole cells (the features' cells fall entirely inside one of them)."""
    return _polygons(np.array([0, 3, 7, 10, 12, 15]) * RES, np.array([0, 2, 5, 9, 12]) * RES)


def pu_polygons_partial():
    """4 x 4 rectangles cutting through cells: amounts weighted by the covered fraction."""
    return _polygons(np.array([0, 3.5, 7.25, 11.6, 15]) * RES, np.array([0, 2.3, 6.5, 9.75, 12]) * RES)


def shapes():
    """Polygons that exercise the exact cell coverage: a triangle, a hole, a multipolygon, a concave star,
    one smaller than a cell, one partly outside the grid, one on cell edges, a rotated square."""
    import geopandas as gpd
    from shapely.affinity import rotate
    from shapely.geometry import MultiPolygon, Polygon, box

    r = RES
    star = Polygon([(7.5 * r + 3.2 * r * np.cos(a) * (1 if k % 2 == 0 else 0.45),
                     6.1 * r + 3.2 * r * np.sin(a) * (1 if k % 2 == 0 else 0.45))
                    for k, a in enumerate(np.linspace(0, 2 * np.pi, 11)[:-1])])
    geoms = [
        Polygon([(0.3 * r, 0.2 * r), (6.7 * r, 1.1 * r), (2.9 * r, 7.45 * r)]),
        Polygon([(1 * r, 8 * r), (6.3 * r, 8 * r), (6.3 * r, 11.6 * r), (1 * r, 11.6 * r)],
                [[(2.2 * r, 9.1 * r), (4.75 * r, 9.1 * r), (3.3 * r, 10.9 * r)]]),
        MultiPolygon([box(9.2 * r, 0.4 * r, 10.9 * r, 2.3 * r), box(12.1 * r, 1.05 * r, 14.6 * r, 3.95 * r)]),
        star,
        box(11.3 * r, 9.2 * r, 11.7 * r, 9.9 * r),
        box(13.4 * r, 7.3 * r, 17.8 * r, 13.9 * r),
        box(8 * r, 9 * r, 10 * r, 12 * r),
        rotate(box(2 * r, 3 * r, 5 * r, 6 * r), 27, origin="centroid"),
    ]
    return gpd.GeoDataFrame({"id": np.arange(1.0, len(geoms) + 1)}, geometry=geoms, crs="EPSG:32723")


# --- prioriactions: a 10 x 10 grid of units, 4 features, 2 threats

PA_SIDE = 10


@cache
def _pa():
    rng = np.random.default_rng(21)
    n = PA_SIDE * PA_SIDE
    feat = rng.random((4, n)) < np.array([0.45, 0.25, 0.6, 0.35])[:, None]
    threat = rng.random((2, n)) < np.array([0.5, 0.4])[:, None]
    return {
        "monitoring": rng.integers(1, 4, n).astype(np.float64),
        "feat": feat,
        "threat": threat,
        "action_cost": rng.integers(1, 5, (2, n)).astype(np.float64),
        "status_pu": np.where(rng.random(n) < 0.04, 2.0, 0.0),
    }


def pa_pu():
    import pandas as pd

    d = _pa()
    n = PA_SIDE * PA_SIDE
    status = d["status_pu"].copy()
    status[[17, 63]] = 3.0  # locked out
    return pd.DataFrame({"id": np.arange(1.0, n + 1), "monitoring_cost": d["monitoring"], "status": status})


def pa_features():
    import pandas as pd

    d = _pa()
    # recoverable where the feature lives with at least one threat on it
    total = (d["feat"] & d["threat"].any(axis=0)).sum(axis=1)
    return pd.DataFrame({"id": np.arange(1.0, 5.0), "target_recovery": np.floor(0.4 * total),
                         "name": [f"feature{k}" for k in range(1, 5)]})


def pa_dist_features():
    import pandas as pd

    feat = _pa()["feat"]
    pu, k = np.nonzero(feat.T)
    return pd.DataFrame({"pu": pu + 1.0, "feature": k + 1.0, "amount": np.ones(pu.size)})


def pa_threats():
    import pandas as pd

    return pd.DataFrame({"id": [1.0, 2.0], "name": ["threat1", "threat2"], "blm_actions": [0.0, 0.0]})


def pa_dist_threats():
    import pandas as pd

    d = _pa()
    pu, t = np.nonzero(d["threat"].T)
    status = np.zeros(pu.size)
    status[[3, 40]] = 2.0  # action locked in
    status[[11, 52]] = 3.0  # action locked out
    return pd.DataFrame({"pu": pu + 1.0, "threat": t + 1.0, "amount": np.ones(pu.size),
                         "action_cost": d["action_cost"][t, pu], "status": status})


def pa_sensitivity():
    import pandas as pd

    f, t = np.meshgrid(np.arange(1.0, 5.0), np.arange(1.0, 3.0))
    return pd.DataFrame({"feature": f.ravel(), "threat": t.ravel()})


def pa_boundary():
    """Shared edges of the grid (length 1), each pair once."""
    import pandas as pd

    ids = np.arange(1, PA_SIDE * PA_SIDE + 1).reshape(PA_SIDE, PA_SIDE).astype(np.float64)
    a = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    b = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    return pd.DataFrame({"id1": a, "id2": b, "boundary": np.ones(a.size)})


# --- spatial: unit x unit data as triplets (i, j, x; 1-based) over the rows of pu_table

def _triplets(i, j, x):
    import pandas as pd

    return pd.DataFrame({"i": np.asarray(i, dtype=np.float64) + 1, "j": np.asarray(j, dtype=np.float64) + 1,
                         "x": np.asarray(x, dtype=np.float64)})


def bnd_table():
    """A boundary matrix of pu_table's units: their raster neighbours with lengths of 7 decimals (round(, 6)
    applies), perimeters on the diagonal."""
    cells = _units()
    rng = np.random.default_rng(31)
    pos = {c: k for k, c in enumerate(cells)}
    i, j, x = [], [], []
    for k, c in enumerate(cells):
        r, q = divmod(int(c), NX)
        for rr, qq in ((r + 1, q), (r, q + 1)):
            if rr < NY and qq < NX and rr * NX + qq in pos:
                v = round(float(rng.uniform(0.5, 2.0)), 7)
                m = pos[rr * NX + qq]
                i += [k, m]
                j += [m, k]
                x += [v, v]
        i.append(k)
        j.append(k)
        x.append(round(float(rng.uniform(4.0, 9.0)), 7))
    return _triplets(i, j, x)


def conn_table():
    """Symmetric connectivity among pu_table's units, a few on the diagonal."""
    n = _units().size
    rng = np.random.default_rng(32)
    i, j, x = [], [], []
    for k in range(n):
        for step in (1, 7):
            m = k + step
            if m < n and rng.random() < 0.6:
                v = float(rng.uniform(0.1, 3.0))
                i += [k, m]
                j += [m, k]
                x += [v, v]
        if rng.random() < 0.1:
            i.append(k)
            j.append(k)
            x.append(float(rng.uniform(0.1, 1.0)))
    return _triplets(i, j, x)


def asym_table():
    """Asymmetric connectivity among pu_table's units (rows: sources), a few on the diagonal."""
    n = _units().size
    rng = np.random.default_rng(33)
    i, j, x = [], [], []
    for k in range(n):
        for step in (1, -3, 9):
            m = k + step
            if 0 <= m < n and rng.random() < 0.5:
                i.append(k)
                j.append(m)
                x.append(float(rng.uniform(0.1, 3.0)))
        if rng.random() < 0.1:
            i.append(k)
            j.append(k)
            x.append(float(rng.uniform(0.1, 1.0)))
    return _triplets(i, j, x)


def penalty_raster():
    """Values for linear penalties and constraints on the grid (NaN outside the planning units' area)."""
    v = 1.0 + 4.0 * smooth_field(41)
    v[0, 0] = np.nan
    return grid(v)


def adj_table():
    """8-direction adjacency of pu_table's units (their raster cells)."""
    cells = _units()
    pos = {c: k for k, c in enumerate(cells)}
    i, j = [], []
    for k, c in enumerate(cells):
        r, q = divmod(int(c), NX)
        for dr in (-1, 0, 1):
            for dq in (-1, 0, 1):
                rr, qq = r + dr, q + dq
                if (dr or dq) and 0 <= rr < NY and 0 <= qq < NX and rr * NX + qq in pos:
                    i.append(k)
                    j.append(pos[rr * NX + qq])
    return _triplets(i, j, np.ones(len(i)))


INPUTS = {f.__name__: f for f in (
    pu_cost, features, shapes, locked_in, locked_out, pu_table, spec_table, rij_table, pu_polygons,
    pu_polygons_partial, pa_pu, pa_features, pa_dist_features, pa_threats, pa_dist_threats, pa_sensitivity,
    pa_boundary, bnd_table, conn_table, asym_table, penalty_raster, adj_table)}


def build_input(name: str):
    return INPUTS[name]()


# ----------------------------------------------------------------------------- cases

RASTER = ("pu_cost", "features")
PA = ("pa_pu", "pa_features", "pa_dist_features", "pa_threats", "pa_dist_threats", "pa_sensitivity", "pa_boundary")


# phylogenies of the five features: tips in their order, and in another (a 5-cycle, not a swap)
TREE_ORDERED = "(((f1:1.5,f2:0.5):1,f3:2):0.75,(f4:1.25,f5:3):0.5);"
TREE_SHUFFLED = "(((f3:1.5,f1:0.5):1,f5:2):0.75,(f2:1.25,f4:3):0.5);"

CASES: dict[str, Case] = {
    # --- core: planning units, min set, targets, locks, decisions
    "minset_raster_relative": Case("core", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_binary_decisions())"""),
    "minset_raster_relative_each": Case("core", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |>
                  add_relative_targets(c(0.1, 0.25, 0.5, 0.2, 0.35)) |> add_binary_decisions())"""),
    "minset_raster_absolute": Case("core", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |>
                  add_absolute_targets(c(1.2, 0.9, 1, 120, 4)) |> add_binary_decisions())"""),
    "minset_raster_locked": Case("core", (*RASTER, "locked_in", "locked_out"), """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_locked_in_constraints(locked_in) |> add_locked_out_constraints(locked_out) |>
                  add_binary_decisions())"""),
    "minset_raster_proportion": Case("core", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_proportion_decisions())""", notes="an LP"),
    "minset_raster_semicontinuous": Case("core", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_semicontinuous_decisions(0.6))"""),
    "minset_table": Case("core", ("pu_table",), """
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_locked_in_constraints("locked_in") |> add_locked_out_constraints("locked_out") |>
                  add_binary_decisions())"""),
    "minset_rij": Case("core", ("pu_table", "spec_table", "rij_table"), """
        reference(problem(pu_table[, c("id", "cost")], spec_table, rij = rij_table, cost_column = "cost") |>
                  add_min_set_objective() |> add_absolute_targets(c(1.2, 0.9, 1, 120, 4)) |>
                  add_binary_decisions())""", notes="Marxan-style tables"),
    "minset_rij_relative": Case("core", ("pu_table", "spec_table", "rij_table"), """
        reference(problem(pu_table[, c("id", "cost")], spec_table, rij = rij_table, cost_column = "cost") |>
                  add_min_set_objective() |> add_relative_targets(0.3) |> add_binary_decisions())"""),
    "minset_table_na_cost": Case("core", ("pu_table",), """
        t <- pu_table; t$cost[c(3, 50, 51, 120)] <- NA
        reference(problem(t, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_min_set_objective() |> add_relative_targets(0.3) |> add_binary_decisions())""",
                                 notes="units without cost are not planning units"),
    "minset_raster_locked_proportion": Case("core", (*RASTER, "locked_in", "locked_out"), """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_locked_in_constraints(locked_in) |> add_locked_out_constraints(locked_out) |>
                  add_proportion_decisions())"""),
    "minset_polygons": Case("core", ("pu_polygons", "features"), """
        reference(problem(pu_polygons, features, cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_binary_decisions())""",
                            notes="amounts: sums of whole cells (exactextractr)"),
    "minset_polygons_partial": Case("core", ("pu_polygons_partial", "features"), """
        reference(problem(pu_polygons_partial, features, cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_binary_decisions())""",
                                    notes="amounts weighted by the covered fraction (exactextractr)"),
    # --- objectives: prioritizr's objectives with a budget, targets (manual, loglinear), feature weights
    "min_shortfall": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_shortfall_objective(2500) |>
                  add_relative_targets(0.4) |> add_binary_decisions())"""),
    "min_shortfall_weights_tiny": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_shortfall_objective(2500) |>
                  add_absolute_targets(c(1.2, 1e-5, 1, 120, 4)) |> add_feature_weights(c(1, 2, 0.5, 3, 1)) |>
                  add_binary_decisions())""", notes="a target of 1e-5 (not above it): 0 in the objective, 1e5 x weight with weights"),
    "min_shortfall_tiny": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_shortfall_objective(2500) |>
                  add_absolute_targets(c(1.2, 1e-5, 1, 120, 4)) |> add_binary_decisions())"""),
    "min_largest_shortfall": Case("objectives", (*RASTER, "locked_in", "locked_out"), """
        reference(problem(pu_cost, features) |> add_min_largest_shortfall_objective(2500) |>
                  add_relative_targets(c(0.3, 0.4, 0.5, 0.2, 0.35)) |> add_locked_in_constraints(locked_in) |>
                  add_locked_out_constraints(locked_out) |> add_binary_decisions())"""),
    "max_utility": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_max_utility_objective(2500) |> add_binary_decisions())"""),
    "max_utility_weights": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_max_utility_objective(2500) |>
                  add_feature_weights(c(5, 1, 2, 0.01, 0.5)) |> add_proportion_decisions())""", notes="an LP"),
    "max_utility_negative": Case("objectives", ("pu_table",), """
        t <- pu_table; t$f3 <- t$f3 - 0.01
        reference(problem(t, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_max_utility_objective(2500) |> add_binary_decisions())""",
                                 notes="negative amounts: amount columns unbounded below"),
    "max_features": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_max_features_objective(2500) |>
                  add_relative_targets(c(0.3, 0.4, 0.5, 0.2, 0.35)) |> add_feature_weights(c(1, 2, 3, 1, 2)) |>
                  add_binary_decisions())"""),
    "max_cover": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_max_cover_objective(800) |> add_binary_decisions())"""),
    "min_penalties": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_penalties_objective(2500) |>
                  add_relative_targets(0.2) |> add_binary_decisions())""", notes="no penalties yet"),
    "targets_manual": Case("objectives", RASTER, """
        t <- data.frame(feature = c("f1", "f2", "f4", "f4", "f5"),
                        type = c("relative", "absolute", "relative", "absolute", "relative"),
                        sense = c(">=", "<=", ">=", "<=", ">="), target = c(0.3, 2.0, 0.2, 120, 0.25))
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_manual_targets(t) |>
                  add_binary_decisions())""", notes="several targets of one feature, a feature without"),
    "targets_manual_equal": Case("objectives", RASTER, """
        t <- data.frame(feature = c("f1", "f3", "f5"), type = c("relative", "absolute", "relative"),
                        sense = c("=", ">=", "="), target = c(0.3, 1.1, 0.25))
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_manual_targets(t) |>
                  add_proportion_decisions())"""),
    "targets_loglinear": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |>
                  add_loglinear_targets(3, 0.6, 300, 0.2) |> add_binary_decisions())"""),
    "targets_loglinear_cap": Case("objectives", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |>
                  add_loglinear_targets(3, 0.6, 300, 0.2, cap_amount = 100, cap_target = 50) |>
                  add_binary_decisions())"""),
    "max_phylo_div": Case("objectives", RASTER, f"""
        tr <- ape::read.tree(text = "{TREE_ORDERED}")
        reference(problem(pu_cost, features) |> add_max_phylo_div_objective(2500, tr) |>
                  add_relative_targets(0.3) |> add_binary_decisions())""",
                          notes="tips in the features' order: prioritizr itself"),
    "max_phylo_div_shuffled": Case("objectives", RASTER, f"""
        tr <- ape::read.tree(text = "{TREE_SHUFFLED}")
        reference(problem(pu_cost, features) |> add_max_phylo_div_objective_fixed(2500, tr) |>
                  add_relative_targets(c(0.3, 0.4, 0.5, 0.2, 0.35)) |> add_feature_weights(c(1, 2, 1, 3, 1)) |>
                  add_binary_decisions())""", notes="tips in another order: prioritizr with its bug fixed"),
    "max_phylo_end": Case("objectives", RASTER, f"""
        tr <- ape::read.tree(text = "{TREE_SHUFFLED}")
        reference(problem(pu_cost, features) |> add_max_phylo_end_objective_fixed(2500, tr) |>
                  add_relative_targets(0.3) |> add_binary_decisions())""",
                          notes="prioritizr with its tip-order bug fixed"),
    "max_phylo_end_ordered": Case("objectives", RASTER, f"""
        tr <- ape::read.tree(text = "{TREE_ORDERED}")
        reference(problem(pu_cost, features) |> add_max_phylo_end_objective(2500, tr) |>
                  add_relative_targets(0.3) |> add_binary_decisions())""",
                                  notes="tips in the features' order: prioritizr itself"),
    # --- spatial: boundary, connectivity, linear penalties and constraints (prioritizr's)
    "boundary_raster": Case("spatial", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_boundary_penalties(0.5) |> add_binary_decisions())"""),
    "boundary_raster_edge_max": Case("spatial", (*RASTER, "locked_in"), """
        reference(problem(pu_cost, features) |> add_max_utility_objective(2500) |>
                  add_boundary_penalties(0.002, edge_factor = 0.3) |> add_locked_in_constraints(locked_in) |>
                  add_binary_decisions())""", notes="a maximisation: the penalty's sign turns"),
    "boundary_polygons": Case("spatial", ("pu_polygons_partial", "features"), """
        reference(problem(pu_polygons_partial, features, cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_boundary_penalties(0.2, edge_factor = 0.75) |>
                  add_binary_decisions())""", notes="terra's sharedPaths and perim"),
    "boundary_matrix": Case("spatial", ("pu_table", "bnd_table"), """
        bm <- Matrix::sparseMatrix(i = bnd_table$i, j = bnd_table$j, x = bnd_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_boundary_penalties(3, data = bm) |> add_binary_decisions())"""),
    "connectivity": Case("spatial", ("pu_table", "conn_table"), """
        cm <- Matrix::sparseMatrix(i = conn_table$i, j = conn_table$j, x = conn_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_connectivity_penalties(5, data = cm) |> add_binary_decisions())"""),
    "connectivity_max": Case("spatial", ("pu_table", "conn_table"), """
        cm <- Matrix::sparseMatrix(i = conn_table$i, j = conn_table$j, x = conn_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_max_features_objective(2500) |> add_relative_targets(0.3) |>
                  add_connectivity_penalties(0.01, data = cm) |> add_binary_decisions())"""),
    "asym_connectivity": Case("spatial", ("pu_table", "asym_table"), """
        am <- Matrix::sparseMatrix(i = asym_table$i, j = asym_table$j, x = asym_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_asym_connectivity_penalties(4, data = am) |> add_binary_decisions())"""),
    "asym_connectivity_max": Case("spatial", ("pu_table", "asym_table"), """
        am <- Matrix::sparseMatrix(i = asym_table$i, j = asym_table$j, x = asym_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_max_utility_objective(2500) |> add_asym_connectivity_penalties(0.05, data = am) |>
                  add_binary_decisions())"""),
    "linear_penalties": Case("spatial", (*RASTER, "penalty_raster"), """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_linear_penalties(7.5, data = penalty_raster) |> add_binary_decisions())"""),
    "linear_constraints": Case("spatial", (*RASTER, "penalty_raster"), """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_linear_constraints(250, "<=", data = penalty_raster) |> add_binary_decisions())"""),
    "spatial_combined": Case("spatial", ("pu_table", "bnd_table", "conn_table"), """
        bm <- Matrix::sparseMatrix(i = bnd_table$i, j = bnd_table$j, x = bnd_table$x, dims = rep(nrow(pu_table), 2))
        cm <- Matrix::sparseMatrix(i = conn_table$i, j = conn_table$j, x = conn_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_min_shortfall_objective(2500) |> add_relative_targets(0.3) |>
                  add_linear_constraints(30, ">=", data = "f4") |> add_connectivity_penalties(0.02, data = cm) |>
                  add_linear_penalties(0.001, data = "f1") |> add_boundary_penalties(0.001, data = bm) |>
                  add_linear_constraints(2, "<=", data = "f2") |> add_locked_in_constraints("locked_in") |>
                  add_binary_decisions())""", notes="penalties and constraints in the order added"),
    "contiguity_raster": Case("spatial", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.2) |>
                  add_contiguity_constraints() |> add_binary_decisions())"""),
    "contiguity_polygons": Case("spatial", ("pu_polygons", "features"), """
        reference(problem(pu_polygons, features, cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.3) |> add_contiguity_constraints() |> add_binary_decisions())""",
                                notes="polygons touching at a corner are adjacent (st_intersects)"),
    "contiguity_matrix": Case("spatial", ("pu_table", "adj_table"), """
        am <- Matrix::sparseMatrix(i = adj_table$i, j = adj_table$j, x = adj_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
                  add_relative_targets(0.2) |> add_contiguity_constraints(data = am) |> add_binary_decisions())"""),
    "neighbors_raster": Case("spatial", RASTER, """
        reference(problem(pu_cost, features) |> add_min_set_objective() |> add_relative_targets(0.3) |>
                  add_neighbor_constraints(2) |> add_binary_decisions())"""),
    "neighbors_matrix": Case("spatial", ("pu_table", "adj_table"), """
        am <- Matrix::sparseMatrix(i = adj_table$i, j = adj_table$j, x = adj_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |>
                  add_max_utility_objective(2500) |> add_neighbor_constraints(5, data = am) |> add_binary_decisions())""",
                             notes="k = 5 clamped to the units with fewer neighbours"),
    "feature_contiguity_raster": Case("spatial", RASTER, """
        reference(problem(pu_cost, features[[2:3]]) |> add_min_set_objective() |> add_relative_targets(0.2) |>
                  add_feature_contiguity_constraints() |> add_binary_decisions())""",
                                      notes="the expanded formulation; two features (Gurobi is slow at gap 0)"),
    "feature_contiguity_matrix": Case("spatial", ("pu_table", "adj_table"), """
        am <- Matrix::sparseMatrix(i = adj_table$i, j = adj_table$j, x = adj_table$x, dims = rep(nrow(pu_table), 2))
        reference(problem(pu_table, features = c("f2", "f3"), cost_column = "cost") |>
                  add_max_features_objective(800) |> add_relative_targets(0.15) |>
                  add_feature_contiguity_constraints(data = am) |> add_binary_decisions())"""),
    # --- matrices: prioritizr's boundary, adjacency, connectivity and proximity matrices (dense)
    "matrices_raster": Case("matrices", ("pu_cost", "penalty_raster"), """
        list(boundary = as.matrix(boundary_matrix(pu_cost)), adjacency = as.matrix(adjacency_matrix(pu_cost)),
             adjacency8 = as.matrix(adjacency_matrix(pu_cost, directions = 8)),
             connectivity = as.matrix(connectivity_matrix(pu_cost, penalty_raster)),
             proximity = as.matrix(proximity_matrix(pu_cost, 150)))"""),
    "matrices_polygons": Case("matrices", ("pu_polygons_partial", "shapes", "penalty_raster"), """
        list(boundary = as.matrix(boundary_matrix(pu_polygons_partial)),
             adjacency = as.matrix(adjacency_matrix(pu_polygons_partial)),
             connectivity = as.matrix(connectivity_matrix(pu_polygons_partial, "cost")),
             connectivity_raster = as.matrix(connectivity_matrix(pu_polygons_partial, penalty_raster)),
             proximity = as.matrix(proximity_matrix(pu_polygons_partial, 120)),
             shapes_boundary = as.matrix(boundary_matrix(shapes)), shapes_adjacency = as.matrix(adjacency_matrix(shapes)))"""),
    # --- portfolio: the cuts portfolio with its no-good cut fixed (prioritizr's uses 2 s, not 2 s - 1), top, gap
    "portfolio_cuts_fixed": Case("portfolio", ("pu_table",), """
        p0 <- problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
          add_relative_targets(0.3) |> add_binary_decisions()
        p <- p0; objs <- c()
        for (k in 1:4) {
          s <- solve(add_gurobi_solver(p, gap = 0, threads = 1, verbose = FALSE))
          v <- s$solution_1
          objs <- c(objs, unname(attr(s, "objective")))
          p <- add_linear_constraints(p, sum(v) - 1, "<=", 2 * v - 1)
        }
        list(objective = objs)""", notes="the correct cut, by add_linear_constraints"),
    "portfolio_top": Case("portfolio", ("pu_table",), """
        p <- problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
          add_relative_targets(0.3) |> add_binary_decisions() |> add_top_portfolio(3) |>
          add_gurobi_solver(gap = 0, threads = 1, verbose = FALSE)
        s <- solve(p)
        list(objective = unname(attr(s, "objective")))"""),
    "portfolio_gap": Case("portfolio", ("pu_table",), """
        p <- problem(pu_table, features = paste0("f", 1:5), cost_column = "cost") |> add_min_set_objective() |>
          add_relative_targets(0.3) |> add_binary_decisions() |> add_gap_portfolio(5, 0.02) |>
          add_gurobi_solver(gap = 0, threads = 1, verbose = FALSE)
        s <- solve(p)
        list(objective = unname(attr(s, "objective")))"""),
    # --- exact: exactextractr's coverage fractions and sums (the amounts of polygon planning units)
    "exact_shapes": Case("exact", ("features", "shapes"), """
        cov <- exactextractr::coverage_fraction(features[[1]], shapes)
        out <- list(sum = as.matrix(exactextractr::exact_extract(features, shapes, fun = "sum", progress = FALSE)))
        for (k in seq_along(cov)) {
          v <- cov[[k]]
          if (!inherits(v, "SpatRaster")) v <- terra::rast(v)
          out[[paste0("coverage_", k)]] <- terra::values(v, mat = FALSE)
        }
        out""", notes="coverage_k: the fraction of every cell of the whole grid (NA outside the feature's window)"),
    # --- actions: prioriactions
    "actions_mincost": Case("actions", PA, """
        actions_reference(prioriactions::problem(pa_data(), model_type = "minimizeCosts"))"""),
    "actions_mincost_blm": Case("actions", PA, """
        thr <- pa_threats; thr$blm_actions <- c(0.5, 0.25)
        actions_reference(prioriactions::problem(pa_data(threats = thr), model_type = "minimizeCosts",
                                                 blm = 1))"""),
    "actions_mincost_conservation": Case("actions", PA, """
        ft <- pa_features; ft$target_conservation <- c(3, 2, 4, 1)
        actions_reference(prioriactions::problem(pa_data(features = ft), model_type = "minimizeCosts"))"""),
    "actions_maxbenefit": Case("actions", PA, """
        actions_reference(prioriactions::problem(pa_data(), model_type = "maximizeBenefits", budget = 60))"""),
    "actions_maxbenefit_blm": Case("actions", PA, """
        actions_reference(prioriactions::problem(pa_data(), model_type = "maximizeBenefits", budget = 60,
                                                 blm = 0.5))"""),
    "actions_mincost_curve3": Case("actions", PA, """
        sens <- pa_sensitivity; sens$a <- 0; sens$b <- 1; sens$c <- 0; sens$d <- 1
        actions_reference(prioriactions::problem(pa_data(sensitivity = sens), model_type = "minimizeCosts",
                                                 curve = 3, segments = 3))""",
                                   notes="power curve: Gurobi general constraints (genconpow)"),
}


# ----------------------------------------------------------------------------- the same problems in Python

FEATURES = [f"f{k}" for k in range(1, 6)]
ABSOLUTE = [1.2, 0.9, 1, 120, 4]


def _raster():
    import landschaft as ls

    return ls.prioritize.problem(pu_cost(), features()).add_min_set_objective()


def _table(t=None):
    import landschaft as ls

    return ls.prioritize.problem(pu_table() if t is None else t, FEATURES, cost_column="cost").add_min_set_objective()


def _rij():
    import landschaft as ls

    return ls.prioritize.problem(pu_table()[["id", "cost"]], spec_table(), cost_column="cost",
                                 rij=rij_table()).add_min_set_objective()


def _polygon_problem(gdf):
    import landschaft as ls

    return ls.prioritize.problem(gdf, features(), cost_column="cost").add_min_set_objective()


def _table_na_cost():
    t = pu_table()
    t.loc[[2, 49, 50, 119], "cost"] = np.nan
    return _table(t)


PY: dict[str, Callable] = {
    "minset_raster_relative": lambda: _raster().add_relative_targets(0.3).add_binary_decisions(),
    "minset_raster_relative_each": lambda: _raster().add_relative_targets([0.1, 0.25, 0.5, 0.2, 0.35])
    .add_binary_decisions(),
    "minset_raster_absolute": lambda: _raster().add_absolute_targets(ABSOLUTE).add_binary_decisions(),
    "minset_raster_locked": lambda: _raster().add_relative_targets(0.3).add_locked_in_constraints(locked_in())
    .add_locked_out_constraints(locked_out()).add_binary_decisions(),
    "minset_raster_proportion": lambda: _raster().add_relative_targets(0.3).add_proportion_decisions(),
    "minset_raster_semicontinuous": lambda: _raster().add_relative_targets(0.3).add_semicontinuous_decisions(0.6),
    "minset_table": lambda: _table().add_relative_targets(0.3).add_locked_in_constraints("locked_in")
    .add_locked_out_constraints("locked_out").add_binary_decisions(),
    "minset_rij": lambda: _rij().add_absolute_targets(ABSOLUTE).add_binary_decisions(),
    "minset_rij_relative": lambda: _rij().add_relative_targets(0.3).add_binary_decisions(),
    "minset_table_na_cost": lambda: _table_na_cost().add_relative_targets(0.3).add_binary_decisions(),
    "minset_raster_locked_proportion": lambda: _raster().add_relative_targets(0.3)
    .add_locked_in_constraints(locked_in()).add_locked_out_constraints(locked_out()).add_proportion_decisions(),
    "minset_polygons": lambda: _polygon_problem(pu_polygons()).add_relative_targets(0.3).add_binary_decisions(),
    "minset_polygons_partial": lambda: _polygon_problem(pu_polygons_partial()).add_relative_targets(0.3)
    .add_binary_decisions(),
}
for _name, _fn in PY.items():
    CASES[_name].py = _fn


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def _rp():
    import landschaft as ls

    return ls.prioritize.problem(pu_cost(), features())


MANUAL = {"feature": ["f1", "f2", "f4", "f4", "f5"], "type": ["relative", "absolute", "relative", "absolute", "relative"],
          "sense": [">=", "<=", ">=", "<=", ">="], "target": [0.3, 2.0, 0.2, 120, 0.25]}
MANUAL_EQ = {"feature": ["f1", "f3", "f5"], "type": ["relative", "absolute", "relative"], "sense": ["=", ">=", "="],
             "target": [0.3, 1.1, 0.25]}


def _negative_table():
    import landschaft as ls

    t = pu_table()
    t["f3"] = t["f3"] - 0.01
    return ls.prioritize.problem(t, FEATURES, cost_column="cost")


PY_OBJECTIVES: dict[str, Callable] = {
    "min_shortfall": lambda: _rp().add_min_shortfall_objective(2500).add_relative_targets(0.4).add_binary_decisions(),
    "min_shortfall_weights_tiny": lambda: _rp().add_min_shortfall_objective(2500)
    .add_absolute_targets([1.2, 1e-5, 1, 120, 4]).add_feature_weights([1, 2, 0.5, 3, 1]).add_binary_decisions(),
    "min_shortfall_tiny": lambda: _rp().add_min_shortfall_objective(2500)
    .add_absolute_targets([1.2, 1e-5, 1, 120, 4]).add_binary_decisions(),
    "min_largest_shortfall": lambda: _rp().add_min_largest_shortfall_objective(2500)
    .add_relative_targets([0.3, 0.4, 0.5, 0.2, 0.35]).add_locked_in_constraints(locked_in())
    .add_locked_out_constraints(locked_out()).add_binary_decisions(),
    "max_utility": lambda: _rp().add_max_utility_objective(2500).add_binary_decisions(),
    "max_utility_weights": lambda: _rp().add_max_utility_objective(2500).add_feature_weights([5, 1, 2, 0.01, 0.5])
    .add_proportion_decisions(),
    "max_utility_negative": lambda: _negative_table().add_max_utility_objective(2500).add_binary_decisions(),
    "max_features": lambda: _rp().add_max_features_objective(2500).add_relative_targets([0.3, 0.4, 0.5, 0.2, 0.35])
    .add_feature_weights([1, 2, 3, 1, 2]).add_binary_decisions(),
    "max_cover": lambda: _rp().add_max_cover_objective(800).add_binary_decisions(),
    "min_penalties": lambda: _rp().add_min_penalties_objective(2500).add_relative_targets(0.2).add_binary_decisions(),
    "targets_manual": lambda: _rp().add_min_set_objective().add_manual_targets(MANUAL).add_binary_decisions(),
    "targets_manual_equal": lambda: _rp().add_min_set_objective().add_manual_targets(MANUAL_EQ)
    .add_proportion_decisions(),
    "targets_loglinear": lambda: _rp().add_min_set_objective().add_loglinear_targets(3, 0.6, 300, 0.2)
    .add_binary_decisions(),
    "targets_loglinear_cap": lambda: _rp().add_min_set_objective()
    .add_loglinear_targets(3, 0.6, 300, 0.2, cap_amount=100, cap_target=50).add_binary_decisions(),
}
for _name, _fn in PY_OBJECTIVES.items():
    CASES[_name].py = _fn

PY_OBJECTIVES.update({
    "max_phylo_div": lambda: _rp().add_max_phylo_div_objective(2500, TREE_ORDERED).add_relative_targets(0.3)
    .add_binary_decisions(),
    "max_phylo_div_shuffled": lambda: _rp().add_max_phylo_div_objective(2500, TREE_SHUFFLED)
    .add_relative_targets([0.3, 0.4, 0.5, 0.2, 0.35]).add_feature_weights([1, 2, 1, 3, 1]).add_binary_decisions(),
    "max_phylo_end": lambda: _rp().add_max_phylo_end_objective(2500, TREE_SHUFFLED).add_relative_targets(0.3)
    .add_binary_decisions(),
    "max_phylo_end_ordered": lambda: _rp().add_max_phylo_end_objective(2500, TREE_ORDERED).add_relative_targets(0.3)
    .add_binary_decisions(),
})
for _name, _fn in PY_OBJECTIVES.items():
    CASES[_name].py = _fn


def _matrix(t, n):
    import scipy.sparse as sp

    return sp.csc_matrix((t["x"].to_numpy(), (t["i"].to_numpy(dtype=np.int64) - 1, t["j"].to_numpy(dtype=np.int64) - 1)),
                         shape=(n, n))


def ls_problem(*args, **kw):
    import landschaft as ls

    return ls.prioritize.problem(*args, **kw)


def _tp():
    import landschaft as ls

    return ls.prioritize.problem(pu_table(), FEATURES, cost_column="cost")


PY_SPATIAL: dict[str, Callable] = {
    "boundary_raster": lambda: _rp().add_min_set_objective().add_relative_targets(0.3).add_boundary_penalties(0.5)
    .add_binary_decisions(),
    "boundary_raster_edge_max": lambda: _rp().add_max_utility_objective(2500)
    .add_boundary_penalties(0.002, edge_factor=0.3).add_locked_in_constraints(locked_in()).add_binary_decisions(),
    "boundary_polygons": lambda: _polygon_problem(pu_polygons_partial()).add_relative_targets(0.3)
    .add_boundary_penalties(0.2, edge_factor=0.75).add_binary_decisions(),
    "boundary_matrix": lambda: _tp().add_min_set_objective().add_relative_targets(0.3)
    .add_boundary_penalties(3, data=_matrix(bnd_table(), len(pu_table()))).add_binary_decisions(),
    "connectivity": lambda: _tp().add_min_set_objective().add_relative_targets(0.3)
    .add_connectivity_penalties(5, data=_matrix(conn_table(), len(pu_table()))).add_binary_decisions(),
    "connectivity_max": lambda: _tp().add_max_features_objective(2500).add_relative_targets(0.3)
    .add_connectivity_penalties(0.01, data=_matrix(conn_table(), len(pu_table()))).add_binary_decisions(),
    "asym_connectivity": lambda: _tp().add_min_set_objective().add_relative_targets(0.3)
    .add_asym_connectivity_penalties(4, data=_matrix(asym_table(), len(pu_table()))).add_binary_decisions(),
    "asym_connectivity_max": lambda: _tp().add_max_utility_objective(2500)
    .add_asym_connectivity_penalties(0.05, data=_matrix(asym_table(), len(pu_table()))).add_binary_decisions(),
    "linear_penalties": lambda: _rp().add_min_set_objective().add_relative_targets(0.3)
    .add_linear_penalties(7.5, data=penalty_raster()).add_binary_decisions(),
    "linear_constraints": lambda: _rp().add_min_set_objective().add_relative_targets(0.3)
    .add_linear_constraints(250, "<=", data=penalty_raster()).add_binary_decisions(),
    "spatial_combined": lambda: _tp().add_min_shortfall_objective(2500).add_relative_targets(0.3)
    .add_linear_constraints(30, ">=", data="f4")
    .add_connectivity_penalties(0.02, data=_matrix(conn_table(), len(pu_table())))
    .add_linear_penalties(0.001, data="f1")
    .add_boundary_penalties(0.001, data=_matrix(bnd_table(), len(pu_table())))
    .add_linear_constraints(2, "<=", data="f2").add_locked_in_constraints("locked_in").add_binary_decisions(),
}
for _name, _fn in PY_SPATIAL.items():
    CASES[_name].py = _fn

PY_SPATIAL.update({
    "contiguity_raster": lambda: _rp().add_min_set_objective().add_relative_targets(0.2)
    .add_contiguity_constraints().add_binary_decisions(),
    "contiguity_polygons": lambda: _polygon_problem(pu_polygons()).add_relative_targets(0.3)
    .add_contiguity_constraints().add_binary_decisions(),
    "contiguity_matrix": lambda: _tp().add_min_set_objective().add_relative_targets(0.2)
    .add_contiguity_constraints(data=_matrix(adj_table(), len(pu_table()))).add_binary_decisions(),
    "neighbors_raster": lambda: _rp().add_min_set_objective().add_relative_targets(0.3)
    .add_neighbors_constraints(2).add_binary_decisions(),
    "neighbors_matrix": lambda: _tp().add_max_utility_objective(2500)
    .add_neighbors_constraints(5, data=_matrix(adj_table(), len(pu_table()))).add_binary_decisions(),
})
for _name, _fn in PY_SPATIAL.items():
    CASES[_name].py = _fn

PY_SPATIAL.update({
    "feature_contiguity_raster": lambda: ls_problem(pu_cost(), features().isel(band=[1, 2])).add_min_set_objective()
    .add_relative_targets(0.2).add_feature_contiguity_constraints().add_binary_decisions(),
    "feature_contiguity_matrix": lambda: ls_problem(pu_table(), ["f2", "f3"], cost_column="cost")
    .add_max_features_objective(800).add_relative_targets(0.15)
    .add_feature_contiguity_constraints(data=_matrix(adj_table(), len(pu_table()))).add_binary_decisions(),
})
for _name, _fn in PY_SPATIAL.items():
    CASES[_name].py = _fn
