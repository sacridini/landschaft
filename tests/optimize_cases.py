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

GROUPS: list[str] = ["core", "objectives", "actions", "exact"]

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


INPUTS = {f.__name__: f for f in (
    pu_cost, features, shapes, locked_in, locked_out, pu_table, spec_table, rij_table, pu_polygons,
    pu_polygons_partial, pa_pu, pa_features, pa_dist_features, pa_threats, pa_dist_threats, pa_sensitivity,
    pa_boundary)}


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
