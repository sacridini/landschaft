# SPDX-License-Identifier: GPL-3.0-or-later
"""MILP prioritization: problems in prioritizr's terms and their compiled models.

A :class:`Problem` holds planning units (raster cells or table rows), the
amount of every feature in every unit and the specification added by its
``add_*`` methods (prioritizr's names). It is immutable: every ``add_*``
returns a new problem, and nothing is computed until :meth:`Problem.compile`
or :meth:`Problem.solve`, so the order of the calls does not matter.

The compiled model (:class:`OptimizationProblem`) is the same, bit for bit, as
prioritizr's ``compile()`` (Hanson et al., prioritizr 8.0.6; GPL-3): the same
rows, columns, coefficients, bounds and right-hand sides.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

__all__ = ["OptimizationProblem", "Problem", "h3_units", "problem"]


# ----------------------------------------------------------------------------- compiled model

@dataclass(frozen=True)
class OptimizationProblem:
    """A compiled model: ``modelsense`` of ``obj @ x`` subject to ``A x (sense) rhs``,
    ``lb <= x <= ub`` and ``x`` of type ``vtype`` (``"B"`` binary, ``"C"`` continuous,
    ``"I"`` integer); the layout of prioritizr's ``compile()``."""

    A: sp.csr_matrix
    obj: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    rhs: np.ndarray
    sense: tuple[str, ...]
    vtype: tuple[str, ...]
    modelsense: str
    row_ids: tuple[str, ...]
    col_ids: tuple[str, ...]

    @property
    def nrow(self) -> int:
        return self.A.shape[0]

    @property
    def ncol(self) -> int:
        return self.A.shape[1]

    @property
    def ncell(self) -> int:
        """Non-zeros of the constraint matrix."""
        return self.A.nnz

    def row_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of every row (-inf / inf when open)."""
        sense = np.asarray(self.sense)
        lo = np.where(np.isin(sense, (">=", "=")), self.rhs, -np.inf)
        hi = np.where(np.isin(sense, ("<=", "=")), self.rhs, np.inf)
        return lo, hi

    def __repr__(self) -> str:
        kinds = ", ".join(f"{n} {t}" for t, n in zip(*np.unique(np.asarray(self.vtype), return_counts=True)))
        return (f"OptimizationProblem({self.modelsense}; {self.nrow:,} rows, {self.ncol:,} columns, "
                f"{self.ncell:,} non-zeros; variables: {kinds})")


# ----------------------------------------------------------------------------- planning units

@dataclass(frozen=True)
class Units:
    """Planning units and the amounts of the features in them."""

    kind: str  # "raster" or "table"
    cost: np.ndarray  # per unit
    rij: sp.csr_matrix  # features x units, no explicit zeros
    totals: np.ndarray  # per feature: the totals relative targets are shares of
    feature_names: tuple[str, ...]
    template: object  # raster: the cost DataArray; table: the DataFrame
    index: np.ndarray  # raster: flat cell of each unit; table: row of each unit

    @property
    def n(self) -> int:
        return self.cost.size


def _sequential_sum(v: np.ndarray) -> float:
    """Sum in order, in double (as terra adds a raster's cells)."""
    return float(np.cumsum(v, dtype=np.float64)[-1]) if v.size else 0.0


def _r_sum(v: np.ndarray) -> float:
    """R's sum() of doubles (80-bit accumulation)."""
    from .. import _core

    return float(_core.nlm_r_sum(np.ascontiguousarray(v, dtype=np.float64)))


def _rij(amounts: np.ndarray) -> sp.csr_matrix:
    """Features x units matrix of the amounts (NaN = 0), zeros left out."""
    a = np.nan_to_num(np.asarray(amounts, dtype=np.float64), nan=0.0)
    m = sp.csr_matrix(a)
    m.eliminate_zeros()
    m.sort_indices()
    return m


def _stack(features) -> xr.DataArray:
    if isinstance(features, xr.DataArray):
        return features if features.ndim == 3 else features.expand_dims(band=[features.name or "feature"])
    if isinstance(features, xr.Dataset):
        return features.to_dataarray("band")
    if isinstance(features, Sequence):
        layers = [_stack(f) for f in features]
        return xr.concat(layers, dim="band")
    raise TypeError("features must be a raster DataArray (band, y, x), a Dataset or a list of rasters")


def _raster_units(cost: xr.DataArray, features) -> Units:
    if cost.ndim == 3 and cost.sizes.get("band", 1) == 1:
        cost = cost.squeeze("band", drop=True)
    if cost.ndim != 2:
        raise ValueError("the cost raster must have one band")
    f = _stack(features)
    if f.shape[-2:] != cost.shape:
        raise ValueError(f"the features ({f.shape[-2:]}) and the cost ({cost.shape}) must be on the same grid")
    for dim in ("y", "x"):
        if dim in cost.coords and dim in f.coords and not np.allclose(cost[dim].values, f[dim].values):
            raise ValueError("the features and the cost must be on the same grid")
    c = np.asarray(cost.values, dtype=np.float64).ravel()
    cells = np.flatnonzero(np.isfinite(c))
    values = np.asarray(f.values, dtype=np.float64).reshape(f.shape[0], -1)
    totals = np.array([_sequential_sum(v[np.isfinite(v)]) for v in values])
    names = tuple(str(b) for b in f["band"].values) if "band" in f.coords else tuple(
        f"feature_{k + 1}" for k in range(f.shape[0]))
    return Units("raster", c[cells], _rij(values[:, cells]), totals, names, cost, cells)


def exact_sums(raster: xr.DataArray, geometries, n_threads: int | None = None) -> np.ndarray:
    """Coverage-weighted sums of every layer over every (multi)polygon: features x layers, as
    exactextractr's ``exact_extract(fun = "sum")`` gives them, bit for bit (exactextract's exact cell
    coverage, float results)."""
    import shapely

    from .. import _core, _threads
    from ..io import _transform

    r = _stack(raster)
    t = _transform(r)
    ny, nx = r.shape[-2:]
    if t.b != 0 or t.d != 0 or t.e >= 0:
        raise ValueError("the features' raster must be north-up")
    dx, dy = t.a, -t.e
    grid = (t.c, t.f - ny * dy, t.c + nx * dx, t.f, dx, dy)
    geoms = np.asarray(geometries, dtype=object)
    kinds = shapely.get_type_id(geoms)
    if not np.isin(kinds, (3, 6)).all():  # Polygon, MultiPolygon
        raise TypeError("planning units must be polygons")
    # every polygon of every feature, in order: rings and polygons as shapely's ragged arrays
    parts, owner = shapely.get_parts(geoms, return_index=True)
    feat_off = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=geoms.size))])
    if parts.size:
        _, xy, (ring_off, poly_off) = shapely.to_ragged_array(parts)
    else:
        xy, ring_off, poly_off = np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros(1, np.int64)
    return _core.io_exact_sums(
        np.ascontiguousarray(r.values, dtype=np.float64), grid, np.ascontiguousarray(xy, dtype=np.float64),
        ring_off.astype(np.int64), poly_off.astype(np.int64), feat_off.astype(np.int64),
        n_threads=_threads.resolve(n_threads))


def h3_units(cost: xr.DataArray, resolution: int, n_threads: int | None = None):
    """H3 hexagons as planning units: the hexagons of an H3 ``resolution``
    that cover a cost raster, with their cost.

    Pass them to :func:`problem` as polygons (``cost_column="cost"``): the
    features' amounts are then summed over every hexagon like any polygon's.

    Parameters
    ----------
    cost
        The cost raster (with a CRS). Every hexagon that covers a cell with a
        cost is a planning unit; its cost is the sum of the costs of the cells
        it covers, each weighted by the exact share of the cell inside it.
    resolution
        The H3 resolution (0-15; 7 is about 5 km², 8 about 0.7 km²).
    n_threads
        Threads for the sums (default: all cores but one).

    Returns
    -------
    geopandas.GeoDataFrame
        One hexagon per row (in the raster's CRS, its six vertices
        reprojected): ``h3`` (the cell's index) and ``cost``.

    Examples
    --------
    >>> import landschaft as ls
    >>> hexes = ls.prioritize.h3_units(cost, 8)                   # doctest: +SKIP
    >>> p = ls.prioritize.problem(hexes, features, cost_column="cost")  # doctest: +SKIP
    """
    import geopandas as gpd
    import shapely

    from .._vector import transform_xy
    from ..bio._grids import _require_h3
    from ..io import _transform

    h3 = _require_h3()
    if cost.ndim == 3 and cost.sizes.get("band", 1) == 1:
        cost = cost.squeeze("band", drop=True)
    if cost.ndim != 2:
        raise ValueError("the cost raster must have one band")
    crs = cost.rio.crs
    if crs is None:
        raise ValueError("the cost raster has no CRS")
    if not 0 <= int(resolution) <= 15:
        raise ValueError("resolution must be from 0 to 15")
    # the raster's footprint (its edges densified) in longitude / latitude
    t = _transform(cost)
    ny, nx = cost.shape
    k = np.linspace(0.0, 1.0, 65)
    cols = np.concatenate([k * nx, np.full(k.size, nx), (1 - k) * nx, np.zeros(k.size)])
    rows = np.concatenate([np.zeros(k.size), k * ny, np.full(k.size, ny), (1 - k) * ny])
    x = t.c + cols * t.a + rows * t.b
    y = t.f + cols * t.d + rows * t.e
    lon, lat = transform_xy(x, y, crs, "EPSG:4326")
    shape = h3.LatLngPoly(list(zip(lat.tolist(), lon.tolist())))
    cells = sorted(h3.h3shape_to_cells_experimental(shape, int(resolution), contain="overlap"))
    if not cells:
        raise ValueError("no H3 cell covers the raster")
    rings = [h3.cell_to_boundary(c) for c in cells]  # (lat, lng) pairs
    hexes = shapely.polygons([[(b, a) for a, b in r] for r in rings])
    gdf = gpd.GeoDataFrame({"h3": cells}, geometry=hexes, crs="EPSG:4326").to_crs(crs)
    values = cost.values.astype(np.float64)
    stack = xr.DataArray(np.stack([values, np.isfinite(values).astype(np.float64)]), dims=("band", *cost.dims),
                         coords={**{d: cost[d] for d in cost.dims}, "band": ["cost", "valid"]})
    stack = stack.rio.write_transform(t).rio.write_crs(crs)
    sums = exact_sums(stack, gdf.geometry.values, n_threads=n_threads)
    keep = sums[:, 1] > 0
    out = gdf.loc[keep].reset_index(drop=True)
    out.insert(1, "cost", sums[keep, 0])
    return out


def _vector_units(x, features, cost_column: str) -> Units:
    if cost_column not in x.columns:
        raise ValueError(f"no cost column {cost_column!r}")
    cost = x[cost_column].to_numpy(dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(cost))
    f = _stack(features)
    geoms = x
    crs = f.rio.crs
    if x.crs is not None and crs is not None and x.crs != crs:  # the amounts in the rasters' CRS
        from .._vector import to_crs

        geoms = to_crs(x, crs)
    amounts = exact_sums(f, geoms.geometry.values).T  # features x units
    amounts[np.abs(amounts) < 1e-10] = 0.0  # prioritizr's fast_extract
    full = _rij(amounts)
    # Matrix's rowSums over every unit: in double, in unit order
    totals = np.array([_sequential_sum(full.getrow(i).data) for i in range(full.shape[0])])
    names = tuple(str(b) for b in f["band"].values) if "band" in f.coords else tuple(
        f"feature_{k + 1}" for k in range(f.shape[0]))
    m = full[:, rows].tocsr()
    m.sort_indices()
    return Units("table", cost[rows], m, totals, names, x, rows)


def _table_units(x: pd.DataFrame, features, cost_column: str, rij) -> Units:
    if cost_column not in x.columns:
        raise ValueError(f"no cost column {cost_column!r}")
    cost = x[cost_column].to_numpy(dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(cost))
    if rij is None:
        names = [features] if isinstance(features, str) else list(features)
        missing = [n for n in names if n not in x.columns]
        if missing:
            raise ValueError(f"no feature columns {missing}")
        amounts = x[names].to_numpy(dtype=np.float64).T
        totals = np.array([_r_sum(np.nan_to_num(a, nan=0.0)) for a in amounts])  # R's colSums
        return Units("table", cost[rows], _rij(amounts[:, rows]), totals, tuple(names), x, rows)
    # Marxan-style tables: features (id, name) and rij (pu, species, amount)
    spec = features
    if not isinstance(spec, pd.DataFrame) or not {"id", "name"} <= set(spec.columns):
        raise ValueError("with rij, features must be a table with columns id and name")
    if not {"pu", "species", "amount"} <= set(rij.columns):
        raise ValueError("rij must have the columns pu, species and amount")
    if "id" not in x.columns:
        raise ValueError("with rij, the planning-unit table needs an id column")
    unit_of = pd.Series(np.arange(len(x)), index=x["id"].to_numpy())
    feat_of = pd.Series(np.arange(len(spec)), index=spec["id"].to_numpy())
    r = unit_of.reindex(rij["pu"].to_numpy()).to_numpy()
    k = feat_of.reindex(rij["species"].to_numpy()).to_numpy()
    if np.isnan(r).any() or np.isnan(k).any():
        raise ValueError("rij refers to planning units or features that are not in the tables")
    a = rij["amount"].to_numpy(dtype=np.float64)
    full = sp.csr_matrix((a, (k.astype(np.int64), r.astype(np.int64))), shape=(len(spec), len(x)))
    full.sum_duplicates()
    # Matrix's rowSums: in double, in planning-unit order
    totals = np.array([_sequential_sum(full.getrow(i).data) for i in range(full.shape[0])])
    m = full[:, rows].tocsr()
    m.eliminate_zeros()
    m.sort_indices()
    return Units("table", cost[rows], m, totals, tuple(str(n) for n in spec["name"]), x, rows)


# ----------------------------------------------------------------------------- problem

@dataclass(frozen=True)
class Problem:
    """A conservation planning problem (prioritizr's ``problem()``): see :func:`problem`."""

    units: Units = field(repr=False)
    objective: tuple | None = None
    targets: tuple | None = None
    locked_in: object = None
    locked_out: object = None
    decisions: tuple = ("binary",)
    solver: tuple = ("highs", {})

    def __repr__(self) -> str:
        u = self.units
        return (f"Problem({u.kind}, {u.n:,} planning units, {len(u.feature_names)} features; "
                f"objective={self.objective and self.objective[0]}, targets={self.targets and self.targets[0]}, "
                f"decisions={self.decisions[0]}, solver={self.solver[0]})")

    def _with(self, **kw) -> Problem:
        return dataclasses.replace(self, **kw)

    # --- objectives
    def add_min_set_objective(self) -> Problem:
        """Minimum set: meet every target at the least cost."""
        return self._with(objective=("min_set",))

    # --- targets
    def add_relative_targets(self, targets) -> Problem:
        """Targets as shares (0-1) of every feature's total amount (one value or one per feature)."""
        return self._with(targets=("relative", self._per_feature(targets)))

    def add_absolute_targets(self, targets) -> Problem:
        """Targets as amounts of every feature (one value or one per feature)."""
        return self._with(targets=("absolute", self._per_feature(targets)))

    def _per_feature(self, v) -> np.ndarray:
        n = len(self.units.feature_names)
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size == 1:
            a = np.repeat(a, n)
        if a.size != n:
            raise ValueError(f"expected one target or {n} (one per feature), got {a.size}")
        return a

    # --- constraints
    def add_locked_in_constraints(self, locked) -> Problem:
        """Planning units that must be selected: a raster (1 = locked), a column name, a boolean
        array over the planning units or their (0-based) indices."""
        return self._with(locked_in=locked)

    def add_locked_out_constraints(self, locked) -> Problem:
        """Planning units that must not be selected (as :meth:`add_locked_in_constraints`)."""
        return self._with(locked_out=locked)

    def _mask(self, locked) -> np.ndarray:
        u = self.units
        if locked is None:
            return np.zeros(u.n, dtype=bool)
        if isinstance(locked, xr.DataArray):
            if u.kind != "raster":
                raise TypeError("a raster of locked units needs raster planning units")
            v = np.asarray(locked.values, dtype=np.float64).ravel()
            if v.size != u.template.size:
                raise ValueError("the raster of locked units must be on the planning units' grid")
            return np.nan_to_num(v[u.index], nan=0.0) != 0
        if isinstance(locked, str):
            if u.kind != "table":
                raise TypeError("a column of locked units needs table planning units")
            return u.template[locked].to_numpy(dtype=bool)[u.index]
        a = np.asarray(locked)
        if a.dtype == bool:
            if a.size != u.n:
                raise ValueError(f"the boolean mask must have one value per planning unit ({u.n})")
            return a.ravel()
        m = np.zeros(u.n, dtype=bool)
        m[a.astype(np.int64).ravel()] = True
        return m

    # --- decisions
    def add_binary_decisions(self) -> Problem:
        """Every planning unit is selected or not."""
        return self._with(decisions=("binary",))

    def add_proportion_decisions(self) -> Problem:
        """Any share (0-1) of every planning unit can be selected (a linear program)."""
        return self._with(decisions=("proportion",))

    def add_semicontinuous_decisions(self, upper_limit: float) -> Problem:
        """Any share of every planning unit up to ``upper_limit`` can be selected."""
        return self._with(decisions=("semicontinuous", float(upper_limit)))

    # --- solvers
    def add_highs_solver(self, gap: float = 0.1, time_limit: float | None = None, n_threads: int | None = None,
                         presolve: str = "choose", verbose: bool = False) -> Problem:
        """Solve on HiGHS (built into landschaft), to a relative ``gap`` (0.1 = 10%)."""
        return self._with(solver=("highs", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                            "presolve": presolve, "verbose": verbose}))

    def add_gurobi_solver(self, gap: float = 0.1, time_limit: float | None = None, n_threads: int | None = None,
                          verbose: bool = False) -> Problem:
        """Solve on Gurobi (needs the ``gurobipy`` package and a Gurobi licence)."""
        return self._with(solver=("gurobi", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                             "verbose": verbose}))

    # --- compile
    def compile(self) -> OptimizationProblem:
        """The model prioritizr's ``compile()`` gives for this problem."""
        u = self.units
        if self.objective is None:
            raise ValueError("the problem has no objective: add one (add_min_set_objective, ...)")
        if self.targets is None:
            raise ValueError("the min set objective needs targets (add_relative_targets, add_absolute_targets)")
        kind, values = self.targets
        rhs = values * u.totals if kind == "relative" else values.copy()
        n = u.n
        lb = np.zeros(n)
        if self.decisions[0] == "binary":
            ub, vtype = np.ones(n), ("B",) * n
        elif self.decisions[0] == "proportion":
            ub, vtype = np.ones(n), ("C",) * n
        else:
            ub, vtype = np.full(n, self.decisions[1]), ("C",) * n
        li, lo = self._mask(self.locked_in), self._mask(self.locked_out)
        if (li & lo).any():
            raise ValueError(f"{int((li & lo).sum())} planning units are both locked in and locked out")
        lb[li] = 1.0
        ub[lo] = 0.0
        nf = len(u.feature_names)
        return OptimizationProblem(
            A=u.rij.copy(), obj=u.cost.copy(), lb=lb, ub=ub, rhs=rhs, sense=(">=",) * nf, vtype=vtype,
            modelsense="min", row_ids=("spp_target",) * nf, col_ids=("pu",) * n)

    # --- solve
    def solve(self, start=None):
        """Solve the problem on its solver (HiGHS unless another was added): a
        :class:`~landschaft.prioritize.Solution`. ``start``: a solution to
        start from (one value per planning unit)."""
        from ._milp_solve import make_solution, solve_model

        m = self.compile()
        if start is not None:
            start = np.asarray(start, dtype=np.float64).ravel()
            if start.size != m.ncol:
                raise ValueError(f"the start solution must have one value per planning unit ({m.ncol})")
        r = solve_model(m, self.solver[0], self.solver[1], start)
        return make_solution(self.units, m, r)


def problem(x, features=None, *, cost_column: str | None = None, rij=None) -> Problem:
    """A conservation planning problem (prioritizr's ``problem()``).

    Parameters
    ----------
    x
        The planning units and their cost: a raster (the cells with a cost are
        the planning units), polygons (a GeoDataFrame) or a table, both with a
        ``cost_column`` (the rows with a cost are the planning units).
    features
        With a raster: the features' rasters on the same grid (a ``(band, y,
        x)`` DataArray, a Dataset or a list of rasters). With polygons: the
        features' rasters, summed over every polygon weighted by the exact
        share of every cell it covers (as exactextractr, which prioritizr
        uses). With a table: the names of its feature columns, or, with
        ``rij``, a table of the features (``id``, ``name``).
    cost_column
        The cost column of a table.
    rij
        Marxan-style amounts (``pu``, ``species``, ``amount``) with a table of
        planning units that has an ``id`` column.

    Examples
    --------
    >>> import landschaft as ls
    >>> p = (ls.prioritize.problem(cost, features)
    ...        .add_min_set_objective()
    ...        .add_relative_targets(0.3)
    ...        .add_binary_decisions())
    >>> sol = p.solve()  # doctest: +SKIP
    """
    if isinstance(x, xr.DataArray):
        if features is None:
            raise ValueError("a raster problem needs the features' rasters")
        return Problem(_raster_units(x, features))
    if isinstance(x, pd.DataFrame):
        if cost_column is None:
            raise ValueError("a table of planning units needs its cost_column")
        rasters = isinstance(features, (xr.DataArray, xr.Dataset)) or (
            isinstance(features, list) and bool(features) and isinstance(features[0], xr.DataArray))
        if rasters:
            if not hasattr(x, "geometry"):
                raise TypeError("raster features need raster or polygon planning units")
            return Problem(_vector_units(x, features, cost_column))
        return Problem(_table_units(x, features, cost_column, rij))
    raise TypeError("x must be a raster DataArray or a table (DataFrame) of planning units")
