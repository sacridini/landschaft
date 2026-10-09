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
from collections.abc import Mapping, Sequence
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
    # node sets that must form one patch (contiguity constraints): the solvers separate cuts on them
    connected: tuple = field(default=(), repr=False, compare=False)
    # corridor networks: the solvers separate node cuts on them
    corridors: tuple = field(default=(), repr=False, compare=False)

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
    features: object = None  # the feature rasters (raster and polygon units), for a Zonation start
    all_rij: object = None  # tables and polygons: features x every row, cost or not (sub-regional totals)
    r_sums: bool = False  # totals are R's sum() (tables of feature columns: colSums), not a double sum
    # management zones (empty: one zone). With zones, cost and rij run over the columns of every zone, zone
    # after zone (column z * n + i; cost NaN: the unit cannot go to the zone), totals is features x zones,
    # zone_rij every zone's features x units and base_totals the totals that targets summed over zones are
    # shares of (the base features of the contributions; None with every zone's own features)
    zone_names: tuple[str, ...] = ()
    zone_rij: tuple = ()
    base_totals: object = None
    zone_source: tuple = ()  # what the zones' layers are made of (sub-regional totals)

    @property
    def nz(self) -> int:
        """The number of zones (1 without zones)."""
        return max(1, len(self.zone_names))

    @property
    def n(self) -> int:
        """The number of planning units."""
        return self.cost.size // self.nz

    @property
    def zoned(self) -> bool:
        return len(self.zone_names) > 0


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


def _layers(features) -> list[tuple[str, xr.DataArray]]:
    """The features as named 2-D rasters, without copying them into one stack."""
    from ..collection import RasterCollection

    if isinstance(features, RasterCollection):  # ls.load_raster([...]): a lazy stack, read block by block
        if not features.same_grid:
            raise ValueError("the feature rasters must share a grid")
        return _layers(features.to_stack())
    if isinstance(features, xr.Dataset):
        return [(str(k), v) for k, v in features.data_vars.items()]
    if isinstance(features, xr.DataArray):
        if features.ndim == 2:  # its name, or its band's (a slice of a stack)
            band = features.coords.get("band")
            name = features.name or (band.item() if band is not None and band.ndim == 0 else "feature")
            return [(str(name), features)]
        if features.ndim != 3:
            raise ValueError("features must be (band, y, x)")
        band = features.dims[0]
        names = features[band].values if band in features.coords else [
            f"feature_{k + 1}" for k in range(features.shape[0])]
        return [(str(n), features[k]) for k, n in enumerate(names)]
    if isinstance(features, Sequence):
        return [item for f in features for item in _layers(f)]
    raise TypeError("features must be a raster DataArray (band, y, x), a Dataset or a list of rasters")


BLOCK_BYTES = 1 << 26  # rows of the features read at once: about 64 MB of doubles


def _raster_units(cost: xr.DataArray, features, n_threads: int | None = None) -> Units:
    from .. import _core, _threads

    if cost.ndim == 3 and cost.sizes.get("band", 1) == 1:
        cost = cost.squeeze("band", drop=True)
    if cost.ndim != 2:
        raise ValueError("the cost raster must have one band")
    layers = _layers(features)
    if not layers:
        raise ValueError("no features")
    for _, f in layers:
        if f.shape != cost.shape:
            raise ValueError(f"the features ({f.shape}) and the cost ({cost.shape}) must be on the same grid")
        for dim in ("y", "x"):
            if dim in cost.coords and dim in f.coords and not np.allclose(cost[dim].values, f[dim].values):
                raise ValueError("the features and the cost must be on the same grid")
    ny, nx = cost.shape
    c = np.asarray(cost.values, dtype=np.float64).ravel()
    cells = np.flatnonzero(np.isfinite(c))
    unit = np.full(c.size, -1, dtype=np.int64)
    unit[cells] = np.arange(cells.size)
    # the features block by block of rows (dask layers computed together)
    acc = _core.OptFeatureMatrix(len(layers))
    threads = _threads.resolve(n_threads)
    step = max(1, BLOCK_BYTES // (8 * nx * len(layers)))
    lazy = [k for k, (_, f) in enumerate(layers) if f.chunks is not None]
    block = None
    for y0 in range(0, ny, step):
        y1 = min(ny, y0 + step)
        if block is None or block.shape[1] != (y1 - y0) * nx:
            block = np.empty((len(layers), (y1 - y0) * nx), dtype=np.float64)
        rows = [f.data[y0:y1] for _, f in layers]
        if lazy:
            import dask

            for k, v in zip(lazy, dask.compute(*[rows[k] for k in lazy])):
                rows[k] = v
        for k, v in enumerate(rows):
            v = np.ma.filled(v.astype(np.float64), np.nan) if np.ma.isMaskedArray(v) else v
            block[k] = np.asarray(v, dtype=np.float64).ravel()
        acc.add(block, unit[y0 * nx:y1 * nx], threads)
    totals, start, index, value = acc.finish()
    rij = sp.csr_matrix((value, index, start), shape=(len(layers), cells.size))
    rij.has_sorted_indices = True
    names = tuple(n for n, _ in layers)
    return Units("raster", c[cells], rij, totals, names, cost, cells, features)


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
    return Units("table", cost[rows], m, totals, names, x, rows, features, full)


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
        return Units("table", cost[rows], _rij(amounts[:, rows]), totals, tuple(names), x, rows,
                     all_rij=_rij(amounts), r_sums=True)
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
    return Units("table", cost[rows], m, totals, tuple(str(n) for n in spec["name"]), x, rows, all_rij=full)


# ----------------------------------------------------------------------------- manual bounds

def _check_bounds_table(data, value_cols: tuple[str, ...]) -> pd.DataFrame:
    """prioritizr's checks of a manual bounded / locked table: ``pu`` whole numbers from 1 (units without
    cost are skipped), finite values, upper >= lower; ``zone`` only as a single zone."""
    d = pd.DataFrame(data).copy()
    what = "manual locked constraints" if value_cols == ("status",) else "manual bounded constraints"
    if len(d) == 0:
        raise ValueError(f"{what}: the table is empty")
    for c in ("pu", *value_cols):
        if c not in d.columns:
            raise ValueError(f"{what}: the table needs a {c!r} column")
        v = pd.to_numeric(d[c], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(v).all():
            raise ValueError(f"{what}: {c!r} must be finite")
        d[c] = v
    pu = d["pu"].to_numpy()
    if (pu != np.round(pu)).any() or pu.min() < 1:
        raise ValueError(f"{what}: 'pu' must hold whole numbers from 1 (1-based unit indices)")
    if value_cols == ("lower", "upper") and (d["upper"] < d["lower"]).any():
        raise ValueError(f"{what}: an upper bound is below its lower bound")
    if "zone" in d.columns:
        d["zone"] = d["zone"].astype(str)
    return d


def _bound_values(u, data, what: str) -> np.ndarray:
    """One value per planning unit (NaN kept): a number, a raster on the units' grid, a column of their
    table, or values per planning unit or per cell / row."""
    if data is None:
        return np.full(u.n, np.nan)
    if np.ndim(data) == 0 and not isinstance(data, str):
        return np.full(u.n, float(data))
    total = u.template.size if u.kind == "raster" else len(u.template)
    if isinstance(data, str):
        if u.kind == "raster":
            raise TypeError(f"{what}: a column name needs table or polygon planning units")
        v = u.template[data].to_numpy(dtype=np.float64)
    elif isinstance(data, xr.DataArray):
        if u.kind != "raster":
            raise TypeError(f"{what}: a raster needs raster planning units")
        v = np.asarray(data.squeeze().values, dtype=np.float64)
        if v.shape != u.template.shape:
            raise ValueError(f"{what}: the raster must be on the planning units' grid")
        v = v.ravel()
    else:
        v = np.asarray(data, dtype=np.float64).ravel()
        if v.size == u.n:
            return v
    if v.size != total:
        raise ValueError(f"{what}: one value per planning unit ({u.n}) or per unit ({total}), got {v.size}")
    return v[u.index]


# ----------------------------------------------------------------------------- problem

@dataclass(frozen=True)
class Problem:
    """A conservation planning problem (prioritizr's ``problem()``): see :func:`problem`.

    Every ``add_*_objective`` that takes a ``budget`` takes one number (the most the selected units may
    cost); with zones, also one budget per zone (a list in the zones' order, or a mapping zone ->
    budget).
    """

    units: Units = field(repr=False)
    objective: tuple | None = None
    targets: tuple | None = None
    locked_in: object = None
    locked_out: object = None
    bounds: tuple = ()  # manual bounded and locked constraints, ("bounded" | "locked", data) in the order added
    decisions: tuple = ("binary",)
    solver: tuple = ("highs", {})
    weights: object = None
    subregional: tuple | None = None
    spatial: tuple = ()  # ("penalty" | "constraint", kind, ...) in the order added
    portfolio: tuple | None = None  # (kind, options)
    # management zones: targets of given zones, ((zone, ...), kind, per-feature values) in the order added
    zone_targets: tuple = ()
    mandatory: bool = False  # every planning unit goes to a zone (add_mandatory_allocation_constraints)

    @property
    def zones(self) -> tuple[str, ...]:
        """The names of the management zones (empty without zones)."""
        return self.units.zone_names

    def __repr__(self) -> str:
        u = self.units
        zones = f", {u.nz} zones {list(u.zone_names)}" if u.zoned else ""
        return (f"Problem({u.kind}, {u.n:,} planning units{zones}, {len(u.feature_names)} features; "
                f"objective={self.objective and self.objective[0]}, targets={self.targets and self.targets[0]}, "
                f"decisions={self.decisions[0]}, solver={self.solver[0]})")

    def _with(self, **kw) -> Problem:
        return dataclasses.replace(self, **kw)

    def _budget(self, budget):
        return _budget(budget, self.units)

    # --- objectives (prioritizr's; a budget is the most the selected units may cost, with zones one or one per zone)
    def add_min_set_objective(self) -> Problem:
        """Minimum set: meet every target at the least cost."""
        return self._with(objective=("min_set",))

    def add_min_shortfall_objective(self, budget: float) -> Problem:
        """Minimum shortfall: within the budget, the least sum of the targets' shortfalls, each as a
        share of its target (weighted by :meth:`add_feature_weights`)."""
        return self._with(objective=("min_shortfall", self._budget(budget)))

    def add_min_largest_shortfall_objective(self, budget: float) -> Problem:
        """Minimum largest shortfall: within the budget, the least of the largest shortfall of any
        target, as a share of the target."""
        return self._with(objective=("min_largest_shortfall", self._budget(budget)))

    def add_max_utility_objective(self, budget: float) -> Problem:
        """Maximum utility: within the budget, the most of the features' amounts (weighted); no
        targets."""
        return self._with(objective=("max_utility", self._budget(budget)))

    def add_max_features_objective(self, budget: float) -> Problem:
        """Maximum features: within the budget, the most targets met (weighted)."""
        return self._with(objective=("max_features", self._budget(budget)))

    def add_max_wtd_sum_objective(self, budget: float) -> Problem:
        """:meth:`add_max_utility_objective` under prioritizr 9's name (the model is 8.0.6's, with the
        small cost term that breaks ties, which prioritizr 9 dropped)."""
        return self.add_max_utility_objective(budget)

    def add_max_n_targets_met_objective(self, budget: float) -> Problem:
        """:meth:`add_max_features_objective` under prioritizr 9's name (the model is 8.0.6's, with the
        small cost term that breaks ties, which prioritizr 9 dropped)."""
        return self.add_max_features_objective(budget)

    def add_max_cover_objective(self, budget: float) -> Problem:
        """Maximum cover: within the budget, the most features held at all (any amount above zero,
        weighted); no targets."""
        return self._with(objective=("max_cover", self._budget(budget)))

    def add_max_phylo_div_objective(self, budget: float, tree) -> Problem:
        """Maximum phylogenetic diversity: within the budget, the most total length of the branches
        of a phylogeny with a target met below them (Faith 1992). ``tree``: an ``ls.bio`` tree
        (:func:`landschaft.bio.read_tree`) or a Newick string, whose tips are the features."""
        return self._with(objective=("max_phylo_div", self._budget(budget), tree))

    def add_max_phylo_end_objective(self, budget: float, tree) -> Problem:
        """Maximum phylogenetic endemism: as :meth:`add_max_phylo_div_objective`, every branch's
        length divided by the total amount of the features below it (Rosauer et al. 2009)."""
        return self._with(objective=("max_phylo_end", self._budget(budget), tree))

    def add_min_extinction_risk_objective(self, budget: float, z=0.25, segments: int = 16) -> Problem:
        """Minimum extinction risk (our extension): within the budget, the least sum over the features
        (weighted) of the risk the species-area relationship gives, 1 - (held / total)^z (Pereira &
        Daily 2006; z = 0.25 by default, one value or one per feature), as a piecewise-linear function
        through ``segments`` + 1 points of the curve (exact at them, slightly above in between). No
        targets."""
        z = np.asarray(z, dtype=np.float64)
        if (z <= 0).any() or (z > 1).any() or not np.isfinite(z).all():
            raise ValueError("z must be in (0, 1]")
        if int(segments) != segments or segments < 1:
            raise ValueError("segments must be a whole number >= 1")
        if z.size not in (1, len(self.units.feature_names)):
            raise ValueError("z: one value or one per feature")
        return self._with(objective=("min_extinction_risk", self._budget(budget), z.copy(), int(segments)))

    def add_min_penalties_objective(self, budget: float) -> Problem:
        """Minimum penalties: meet the targets within the budget with the least penalties (the
        penalties come in a later phase: until then any solution that meets them is optimal)."""
        return self._with(objective=("min_penalties", self._budget(budget)))

    # --- targets
    def add_relative_targets(self, targets, zone=None) -> Problem:
        """Targets as shares (0-1) of every feature's total amount (one value or one per feature).

        With zones, the targets are met by the sum over the zones of the features' amounts in them,
        each counted by its zone's contribution: a share of the feature's total (the base features
        of the contributions; with every zone's own features, the sum of its totals in the zones, as
        prioritizr). ``zone``: targets of one zone (shares of the feature's total in that zone) or of a
        list of zones, added to the others (targets of the same zones replace the earlier ones). A
        matrix (features x zones) gives every zone its own targets (prioritizr's form).
        """
        return self._targets("relative", targets, zone)

    def add_absolute_targets(self, targets, zone=None) -> Problem:
        """Targets as amounts of every feature (one value or one per feature); ``zone`` as in
        :meth:`add_relative_targets`."""
        return self._targets("absolute", targets, zone)

    def _targets(self, kind: str, targets, zone) -> Problem:
        if zone is None:
            return self._with(targets=(kind, self._per_feature(targets)))
        from ._zones import zone_key

        key = zone_key(self.units, zone)
        a = np.asarray(targets, dtype=np.float64)
        if a.ndim > 1:
            raise ValueError("targets of given zones: one value or one per feature")
        new = (key, kind, self._per_feature(a))
        keys = [t[0] for t in self.zone_targets]
        if key in keys:  # the same zones: replaced where they were
            k = keys.index(key)
            return self._with(zone_targets=(*self.zone_targets[:k], new, *self.zone_targets[k + 1:]))
        return self._with(zone_targets=(*self.zone_targets, new))

    def add_manual_targets(self, targets: pd.DataFrame) -> Problem:
        """Targets from a table, one row per target: ``feature`` (name), ``type`` (``"relative"`` or
        ``"absolute"``), ``target`` and, optionally, ``sense`` (``">="``, the default, ``"<="`` or
        ``"="``). A feature may have several targets, or none. With zones, an optional ``zone``
        column: a zone or a list of zones whose amounts the target adds up (default: every zone; a
        relative target as in :meth:`add_relative_targets`)."""
        t = pd.DataFrame(targets)
        for col in ("feature", "type", "target"):
            if col not in t.columns:
                raise ValueError(f"the targets table needs a {col!r} column")
        if "zone" in t.columns:
            from ._zones import zone_key

            t["zone"] = [zone_key(self.units, z) for z in t["zone"]]
        extra = set(t.columns) - {"feature", "type", "target", "sense", "zone"}
        if extra:
            raise ValueError(f"unknown columns in the targets table: {sorted(extra)}")
        if not np.isfinite(t["target"].to_numpy(dtype=np.float64)).all():
            raise ValueError("targets must be finite")
        return self._with(targets=("manual", t.reset_index(drop=True).copy()))

    def add_subregional_targets(self, regions, targets: pd.DataFrame) -> Problem:
        """Targets within sub-regions (an extension: prioritizr has none), added to the other
        targets: one row per target, ``region``, ``feature`` (name), ``target``, optionally ``type``
        (``"relative"``, the default, or ``"absolute"``) and ``sense`` (``">="``, ``"<="``, ``"="``).
        A relative target is a share of the feature's total in the region, over every unit with or
        without a cost, as the other relative targets are. With zones, an optional ``zone`` column (a
        zone or a list of zones, default every zone), as in :meth:`add_relative_targets`.

        Parameters
        ----------
        regions
            The region of every unit: with raster units, a raster on their grid (NaN: no region);
            with tables and polygons, a column name or one value per row (missing: no region).
        targets
            The targets (a DataFrame or a dict of columns).
        """
        t = pd.DataFrame(targets)
        for col in ("region", "feature", "target"):
            if col not in t.columns:
                raise ValueError(f"the targets table needs a {col!r} column")
        if "zone" in t.columns:
            from ._zones import zone_key

            for z in t["zone"]:
                zone_key(self.units, z)
        extra = set(t.columns) - {"region", "feature", "type", "target", "sense", "zone"}
        if extra:
            raise ValueError(f"unknown columns in the targets table: {sorted(extra)}")
        if not np.isfinite(t["target"].to_numpy(dtype=np.float64)).all():
            raise ValueError("targets must be finite")
        return self._with(subregional=(regions, t.reset_index(drop=True).copy()))

    def add_loglinear_targets(self, lower_bound_amount: float, lower_bound_target: float,
                              upper_bound_amount: float, upper_bound_target: float,
                              cap_amount: float | None = None, cap_target: float | None = None,
                              abundances=None) -> Problem:
        """Targets that fall log-linearly with a feature's total amount (Rodrigues et al. 2004):
        ``lower_bound_target`` (a share) up to ``lower_bound_amount``, ``upper_bound_target`` from
        ``upper_bound_amount``, linear in log10(amount) between them; with a cap, features of at
        least ``cap_amount`` get the absolute target ``cap_target``. ``abundances``: the amounts
        (default: every feature's total)."""
        for v in (lower_bound_amount, upper_bound_amount):
            if not (np.isfinite(v) and v >= 0):
                raise ValueError("the bound amounts must be finite and >= 0")
        for v in (lower_bound_target, upper_bound_target):
            if not 0 <= v <= 1:
                raise ValueError("the bound targets must be from 0 to 1")
        if lower_bound_amount > upper_bound_amount:
            raise ValueError("the lower bound amount must not exceed the upper bound amount")
        if (cap_amount is None) != (cap_target is None):
            raise ValueError("give both cap_amount and cap_target, or neither")
        a = None if abundances is None else self._per_feature(abundances)
        if a is not None and (a < 0).any():
            raise ValueError("abundances must be >= 0")
        return self._with(targets=("loglinear", (float(lower_bound_amount), float(lower_bound_target),
                                                 float(upper_bound_amount), float(upper_bound_target),
                                                 cap_amount, cap_target, a)))

    # --- penalties
    def add_feature_weights(self, weights) -> Problem:
        """Weights of the features (of the targets, when there are targets) in the objectives that
        add them up: min shortfall, max utility, max features, max cover (ignored, with a warning,
        by min set and min largest shortfall). With zones and no targets (max utility, max cover): one
        weight per feature (the same in every zone) or a features x zones matrix."""
        w = np.asarray(weights, dtype=np.float64)
        w = w if (w.ndim == 2 and self.units.zoned and w.shape[1] > 1) else w.ravel()
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weights must be finite and >= 0")
        return self._with(weights=w)

    def _per_feature(self, v) -> np.ndarray:
        n = len(self.units.feature_names)
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 2 and self.units.zoned:  # features x zones: every zone's own targets (prioritizr's matrix)
            if a.shape != (n, self.units.nz):
                raise ValueError(f"a matrix of targets must be features x zones ({n} x {self.units.nz})")
            return a.copy()
        a = a.ravel()
        if a.size == 1:
            a = np.repeat(a, n)
        if a.size != n:
            raise ValueError(f"expected one target or {n} (one per feature), got {a.size}")
        return a

    # --- spatial and linear penalties (prioritizr's names; a penalty is added to the objective, weighed
    # against it: a cost to minimise, a loss in the objectives that maximise)
    def _add(self, *spec) -> Problem:
        return self._with(spatial=(*self.spatial, spec))

    def add_boundary_penalties(self, penalty: float, edge_factor: float = 0.5, data=None) -> Problem:
        """Penalise the boundary of the selection (more compact solutions): ``penalty`` per unit of
        exposed boundary length. ``edge_factor``: the share of the boundary at the edge of the study
        area (not shared with another planning unit) that counts. ``data``: a boundary matrix of every
        unit (shared lengths, every unit's perimeter on the diagonal); by default from the planning
        units' raster grid or polygons."""
        if not 0 <= edge_factor <= 1:
            raise ValueError("edge_factor must be from 0 to 1")
        return self._add("penalty", "boundary", float(penalty), float(edge_factor), data)

    def add_connectivity_penalties(self, penalty: float, data) -> Problem:
        """Reward selecting connected units: ``penalty`` x the connectivity of every pair selected
        together (a symmetric matrix of every unit; the diagonal rewards a unit alone)."""
        return self._add("penalty", "connectivity", float(penalty), data)

    def add_asym_connectivity_penalties(self, penalty: float, data) -> Problem:
        """Penalise asymmetric connections that the selection breaks: ``penalty`` x the connectivity
        from a selected unit to one not selected (a matrix of every unit, rows the sources)."""
        return self._add("penalty", "asym_connectivity", float(penalty), data)

    def add_linear_penalties(self, penalty: float, data) -> Problem:
        """Penalise selecting units by a value of each (``penalty`` x value): a raster on the units'
        grid, a column name, or one value per unit."""
        return self._add("penalty", "linear", float(penalty), data)

    def add_linear_constraints(self, threshold: float, sense: str, data) -> Problem:
        """The sum over the selected units of a value of each (as :meth:`add_linear_penalties`) must
        be ``>=``, ``<=`` or ``=`` the ``threshold``."""
        if sense not in (">=", "<=", "="):
            raise ValueError("sense must be '>=', '<=' or '='")
        return self._add("constraint", "linear", float(threshold), sense, data)

    def add_contiguity_constraints(self, data=None) -> Problem:
        """The selected units must form one contiguous patch. ``data``: an adjacency matrix of every
        unit; by default units that share an edge (raster cells, 4 directions) or that touch (polygons).

        The model is exact: a spanning arborescence of the selected units, with a flow that rules out
        cycles. prioritizr's formulation (a tree whose edges always run from a unit to one of a lower
        index) is a restriction: it misses patches in which a unit has no selected neighbour of a higher
        index, so it can return a costlier patch than the best, or call a feasible problem infeasible.
        On HiGHS the problem is solved in three steps: a first patch from that restriction, a bound from
        directed cuts, then the exact MIP from the first patch. Contiguity stays the hardest spatial
        rule: give it a time limit on more than a few hundred units."""
        return self._add("constraint", "contiguity", data)

    def add_neighbors_constraints(self, k: int, clamp: bool = True, data=None) -> Problem:
        """Every selected unit must have at least ``k`` selected neighbours (``clamp``: at most as many
        as it has). ``data``: an adjacency matrix of every unit, by default as
        :meth:`add_contiguity_constraints`."""
        if int(k) != k or k < 0:
            raise ValueError("k must be a whole number >= 0")
        return self._add("constraint", "neighbors", int(k), bool(clamp), data)

    def add_neighbor_constraints(self, k: int, clamp: bool = True, data=None) -> Problem:
        """:meth:`add_neighbors_constraints` under the name of prioritizr's newer versions."""
        return self.add_neighbors_constraints(k, clamp, data)

    def add_feature_contiguity_constraints(self, data=None) -> Problem:
        """Every feature's units (where it is counted towards its targets) must form one contiguous
        patch (the expanded formulation: a column per feature and unit; the exact model of
        :meth:`add_contiguity_constraints` for every feature). ``data``: an adjacency matrix of every
        unit for all features, a list of one per feature, or None (as
        :meth:`add_contiguity_constraints`)."""
        return self._add("constraint", "feature_contiguity", data)

    def add_corridor_constraints(self, terminals, resistance=None, penalty: float = 1.0, data=None) -> Problem:
        """Corridors (an extension, after prioripy): the terminal units of every network are selected
        and joined by selected units (a flow from the first terminal to each of the others).

        Parameters
        ----------
        terminals
            The terminals of one network (a raster on the units' grid, a column name or one value per
            unit; > 0 marks a terminal), or a list of networks.
        resistance
            Optional: values per unit (as ``terminals``; an edge gets the mean of its two units) or an
            ``ls.cost`` :class:`~landschaft.cost.Transition` (raster units; its connections are the
            edges, each with the reciprocal of its conductance). The flow's ``penalty`` x resistance
            is added to the objective, so corridors run through low resistance.
        data
            An adjacency matrix of every unit (default: as :meth:`add_contiguity_constraints`).
        """
        return self._add("constraint", "corridors", terminals, resistance, float(penalty), data)

    # --- constraints
    def add_locked_in_constraints(self, locked) -> Problem:
        """Planning units that must be selected: a raster (1 = locked), a column name, a boolean
        array over the planning units or their (0-based) indices. With zones, a mapping zone -> any of
        these: the units locked into each zone (a unit in one zone at most, and only where it has a
        cost)."""
        return self._with(locked_in=locked)

    def add_locked_out_constraints(self, locked) -> Problem:
        """Planning units that must not be selected (as :meth:`add_locked_in_constraints`; with zones,
        a mapping zone -> the units kept out of that zone)."""
        return self._with(locked_out=locked)

    def add_mandatory_allocation_constraints(self) -> Problem:
        """With zones: every planning unit goes to a zone (by default a unit may go to none)."""
        if not self.units.zoned:
            raise ValueError("mandatory allocation needs a problem with zones")
        return self._with(mandatory=True)

    def add_manual_bounded_constraints(self, data: pd.DataFrame | None = None, *, lower=None,
                                       upper=None, zone=None) -> Problem:
        """The selected share of given planning units must lie between a lower and an upper bound
        (prioritizr's ``add_manual_bounded_constraints``; prioripy's ``add_continuous_bounds``).

        Parameters
        ----------
        data
            prioritizr's table: a ``pu`` column (the 1-based index of the unit among *all* units: the
            cell of the raster, row-major, or the row of the table, with or without cost; units
            without cost are skipped), ``lower`` and ``upper``. A unit listed twice keeps its last
            bounds.
        lower, upper
            The same as values instead of a table: a number for every planning unit, a raster on the
            planning units' grid, a column of their table, or one value per planning unit (or per
            cell or row). A unit with NaN in both is left alone; a NaN (or a side not given) takes the
            decisions' own bound (0, or 1 or the semicontinuous limit).
        zone
            With zones: the zone of the values (the table takes a ``zone`` column instead). A unit
            without a cost in a zone stays out of it (a lower bound above 0 there is an error).

        Notes
        -----
        The bounds overwrite the units' bounds in the order the manual constraints were added, after
        :meth:`add_locked_in_constraints` and :meth:`add_locked_out_constraints`. They must lie within
        the decisions' bounds, and with binary decisions be 0 or 1 (prioritizr accepts a share there and
        the solver rounds it): both raise an error when the problem is compiled.
        """
        if data is None and lower is None and upper is None:
            raise ValueError("give a table (data) or lower= / upper=")
        if data is not None and (lower is not None or upper is not None):
            raise ValueError("give either a table (data) or lower= / upper=, not both")
        spec = _check_bounds_table(data, ("lower", "upper")) if data is not None else (lower, upper)
        return self._with(bounds=(*self.bounds, ("bounded", self._zone_spec(spec, zone, data is not None))))

    def add_manual_locked_constraints(self, data: pd.DataFrame | None = None, *, status=None,
                                      zone=None) -> Problem:
        """Fix the selected share of given planning units (prioritizr's ``add_manual_locked_constraints``):
        a ``status`` of 1 locks a unit in, 0 out, a share in between (proportion decisions) fixes it.

        Parameters
        ----------
        data
            prioritizr's table: ``pu`` (as in :meth:`add_manual_bounded_constraints`) and ``status``.
        status
            The same as values: a number, a raster on the planning units' grid, a column of their table
            or one value per planning unit; NaN leaves a unit alone.
        zone
            With zones: as in :meth:`add_manual_bounded_constraints`.
        """
        if (data is None) == (status is None):
            raise ValueError("give either a table (data) or status=")
        spec = _check_bounds_table(data, ("status",)) if data is not None else (status, status)
        return self._with(bounds=(*self.bounds, ("locked", self._zone_spec(spec, zone, data is not None))))

    def _zone_spec(self, spec, zone, table: bool):
        """A manual bound's specification with its zone: with zones, the table's zone column as zone
        indices, or ``(values, zone index)``."""
        u = self.units
        if not u.zoned:
            if zone is not None or (table and "zone" in spec.columns and spec["zone"].nunique() > 1):
                raise ValueError("zones are given but the problem has none")
            return spec
        from ._zones import zone_key

        if table:
            if zone is not None:
                raise ValueError("give the zones in the table's zone column")
            if "zone" not in spec.columns:
                raise ValueError("with zones, the table needs a zone column")
            spec = spec.copy()
            spec["zone"] = [zone_key(u, z, single=True)[0] for z in spec["zone"]]
            return spec
        if zone is None:
            raise ValueError("with zones, give the zone of the values (zone=)")
        return (spec, zone_key(u, zone, single=True)[0])

    def _manual_bounds(self, lb: np.ndarray, ub: np.ndarray, binary: bool, limit: float) -> None:
        """Apply the manual bounded and locked constraints to the units' bounds (in place; with zones,
        the bounds of every unit and zone, zone after zone)."""
        u = self.units
        n = u.n
        for kind, spec in self.bounds:
            what = f"manual {kind} constraints"
            zcol = None
            if u.zoned and not isinstance(spec, pd.DataFrame):
                spec, zcol = spec
            if isinstance(spec, pd.DataFrame):
                total = u.template.size if u.kind == "raster" else len(u.template)
                pu = spec["pu"].to_numpy(dtype=np.float64)
                if pu.max(initial=0) > total:
                    raise ValueError(f"{what}: pu goes up to {int(pu.max())}, but there are {total} units")
                pos = np.full(total + 1, -1, dtype=np.int64)  # 1-based unit -> planning unit (prioritizr's match)
                pos[u.index + 1] = np.arange(u.n)
                j = pos[pu.astype(np.int64)]
                keep = j >= 0
                j = j[keep]
                cols = ("status", "status") if kind == "locked" else ("lower", "upper")
                lo = spec[cols[0]].to_numpy(dtype=np.float64)[keep]
                hi = spec[cols[1]].to_numpy(dtype=np.float64)[keep]
                if u.zoned:  # the column of the unit in its zone
                    j = spec["zone"].to_numpy(dtype=np.int64)[keep] * n + j
            else:
                a = _bound_values(u, spec[0], what)
                b = a if kind == "locked" else _bound_values(u, spec[1], what)
                keep = ~(np.isnan(a) & np.isnan(b))
                j = np.flatnonzero(keep)
                lo = np.where(np.isnan(a[keep]), 0.0, a[keep])
                hi = np.where(np.isnan(b[keep]), limit, b[keep])
                if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                    raise ValueError(f"{what}: the bounds must be finite")
                if (hi < lo).any():
                    raise ValueError(f"{what}: an upper bound is below its lower bound")
                if zcol is not None:
                    j = zcol * n + j
            if lo.size and (lo.min() < 0 or hi.max() > limit):
                raise ValueError(f"{what}: the bounds must lie within the decisions' bounds [0, {limit:g}]")
            if binary and lo.size and not (np.isin(lo, (0.0, 1.0)).all() and np.isin(hi, (0.0, 1.0)).all()):
                raise ValueError(f"{what}: with binary decisions the bounds must be 0 or 1 (use proportion "
                                 "decisions for shares of units)")
            # prioritizr's loop: in table order, a unit listed again keeps its last bounds
            lb[j] = lo
            ub[j] = hi

    def _unit_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The planning units' bounds as compiled: the decisions', the locked units', the manual ones
        (with zones: of every unit and zone, zone after zone; a unit without a cost in a zone stays out
        of it)."""
        u = self.units
        ncol = u.cost.size
        limit = 1.0 if self.decisions[0] != "semicontinuous" else float(self.decisions[1])
        lb, ub = np.zeros(ncol), np.full(ncol, limit)
        li, lo = self._mask(self.locked_in), self._mask(self.locked_out)
        if (li & lo).any():
            raise ValueError(f"{int((li & lo).sum())} planning units are both locked in and locked out")
        lb[li] = 1.0
        ub[lo] = 0.0
        if self.bounds:
            self._manual_bounds(lb, ub, self.decisions[0] == "binary", limit)
        closed = ~np.isfinite(u.cost)
        if closed.any():
            bad = closed & (lb > 0)
            if bad.any():
                raise ValueError(f"{int(bad.sum())} planning units are locked or bounded into zones where they "
                                 "have no cost")
            ub[closed] = 0.0
        return lb, ub

    def _mask(self, locked) -> np.ndarray:
        u = self.units
        if u.zoned:
            from ._zones import zone_mask

            return zone_mask(self, locked)
        return unit_mask(u, locked)

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

    # --- portfolios (several solutions)
    def add_cuts_portfolio(self, number_solutions: int = 10) -> Problem:
        """The best ``number_solutions`` distinct solutions, one solve after another, each forbidding
        the earlier ones (binary decisions)."""
        from ._milp_portfolio import check_portfolio

        return self._with(portfolio=check_portfolio("cuts", number_solutions))

    def add_top_portfolio(self, number_solutions: int = 10) -> Problem:
        """As :meth:`add_cuts_portfolio`, every solve to optimality (gap 0)."""
        from ._milp_portfolio import check_portfolio

        return self._with(portfolio=check_portfolio("top", number_solutions))

    def add_gap_portfolio(self, number_solutions: int = 10, pool_gap: float = 0.1) -> Problem:
        """Up to ``number_solutions`` distinct solutions whose objective is within ``pool_gap`` (relative)
        of the optimum's."""
        from ._milp_portfolio import check_portfolio

        return self._with(portfolio=check_portfolio("gap", number_solutions, pool_gap=float(pool_gap)))

    def add_shuffle_portfolio(self, number_solutions: int = 10, remove_duplicates: bool = True,
                              seed: int | None = None) -> Problem:
        """``number_solutions`` solves of the model with its planning units (and other columns) in random
        orders: different solutions of equal quality when there are ties (``remove_duplicates``: keep
        distinct ones)."""
        from ._milp_portfolio import check_portfolio

        return self._with(portfolio=check_portfolio("shuffle", number_solutions,
                                                    remove_duplicates=bool(remove_duplicates), seed=seed))

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

    def add_anneal_solver(self, num_reps: int = 10, num_iters: int | None = None, start_temp: float = -1.0,
                          cool_fac: float = 0.0, num_temp: int = 10_000, spf=(1.0, 2.0), itimp_type: int = 0,
                          time_limit: float | None = None, seed: int = 1, n_threads: int | None = None) -> Problem:
        """Solve on Marxan's simulated annealing (:func:`~landschaft.prioritize.marxan`'s engine), for
        minimum-set problems that are hard for a MILP solver: boundary, connectivity and linear penalties,
        targets, locked units and other ``">="`` rows on the planning units, binary decisions.

        The problem is Marxan's exactly (prioritizr's boundary penalty is Marxan's BLM, its edge factor
        Marxan's fixed boundaries); every run of annealing and iterative improvement is then repaired to
        meet every target and trimmed of the units the targets do not need, and the solution is the run
        with the lowest objective. Its ``bound`` is the linear relaxation without the boundary terms
        (a lower bound on the optimum) and ``status`` is ``"optimal"`` only when the gap closes,
        ``"heuristic"`` otherwise (``"time_limit"`` if the annealing was cut short).

        Parameters
        ----------
        num_reps
            Runs (in parallel), from different random reserves.
        num_iters
            Annealing iterations per run (default: 1000 per planning unit, at least a million); with a
            ``time_limit``, the runs take as many as fit (the schedule spread over their share of it).
        start_temp, cool_fac, num_temp
            Marxan's schedule: ``start_temp`` < 0 for Connolly's (the default) or the adaptive (-2)
            one, else the initial temperature and its cooling factor; ``num_temp`` temperature
            decreases.
        spf
            Every feature's penalty factor during the annealing (Marxan's SPF): a shortfall costs
            ``spf`` x the cost of meeting the target alone. Low values let the annealing roam and the
            repair meets the targets anyway; several values: the runs take them in turn (run ``r`` the
            ``r % len(spf)``-th), the best run wins.
        itimp_type
            Marxan's iterative improvement after the annealing (0 normal, 3 twice).
        time_limit
            Seconds for the runs: every run's annealing gets its share (the runs in waves of
            ``n_threads``) and cools over it, then improvement and repair.
        seed
            The same seed gives the same solution on any number of threads.
        n_threads
            Runs in parallel (default: all cores but one).
        """
        if num_reps < 1:
            raise ValueError("num_reps must be at least 1")
        if itimp_type not in (0, 3):
            raise ValueError("itimp_type must be 0 or 3")
        return self._with(solver=("anneal", {
            "num_reps": int(num_reps), "num_iters": None if num_iters is None else int(num_iters),
            "start_temp": float(start_temp), "cool_fac": float(cool_fac), "num_temp": int(num_temp),
            "spf": tuple(float(v) for v in np.atleast_1d(spf)), "itimp_type": int(itimp_type), "time_limit": time_limit, "seed": int(seed),
            "n_threads": n_threads}))

    # --- compile
    def compile(self) -> OptimizationProblem:
        """The model prioritizr's ``compile()`` gives for this problem."""
        from ._milp_compile import compile_problem

        return compile_problem(self)[0]

    # --- solve
    def solve(self, start=None):
        """Solve the problem on its solver (HiGHS unless another was added): a
        :class:`~landschaft.prioritize.Solution`.

        Parameters
        ----------
        start
            A solution to start from: one value per planning unit, or a
            Zonation ranking to build one from: ``"zonation"`` (Zonation's
            CAZ2 on the problem's features, with the cost raster if the units
            are cells), a :class:`~landschaft.prioritize.ZonationResult` or a
            rank raster. The units go in from the highest rank down until
            every target is met (locked-in units always, locked-out never);
            the solver then drops those not needed and goes on from there.
        """
        from ._milp_compile import compile_problem
        from ._milp_solve import make_solution, solve_model

        m, targets = compile_problem(self)
        zoned = self.units.zoned
        if zoned and self.solver[0] == "anneal":
            raise ValueError("the anneal solver does not support zones yet: use HiGHS or Gurobi")
        if zoned and self.portfolio is not None:
            raise ValueError("portfolios do not support zones yet")
        if self.portfolio is not None:
            if start is not None:
                raise ValueError("a start solution and a portfolio cannot be combined")
            if self.solver[0] == "anneal":
                raise ValueError("portfolios need the HiGHS or the Gurobi solver")
            from ._milp_portfolio import solve_portfolio

            return solve_portfolio(self, m, targets)
        if start is not None:
            start = self._start(start, m)
        r = solve_model(m, self.solver[0], self.solver[1], start, n_units=None if zoned else self.units.n,
                        engines=not zoned)
        return make_solution(self.units, m, r, targets)

    def _start(self, start, m: OptimizationProblem) -> np.ndarray:
        from ._zonation import ZonationResult

        if self.units.zoned and (isinstance(start, (str, ZonationResult)) or (
                isinstance(start, xr.DataArray) and start.ndim >= 2)):
            raise ValueError("with zones, the start solution is one value per planning unit and zone")
        if isinstance(start, str):
            if start != "zonation":
                raise ValueError(f"unknown start {start!r} (a solution, 'zonation' or a Zonation result)")
            start = _zonation_rank(self.units)
        if isinstance(start, ZonationResult):
            start = start.rank
        if isinstance(start, xr.DataArray) and start.ndim >= 2:
            return _start_from_rank(m, _unit_ranks(self.units, start))
        start = np.asarray(start, dtype=np.float64)
        if self.units.zoned and start.ndim == 2:  # units x zones -> the columns, zone after zone
            start = start.T
        start = start.ravel()
        n = self.units.cost.size
        if start.size == n and m.ncol > n:  # the other columns from the units: the solver completes them
            start = np.concatenate([start, np.zeros(m.ncol - n)])
        if start.size != m.ncol:
            raise ValueError(f"the start solution must have one value per planning unit ({self.units.n})"
                             + (f" and zone ({self.units.nz})" if self.units.zoned else ""))
        return start


def unit_mask(u: Units, locked) -> np.ndarray:
    """Locked planning units as a mask: a raster (non-zero), a column name, a boolean array over the
    planning units or their (0-based) indices."""
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


def _budget(budget, units=None):
    """A budget: one number, or with zones one per zone (a sequence in the zones' order or a mapping
    zone -> budget), the most the units selected in that zone may cost."""
    zoned = units is not None and units.zoned
    if isinstance(budget, Mapping):
        if not zoned:
            raise ValueError("budgets per zone need a problem with zones")
        from ._zones import zone_key

        b = np.full(units.nz, np.nan)
        for z, v in budget.items():
            b[zone_key(units, z, single=True)[0]] = float(v)
        if np.isnan(b).any():
            raise ValueError(f"give a budget for every zone ({list(units.zone_names)})")
    else:
        b = np.asarray(budget, dtype=np.float64).ravel()
    if b.size != 1 and not (zoned and b.size == units.nz):
        raise ValueError("one budget" + (f", or one per zone ({units.nz})" if zoned else " (problems have one zone)"))
    if not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError("the budget must be finite and > 0")
    return float(b[0]) if b.size == 1 else tuple(float(v) for v in b)


def _zonation_rank(u: Units) -> xr.DataArray:
    """Zonation's CAZ2 ranking of the problem's features (with the cost raster for cell units)."""
    from ._zonation import zonation

    if u.features is None:
        raise ValueError("a Zonation start needs feature rasters (raster or polygon planning units)")
    layers = [f for _, f in _layers(u.features)]
    return zonation(layers, cost=u.template if u.kind == "raster" else None).rank


def _unit_ranks(u: Units, rank: xr.DataArray) -> np.ndarray:
    """The rank of every planning unit: its cell's, or a polygon's mean over the cells it covers."""
    if rank.ndim == 3 and rank.shape[0] == 1:
        rank = rank[0]
    if rank.ndim != 2:
        raise ValueError("the rank raster must have one band")
    if u.kind == "raster":
        if rank.shape != u.template.shape:
            raise ValueError(f"the rank raster ({rank.shape}) is not on the planning units' grid "
                             f"({u.template.shape})")
        return np.asarray(rank.values, dtype=np.float64).ravel()[u.index]
    if not hasattr(u.template, "geometry"):
        raise ValueError("a rank raster needs raster or polygon planning units")
    geoms = u.template.iloc[u.index]
    crs = rank.rio.crs
    if geoms.crs is not None and crs is not None and geoms.crs != crs:
        from .._vector import to_crs

        geoms = to_crs(geoms, crs)
    v = np.asarray(rank.values, dtype=np.float64)
    ok = np.isfinite(v)
    stack = xr.DataArray(np.stack([np.where(ok, v, 0.0), ok.astype(np.float64)]), dims=("band", *rank.dims),
                         coords={**{d: rank[d] for d in rank.dims if d in rank.coords}, "band": ["rank", "n"]})
    if crs is not None:
        stack = stack.rio.write_crs(crs)
    from ..io import _transform

    stack = stack.rio.write_transform(_transform(rank))
    sums = exact_sums(stack, geoms.geometry.values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sums[:, 1] > 0, sums[:, 0] / sums[:, 1], np.nan)


def _start_from_rank(m: OptimizationProblem, rank: np.ndarray) -> np.ndarray:
    """Units from the highest rank down (ties: unit order) until every row is met; locked-in units
    first, locked-out never. Units without a rank come last."""
    if m.modelsense != "min" or any(s != ">=" for s in m.sense):
        raise ValueError("a start from a ranking needs a minimum set problem")
    x = np.where(m.lb > 0, m.lb, 0.0)
    act = m.A @ x
    need = np.flatnonzero(act < m.rhs)
    if need.size:
        r = np.where(np.isfinite(rank), rank, -np.inf)
        order = np.argsort(-r, kind="stable")
        order = order[(m.ub[order] > x[order])]
        # every row's amounts in rank order (positions in `order`): the first position that meets it
        gain = sp.csr_matrix(m.A[need][:, order].multiply((m.ub - x)[order]))
        gain.sort_indices()
        reach = 0
        for i, t in enumerate(m.rhs[need]):
            row = slice(gain.indptr[i], gain.indptr[i + 1])
            cum = np.cumsum(gain.data[row]) + act[need[i]]
            hit = np.flatnonzero(cum >= t - 1e-9 * max(1.0, abs(t)))
            if not hit.size:
                raise ValueError("the targets cannot be met by the planning units")
            reach = max(reach, int(gain.indices[row][hit[0]]))
        take = order[: reach + 1]
        x[take] = m.ub[take]
    return x


def problem(x, features=None, *, cost_column=None, rij=None, contribution=None, zones=None) -> Problem:
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
        planning units that has an ``id`` column (with zones, a ``zone``
        column too).
    contribution
        Management zones: every planning unit goes to at most one zone, each
        with its own cost (``x``: a mapping zone -> cost raster, a raster with
        a band per zone, or ``cost_column`` a mapping zone -> column). The
        features are given once, and ``contribution`` says how much of them
        counts in each zone: a mapping zone -> a number, one number per
        feature, or a raster (a column with tables) of factors per unit
        (zones not listed count fully). Instead, ``features`` may be a
        mapping zone -> that zone's own features (prioritizr's ``zones()``).
    zones
        With ``rij`` and a ``zone`` column: the zones table (``id``,
        ``name``), in the order of the cost columns.

    Examples
    --------
    >>> import landschaft as ls
    >>> cost = ls.load_raster(ls.prioritize.example_data("cost"))
    >>> features = ls.load_raster(ls.prioritize.example_data("features"))
    >>> p = (ls.prioritize.problem(cost, features)
    ...        .add_min_set_objective()
    ...        .add_relative_targets(0.3)
    ...        .add_binary_decisions())
    >>> sol = p.solve()
    >>> sol.status
    'optimal'
    >>> bool(sol.coverage["met"].all())
    True
    """
    from ._zones import wants_zones, zone_units

    if wants_zones(x, cost_column, contribution, zones) or isinstance(features, Mapping):
        return Problem(zone_units(x, features, cost_column=cost_column, rij=rij, contribution=contribution,
                                  zones=zones))
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
