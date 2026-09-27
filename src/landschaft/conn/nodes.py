# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat-graph nodes and distances from rasters and polygons (Makurhini's nodesfile / distancefile)."""

from __future__ import annotations

from .._threads import threaded

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import _core

# Makurhini's unit_convert factors from square metres and metres.
AREA_UNITS = {"m2": 1.0, "km2": 0.000001, "ha": 0.0001, "Dam2": 0.01, "inch2": 1550.0, "foot2": 10.7639,
              "yard2": 1.19599, "mile2": 3.861e-7}
DISTANCE_UNITS = {"m": 1.0, "km": 0.001, "inch": 39.3701, "foot": 3.28084, "yard": 1.09361, "mile": 0.000621371}
DISTANCE_TYPES = ("centroid", "edge", "least-cost", "commute-time")
COST_TYPES = ("least-cost", "commute-time")
COST_FUNCTIONS = ("makurhini", "mean")
# Each least-cost thread keeps a cost (8 bytes) and a flag per cell.
_COST_MEMORY = 8e9


def _factor(table: dict, unit: str, what: str) -> float:
    if unit not in table:
        raise ValueError(f"unknown {what} unit {unit!r}; choose from {', '.join(table)}")
    return table[unit]


def _threads(n_threads) -> int:
    return 0 if n_threads is None else int(n_threads)


def _is_vector(nodes) -> bool:
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover - vector support is optional
        return False
    return isinstance(nodes, (gpd.GeoDataFrame, gpd.GeoSeries))


@dataclass
class RasterPatches:
    """A raster of patch ids read for a habitat graph."""

    values: np.ndarray        # int32 ids
    mask: np.ndarray          # True outside the patches
    res: tuple[float, float]
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax as terra reads them
    template: Any             # the DataArray, for rasters of node values
    nodes: dict | None = None  # raster_nodes result

    def read_nodes(self, n_threads=None, centroids: bool = False) -> dict:
        if self.nodes is None or (centroids and "cx" not in self.nodes):
            xmin, xmax, ymin, ymax = self.extent
            na = self.mask.view(np.uint8) if self.mask.any() else None
            self.nodes = _core.conn_raster_nodes(self.values, na, xmin, xmax, ymin, ymax, centroids, True,
                                                 _threads(n_threads))
        return self.nodes


def read_patches(nodes, *, habitat=None, directions: int = 8, nodata=None, band=None) -> RasterPatches:
    """A raster of patch ids, or the patches of the ``habitat`` classes of a map."""
    from ..lsm.sampling import _grid

    values, mask, res, transform = _grid(nodes, None, nodata, band)
    template = None
    if not isinstance(nodes, np.ndarray) and not np.ma.isMaskedArray(nodes):
        from ..lsm.landscape import _read

        template = _read(nodes, band) if isinstance(nodes, (str, bytes)) or hasattr(nodes, "__fspath__") else nodes
    if habitat is not None:
        from ..lsm import Landscape

        keep = np.isin(values, np.atleast_1d(habitat)) & ~mask
        if not keep.any():
            raise ValueError("no cell of the habitat classes")
        labels = np.asarray(Landscape(np.ma.masked_array(keep.astype(np.int32), mask=~keep), res=res,
                                      directions=directions).labels())
        values = labels.astype(np.int32)
        mask = ~keep
    ny, nx = values.shape
    # terra's extent: origin plus the cell counts times the cell sizes.
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + nx * transform.a
    ymin = ymax + ny * transform.e
    return RasterPatches(values, mask, res, (xmin, xmax, ymin, ymax), template)


def _raster_attributes(p: RasterPatches, attribute, weighted: bool, area_unit: str, n_threads) -> pd.DataFrame:
    r = p.read_nodes(n_threads)
    ids = r["ids"]
    xmin, xmax = p.extent[0], p.extent[1]
    res_x = (xmax - xmin) / p.values.shape[1]  # terra::res
    nres = (res_x**2) * _factor(AREA_UNITS, area_unit, "area")
    if attribute is None:
        values = r["cells"] * nres
    else:
        attribute = np.asarray(attribute, dtype=np.float64)
        if attribute.shape != ids.shape:
            raise ValueError(f"attribute must have one value per patch ({ids.size})")
        values = r["cells"] * nres * attribute if weighted else attribute
    return pd.DataFrame({"id": ids, "attribute": values})


@threaded
def gdal_areas(geometries) -> np.ndarray:
    """Areas as sf's st_area (GDAL's shoelace sums, in order) computes them.

    Points and lines have no area (0).
    """
    import shapely

    geoms = np.asarray(geometries)
    polygonal = np.isin(shapely.get_type_id(geoms), (3, 6))  # Polygon, MultiPolygon
    parts, part_geom = shapely.get_parts(np.where(polygonal, geoms, None), return_index=True)
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    coords, coord_ring = shapely.get_coordinates(rings, return_index=True)
    starts = lambda owner, n: np.searchsorted(owner, np.arange(n + 1)).astype(np.int64)
    return _core.conn_gdal_areas(np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1]),
                                 starts(coord_ring, len(rings)), starts(ring_part, len(parts)),
                                 starts(part_geom, len(geoms)))


def _vector_frame(nodes, id):
    import geopandas as gpd

    if isinstance(nodes, gpd.GeoSeries):
        nodes = gpd.GeoDataFrame(geometry=nodes)
    if id is None:
        ids = np.arange(1, len(nodes) + 1)
    else:
        if id not in nodes.columns:
            raise ValueError(f"no column {id!r} in the nodes")
        ids = nodes[id].to_numpy()
    if len(nodes) < 1:
        raise ValueError("a habitat graph needs at least one node")
    return nodes, ids


def node_table(nodes, *, attribute=None, weighted: bool = False, area_unit: str = "m2", id: str | None = None,
               habitat=None, directions: int = 8, nodata=None, band=None, n_threads=None) -> pd.DataFrame:
    """Node ids and attributes, as Makurhini's ``nodesfile``.

    Parameters
    ----------
    nodes
        A raster of patch ids (path, DataArray or array; NA or ``nodata`` =
        no habitat) or polygons / points (GeoDataFrame).
    attribute
        Raster: one value per patch (in the order of the sorted ids) used as
        the attribute, or multiplied by the patch area when ``weighted``.
        Polygons: a column name. By default the area.
    area_unit
        Unit of the areas: ``"m2"``, ``"ha"``, ``"km2"``... (Makurhini's
        factors; the map units must be metres).
    id
        Polygons: the column with the node ids (default ``1 .. n``).
    habitat, directions
        Raster: the classes whose patches (``directions``-connected) are the
        nodes, instead of a raster of patch ids.

    Returns
    -------
    pandas.DataFrame
        ``id`` and ``attribute``, one row per node.

    Examples
    --------
    Forest patches (classes 23-25) of the example map, areas in hectares:

    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    >>> nodes = ls.conn.node_table(landcover, habitat=[23, 24, 25], area_unit="ha")
    >>> len(nodes)
    158
    >>> nodes.head(2)
       id  attribute
    0   1     73.250
    1   2     89.125
    """
    if _is_vector(nodes):
        import shapely

        frame, ids = _vector_frame(nodes, id)
        if attribute is None:
            values = gdal_areas(frame.geometry.values) * _factor(AREA_UNITS, area_unit, "area")
        else:
            if isinstance(attribute, str):
                values = frame[attribute].to_numpy(np.float64)
            else:
                values = np.asarray(attribute, dtype=np.float64)
            if weighted:
                values = values * gdal_areas(frame.geometry.values) * _factor(AREA_UNITS, area_unit, "area")
        return pd.DataFrame({"id": ids, "attribute": values})
    patches = nodes if isinstance(nodes, RasterPatches) else read_patches(
        nodes, habitat=habitat, directions=directions, nodata=nodata, band=band)
    return _raster_attributes(patches, attribute, weighted, area_unit, n_threads)


def vector_centroids(geometries) -> np.ndarray:
    """Makurhini's st_centroid_within_poly: the centroid of the largest part,
    or a point on the surface when that centroid is not within the geometry."""
    import shapely

    geoms = np.asarray(geometries)
    parts, index = shapely.get_parts(geoms, return_index=True)
    largest = geoms.copy()
    multi = shapely.get_num_geometries(geoms) > 1
    if multi.any():
        area = shapely.area(parts)
        order = np.lexsort((-area, index))  # by geometry, largest first (first of equals)
        first = np.unique(index[order], return_index=True)[1]
        chosen = parts[order][first]
        largest[multi] = chosen[multi]
    points = shapely.centroid(largest)
    inside = shapely.within(points, geoms)
    points[~inside] = shapely.point_on_surface(geoms[~inside])
    return points


@dataclass
class Resistance:
    values: np.ndarray  # float64, NaN outside
    extent: tuple[float, float, float, float]


def read_resistance(resistance, *, band=None, nodata=None, positive: bool = True) -> Resistance:
    """A resistance raster (path, DataArray or array) as float64 with NaN outside."""
    import xarray as xr

    if isinstance(resistance, Resistance):
        return resistance
    if isinstance(resistance, (str, bytes)) or hasattr(resistance, "__fspath__"):
        from ..io import load_raster

        resistance = load_raster(resistance, band=band, validate=None)
    if isinstance(resistance, xr.DataArray):
        da = resistance.squeeze(drop=True) if "band" in resistance.dims else resistance
        if nodata is None:
            nodata = da.rio.nodata
        values = np.asarray(da.values, dtype=np.float64)
        try:
            t = da.rio.transform()
        except Exception:
            t = None
    else:
        values = np.asarray(np.ma.filled(np.ma.asarray(resistance, dtype=np.float64), np.nan), dtype=np.float64)
        t = None
    if values.ndim != 2:
        raise ValueError("the resistance must be a 2-D raster")
    values = values.copy()
    if nodata is not None and not np.isnan(nodata):
        values[values == nodata] = np.nan
    ny, nx = values.shape
    if t is None:  # as terra::rast(matrix): unit cells, origin at 0, first row on top
        xmin, xmax, ymin, ymax = 0.0, float(nx), 0.0, float(ny)
    else:
        if t.b != 0 or t.d != 0:
            raise ValueError("rotated rasters are not supported")
        xmin, ymax = t.c, t.f
        xmax, ymin = xmin + nx * t.a, ymax + ny * t.e
    finite = values[np.isfinite(values)]
    if positive and finite.size and finite.min() <= 0:
        raise ValueError("resistances must be positive")
    return Resistance(np.ascontiguousarray(values), (xmin, xmax, ymin, ymax))


def cells_of(x: np.ndarray, y: np.ndarray, shape: tuple[int, int], extent) -> np.ndarray:
    """raster::cellFromXY (0-based, -1 outside)."""
    ny, nx = shape
    xmin, xmax, ymin, ymax = extent
    yres_inv = ny / (ymax - ymin)
    xres_inv = nx / (xmax - xmin)
    row = np.floor((ymax - y) * yres_inv)
    row = np.where(y == ymin, ny - 1, row)
    col = np.floor((x - xmin) * xres_inv)
    col = np.where(x == xmax, nx - 1, col)
    outside = (row < 0) | (row >= ny) | (col < 0) | (col >= nx) | np.isnan(row) | np.isnan(col)
    return np.where(outside, -1, row * nx + col).astype(np.int64)


def distance_table(nodes, *, distance: str = "centroid", id: str | None = None, max_distance: float | None = None,
                   distance_unit: str = "m", habitat=None, directions: int = 8, nodata=None, band=None,
                   resistance=None, cost_directions: int = 8, cost_function: str = "makurhini",
                   unreachable: str = "euclidean", n_threads=None) -> pd.DataFrame:
    """Euclidean distances between nodes, as Makurhini's ``distancefile``.

    Parameters
    ----------
    nodes
        A raster of patch ids or polygons / points (see :func:`node_table`).
    distance
        ``"centroid"``: between centroids (rasters: the mean of the cell
        centres; polygons: the centroid of the largest part, or a point on
        the surface when it falls outside). ``"edge"``: between the closest
        edges (0 for touching patches). ``"least-cost"``: the cheapest path
        between the cells of the centroids over ``resistance``, as
        gdistance's ``costDistance`` does for Makurhini: the cost of a step
        between neighbouring cells (``cost_directions`` 4 or 8) is the step
        length (between cell centres) over the conductance, 1 / the
        resistance of the cell of larger index (``cost_function="makurhini"``,
        Makurhini's default) or 1 / the mean of both (``"mean"``). Distances are
        in cost units (resistance × map units); ``distance_unit`` does not apply.
        ``"commute-time"``: gdistance's ``commuteDistance`` over the same
        conductances (1 / the step cost): the expected time of a random walk
        from one node's cell to the other's and back, the effective resistance
        between them (as in Circuitscape) times the total conductance.
    max_distance
        Keep only the pairs up to this distance (in ``distance_unit``).
        Makurhini's ``threshold`` keeps them this way in pair tables.
    distance_unit
        ``"m"``, ``"km"``... (the map units must be metres).
    resistance
        The resistance raster of ``"least-cost"`` (path, DataArray or array;
        NA cells cannot be crossed).
    unreachable
        ``"least-cost"`` or ``"commute-time"`` pairs without a path: ``"euclidean"`` gives them the
        distance between their centroids, as Makurhini does, ``"drop"``
        leaves them unconnected.

    Returns
    -------
    pandas.DataFrame
        ``from``, ``to`` (node ids, ``from`` first in node order) and ``distance``.

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    >>> pairs = ls.conn.distance_table(landcover, habitat=[23, 24, 25], distance="edge",
    ...                                max_distance=1000)
    >>> len(pairs)
    126
    >>> pairs.head(2)
       from  to    distance
    0     3  13  989.002022
    1     4   5  206.155281
    """
    if distance not in DISTANCE_TYPES:
        raise ValueError(f"distance must be one of {DISTANCE_TYPES}, got {distance!r}")
    if distance in COST_TYPES:
        return _least_cost_table(nodes, id=id, max_distance=max_distance, habitat=habitat, directions=directions,
                                 nodata=nodata, band=band, resistance=resistance, cost_directions=cost_directions,
                                 cost_function=cost_function, unreachable=unreachable, n_threads=n_threads,
                                 kind=distance)
    factor = _factor(DISTANCE_UNITS, distance_unit, "distance")
    # The limit in metres, a hair above so the converted distance decides.
    limit = np.inf if max_distance is None else float(max_distance) / factor * (1 + 1e-12)
    threads = _threads(n_threads)
    if _is_vector(nodes):
        import shapely

        frame, ids = _vector_frame(nodes, id)
        geoms = frame.geometry.values
        if distance == "centroid":
            points = vector_centroids(geoms)
            r = _core.conn_point_distances(shapely.get_x(points), shapely.get_y(points), limit, threads)
        else:
            n = len(geoms)
            if np.isfinite(limit):
                tree = shapely.STRtree(geoms)
                a, b = tree.query(geoms, predicate="dwithin", distance=limit)
            else:
                a, b = np.triu_indices(n, 1)
            keep = a < b
            a, b = a[keep], b[keep]
            order = np.lexsort((b, a))
            a, b = a[order], b[order]
            r = {"i": a, "j": b, "distance": shapely.distance(geoms[a], geoms[b])}
    else:
        patches = nodes if isinstance(nodes, RasterPatches) else read_patches(
            nodes, habitat=habitat, directions=directions, nodata=nodata, band=band)
        nd = patches.read_nodes(threads, centroids=distance == "centroid")
        ids = nd["ids"]
        if distance == "centroid":
            r = _core.conn_point_distances(nd["cx"], nd["cy"], limit, threads)
        else:
            xmin, xmax, ymin, ymax = patches.extent
            ny, nx = patches.values.shape
            r = _core.conn_raster_edge_distances(nd["node"], len(ids), (xmax - xmin) / nx, (ymax - ymin) / ny, limit,
                                                 threads)
    d = np.asarray(r["distance"], dtype=np.float64) * factor
    table = pd.DataFrame({"from": np.asarray(ids)[r["i"]], "to": np.asarray(ids)[r["j"]], "distance": d})
    if max_distance is not None:
        table = table[table["distance"] <= max_distance].reset_index(drop=True)
    return table


def _node_points(nodes, *, id, habitat, directions, nodata, band, n_threads):
    """Node ids and centroid coordinates (Makurhini's)."""
    if _is_vector(nodes):
        import shapely

        frame, ids = _vector_frame(nodes, id)
        points = vector_centroids(frame.geometry.values)
        return np.asarray(ids), shapely.get_x(points), shapely.get_y(points)
    patches = nodes if isinstance(nodes, RasterPatches) else read_patches(
        nodes, habitat=habitat, directions=directions, nodata=nodata, band=band)
    nd = patches.read_nodes(n_threads, centroids=True)
    return nd["ids"], nd["cx"], nd["cy"]


def _commute_distances(res, cells: np.ndarray, directions: int, function: str):
    """Commute-time distances between the node cells (pairs i < j in the same component), and which nodes
    are on the conductance graph."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu

    xmin, xmax, ymin, ymax = res.extent
    e = _core.conn_conductance(res.values, xmin, xmax, ymin, ymax, directions, function)
    a, b, c = e["a"], e["b"], e["conductance"]
    empty = pd.DataFrame({"i": np.zeros(0, np.int64), "j": np.zeros(0, np.int64), "distance": np.zeros(0)})
    if a.size == 0:
        return empty, np.zeros(cells.size, dtype=bool)
    # Cells with a conductance (gdistance's transitionSolidify), numbered 0..m-1.
    used = np.unique(np.concatenate([a, b]))
    ia, ib = np.searchsorted(used, a), np.searchsorted(used, b)
    m = used.size
    volume = 2.0 * float(c.sum())  # sum(transitionMatrix): both triangles
    adjacency = sparse.coo_matrix((np.concatenate([c, c]), (np.concatenate([ia, ib]), np.concatenate([ib, ia]))),
                                  shape=(m, m)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - adjacency).tocsc()
    n_comp, label = connected_components(adjacency, directed=False)
    pos = np.searchsorted(used, cells)
    inside = (cells >= 0) & (pos < m) & (used[np.clip(pos, 0, m - 1)] == cells)
    vertex = np.where(inside, pos, -1)
    frames = []
    for comp in np.unique(label[vertex[inside]]):
        nodes = np.flatnonzero(inside & (label[np.clip(vertex, 0, None)] == comp))
        if nodes.size < 2:
            continue
        verts = np.flatnonzero(label == comp)
        targets = np.unique(vertex[nodes])  # distinct cells
        local = np.searchsorted(verts, targets)
        ground = local[-1]
        keep = np.ones(verts.size, dtype=bool)
        keep[ground] = False
        sub = laplacian[verts][:, verts][keep][:, keep].tocsc()
        # Row of each target in the grounded system (the ground has none).
        rows = np.searchsorted(np.flatnonzero(keep), local[:-1])
        k = rows.size
        G = np.zeros((k + 1, k + 1))  # Laplacian's inverse on the targets, the ground last (0)
        if k:
            lu = splu(sub, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
            block = max(1, min(64, int(2e7 // max(1, sub.shape[0]))))
            for s in range(0, k, block):
                cols = rows[s:s + block]
                rhs = np.zeros((sub.shape[0], cols.size))
                rhs[cols, np.arange(cols.size)] = 1.0
                G[:k, s:s + cols.size] = lu.solve(rhs)[rows]
        G = (G + G.T) / 2
        d = np.diag(G)
        R = (d[:, None] + d[None, :] - 2 * G) * volume
        np.fill_diagonal(R, 0.0)
        where = np.searchsorted(targets, vertex[nodes])
        i, j = np.triu_indices(nodes.size, 1)
        frames.append(pd.DataFrame({"i": nodes[i], "j": nodes[j], "distance": R[where[i], where[j]]}))
    if not frames:
        return empty, inside
    return pd.concat(frames).sort_values(["i", "j"], kind="stable").reset_index(drop=True), inside


def _least_cost_table(nodes, *, id, max_distance, habitat, directions, nodata, band, resistance, cost_directions,
                      cost_function, unreachable, n_threads, kind="least-cost") -> pd.DataFrame:
    if resistance is None:
        raise ValueError(f"{kind} distances need a resistance raster (resistance=)")
    if cost_function not in COST_FUNCTIONS:
        raise ValueError(f"cost_function must be one of {COST_FUNCTIONS}, got {cost_function!r}")
    if cost_directions not in (4, 8):
        raise ValueError("cost_directions must be 4 or 8")
    if unreachable not in ("euclidean", "drop"):
        raise ValueError("unreachable must be 'euclidean' or 'drop'")
    ids, x, y = _node_points(nodes, id=id, habitat=habitat, directions=directions, nodata=nodata, band=band,
                             n_threads=n_threads)
    res = read_resistance(resistance)
    cells = cells_of(x, y, res.values.shape, res.extent)
    limit = np.inf if max_distance is None else float(max_distance)
    xmin, xmax, ymin, ymax = res.extent
    if kind == "commute-time":
        table, on_graph = _commute_distances(res, cells, cost_directions, cost_function)
        table = table[table["distance"] <= limit]
    else:
        cap = max(1, int(_COST_MEMORY // (9 * res.values.size + 1)))
        r = _core.conn_least_cost(res.values, xmin, xmax, ymin, ymax, cells, cost_directions, cost_function, limit,
                                  _threads(n_threads), cap)
        table = pd.DataFrame({"i": r["i"], "j": r["j"], "distance": r["distance"]})
    if unreachable == "euclidean":
        # Makurhini: pairs with an infinite cost (no path at all: different
        # components of the resistance, or a node outside it or on an NA
        # cell) get their centroid distance. Pairs only beyond the limit do not.
        from scipy import ndimage

        structure = np.ones((3, 3), bool) if cost_directions == 8 else None
        labels = ndimage.label(np.isfinite(res.values), structure=structure)[0].ravel()
        component = np.where(cells >= 0, labels[np.clip(cells, 0, None)], 0)
        e = _core.conn_point_distances(np.ascontiguousarray(x), np.ascontiguousarray(y), limit, _threads(n_threads))
        ei, ej = e["i"].astype(np.int64), e["j"].astype(np.int64)
        cut_off = (component[ei] == 0) | (component[ei] != component[ej])
        if kind == "commute-time":
            # gdistance leaves out the nodes off the conductance graph (NA);
            # it fails on separate components, which get the centroid distance.
            cut_off &= on_graph[ei] & on_graph[ej]
        n = len(ids)
        found = np.isin(ei * n + ej, table["i"].to_numpy(np.int64) * n + table["j"].to_numpy(np.int64))
        fallback = cut_off & ~found
        if fallback.any():
            euclid = pd.DataFrame({"i": ei[fallback], "j": ej[fallback], "distance": e["distance"][fallback]})
            table = pd.concat([table, euclid]).sort_values(["i", "j"], kind="stable")
    out = pd.DataFrame({"from": ids[table["i"].to_numpy()], "to": ids[table["j"].to_numpy()],
                        "distance": table["distance"].to_numpy(np.float64)})
    if max_distance is not None:
        out = out[out["distance"] <= max_distance]
    return out.reset_index(drop=True)
