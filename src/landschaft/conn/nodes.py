# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat-graph nodes and distances from rasters and polygons (Makurhini's nodesfile / distancefile)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import _core

# Makurhini's unit_convert factors from square metres and metres.
AREA_UNITS = {"m2": 1.0, "km2": 0.000001, "ha": 0.0001, "Dam2": 0.01, "inch2": 1550.0, "foot2": 10.7639,
              "yard2": 1.19599, "mile2": 3.861e-7}
DISTANCE_UNITS = {"m": 1.0, "km": 0.001, "inch": 39.3701, "foot": 3.28084, "yard": 1.09361, "mile": 0.000621371}
DISTANCE_TYPES = ("centroid", "edge")


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
    if len(nodes) < 2:
        raise ValueError("a habitat graph needs at least two nodes")
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


def distance_table(nodes, *, distance: str = "centroid", id: str | None = None, max_distance: float | None = None,
                   distance_unit: str = "m", habitat=None, directions: int = 8, nodata=None, band=None,
                   n_threads=None) -> pd.DataFrame:
    """Euclidean distances between nodes, as Makurhini's ``distancefile``.

    Parameters
    ----------
    nodes
        A raster of patch ids or polygons / points (see :func:`node_table`).
    distance
        ``"centroid"``: between centroids (rasters: the mean of the cell
        centres; polygons: the centroid of the largest part, or a point on
        the surface when it falls outside). ``"edge"``: between the closest
        edges (0 for touching patches).
    max_distance
        Keep only the pairs up to this distance (in ``distance_unit``).
        Makurhini's ``threshold`` keeps them this way in pair tables.
    distance_unit
        ``"m"``, ``"km"``... (the map units must be metres).

    Returns
    -------
    pandas.DataFrame
        ``from``, ``to`` (node ids, ``from`` first in node order) and ``distance``.
    """
    if distance not in DISTANCE_TYPES:
        raise ValueError(f"distance must be one of {DISTANCE_TYPES}, got {distance!r}")
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
