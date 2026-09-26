# SPDX-License-Identifier: GPL-3.0-or-later
"""Focal connectivity of every node in its neighbourhood, as Makurhini's MK_Focal_nodes (Latorre-Cárdenas et al. 2023)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .graph import HabitatGraph, _as_list
from .nodes import AREA_UNITS, _factor, _threads, _vector_frame, gdal_areas
from .protconn import QUAD_SEGS


def _bbox_area(bounds: np.ndarray) -> float:
    """Area of the bounding box of geometries as sf computes it (st_bbox -> st_as_sfc -> st_area)."""
    import shapely

    x0, y0 = bounds[:, 0].min(), bounds[:, 1].min()
    x1, y1 = bounds[:, 2].max(), bounds[:, 3].max()
    ring = shapely.polygons([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    return float(gdal_areas([ring])[0])


def focal_nodes(nodes, *, distance, search_buffer, metric: str = "IIC", probability: float | None = None,
                pij_min: float | None = 0.01, id: str | None = None, attribute=None, area_unit: str = "ha",
                threshold: float | None = None, n_threads: int | None = None) -> pd.DataFrame:
    """IIC or PC of every node's neighbourhood and the node's importance in it (``MK_Focal_nodes``).

    For every node (the focal node), the nodes whose polygons intersect its
    buffer of ``search_buffer`` form a graph (distances between their points on
    surface, the landscape area the area of their bounding box); with three or
    more nodes its IIC or PC and the focal node's dIIC or dPC (and fractions) are
    those of Makurhini's ``MK_dPCIIC``.

    Parameters
    ----------
    nodes
        Polygons (GeoDataFrame or GeoSeries), in a projected CRS.
    distance
        Dispersal distance (or distances) of the links or of ``probability``.
    search_buffer
        Radius of the neighbourhood, one for all distances or one per distance.
    metric
        ``"IIC"`` (Makurhini's default) or ``"PC"``.
    probability
        Probability at ``distance`` for PC (default 0.5).
    pij_min
        Probability cut of PC (Makurhini's 0.01).
    id
        Column with the node ids (default 1..n).
    attribute
        Column name or values of the node attribute (default: the area in
        ``area_unit``).
    area_unit
        Unit of the areas and of the landscape areas (Makurhini: always ha).
    threshold
        Pairs farther apart than this are not connected (Makurhini's ``threshold``).
    n_threads
        OpenMP threads.

    Returns
    -------
    pandas.DataFrame
        One row per node and distance: ``id``, ``distance``, ``attribute``,
        ``n_nodes`` (the neighbourhood), ``EC(IIC)``, ``IIC``, ``dIIC``,
        ``dIICintra``, ``dIICflux``, ``dIICconnector`` (PC: the same with PC)
        and ``IComp`` = IIC × dIIC, the composite connectivity index. Nodes
        with fewer than three nodes around them have NaN.

    Notes
    -----
    Makurhini simplifies the focal polygon (mapshaper, 90 % of the vertices)
    before buffering it; landschaft buffers the polygon itself, which only
    changes the neighbourhood when a node lies at the very edge of the buffer.
    """
    import shapely

    if metric not in ("IIC", "PC"):
        raise ValueError(f"metric must be 'IIC' or 'PC', got {metric!r}")
    frame, ids = _vector_frame(nodes, id)
    geoms = frame.geometry.values
    n = len(geoms)
    factor = _factor(AREA_UNITS, area_unit, "area")
    if attribute is None:
        attr = gdal_areas(geoms) * factor
    elif isinstance(attribute, str):
        attr = frame[attribute].to_numpy(np.float64)
    else:
        attr = np.asarray(attribute, dtype=np.float64)
        if attr.shape != (n,):
            raise ValueError("attribute must have one value per node")
    distances = _as_list(distance)
    buffers = np.broadcast_to(np.asarray(search_buffer, dtype=np.float64), (len(distances),))
    if metric == "PC" and probability is None:
        probability = 0.5
    points = shapely.point_on_surface(geoms)
    xy = shapely.get_coordinates(points)
    threads = _threads(n_threads)
    tree = shapely.STRtree(geoms)
    # Neighbourhoods: every search buffer, within the largest (as Makurhini).
    around = {}
    for sb in np.unique(buffers):
        a, b = tree.query(shapely.buffer(geoms, sb, quad_segs=QUAD_SEGS), predicate="intersects")
        around[sb] = (a, b)
    bounds = shapely.bounds(geoms)
    widest = around[buffers.max()]
    starts = {sb: np.searchsorted(a, np.arange(n + 1)) for sb, (a, _) in around.items()}
    cols = [f"EC({metric})", metric, f"d{metric}", f"d{metric}intra", f"d{metric}flux", f"d{metric}connector"]
    rows = []
    for k, (d, sb) in enumerate(zip(distances, buffers)):
        a, b = around[sb]
        st = starts[sb]
        if sb != buffers.max():
            wb = widest[1]
            wst = starts[buffers.max()]
        for i in range(n):
            members = np.sort(b[st[i]:st[i + 1]])
            if sb != buffers.max():
                members = np.intersect1d(members, wb[wst[i]:wst[i + 1]])
            row = {"id": ids[i], "distance": d, "attribute": attr[i], "n_nodes": members.size}
            if members.size <= 2:
                row.update(dict.fromkeys(cols, np.nan))
                row["IComp"] = np.nan
                rows.append(row)
                continue
            sub = xy[members]
            e = _core.conn_point_distances(np.ascontiguousarray(sub[:, 0]), np.ascontiguousarray(sub[:, 1]), np.inf,
                                           threads)
            m = members.size
            D = np.zeros((m, m))
            D[e["i"], e["j"]] = D[e["j"], e["i"]] = e["distance"]
            if threshold is not None:
                D[D > threshold] = np.nan
            la = _bbox_area(bounds[members]) * factor
            g = HabitatGraph(attr[members], D, landscape_area=la, n_threads=n_threads)
            overall = g.overall([metric], distance=d, probability=probability, pij_min=pij_min).iloc[0]
            focal = int(np.searchsorted(members, i))
            imp = g.node_importance([metric], distance=d, probability=probability, pij_min=pij_min).iloc[focal]
            row[f"EC({metric})"] = overall[f"EC({metric})"]
            row[metric] = overall[metric]
            for c in cols[2:]:
                row[c] = imp[c]
            row["IComp"] = row[metric] * row[f"d{metric}"]
            rows.append(row)
    return pd.DataFrame(rows)
