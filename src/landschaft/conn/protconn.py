# SPDX-License-Identifier: GPL-3.0-or-later
"""Protected Connected land (ProtConn) and its fractions, as Makurhini's MK_ProtConn (Saura et al. 2017, 2018)."""

from __future__ import annotations

from .._threads import threaded

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from .graph import HabitatGraph, _as_list, r_sum
from .nodes import AREA_UNITS, gdal_areas

# st_buffer's default: 30 segments per quarter circle.
QUAD_SEGS = 30

INDICATORS = ["ECA", "PC", "LA", "Protected_surface", "Prot", "ProtConn", "ProtUnconn", "RelConn",
              "ProtUnconn_Design", "ProtConn_Bound", "Unprotected", "ProtConn_Prot", "ProtConn_Trans",
              "ProtConn_Unprot", "ProtConn_Within", "ProtConn_Contig", "ProtConn_Within_land",
              "ProtConn_Contig_land", "ProtConn_Unprot_land", "ProtConn_Trans_land"]


@dataclass
class ProtConnNodes:
    """The nodes of a ProtConn analysis of one region (:func:`protconn_nodes`).

    Attributes
    ----------
    nodes : geopandas.GeoDataFrame
        ``type`` ("Non-Transboundary", "Transboundary" or "Region") and
        ``attribute`` (the area in the region; 0 for the others).
    pieces : numpy.ndarray
        Areas of the protected areas' pieces in the region before they are
        dissolved (ProtConn[Within]).
    clipped : geopandas.GeoDataFrame
        Every protected area cut to the region (the nodes of the importance of
        each area).
    la : float
        The area of the region.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> nodes = ls.conn.protconn_nodes(reserves, region)
    >>> len(nodes.nodes), len(nodes.pieces)
    (4, 4)
    """

    nodes: object
    pieces: np.ndarray
    clipped: object
    la: float


def _explode(geom) -> list:
    import shapely

    if geom is None or shapely.is_empty(geom):
        return []
    parts = shapely.get_parts(geom)
    return [p for p in parts if shapely.get_type_id(p) == 3 and not shapely.is_empty(p)]


def protconn_nodes(protected_areas, region, *, transboundary: float | None = None,
                   transboundary_type: str = "nodes", bound: bool = False, area_unit: str = "m2",
                   preselect: bool = True) -> ProtConnNodes:
    """Nodes of ProtConn in a region, as Makurhini's ``Protconn_nodes`` builds them.

    The protected areas that overlap the region are cut to it, dissolved and
    split into separate polygons (the nodes, with their area); with
    ``transboundary`` (a distance) the parts of those areas outside the
    region and the other protected areas within that distance of them
    (``transboundary_type="nodes"``) or of the region (``"region"``), cut to
    that zone, become transboundary nodes (attribute 0: stepping stones);
    ``bound`` adds the parts of the region itself (ProtConn[Bound]), as
    Makurhini does only when there are transboundary nodes.

    ``preselect=False`` uses all the given areas as candidates (the grid's
    selection, made once for its whole region).

    Makurhini simplifies the polygons of the zones with mapshaper (5 % of the
    vertices) before buffering them; landschaft buffers the exact polygons
    (with GEOS' 30 segments per quarter circle, as sf).

    Parameters
    ----------
    protected_areas
        Polygons of the protected areas (GeoDataFrame, projected in metres).
    region
        The region (one or more polygons, merged).
    transboundary, transboundary_type, bound, area_unit
        As in :func:`protconn`.
    preselect
        Only use the protected areas near the region.

    Returns
    -------
    ProtConnNodes

    See Also
    --------
    protconn_indicators : ProtConn from these nodes and their distances.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> nodes = ls.conn.protconn_nodes(reserves, region, area_unit="ha")
    >>> nodes.nodes["type"].tolist()
    ['Non-Transboundary', 'Non-Transboundary', 'Non-Transboundary', 'Non-Transboundary']
    >>> float(nodes.la)
    12100.0
    """
    import geopandas as gpd
    import shapely

    if transboundary_type not in ("nodes", "region"):
        raise ValueError("transboundary_type must be 'nodes' or 'region'")
    factor = AREA_UNITS[area_unit]
    crs = getattr(protected_areas, "crs", None)
    region_geoms = np.asarray(region.geometry.values if hasattr(region, "geometry") else region)
    region_geom = shapely.make_valid(shapely.union_all(region_geoms))
    la = float(r_sum(gdal_areas(region_geoms))) * factor
    pas = np.asarray(protected_areas.geometry.values if hasattr(protected_areas, "geometry") else protected_areas)
    pas = shapely.make_valid(pas)
    # input_grid: the protected areas within `transboundary` of the region
    # (MK_Connect_grid selects them once for the whole region: preselect=False).
    if preselect:
        zone = region_geom if not transboundary else shapely.buffer(region_geom, transboundary, quad_segs=QUAD_SEGS)
        y = pas[shapely.intersects(pas, zone)]
    else:
        y = pas
    inside = shapely.intersects(y, region_geom)
    y1, y2 = y[inside], y[~inside]
    empty = gpd.GeoDataFrame({"type": [], "attribute": []}, geometry=[], crs=crs)
    if len(y1) == 0:
        return ProtConnNodes(empty, np.zeros(0), empty, la)
    f1 = shapely.intersection(y1, region_geom)
    keep = ~shapely.is_empty(f1)
    f1 = f1[keep]
    clipped = gpd.GeoDataFrame({"attribute": gdal_areas(f1) * factor}, geometry=f1, crs=crs)
    pieces = [p for g in f1 for p in _explode(g)]
    f2 = _explode(shapely.buffer(shapely.union_all(f1), 0))
    types = ["Non-Transboundary"] * len(f2)
    geoms = list(f2)
    attributes = list(gdal_areas(np.array(f2, dtype=object)) * factor)
    if len(y2) > 0 and transboundary:
        f3 = shapely.difference(y1[keep], region_geom)
        if transboundary_type == "nodes":
            mask = shapely.buffer(shapely.union_all(y1), transboundary, quad_segs=QUAD_SEGS)
        else:
            mask = shapely.buffer(region_geom, transboundary, quad_segs=QUAD_SEGS)
        f4 = shapely.intersection(y2[shapely.intersects(y2, mask)], mask)
        parts = [g for g in list(f3) + list(f4) if not shapely.is_empty(g)]
        if parts:
            f5 = _explode(shapely.buffer(shapely.union_all(np.array(parts, dtype=object)), 0))
            geoms += f5
            types += ["Transboundary"] * len(f5)
            attributes += [0.0] * len(f5)
            # Makurhini adds the region itself only next to transboundary nodes
            # (without them ProtConn[Bound] is NA).
            if bound:
                region_parts = _explode(region_geom)
                geoms += region_parts
                types += ["Region"] * len(region_parts)
                attributes += [0.0] * len(region_parts)
    nodes = gpd.GeoDataFrame({"type": types, "attribute": attributes}, geometry=geoms, crs=crs)
    nodes.index = pd.RangeIndex(1, len(nodes) + 1, name="id")
    return ProtConnNodes(nodes, gdal_areas(np.array(pieces, dtype=object)) * factor, clipped, la)


def _pair_matrix(n: int, distances) -> np.ndarray:
    if isinstance(distances, pd.DataFrame):
        m = np.full((n, n), np.nan)
        i = distances.iloc[:, 0].to_numpy(int) - 1
        j = distances.iloc[:, 1].to_numpy(int) - 1
        m[i, j] = m[j, i] = distances.iloc[:, 2].to_numpy(float)
        np.fill_diagonal(m, 0.0)
        return m
    m = np.asarray(distances, dtype=np.float64)
    if m.shape != (n, n):
        raise ValueError("distances must be an n × n matrix or a (from, to, distance) table over the nodes")
    return m


def _pstar(dist: np.ndarray, keep: np.ndarray, distance: float, probability: float | None,
           pij_min: float | None = None) -> np.ndarray:
    """Most probable paths between the nodes where `keep` (MK: exp(d log(p) / t), cut below pij_min)."""
    idx = np.flatnonzero(keep)
    n = len(idx)
    if n == 0:
        return np.zeros((0, 0))
    sub = dist[np.ix_(idx, idx)]
    i, j = np.triu_indices(n, 1)
    value = sub[i, j]
    if probability is None:
        # Makurhini: exp(-d / t), a probability of exp(-1) at the distance.
        probability = float(np.exp(-1.0))
    cut = 0.0 if pij_min is None else float(pij_min)
    if np.isinf(cut):  # every link cut (Makurhini's grid default): each node alone
        return np.eye(n)
    D = _core.conn_all_pairs(i.astype(np.int32), j.astype(np.int32), np.ascontiguousarray(value), n, "distance",
                             float(distance), float(probability), cut, np.inf, False, False, 0)
    return np.exp(-D)


def protconn_indicators(nodes, distances, *, distance: float, la: float, pieces=None,
                        probability: float | None = 0.5, transboundary_weight: float = 1.0,
                        pij_min: float | None = None) -> dict:
    """ProtConn and its fractions from prepared nodes and distances (Makurhini's ``get_protconn_grid``).

    Parameters
    ----------
    nodes
        Table with ``type`` ("Non-Transboundary", "Transboundary", "Region")
        and ``attribute``, one row per node (e.g. :func:`protconn_nodes`).
    distances
        Distances between the nodes: ``n × n`` matrix (NaN = none) or
        ``(from, to, distance)`` table with node positions 1..n.
    distance, probability
        The dispersal distance and its probability (``None``: exp(-1)).
    la
        The area of the region.
    pieces
        Areas of the protected pieces before dissolving (ProtConn[Within]);
        default: the node attributes.
    transboundary_weight
        a_i a_j of the pairs with a transboundary (or region) node: Makurhini
        uses 1; 0 counts them as stepping stones only (Saura et al. 2017).
    pij_min
        Links with a lower probability are cut (None: none).

    Returns
    -------
    dict
        The indicators of :func:`protconn` (``ECA``, ``PC``, ``Prot``,
        ``ProtConn``, its fractions...) for this distance.

    Examples
    --------
    Distances between the nodes of :func:`protconn_nodes`, then the indicators:

    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> import numpy as np
    >>> nodes = ls.conn.protconn_nodes(reserves, region)
    >>> geometry = nodes.nodes.geometry
    >>> distances = np.array([[a.distance(b) if i != j else np.nan for j, b in enumerate(geometry)]
    ...                       for i, a in enumerate(geometry)])
    >>> result = ls.conn.protconn_indicators(nodes.nodes, distances, distance=2000, la=nodes.la,
    ...                                      pieces=nodes.pieces)
    >>> round(float(result["ProtConn"]), 2)
    5.25
    """
    types = np.asarray(nodes["type"])
    a = np.asarray(nodes["attribute"], dtype=np.float64)
    n = len(a)
    out = dict.fromkeys(INDICATORS, np.nan)
    non = types == "Non-Transboundary"
    tr = types == "Transboundary"
    region = types == "Region"
    area1 = a[non]
    pieces = area1 if pieces is None else np.asarray(pieces, dtype=np.float64)
    out["LA"] = la
    if n == 0 or not non.any():
        out.update(ECA=0.0, PC=0.0, Protected_surface=0.0, Prot=0.0, ProtConn=0.0, ProtUnconn=0.0,
                   Unprotected=100.0)
        return out
    dist = _pair_matrix(n, distances)
    w = float(transboundary_weight)

    def numerator(keep: np.ndarray, weighted: np.ndarray) -> float:
        p = _pstar(dist, keep, distance, probability, pij_min)
        b = a[keep]
        aiaj = np.outer(b, b)
        special = weighted[keep]
        aiaj[special, :] = w
        aiaj[:, special] = w
        return r_sum((aiaj * p).ravel(order="F"))

    graph = ~region
    pcnum = numerator(graph, tr)
    eca = float(np.sqrt(pcnum))
    protected = r_sum(area1)
    out.update(ECA=eca, PC=pcnum / la**2, Protected_surface=protected)
    out["Prot"] = 100 * (protected / la)
    out["ProtConn"] = 100 * (eca / la)
    out["ProtUnconn"] = out["Prot"] - out["ProtConn"]
    out["RelConn"] = 100 * (out["ProtConn"] / out["Prot"])
    out["Unprotected"] = 100 - out["Prot"]
    out["ProtConn_Prot"] = 100 * ((100 * (np.sqrt(r_sum(area1**2)) / la)) / out["ProtConn"])
    if not tr.any():
        out["ProtConn_Trans"] = 0.0
    else:
        ecan = float(np.sqrt(numerator(non, np.zeros(n, bool))))
        out["ProtConn_Trans"] = 100 * ((100 * ((eca - ecan) / la)) / out["ProtConn"])
    out["ProtConn_Unprot"] = max(100 - out["ProtConn_Prot"] - out["ProtConn_Trans"], 0.0)
    within1 = (np.sqrt(r_sum(pieces**2)) / la) * 100
    within2 = np.sqrt(protected / r_sum(pieces))
    out["ProtConn_Within"] = min(100 * ((within1 * within2) / out["ProtConn"]), 100.0)
    out["ProtConn_Contig"] = max(100 - out["ProtConn_Within"], 0.0)
    for part in ("Within", "Contig", "Unprot", "Trans"):
        out[f"ProtConn_{part}_land"] = out[f"ProtConn_{part}"] * (out["ProtConn"] / 100)
    if region.any():
        design = float(np.sqrt(numerator(np.ones(n, bool), ~non)))
        out["ProtUnconn_Design"] = (100 * (design / la)) - (100 * (eca / la))
        out["ProtConn_Bound"] = out["Prot"] - out["ProtUnconn_Design"]
    return out


def _single_node(area: float, la: float) -> dict:
    """MK_ProtConn with one (dissolved) protected node in the region: its fixed values."""
    out = dict.fromkeys(INDICATORS, np.nan)
    full = area >= la
    pct = 100 * (area / la)
    out.update(ECA=la if full else area, PC=1.0 if full else area / la**2, LA=la, Protected_surface=area,
               Prot=min(pct, 100.0), Unprotected=max(100 - pct, 0.0), ProtConn=min(pct, 100.0), ProtUnconn=0.0,
               ProtUnconn_Design=0.0, ProtConn_Bound=min(pct, 100.0), ProtConn_Prot=100.0, ProtConn_Within=100.0)
    return out


def _node_distances(prep: ProtConnNodes, distance_type: str, resistance, cost_kwargs) -> np.ndarray:
    import shapely

    nodes = prep.nodes
    n = len(nodes)
    geoms = nodes.geometry.values
    if distance_type == "edge":
        i, j = np.triu_indices(n, 1)
        d = shapely.distance(geoms[i], geoms[j])
        m = np.zeros((n, n))
        m[i, j] = m[j, i] = d
    elif distance_type == "least-cost":
        from .nodes import distance_table

        frame = nodes.reset_index()
        t = distance_table(frame, id="id", distance="least-cost", resistance=resistance, **cost_kwargs)
        m = _pair_matrix(n, t)
    else:
        raise ValueError("ProtConn distances are 'edge' or 'least-cost'")
    # protconn_dist: the region node is 0 from the transboundary nodes (one region
    # part), or each region part 0 from the nodes on it.
    types = nodes["type"].to_numpy()
    region = np.flatnonzero(types == "Region")
    if len(region) == 1:
        other = np.flatnonzero(types != "Non-Transboundary")
        m[region[0], other] = m[other, region[0]] = 0.0
    elif len(region) > 1:
        non = np.flatnonzero(types == "Non-Transboundary")
        for r in region:
            on = non[shapely.intersects(geoms[non], geoms[r])]
            m[r, on] = m[on, r] = 0.0
    return m


def protconn(protected_areas, region, *, distance, probability: float | None = 0.5,
             transboundary: float | None = None, transboundary_type: str = "nodes", bound: bool = False,
             area_unit: str = "m2", distance_type: str = "edge", resistance=None, delta: bool = False,
             transboundary_weight: float = 1.0, cost_directions: int = 8, cost_function: str = "makurhini",
             preselect: bool = True, pij_min: float | None = None):
    """Protected Connected land (ProtConn) of a region, as Makurhini's ``MK_ProtConn``.

    Parameters
    ----------
    protected_areas
        Polygons of the protected areas (GeoDataFrame, projected in metres).
    region
        The region (one or more polygons, merged).
    distance
        Dispersal distance(s) (one row per distance).
    probability
        Dispersal probability at the distance (default 0.5).
    transboundary
        Distance of the transboundary zone (None: no transboundary nodes).
    transboundary_type
        ``"nodes"`` (the zone around the region's protected areas, Saura et
        al. 2017) or ``"region"`` (around the region).
    bound
        Also the ProtUnconn[Design] and ProtConn[Bound] fractions.
    area_unit
        Unit of the areas (``"m2"``, ``"ha"``, ``"km2"``...).
    distance_type
        ``"edge"`` (between polygon edges) or ``"least-cost"`` over
        ``resistance`` (between centroids, gdistance's cost).
    delta
        Also return the importance of every protected area: ``dProt``,
        ``dProtConn`` and ``varProtConn`` once it is removed.
    transboundary_weight
        a_i a_j of the pairs with transboundary nodes (Makurhini: 1; 0 for
        Saura's definition, stepping stones only).
    resistance, cost_directions, cost_function
        Resistance raster of ``distance_type="least-cost"``, its neighbours (8
        or 4) and conductance rule, as in :func:`distance_table`.
    preselect
        Only use the protected areas near the region (``False``: all of them).
    pij_min
        Links with a lower probability are cut (``None``: none).

    Returns
    -------
    pandas.DataFrame, or (DataFrame, GeoDataFrame) with ``delta``
        One row per distance: ``distance``, ``ECA``, ``PC``, ``LA``,
        ``Protected_surface``, ``Prot``, ``ProtConn``, ``ProtUnconn``,
        ``RelConn``, ``ProtUnconn_Design``, ``ProtConn_Bound``, ``Unprotected``
        and the fractions ``ProtConn_Prot``, ``_Trans``, ``_Unprot``,
        ``_Within``, ``_Contig`` and their ``_land`` shares (in % of the region).

    See Also
    --------
    protconn_regions : one row per region.
    protconn_raster : protected areas given as a raster.
    connectivity_grid : ProtConn of every cell of a grid.

    Examples
    --------
    Four protected areas in a region of 11 x 11 km:

    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> table = ls.conn.protconn(reserves, region, distance=2000, probability=0.5)
    >>> table[["distance", "Prot", "ProtConn"]].round(2)
       distance  Prot  ProtConn
    0      2000  6.82      5.25

    With ``delta``, the loss of ProtConn when each area is removed:

    >>> table, areas = ls.conn.protconn(reserves, region, distance=2000, delta=True)
    >>> "dProtConn" in areas.columns
    True
    """
    prep = protconn_nodes(protected_areas, region, transboundary=transboundary,
                          transboundary_type=transboundary_type, bound=bound, area_unit=area_unit,
                          preselect=preselect)
    cost_kwargs = {"cost_directions": cost_directions, "cost_function": cost_function}
    dist = _node_distances(prep, distance_type, resistance, cost_kwargs) if len(prep.nodes) else np.zeros((0, 0))
    non = prep.nodes["type"] == "Non-Transboundary" if len(prep.nodes) else np.zeros(0, bool)
    # Makurhini: one dissolved node, or one protected area (whatever its parts),
    # is a single node of area min(area, LA).
    single = int(non.sum()) == 1 or len(prep.clipped) == 1
    rows = []
    for d in _as_list(distance):
        row = {"distance": d}
        if single:
            row.update(_single_node(min(r_sum(prep.nodes.loc[non, "attribute"].to_numpy()), prep.la), prep.la))
        else:
            row.update(protconn_indicators(prep.nodes, dist, distance=d, la=prep.la, pieces=prep.pieces,
                                           probability=probability, transboundary_weight=transboundary_weight,
                                           pij_min=pij_min))
        rows.append(row)
    table = pd.DataFrame(rows)
    if not delta:
        return table
    return table, _delta(prep, distance, probability, distance_type, resistance, cost_kwargs)


def _delta(prep: ProtConnNodes, distance, probability, distance_type, resistance, cost_kwargs):
    """Importance of every protected area in the region (Makurhini's delta_ProtConn)."""
    import geopandas as gpd

    clipped = prep.clipped
    trans = prep.nodes[prep.nodes["type"] == "Transboundary"]
    both = gpd.GeoDataFrame({"type": ["Non-Transboundary"] * len(clipped) + ["Transboundary"] * len(trans),
                             "attribute": np.concatenate([clipped["attribute"].to_numpy(), np.zeros(len(trans))])},
                            geometry=list(clipped.geometry.values) + list(trans.geometry.values), crs=clipped.crs)
    both.index = pd.RangeIndex(1, len(both) + 1, name="id")
    sub = ProtConnNodes(both, prep.pieces, clipped, prep.la)
    dist = _node_distances(sub, distance_type, resistance, cost_kwargs)
    a = both["attribute"].to_numpy()
    total = r_sum(a)
    frames = []
    for d in _as_list(distance):
        n = len(a)
        i, j = np.triu_indices(n, 1)
        g = HabitatGraph(a, pd.DataFrame({"from": i + 1, "to": j + 1, "d": dist[i, j]}))
        imp = g.node_importance("PC", distance=d, probability=0.5 if probability is None else probability,
                                pij_min=0.0)
        keep = np.arange(len(clipped))
        dpc = imp["dPC"].to_numpy()[keep]
        eca_ratio = np.sqrt(np.clip(1 - dpc / 100, 0, None))
        protconn1 = 100 * np.sqrt(g.overall("PC", distance=d, probability=probability or 0.5,
                                            pij_min=0.0).iloc[0]["PCnum"]) / prep.la
        frame = clipped.copy()
        frame["distance"] = d
        frame["dProt"] = 100 * a[keep] / total
        frame["dProtConn"] = 100 * (1 - eca_ratio)
        frame["varProtConn"] = protconn1 * (1 - eca_ratio)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def protconn_regions(protected_areas, regions, *, id: str | None = None, **kwargs) -> pd.DataFrame:
    """ProtConn of several regions, one row per region and distance (Makurhini's ``MK_ProtConnMult``).

    Each region (a row of ``regions``) is analysed as :func:`protconn` does, with
    the same protected areas.

    Parameters
    ----------
    protected_areas
        Polygons of the protected areas (GeoDataFrame, projected in metres).
    regions
        GeoDataFrame with one region per row.
    id
        Column of the region ids (default ``1 .. n``).
    **kwargs
        Options of :func:`protconn` (``distance`` is required).

    Returns
    -------
    pandas.DataFrame
        ``region`` and the columns of :func:`protconn`, one row per region and
        distance.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> regions = gpd.GeoDataFrame({"zone": ["west", "east"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(-1000, -1000, 4000, 10000), shapely.box(4000, -1000, 12000, 12000)])
    >>> table = ls.conn.protconn_regions(reserves, regions, id="zone", distance=2000)
    >>> table[["region", "distance", "Prot", "ProtConn"]].round(2)
      region  distance   Prot  ProtConn
    0   west      2000  11.82     10.17
    1   east      2000   4.57      3.95
    """
    ids = np.arange(1, len(regions) + 1) if id is None else regions[id].to_numpy()
    frames = []
    for k in range(len(regions)):
        table = protconn(protected_areas, regions.iloc[[k]], **kwargs)
        if isinstance(table, tuple):
            table = table[0]
        table.insert(0, "region", ids[k])
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _rasterize(geoms, transform, shape, values=None) -> np.ndarray:
    """terra::rasterize of polygons: the cells whose centre is inside (0 outside)."""
    from rasterio import features

    if values is None:
        values = [1] * len(geoms)
    shapes = [(g, int(v)) for g, v in zip(geoms, values) if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.int32)
    return features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=False,
                              dtype="int32")


def _cell_polygons(mask: np.ndarray, transform):
    """The cells of a mask as polygons (terra::as.polygons, dissolved)."""
    import shapely
    from rasterio import features

    polys = [shapely.geometry.shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask,
                                                                    transform=transform) if v == 1]
    return shapely.union_all(np.array(polys, dtype=object)) if polys else None


@threaded
def protconn_raster(protected_areas, region, *, distance, probability: float | None = 0.5,
                    transboundary: float | None = None, transboundary_type: str = "nodes", bound: bool = False,
                    area_unit: str = "m2", pij_min: float | None = None, nodata=None, band=None) -> pd.DataFrame:
    """ProtConn of a region from a raster of protected areas, as Makurhini's ``MK_ProtConn_raster``.

    Parameters
    ----------
    protected_areas
        Raster of protected-area ids (NA / ``nodata`` elsewhere).
    region
        Polygons of the region (rasterized on the protected areas' grid,
        cells whose centre is inside; with ``bound`` each part is a region
        node), or a raster of it (its 4-connected patches).
    distance, probability, transboundary, transboundary_type, bound, area_unit, pij_min
        As in :func:`protconn`.
    nodata, band
        Reading options of the rasters.

    Returns
    -------
    pandas.DataFrame
        The columns of :func:`protconn`, one row per distance.

    Notes
    -----
    Every protected-area id inside the region is a node (not dissolved), its
    attribute the area of its cells there; transboundary nodes are the ids
    outside the region within ``transboundary`` of the region's protected
    cells (``"nodes"``) or of the region (``"region"``), when there are at
    least two of them (as Makurhini). Distances run between the nodes'
    centroids (R's means of the cell centres). Makurhini's quirks are kept:
    ProtConn[Within] sums every node's attribute (transboundary and region
    ones included). Makurhini pairs the attributes (by id) with centroids in
    the order ids first appear in the raster; landschaft keeps each id with
    its own centroid (the same when ids appear in increasing order).

    Examples
    --------
    The broad-leaved forest of the example map as protected areas (each patch
    with its own id):

    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    >>> import geopandas as gpd
    >>> import shapely
    >>> labels = ls.Landscape(landcover.where(landcover == 23)).labels()
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(*landcover.rio.bounds())], crs=landcover.rio.crs)
    >>> table = ls.conn.protconn_raster(labels, region, distance=1000, area_unit="ha")
    >>> table[["distance", "Prot", "ProtConn"]].round(2)
       distance  Prot  ProtConn
    0      1000  7.71      2.76
    """
    import geopandas as gpd
    import shapely

    from .nodes import read_patches

    if transboundary_type not in ("nodes", "region"):
        raise ValueError("transboundary_type must be 'nodes' or 'region'")
    patches = read_patches(protected_areas, nodata=nodata, band=band)
    values, mask = patches.values, patches.mask
    ny, nx = values.shape
    xmin, xmax, ymin, ymax = patches.extent
    from rasterio.transform import Affine

    transform = Affine((xmax - xmin) / nx, 0.0, xmin, 0.0, -(ymax - ymin) / ny, ymax)
    res_x = (xmax - xmin) / nx
    factor = AREA_UNITS[area_unit]
    cell_area = (res_x**2) * factor
    if isinstance(region, (gpd.GeoDataFrame, gpd.GeoSeries)) or hasattr(region, "geometry"):
        geoms = np.asarray(region.geometry.values)
        if bound:
            parts = [p for g in geoms for p in _explode(g)]
            region_ids = _rasterize(parts, transform, (ny, nx), range(1, len(parts) + 1))
        else:
            region_ids = _rasterize(list(geoms), transform, (ny, nx))
    else:
        from scipy import ndimage

        rp = read_patches(region, nodata=nodata)
        inside = ~rp.mask
        region_ids = ndimage.label(inside)[0].astype(np.int32)  # terra::patches, 4 directions
    in_region = region_ids > 0
    pa = ~mask

    def nodes_of(cells: np.ndarray, kind: str):
        v = np.where(cells, values, 0)
        m = (~cells).view(np.uint8)
        r = _core.conn_raster_nodes(np.ascontiguousarray(v, dtype=np.int32), np.ascontiguousarray(m), xmin, xmax,
                                    ymin, ymax, True, False, 0)
        return pd.DataFrame({"type": kind, "attribute": r["cells"] * cell_area, "x": r["cx"], "y": r["cy"]})

    non = nodes_of(pa & in_region, "Non-Transboundary")
    la = r_sum(np.bincount(region_ids[in_region])[1:] * cell_area)
    rows = []
    if len(non) == 0:
        for d in _as_list(distance):
            row = dict.fromkeys(INDICATORS, np.nan)
            row.update(distance=d, LA=la, Protected_surface=0.0, Prot=0.0, Unprotected=100.0)
            rows.append(row)
        return pd.DataFrame(rows)[["distance"] + INDICATORS]
    if len(non) == 1:
        for d in _as_list(distance):
            row = {"distance": d}
            row.update(_single_node(min(float(non["attribute"].iloc[0]), la), la))
            rows.append(row)
        return pd.DataFrame(rows)
    frames = [non]
    if transboundary:
        reference = (pa & in_region) if transboundary_type == "nodes" else in_region
        poly = _cell_polygons(reference, transform)
        zone = _rasterize([shapely.buffer(shapely.buffer(poly, 0), transboundary, quad_segs=QUAD_SEGS)], transform,
                          (ny, nx)) > 0
        trans = nodes_of(pa & ~in_region & zone, "Transboundary")
        if len(trans) > 1:  # Makurhini keeps them only when there are two or more
            frames.append(trans)
            if bound:
                reg = pd.DataFrame({"type": "Region",
                                    "attribute": np.bincount(region_ids[in_region])[1:] * cell_area})
                r = _core.conn_raster_nodes(np.ascontiguousarray(region_ids), np.ascontiguousarray((~in_region).view(
                    np.uint8)), xmin, xmax, ymin, ymax, True, False, 0)
                reg["x"], reg["y"] = r["cx"], r["cy"]
                frames.append(reg)
    nodes = pd.concat(frames, ignore_index=True)
    n = len(nodes)
    e = _core.conn_point_distances(nodes["x"].to_numpy(), nodes["y"].to_numpy(), np.inf, 0)
    m = np.zeros((n, n))
    m[e["i"], e["j"]] = m[e["j"], e["i"]] = e["distance"]
    types = nodes["type"].to_numpy()
    region_nodes = np.flatnonzero(types == "Region")
    if len(region_nodes) == 1:
        other = np.flatnonzero(types != "Non-Transboundary")
        m[region_nodes[0], other] = m[other, region_nodes[0]] = 0.0
    for d in _as_list(distance):
        row = {"distance": d}
        row.update(protconn_indicators(nodes, m, distance=d, la=la, pieces=nodes["attribute"].to_numpy(),
                                       probability=probability, pij_min=pij_min))
        rows.append(row)
    return pd.DataFrame(rows)
