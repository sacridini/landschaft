# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat connectivity graphs: Conefor Sensinode's indices and Makurhini's node importances."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .. import _core


def r_sum(values) -> float:
    """R's sum() (80-bit accumulation)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    return float(_core.lsm_r_statistics(v.tolist())[0])

CONVENTIONS = ("makurhini", "conefor")
KINDS = ("distance", "probability", "link")
METRICS = ("A", "NC", "NL", "H", "CCP", "LCP", "IIC", "F", "AWF", "PC", "BC", "BCIIC", "BCPC")
LINK_METRICS = ("NC", "NL", "H", "CCP", "LCP", "IIC", "BC", "BCIIC")
PROBABILITY_METRICS = ("F", "AWF", "PC", "BCPC")
# Betweenness metrics: node values only, computed when asked for.
BETWEENNESS = ("BC", "BCIIC", "BCPC")

# Makurhini stores the hop count of unconnected pairs as this number, so the
# overall IIC counts a_i a_j / (1 + 16443701) for each of them.
MAKURHINI_UNCONNECTED = 16443701.0


def _metric_list(metrics, kind: str) -> list[str]:
    if metrics is None:
        chosen = [m for m in METRICS if m not in BETWEENNESS]
        if kind == "link":
            chosen = [m for m in chosen if m not in PROBABILITY_METRICS]
        return chosen
    if isinstance(metrics, str):
        metrics = [metrics]
    chosen = []
    for m in metrics:
        key = str(m).upper()
        if key not in METRICS:
            raise ValueError(f"unknown metric {m!r}; choose from {', '.join(METRICS)}")
        if kind == "link" and key in PROBABILITY_METRICS:
            raise ValueError(f"{key} needs distances or probabilities, not links")
        if key not in chosen:
            chosen.append(key)
    return chosen


def _as_list(value) -> list:
    if value is None:
        return [None]
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


class HabitatGraph:
    """Habitat patches (nodes with an attribute) and the connections between them.

        The indices are those of Conefor Sensinode (Saura & Torné 2009): NC, NL, H,
        CCP, LCP, IIC, F, AWF and PC, with the importance of every node (dI, the
        loss of each index without the node) and the intra, flux and connector
        fractions of dIIC and dPC, as Makurhini's ``MK_dPCIIC`` reports them.

        Parameters
        ----------
        attributes
            Attribute of every node (usually its area).
        connections
            Either a square ``n × n`` matrix (the upper triangle is read; NaN = no
            connection; a square array is always read as a matrix) or a table of pairs: a DataFrame or ``(m, 3)`` array with
            the two node ids and the value. Pairs missing from a table are not
            connected.
        ids
            Node ids (default ``1 .. n``); pair tables refer to them.
        kind
            What the values are: ``"distance"`` (the default), ``"probability"``
            of direct dispersal, or ``"link"`` (1 = linked, 0 = not), as
            Conefor's connection types.
        landscape_area
            LA, the attribute of a landscape fully covered by habitat: IIC, PC and
            LCP are their numerators divided by LA². Without it Makurhini reports
            only the numerators (the indices are NaN here), Conefor uses LA = 1.
        convention
            ``"makurhini"`` (default) follows ``MK_dPCIIC``: pairs closer than the
            distance are linked (``d < t``), probabilities
            ``exp(d log(P) / t)`` below ``pij_min`` (default 0.01) are dropped and
            unconnected pairs count in IIC as 16443701 links apart.
            ``"conefor"`` follows Conefor Sensinode: links up to the distance
            (``d <= t``), no probability cut by default (``pij_min`` is its
            minProb heuristic: probabilities up to it are dropped and paths no
            more probable are not extended) and unconnected pairs add nothing.
        restoration
            1 for the existing nodes and 0 for nodes that could be added (for
            restoration), as Makurhini's ``restoration`` column; then
            :meth:`node_importance` also rates the nodes to add (``dPCres``,
            ``dIICres``: the gain in % of the graph with them, Makurhini; with
            ``convention="conefor"``, Conefor's "adding nodes" mode: the other
            indices are those of the existing nodes and every index gets its
            ``d...add`` gain in % of the existing graph).
        n_threads
            Threads (default: all cores but one).

        See Also
        --------
        HabitatGraph.from_raster : nodes and distances from a map.
        HabitatGraph.from_vector : nodes and distances from polygons.
        HabitatGraph.from_conefor : Conefor's input files.

        Examples
        --------
        Four patches (areas in ha) and the distances between them (m):

        >>> import numpy as np
        >>> import landschaft as ls
        >>> areas = [100, 50, 20, 80]
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph(areas, distances, landscape_area=10_000)
        >>> graph.n_nodes
        4
        >>> table = graph.overall(["NC", "PC"], distance=1000, probability=0.5)
        >>> int(table["NC"].iloc[0]), round(float(table["EC(PC)"].iloc[0]), 1)
        (2, 172.3)

        The same as a table of pairs:

        >>> pairs = [(1, 2, 800), (2, 3, 600), (1, 3, 1500)]
        >>> sparse = ls.HabitatGraph(areas, pairs, landscape_area=10_000)
    """

    def __init__(
        self,
        attributes,
        connections,
        *,
        ids=None,
        kind: str = "distance",
        landscape_area: float | None = None,
        convention: str = "makurhini",
        restoration=None,
        n_threads: int | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
        a = np.asarray(attributes, dtype=np.float64).ravel()
        if a.size < 1:
            raise ValueError("a habitat graph needs at least one node")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError("node attributes must be finite and non-negative")
        n = a.size
        ids = np.arange(1, n + 1) if ids is None else np.asarray(ids)
        if ids.shape != (n,):
            raise ValueError("ids must have one entry per node")
        if pd.Index(ids).has_duplicates:
            raise ValueError("node ids must be unique")
        self.ids = ids
        self.attributes = a
        self.kind = kind
        self.landscape_area = None if landscape_area is None else float(landscape_area)
        self.convention = convention
        self.n_threads = 0 if n_threads is None else int(n_threads)
        self.habitat_area = None  # set by from_raster / from_vector (Makurhini's habitat area)
        self._pi, self._pj, self._value, self._dense = self._pairs(connections)
        if restoration is None:
            self.restoration = None
        else:
            flags = np.asarray(restoration)
            if flags.shape != (n,) or not np.isin(flags, (0, 1)).all():
                raise ValueError("restoration must be 1 (existing node) or 0 (node to add) for every node")
            if not (flags == 1).any():
                raise ValueError("restoration: no existing node")
            self.restoration = flags.astype(np.int8)

    # ------------------------------------------------------------------ input
    def _pairs(self, connections):
        n = self.attributes.size
        if isinstance(connections, pd.DataFrame):
            if connections.shape[1] != 3:
                raise ValueError("a pair table needs three columns: from, to, value")
            table = connections.to_numpy()
        else:
            table = np.asarray(connections)
            if table.ndim == 2 and table.shape == (n, n):
                m = table.astype(np.float64)
                i, j = np.triu_indices(n, 1)
                return i.astype(np.int32), j.astype(np.int32), np.ascontiguousarray(m[i, j]), True
            if table.ndim != 2 or table.shape[1] != 3:
                raise ValueError("connections must be an n × n matrix or a (from, to, value) table")
        index = pd.Index(self.ids)
        fi = index.get_indexer(table[:, 0])
        fj = index.get_indexer(table[:, 1])
        if np.any(fi < 0) or np.any(fj < 0):
            raise ValueError("the pair table refers to ids that are not nodes")
        v = table[:, 2].astype(np.float64)
        keep = fi != fj
        fi, fj, v = fi[keep], fj[keep], v[keep]
        lo, hi = np.minimum(fi, fj), np.maximum(fi, fj)
        frame = pd.DataFrame({"i": lo, "j": hi, "v": v}).drop_duplicates()
        if frame.duplicated(["i", "j"]).any():
            raise ValueError("a pair appears twice with different values")
        frame = frame.sort_values(["i", "j"], kind="stable")
        return (frame["i"].to_numpy(np.int32), frame["j"].to_numpy(np.int32), frame["v"].to_numpy(np.float64), False)

    @classmethod
    def from_raster(cls, nodes, *, attribute=None, weighted: bool = False, area_unit: str = "m2",
                    distance: str = "centroid", max_distance: float | None = None, distance_unit: str = "m",
                    habitat=None, directions: int = 8, nodata=None, band=None, resistance=None,
                    cost_directions: int = 8, cost_function: str = "makurhini", unreachable: str = "euclidean",
                    **kwargs) -> "HabitatGraph":
        """Nodes from a raster of patch ids (or the patches of ``habitat`` classes).

                The attributes come from :func:`~landschaft.conn.node_table` (the patch
                areas in ``area_unit`` by default) and the connections from
                :func:`~landschaft.conn.distance_table` (``distance="centroid"``,
                ``"edge"``, ``"least-cost"`` or ``"commute-time"`` over ``resistance``, pairs up to
                ``max_distance`` in ``distance_unit``), as
                Makurhini's ``MK_dPCIIC`` does with raster nodes. Other arguments go
                to :class:`HabitatGraph`; :meth:`to_raster` paints node values on the
                patches.

            Parameters
            ----------
            nodes
                Raster of patch ids (NaN / NoData = no habitat), or of classes with
                ``habitat``.
            attribute
                Attribute per patch (table or Series by id); default: the patch area.
            weighted
                Multiply the given attribute by the patch area.
            area_unit
                ``"m2"``, ``"ha"`` or ``"km2"`` for the areas.
            distance
                ``"centroid"``, ``"edge"``, ``"least-cost"`` or ``"commute-time"``.
            max_distance, distance_unit
                Keep the pairs up to this distance (``"m"`` or ``"km"``).
            habitat
                Class values whose patches are the nodes (patches are labelled with
                ``directions`` 8 or 4).
            nodata, band
                Reading options.
            resistance
                Resistance raster for least-cost and commute-time distances.
            cost_directions, cost_function, unreachable
                Neighbours of the cost surface (8 or 4), how the conductance between
                two cells is taken (``"makurhini"`` or ``"mean"``), and what unreachable
                pairs get (``"euclidean"``: their centroid distance, or ``"drop"``).
            **kwargs
                :class:`HabitatGraph` options (``landscape_area``, ``convention``...).

            Returns
            -------
            HabitatGraph

            Examples
            --------
            >>> import landschaft as ls
            >>> landcover = ls.load_raster(ls.example_data("clc"))
            >>> graph = ls.HabitatGraph.from_raster(landcover, habitat=[23, 24, 25], area_unit="ha",
            ...                                     distance="edge", max_distance=5000,
            ...                                     landscape_area=landcover.size * 25 * 25 / 1e4)
            >>> graph.n_nodes
            158
            >>> importance = graph.node_importance("PC", distance=1000, probability=0.5)
            >>> dpc_map = graph.to_raster(importance, "dPC")
        """
        from .nodes import distance_table, node_table, read_patches

        patches = read_patches(nodes, habitat=habitat, directions=directions, nodata=nodata, band=band)
        threads = kwargs.get("n_threads")
        table = node_table(patches, attribute=attribute, weighted=weighted, area_unit=area_unit, n_threads=threads)
        pairs = distance_table(patches, distance=distance, max_distance=max_distance, distance_unit=distance_unit,
                               resistance=resistance, cost_directions=cost_directions, cost_function=cost_function,
                               unreachable=unreachable, n_threads=threads)
        graph = cls(table["attribute"].to_numpy(), pairs, ids=table["id"].to_numpy(), **kwargs)
        graph._patches = patches
        areas = node_table(patches, area_unit=area_unit, n_threads=threads)["attribute"].to_numpy()
        graph.habitat_area = r_sum(areas)  # MK_dECA: sum of the cells' areas
        return graph

    @classmethod
    def from_vector(cls, nodes, *, id: str | None = None, attribute=None, weighted: bool = False,
                    area_unit: str = "m2", distance: str = "centroid", max_distance: float | None = None,
                    distance_unit: str = "m", resistance=None, cost_directions: int = 8,
                    cost_function: str = "makurhini", unreachable: str = "euclidean", **kwargs) -> "HabitatGraph":
        """Nodes from polygons or points (GeoDataFrame), as Makurhini's ``MK_dPCIIC``.

                ``id`` names the column of node ids, ``attribute`` a column (default:
                the area in ``area_unit``); distances are ``"centroid"``, ``"edge"``
                (GEOS), ``"least-cost"`` or ``"commute-time"`` over ``resistance``, pairs up to ``max_distance``. :meth:`to_geodataframe` joins
                node values to the geometries.

            Parameters
            ----------
            nodes
                GeoDataFrame of polygons or points, in a metric CRS.
            id
                Column of node ids (default: 1..n).
            attribute, weighted, area_unit, distance, max_distance, distance_unit
                As in :meth:`from_raster`.
            resistance, cost_directions, cost_function, unreachable
                As in :meth:`from_raster`.
            **kwargs
                :class:`HabitatGraph` options.

            Returns
            -------
            HabitatGraph

            Examples
            --------
            >>> import geopandas as gpd
            >>> import shapely
            >>> import landschaft as ls
            >>> patches = gpd.GeoDataFrame({"name": ["a", "b", "c"]}, crs="EPSG:3035", geometry=[
            ...     shapely.box(0, 0, 500, 500), shapely.box(900, 0, 1400, 400), shapely.box(3000, 0, 3300, 300)])
            >>> graph = ls.HabitatGraph.from_vector(patches, id="name", area_unit="ha", distance="edge")
            >>> graph.components(distance=1000).tolist()
            [1, 1, 2]
        """
        from .nodes import distance_table, node_table

        threads = kwargs.get("n_threads")
        table = node_table(nodes, attribute=attribute, weighted=weighted, area_unit=area_unit, id=id,
                           n_threads=threads)
        pairs = distance_table(nodes, distance=distance, id=id, max_distance=max_distance,
                               distance_unit=distance_unit, resistance=resistance, cost_directions=cost_directions,
                               cost_function=cost_function, unreachable=unreachable, n_threads=threads)
        graph = cls(table["attribute"].to_numpy(), pairs, ids=table["id"].to_numpy(), **kwargs)
        graph._vector = nodes
        from .nodes import AREA_UNITS, gdal_areas

        geoms = nodes.geometry.values if hasattr(nodes, "geometry") else np.asarray(nodes)
        graph.habitat_area = r_sum(gdal_areas(geoms)) * AREA_UNITS[area_unit]  # MK_dECA: sum, then the unit
        return graph

    @classmethod
    def from_samc(cls, nodes, resistance, *, dispersal_distance: float | None = None, absorption=None,
                  directions: int = 8, cost_function: str = "mean", vertices: int | None = 10,
                  vertex_fraction: float | None = None, settle: str = "sum", id: str | None = None, attribute=None,
                  area_unit: str = "m2", **kwargs) -> "HabitatGraph":
        """Polygons connected by settlement probabilities of a spatial absorbing Markov chain (``MK_dPC_SAMC``).

                The probabilities are :func:`landschaft.conn.settlement_probabilities`
                (samc's random walk over ``resistance``, absorbed with ``absorption``
                or Fletcher et al.'s rate for ``dispersal_distance``); the attribute is
                the area in ``area_unit`` or a column. The graph has
                ``kind="probability"``: ``node_importance("PC", pij_min=0.001)`` is
                Makurhini's ``MK_dPC_SAMC`` (which cuts at 0.001).

            Parameters
            ----------
            nodes
                GeoDataFrame of habitat polygons.
            resistance
                Resistance raster the random walk moves on.
            dispersal_distance, absorption
                Mean dispersal distance (for the default absorption rate) or the
                absorption per cell.
            directions, cost_function
                Neighbours of the walk and how the step probabilities are derived.
            vertices, vertex_fraction, settle, id, attribute, area_unit
                Boundary vertices sampled per polygon, how settlement is summed, node
                ids and attributes; see :func:`~landschaft.conn.settlement_probabilities`.
            **kwargs
                :class:`HabitatGraph` options.

            Returns
            -------
            HabitatGraph
                With ``kind="probability"``.

            Examples
            --------
            >>> import geopandas as gpd
            >>> import numpy as np
            >>> import shapely
            >>> import xarray as xr
            >>> import landschaft as ls
            >>> from rasterio.transform import from_origin
            >>> resistance = xr.DataArray(np.ones((60, 60)), dims=("y", "x")).rio.write_transform(
            ...     from_origin(0, 600, 10, 10)).rio.write_crs("EPSG:3035")
            >>> patches = gpd.GeoDataFrame(geometry=[shapely.box(50, 50, 150, 150), shapely.box(400, 400, 520, 520)],
            ...                            crs="EPSG:3035")
            >>> graph = ls.HabitatGraph.from_samc(patches, resistance, dispersal_distance=300, area_unit="ha")
            >>> graph.n_nodes
            2
        """
        from .nodes import node_table
        from .samc import settlement_probabilities

        table = node_table(nodes, attribute=attribute, area_unit=area_unit, id=id, n_threads=kwargs.get("n_threads"))
        ids, P = settlement_probabilities(nodes, resistance, dispersal_distance=dispersal_distance,
                                          absorption=absorption, directions=directions, cost_function=cost_function,
                                          vertices=vertices, vertex_fraction=vertex_fraction, settle=settle, id=id)
        np.fill_diagonal(P, 0.0)
        graph = cls(table["attribute"].to_numpy(), P, ids=ids, kind="probability", **kwargs)
        graph._vector = nodes
        return graph

    def to_raster(self, values, name: str | None = None):
        """Node values painted on the patches of :meth:`from_raster`.

        Parameters
        ----------
        values
            One value per node (in node order), a Series indexed by node id, or a
            table such as :meth:`node_importance` (with ``name`` naming the column).
        name
            Column of ``values`` to paint.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            The patch map with the values, NaN elsewhere.

        Examples
        --------
        >>> import landschaft as ls
        >>> landcover = ls.load_raster(ls.example_data("clc"))
        >>> graph = ls.HabitatGraph.from_raster(landcover, habitat=[23, 24, 25], area_unit="ha",
        ...                                     distance="edge", max_distance=5000,
        ...                                     landscape_area=landcover.size * 25 * 25 / 1e4)
        >>> components = graph.components(distance=1000)
        >>> component_map = graph.to_raster(components)
        """
        patches = getattr(self, "_patches", None)
        if patches is None:
            raise ValueError("to_raster needs a graph made by HabitatGraph.from_raster")
        per_node = self._per_node(values, name)
        node = patches.read_nodes(self.n_threads)["node"]
        out = np.full(node.shape, np.nan)
        inside = node >= 0
        out[inside] = per_node[node[inside]]
        template = patches.template
        if template is not None and hasattr(template, "coords"):
            import xarray as xr

            da = template.squeeze(drop=True) if "band" in template.dims else template
            result = xr.DataArray(out, coords=da.coords, dims=da.dims, name=name or "value")
            try:
                from ..io import _with_georef

                if da.rio.crs is not None:  # without a CRS, the result was left bare
                    result = _with_georef(result, crs=da.rio.crs, nodata=np.nan)
            except Exception:
                pass
            return result
        return out

    def to_geodataframe(self, values, name: str | None = None):
        """Node values joined to the geometries of :meth:`from_vector`.

        Parameters
        ----------
        values
            A :meth:`node_importance` table (all its columns; one block per
            distance), a Series indexed by node id, or one value per node.
        name
            Name of the column for a plain sequence of values.

        Returns
        -------
        geopandas.GeoDataFrame

        Examples
        --------
        >>> import geopandas as gpd
        >>> import shapely
        >>> import landschaft as ls
        >>> patches = gpd.GeoDataFrame(crs="EPSG:3035", geometry=[
        ...     shapely.box(0, 0, 500, 500), shapely.box(900, 0, 1400, 400), shapely.box(3000, 0, 3300, 300)])
        >>> graph = ls.HabitatGraph.from_vector(patches, area_unit="ha", distance="edge",
        ...                                     landscape_area=100)
        >>> table = graph.to_geodataframe(graph.node_importance("PC", distance=1000, probability=0.5))
        >>> "dPC" in table.columns
        True
        """
        nodes = getattr(self, "_vector", None)
        if nodes is None:
            raise ValueError("to_geodataframe needs a graph made by HabitatGraph.from_vector")
        import geopandas as gpd

        geometry = nodes.geometry.values if isinstance(nodes, gpd.GeoDataFrame) else np.asarray(nodes.values)
        crs = nodes.crs
        if isinstance(values, pd.DataFrame) and "id" in values.columns:
            position = pd.Index(self.ids).get_indexer(values["id"])
            return gpd.GeoDataFrame(values.reset_index(drop=True), geometry=geometry[position], crs=crs)
        per_node = self._per_node(values, name)
        frame = pd.DataFrame({"id": self.ids, name or "value": per_node})
        return gpd.GeoDataFrame(frame, geometry=geometry, crs=crs)

    def _per_node(self, values, name) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            if name is None:
                raise ValueError("give the column to use (name=)")
            if values["id"].duplicated().any():
                raise ValueError("the table has several rows per node (several distances): select one")
            values = pd.Series(values[name].to_numpy(), index=values["id"].to_numpy())
        if isinstance(values, pd.Series) and not isinstance(values.index, pd.RangeIndex):
            return values.reindex(self.ids).to_numpy(dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (self.n_nodes,):
            raise ValueError(f"expected one value per node ({self.n_nodes})")
        return v

    @classmethod
    def from_conefor(cls, node_file, connection_file, **kwargs) -> "HabitatGraph":
        """Read Conefor's input files.

                ``node_file`` holds ``id attribute`` per line; ``connection_file`` either
                ``id id value`` per line (pairs not listed are not connected) or the
                ``n × n`` matrix in the order of the nodes. Other arguments go to
                :class:`HabitatGraph` (``kind``, ``convention``, ``landscape_area``...).

            Parameters
            ----------
            node_file
                Node file: ``id attribute`` per line.
            connection_file
                Connection file: ``id id value`` per line, or an ``n x n`` matrix.
            **kwargs
                :class:`HabitatGraph` options.

            Returns
            -------
            HabitatGraph

            Examples
            --------
            >>> import numpy as np
            >>> import landschaft as ls
            >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
            ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
            >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
            >>> graph.to_conefor("nodes.txt", "distances.txt")
            >>> again = ls.HabitatGraph.from_conefor("nodes.txt", "distances.txt", landscape_area=10_000)
            >>> again.n_nodes
            4
        """
        nodes = np.loadtxt(node_file, dtype=np.float64, ndmin=2)
        if nodes.shape[1] < 2:
            raise ValueError("a Conefor node file has an id and an attribute per line")
        ids = nodes[:, 0]
        if np.all(ids == np.round(ids)):
            ids = ids.astype(np.int64)
        values = np.loadtxt(connection_file, dtype=np.float64, ndmin=2)
        n = nodes.shape[0]
        if values.shape != (n, n) and values.shape[1] != 3:
            raise ValueError("a Conefor connection file holds 'id id value' lines or an n × n matrix")
        if values.shape[1] == 3 and values.shape != (n, n):
            values = pd.DataFrame({"from": values[:, 0].astype(ids.dtype), "to": values[:, 1].astype(ids.dtype),
                                   "value": values[:, 2]})
        return cls(nodes[:, 1], values, ids=ids, **kwargs)

    def to_conefor(self, node_file, connection_file) -> None:
        """Write Conefor's input files.

        Parameters
        ----------
        node_file
            ``id attribute`` lines.
        connection_file
            ``id id value`` lines, connected pairs only.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> graph.to_conefor("nodes.txt", "distances.txt")
        >>> open("nodes.txt").read().split()[:2]
        ['1', '100.0']
        """
        with open(node_file, "w", encoding="ascii") as f:
            for i, a in zip(self.ids, self.attributes):
                f.write(f"{i}\t{float(a)!r}\n")
        keep = ~np.isnan(self._value)
        with open(connection_file, "w", encoding="ascii") as f:
            for i, j, v in zip(self.ids[self._pi[keep]], self.ids[self._pj[keep]], self._value[keep]):
                f.write(f"{i}\t{j}\t{float(v)!r}\n")

    @property
    def n_nodes(self) -> int:
        """Number of nodes.

        Examples
        --------
        >>> import landschaft as ls
        >>> ls.HabitatGraph([1.0, 2.0], [[0, 5], [5, 0]]).n_nodes
        2
        """
        return int(self.attributes.size)

    def __repr__(self) -> str:
        return (f"<HabitatGraph: {self.n_nodes} nodes, {self._pi.size} pairs ({self.kind}), "
                f"convention={self.convention!r}>")

    # ------------------------------------------------------------- scenarios
    def _defaults(self, distance, probability, pij_min):
        if self.kind == "distance" and distance is None:
            if self.convention == "conefor":
                raise ValueError("give the distance (threshold) of the links and probabilities")
            # MK_dPCIIC: the median of the full distance matrix (both
            # triangles and the zero diagonal).
            v = self._value[~np.isnan(self._value)]
            distance = float(np.median(np.concatenate([v, v, np.zeros(self.n_nodes)])))
        if probability is None:
            probability = 0.5
        if pij_min is None:
            pij_min = 0.01 if self.convention == "makurhini" else 0.0
        return distance, float(probability), float(pij_min)

    def _scenario(self, metrics, distance, probability, pij_min):
        """Threshold, probability, pij_min and Conefor's expansion limit of a scenario."""
        distance, probability, pij_min = self._defaults(distance, probability, pij_min)
        if self.kind == "probability" and any(m in LINK_METRICS for m in metrics) and distance is None:
            raise ValueError("links between probabilities need a minimum probability (distance=)")
        expand = np.inf
        if self.convention == "conefor" and pij_min > 0:
            expand = float(-np.log(pij_min))
        threshold = np.nan if distance is None else float(distance)
        if any(m in PROBABILITY_METRICS for m in metrics) and self.kind == "distance" and not threshold > 0:
            raise ValueError("the distance of the probability must be positive")
        return distance, threshold, probability, pij_min, expand

    def _run(self, metrics, distance, probability, pij_min, importance, edges=False):
        distance, threshold, probability, pij_min, expand = self._scenario(metrics, distance, probability, pij_min)
        want_links = any(m in LINK_METRICS for m in metrics)
        want_prob = any(m in PROBABILITY_METRICS for m in metrics)
        conefor = self.convention == "conefor"
        betweenness = importance and any(m in BETWEENNESS for m in metrics)
        r = _core.conn_analyse(self._pi, self._pj, self._value, self.attributes, self.kind, threshold, probability,
                               pij_min, expand, conefor, want_links, want_prob, importance, edges, betweenness,
                               self.n_threads)
        return distance, r

    def _area(self) -> float | None:
        if self.landscape_area is not None:
            return self.landscape_area
        return 1.0 if self.convention == "conefor" else None

    # --------------------------------------------------------------- results
    def overall(self, metrics: Sequence[str] | str | None = None, *, distance=None, probability: float | None = None,
                pij_min: float | None = None) -> pd.DataFrame:
        """Indices of the whole graph, one row per distance.

        Parameters
        ----------
        metrics
            Any of ``"NC", "NL", "H", "CCP", "LCP", "IIC", "F", "AWF", "PC"``
            (default: all that the connection kind allows).
        distance
            Distance (or distances) of the links and of ``probability``
            (Makurhini's ``distance_thresholds``; with ``kind="probability"``
            the minimum probability of a link). Makurhini's default is the
            median distance.
        probability
            Dispersal probability at ``distance`` (default 0.5).
        pij_min
            Probability cut (default 0.01 for Makurhini, none for Conefor).

        Returns
        -------
        pandas.DataFrame
            ``distance`` and the indices: NC, NL, H, CCP, LCP, LCPnum, IIC,
            IICnum, EC(IIC), IICintra(%), IICdirect(%), IICstep(%), F, AWF, PC,
            PCnum, EC(PC), PCintra(%), PCdirect(%), PCstep(%).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> table = graph.overall(["NC", "IIC", "PC"], distance=[500, 1000], probability=0.5)
        >>> table["distance"].tolist()
        [500, 1000]
        """
        chosen = _metric_list(metrics, self.kind)
        # Conefor's "adding nodes": the indices are those of the existing nodes.
        graph = self
        if self.restoration is not None and self.convention == "conefor":
            graph = self._subgraph(self.restoration == 1)
        rows = [graph._overall_row(chosen, d, probability, pij_min) for d in _as_list(distance)]
        return pd.DataFrame(rows)

    def _overall_row(self, chosen, distance, probability, pij_min) -> dict:
        d, r = self._run(chosen, distance, probability, pij_min, importance=False)
        a = self.attributes
        la = self._area()
        total = float(a.sum())
        intra = float(np.dot(a, a))
        out: dict = {"distance": np.nan if d is None else d}
        if "A" in chosen:
            out["A"] = total
        if "NC" in chosen:
            out["NC"] = int(r["nc"])
        if "NL" in chosen:
            out["NL"] = int(r["nl"])
        if "H" in chosen:
            out["H"] = r["h"]
        if "CCP" in chosen:
            out["CCP"] = r["comp_sq"] / (total * total)
        if "LCP" in chosen:
            out["LCP"] = np.nan if la is None else r["comp_sq"] / (la * la)
            out["LCPnum"] = r["comp_sq"]
        if "IIC" in chosen:
            reach = r["iic_reach"]
            num = reach + self._unconnected(r["unreach_aa"])
            out.update(self._index("IIC", num, la))
            direct = r["iic_direct"]
            out.update(self._fractions("IIC", num, intra, direct, reach - intra - direct))
        if "F" in chosen:
            out["F"] = float(r["f_row"].sum())
        if "AWF" in chosen:
            out["AWF"] = float(np.dot(a, r["awf_row"]))
        if "PC" in chosen:
            num = r["pc_num"]
            out.update(self._index("PC", num, la))
            direct = r["pc_direct"]
            out.update(self._fractions("PC", num, intra, direct, num - intra - direct))
        return out

    def _unconnected(self, aa):
        if self.convention == "makurhini":
            return aa / (1.0 + MAKURHINI_UNCONNECTED)
        return 0.0 * aa

    def _index(self, name: str, num: float, la: float | None) -> dict:
        ec = float(np.sqrt(num))
        if la is None:
            value = np.nan
        else:
            value = num / (la * la)
            if self.convention == "makurhini":  # MK_dPCIIC caps both
                value = min(value, 1.0)
                ec = min(ec, la)
        return {name: value, f"{name}num": num, f"EC({name})": ec}

    @staticmethod
    def _fractions(name: str, num: float, intra: float, direct: float, step: float) -> dict:
        return {f"{name}intra(%)": 100 * intra / num, f"{name}direct(%)": 100 * direct / num,
                f"{name}step(%)": 100 * step / num}

    def node_importance(self, metrics: Sequence[str] | str | None = None, *, distance=None,
                        probability: float | None = None, pij_min: float | None = None) -> pd.DataFrame:
        """Importance of every node: the loss of each index (in %) without it.

        Parameters are those of :meth:`overall`.

        Returns
        -------
        pandas.DataFrame
            ``id``, ``distance`` and, for the metrics asked: dA, dNC, dNL, dH,
            dCCP, dLCP, dIIC with dIICintra/dIICflux/dIICconnector, dF, dAWF,
            dPC with dPCintra/dPCflux/dPCconnector and varPC (the loss of PC
            itself). Several distances give one block of rows per distance.
            Asked for by name (not in the default), Conefor's betweenness
            centralities: ``"BC"`` (Freeman's, normalised), ``"BCIIC"``
            (``varBC_IIC``: over ordered pairs the share of their shortest
            paths through the node times a_i a_j / (1 + nl_ij); ``dBC_IIC`` in %
            of IICnum) and ``"BCPC"`` (``varBC_PC``: a_i a_j p*_ij of the pairs
            whose most probable path runs through it; ``dBC_PC`` in % of PCnum),
            Bodin & Saura (2010).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> importance = graph.node_importance("PC", distance=1000, probability=0.5)
        >>> importance.columns.tolist()
        ['id', 'distance', 'dPC', 'dPCintra', 'dPCflux', 'dPCconnector', 'varPC']
        """
        chosen = _metric_list(metrics, self.kind)
        frames = [self._importance_frame(chosen, d, probability, pij_min) for d in _as_list(distance)]
        return pd.concat(frames, ignore_index=True)

    def _importance_frame(self, chosen, distance, probability, pij_min) -> pd.DataFrame:
        if self.restoration is not None:
            return self._restoration_frame(chosen, distance, probability, pij_min)
        return self._removal_frame(chosen, distance, probability, pij_min)

    def _removal_frame(self, chosen, distance, probability, pij_min) -> pd.DataFrame:
        d, r = self._run(chosen, distance, probability, pij_min, importance=True)
        a = self.attributes
        la = self._area()
        total = float(a.sum())
        a2 = a * a
        cols: dict = {"id": self.ids, "distance": np.full(a.size, np.nan if d is None else d)}
        with np.errstate(divide="ignore", invalid="ignore"):
            if "A" in chosen:
                cols["dA"] = 100 * a / total
            if any(m in chosen for m in LINK_METRICS):
                comp = r["component"]
                comp_area = np.bincount(comp, weights=a)[comp]
                rest_sq = r["comp_sq"] - comp_area * comp_area + r["pieces_sq"]
            if "NC" in chosen:
                cols["dNC"] = 100.0 * (1 - r["pieces"]) / r["nc"]
            if "NL" in chosen:
                nl = r["nl"]
                cols["dNL"] = 100.0 * r["degree"] / nl if nl else np.zeros(a.size)
            if "H" in chosen:
                h = r["h"]
                cols["dH"] = 100 * (r["h_row"] + r["h_conn"]) / h if h else np.zeros(a.size)
            if "CCP" in chosen:
                ccp = r["comp_sq"] / (total * total)
                ccp_del = rest_sq / ((total - a) * (total - a))
                cols["dCCP"] = 100 * (ccp - ccp_del) / ccp
            if "LCP" in chosen:
                cols["dLCP"] = 100 * (r["comp_sq"] - rest_sq) / r["comp_sq"]
            if "IIC" in chosen:
                reach_row = r["iic_row"]
                num = r["iic_reach"] + self._unconnected(r["unreach_aa"])
                flux_reach = 2 * (a * reach_row - a2)
                conn = r["iic_conn"]
                if self.convention == "makurhini":
                    # MK_dPCIIC: the whole graph counts unconnected pairs, the
                    # graph without the node does not; the flux counts the
                    # node's own unconnected pairs.
                    own = self._unconnected(a * r["unreach_a"])
                    whole = self._unconnected(r["unreach_aa"])
                    delta = a2 + flux_reach + conn + whole
                    flux = flux_reach + 2 * own
                    connector = np.maximum(conn + whole - 2 * own, 0.0)
                else:
                    delta = a2 + flux_reach + conn
                    flux = flux_reach
                    connector = conn
                cols["dIIC"] = 100 * delta / num
                cols["dIICintra"] = 100 * a2 / num
                cols["dIICflux"] = 100 * flux / num
                cols["dIICconnector"] = 100 * connector / num
            if "F" in chosen:
                f = r["f_row"]
                cols["dF"] = 100 * 2 * f / f.sum()
            if "AWF" in chosen:
                w = a * r["awf_row"]
                cols["dAWF"] = 100 * 2 * w / w.sum()
            if "PC" in chosen:
                num = r["pc_num"]
                flux = 2 * (a * r["pc_row"] - a2)
                conn = r["pc_conn"]
                delta = a2 + flux + conn
                cols["dPC"] = 100 * delta / num
                cols["dPCintra"] = 100 * a2 / num
                cols["dPCflux"] = 100 * flux / num
                cols["dPCconnector"] = 100 * conn / num
                cols["varPC"] = delta / (la * la) if la is not None else np.full(a.size, np.nan)
            n = a.size
            if "BC" in chosen:
                # Conefor's BC: Freeman's, over pairs, normalised by (n - 1)(n - 2) / 2.
                cols["BC"] = (r["bc"] / 2) / ((n - 1) * (n - 2) / 2) if n > 2 else np.zeros(n)
            if "BCIIC" in chosen:
                cols["dBC_IIC"] = 100 * r["bc_iic"] / r["iic_reach"]
                cols["varBC_IIC"] = r["bc_iic"]
            if "BCPC" in chosen:
                cols["dBC_PC"] = 100 * r["bc_pc"] / r["pc_num"]
                cols["varBC_PC"] = r["bc_pc"]
        return pd.DataFrame(cols)

    def components(self, distance=None) -> pd.Series:
        """Connected component of every node.

        Parameters
        ----------
        distance
            Links are the pairs closer than this (distance graphs).

        Returns
        -------
        pandas.Series
            Component number (1, 2, ...) indexed by node id.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> graph.components(distance=1000).tolist()
        [1, 1, 1, 2]
        """
        _, r = self._run(["NC"], distance, None, None, importance=False)
        return pd.Series(r["component"] + 1, index=pd.Index(self.ids, name="id"), name="component")

    # ------------------------------------------------------------ subgraphs
    def _subgraph(self, keep: np.ndarray) -> "HabitatGraph":
        """The graph of the nodes where ``keep`` (same kind, convention and area)."""
        index = np.full(self.n_nodes, -1, dtype=np.int64)
        index[keep] = np.arange(int(keep.sum()))
        both = keep[self._pi] & keep[self._pj]
        sub = HabitatGraph.__new__(HabitatGraph)
        sub.ids = self.ids[keep]
        sub.attributes = self.attributes[keep]
        sub.kind = self.kind
        sub.landscape_area = self.landscape_area
        sub.convention = self.convention
        sub.n_threads = self.n_threads
        sub.restoration = None
        sub.habitat_area = None
        sub._pi = index[self._pi[both]].astype(np.int32)
        sub._pj = index[self._pj[both]].astype(np.int32)
        sub._value = self._value[both]
        sub._dense = False
        return sub

    def _lengths(self, values, threshold, probability, pij_min, links: bool, cut: bool = True) -> np.ndarray:
        return _core.conn_pair_lengths(np.ascontiguousarray(values, dtype=np.float64), self.kind, threshold,
                                       probability, pij_min, self.convention == "conefor", links, cut)

    def _all_pairs(self, threshold, probability, pij_min, expand, links: bool) -> np.ndarray:
        n = self.n_nodes
        if n * n > 4e8:
            raise MemoryError(f"all-pairs lengths of {n} nodes do not fit in memory")
        return _core.conn_all_pairs(self._pi, self._pj, self._value, n, self.kind, threshold, probability, pij_min,
                                    expand, self.convention == "conefor", links, self.n_threads)

    @staticmethod
    def _numerator(D: np.ndarray, a: np.ndarray, links: bool) -> float:
        with np.errstate(divide="ignore", over="ignore"):
            f = np.where(np.isfinite(D), 1.0 / (1.0 + D), 0.0) if links else np.exp(-D)
        return float(a @ f @ a)

    # ---------------------------------------------------------- restoration
    def _restoration_frame(self, chosen, distance, probability, pij_min) -> pd.DataFrame:
        base = self.restoration == 1
        cand = ~base
        existing = self._subgraph(base)
        a = self.attributes
        if self.convention == "makurhini":
            frame = self._removal_frame(chosen, distance, probability, pij_min)
            for metric in ("IIC", "PC"):
                if metric not in chosen:
                    continue
                gain, num_base, _ = self._addition_gains(existing, base, metric, distance, probability, pij_min)
                col = np.zeros(self.n_nodes)
                col[cand] = 100 * gain / (num_base + gain)  # MK_dPCIIC: in % of the graph with the node
                frame[f"d{metric}res"] = col
            return frame
        # Conefor's "adding nodes": the existing graph, and the gain of each node to add.
        removal = existing._removal_frame(chosen, distance, probability, pij_min)
        frame = pd.DataFrame({"id": self.ids, "distance": np.full(self.n_nodes, removal["distance"].iloc[0])})
        for col in removal.columns[2:]:
            values = np.full(self.n_nodes, np.nan)
            values[base] = removal[col].to_numpy()
            frame[col] = values
        adds = self._conefor_additions(existing, base, chosen, distance, probability, pij_min)
        for col, values in adds.items():
            full = np.full(self.n_nodes, np.nan)
            full[cand] = values
            frame[col] = full
        return frame

    def _candidate_edges(self, base, lengths):
        """Edges from every node to add to the existing nodes: CSR over the candidates."""
        cand_index = np.full(self.n_nodes, -1, dtype=np.int64)
        cand_index[~base] = np.arange(int((~base).sum()))
        base_index = np.full(self.n_nodes, -1, dtype=np.int64)
        base_index[base] = np.arange(int(base.sum()))
        ok = ~np.isnan(lengths)
        rows_c, rows_b = [], []
        for c_end, b_end in ((self._pi, self._pj), (self._pj, self._pi)):
            m = ok & ~base[c_end] & base[b_end]
            rows_c.append(cand_index[c_end[m]])
            rows_b.append(np.column_stack([base_index[b_end[m]], lengths[m]]))
        c = np.concatenate(rows_c)
        b = np.concatenate(rows_b)
        order = np.argsort(c, kind="stable")
        c, b = c[order], b[order]
        start = np.searchsorted(c, np.arange(int((~base).sum()) + 1)).astype(np.int64)
        return start, b[:, 0].astype(np.int32), np.ascontiguousarray(b[:, 1])

    def _addition_gains(self, existing, base, metric, distance, probability, pij_min):
        links = metric == "IIC"
        dist, threshold, prob, pmin, expand = self._scenario([metric], distance, probability, pij_min)
        D = existing._all_pairs(threshold, prob, pmin, expand, links)
        lengths = self._lengths(self._value, threshold, prob, pmin, links)
        start, node, length = self._candidate_edges(base, lengths)
        g = _core.conn_addition_gains(D, existing.attributes, start, node, length, self.attributes[~base], links,
                                      self.n_threads)
        return g["num"], self._numerator(D, existing.attributes, links), g

    def _conefor_additions(self, existing, base, chosen, distance, probability, pij_min) -> dict:
        a_base = existing.attributes
        a_cand = self.attributes[~base]
        total = float(a_base.sum())
        out = {}
        dist, threshold, prob, pmin, expand = self._scenario(chosen, distance, probability, pij_min)
        if "A" in chosen:
            out["dAadd"] = 100 * a_cand / total
        if any(m in chosen for m in LINK_METRICS):
            _, r = existing._run(["NC"], distance, probability, pij_min, importance=False)
            comp = r["component"]
            comp_area = np.bincount(comp, weights=a_base, minlength=r["nc"])
            lengths = self._lengths(self._value, threshold, prob, pmin, True)
            start, node, _ = self._candidate_edges(base, lengths)
            degree = np.diff(start)
            touched = [np.unique(comp[node[start[k]:start[k + 1]]]) for k in range(len(a_cand))]
            nc_add = np.array([r["nc"] + 1 - len(t) for t in touched], dtype=float)
            merged = np.array([a_cand[k] + comp_area[t].sum() for k, t in enumerate(touched)])
            sq_add = np.array([r["comp_sq"] - (comp_area[t] ** 2).sum() for t in touched]) + merged**2
            if "NC" in chosen:
                out["dNCadd"] = 100.0 * (nc_add - r["nc"]) / r["nc"]
            if "NL" in chosen:
                out["dNLadd"] = 100.0 * degree / r["nl"] if r["nl"] else np.zeros(len(a_cand))
            if "CCP" in chosen:
                ccp = r["comp_sq"] / total**2
                out["dCCPadd"] = 100 * (sq_add / (total + a_cand) ** 2 - ccp) / ccp
            if "LCP" in chosen:
                out["dLCPadd"] = 100 * (sq_add - r["comp_sq"]) / r["comp_sq"]
            if "IIC" in chosen or "H" in chosen:
                gain, num_base, g = self._addition_gains(existing, base, "IIC", distance, probability, pij_min)
                if "H" in chosen:
                    _, rh = existing._run(["H"], distance, probability, pij_min, importance=False)
                    out["dHadd"] = 100 * g["h"] / rh["h"] if rh["h"] else np.zeros(len(a_cand))
                if "IIC" in chosen:
                    out["dIICadd"] = 100 * gain / num_base
        if any(m in chosen for m in PROBABILITY_METRICS):
            uncut = self._lengths(self._value, threshold, prob, 0.0, False, cut=False)
            p = np.exp(-uncut)
            p = np.where(np.isnan(p), 0.0, p)
            cand_pos = np.full(self.n_nodes, -1, dtype=np.int64)
            cand_pos[~base] = np.arange(len(a_cand))
            f_add = np.zeros(len(a_cand))
            awf_add = np.zeros(len(a_cand))
            for c_end, b_end in ((self._pi, self._pj), (self._pj, self._pi)):
                m = ~base[c_end] & base[b_end]
                np.add.at(f_add, cand_pos[c_end[m]], p[m])
                np.add.at(awf_add, cand_pos[c_end[m]], self.attributes[b_end[m]] * p[m])
            if "F" in chosen or "AWF" in chosen:
                _, rf = existing._run(["F"], distance, probability, pij_min, importance=False)
                if "F" in chosen:
                    out["dFadd"] = 100 * 2 * f_add / rf["f_row"].sum()
                if "AWF" in chosen:
                    out["dAWFadd"] = 100 * 2 * a_cand * awf_add / float(a_base @ rf["awf_row"])
            if "PC" in chosen:
                gain, num_base, _ = self._addition_gains(existing, base, "PC", distance, probability, pij_min)
                out["dPCadd"] = 100 * gain / num_base
        return out

    # --------------------------------------------------------------- links
    def link_importance(self, metric: str = "PC", *, distance=None, probability: float | None = None,
                        pij_min: float | None = None) -> pd.DataFrame:
        """Importance of every link: the loss of the index (in %) once it is removed.

        Makurhini's ``MK_dPCIIC_links`` (``removal``): the links of the IIC
        graph, or the edges of the PC graph (pairs above ``pij_min``).
        Other pairs change nothing (0; for IIC Makurhini reports its constant
        share of the unconnected pairs, as for the links). Parameters as in
        :meth:`overall`.

        Returns
        -------
        pandas.DataFrame
            ``from``, ``to`` (node ids), ``distance`` and ``dIIC_removal`` /
            ``dPC_removal``.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> links = graph.link_importance("PC", distance=1000, probability=0.5)
        >>> links.columns.tolist()
        ['from', 'to', 'distance', 'dPC_removal']
        """
        metric = str(metric).upper()
        if metric not in ("IIC", "PC"):
            raise ValueError("metric must be 'IIC' or 'PC'")
        d, r = self._run([metric], distance, probability, pij_min, importance=False, edges=True)
        if metric == "IIC":
            whole = self._unconnected(r["unreach_aa"])
            num = r["iic_reach"] + whole
            delta = r["edge_iic"] + whole
            i, j = r["link_i"], r["link_j"]
        else:
            num = r["pc_num"]
            delta = r["edge_pc"]
            i, j = r["edge_i"], r["edge_j"]
        return pd.DataFrame({"from": self.ids[i], "to": self.ids[j], "distance": np.nan if d is None else d,
                             f"d{metric}_removal": 100 * delta / num})

    def link_change(self, changes, metric: str = "PC", *, distance=None, probability: float | None = None,
                    pij_min: float | None = None) -> pd.DataFrame:
        """Change of the index when the distance of a pair changes (Makurhini's ``MK_dPCIIC_links(change=)``).

        Parameters
        ----------
        changes
            New values of some pairs: a ``(from, to, value)`` table, or an
            ``n × n`` matrix (the pairs whose value differs are evaluated).
        metric, distance, probability, pij_min
            As in :meth:`link_importance`.

        Returns
        -------
        pandas.DataFrame
            ``from``, ``to``, ``distance`` and ``dIIC_change`` / ``dPC_change``:
            ``100 (I - I') / I`` as Makurhini (negative for a gain). IIC: a
            pair that comes closer than ``distance`` gains a link, any other
            change counts 0. PC: the pair's edge takes the probability of its
            new value (not cut by ``pij_min``).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> changes = [(1, 2, 2000)]                    # the pair 1-2 becomes 2 km apart
        >>> change = graph.link_change(changes, "PC", distance=1000, probability=0.5)
        >>> list(change.columns)
        ['from', 'to', 'distance', 'dPC_change']
        """
        metric = str(metric).upper()
        if metric not in ("IIC", "PC"):
            raise ValueError("metric must be 'IIC' or 'PC'")
        links = metric == "IIC"
        ci, cj, cv = self._change_pairs(changes)
        dist, threshold, prob, pmin, expand = self._scenario([metric], distance, probability, pij_min)
        # Current value and edge of each changed pair.
        key = self._pi.astype(np.int64) * self.n_nodes + self._pj
        wanted = ci.astype(np.int64) * self.n_nodes + cj
        pos = np.minimum(np.searchsorted(key, wanted), max(len(key) - 1, 0))
        present = key[pos] == wanted if len(key) else np.zeros(len(ci), bool)
        old_value = np.where(present, self._value[pos], np.nan) if len(key) else np.full(len(ci), np.nan)
        old_len = self._lengths(old_value, threshold, prob, pmin, links)
        new_len = self._lengths(cv, threshold, prob, pmin, links, cut=links)
        D = self._all_pairs(threshold, prob, pmin, expand, links)
        a = self.attributes
        num_graph = self._numerator(D, a, links)
        if links:
            num = num_graph + (self._unconnected(self._run(["IIC"], distance, probability, pij_min,
                                                           importance=False)[1]["unreach_aa"]))
        else:
            num = num_graph
        out = np.zeros(len(ci))
        differs = ~(cv == old_value)
        if links:
            apply = differs & ~np.isnan(new_len)
            new_link = apply & np.isnan(old_len)
            if new_link.any():
                g = _core.conn_edge_gains(D, a, ci[new_link].astype(np.int32), cj[new_link].astype(np.int32),
                                          np.ones(int(new_link.sum())), True, self.n_threads)
                out[new_link] = g["num"]  # the gains, before the % below
            # MK: a link added over an existing one changes nothing, but the
            # graph without unconnected pairs is still compared with the whole.
            out[apply] = (num - (num_graph + out[apply])) / num * 100
        else:
            better = differs & ~np.isnan(new_len) & (np.isnan(old_len) | (new_len < old_len))
            if better.any():
                g = _core.conn_edge_gains(D, a, ci[better].astype(np.int32), cj[better].astype(np.int32),
                                          new_len[better], False, self.n_threads)
                out[better] = -100 * g["num"] / num
            worse = differs & ~better & ~np.isnan(old_len)
            for k in np.flatnonzero(worse):
                out[k] = 100 * (num - self._replaced(k, pos[k], new_len[k], threshold, prob, pmin, expand)) / num
        return pd.DataFrame({"from": self.ids[ci], "to": self.ids[cj], "distance": np.nan if dist is None else dist,
                             f"d{metric}_change": out})

    def _replaced(self, k, pos, new_len, threshold, prob, pmin, expand) -> float:
        """PC numerator once the edge of pair `pos` takes the length new_len (possibly longer)."""
        value = self._value.copy()
        value[pos] = np.nan
        sub = HabitatGraph.__new__(HabitatGraph)
        sub.__dict__.update(self.__dict__)
        sub._value = value
        D = sub._all_pairs(threshold, prob, pmin, expand, False)
        num = self._numerator(D, self.attributes, False)
        if np.isnan(new_len):
            return num
        g = _core.conn_edge_gains(D, self.attributes, self._pi[pos:pos + 1].astype(np.int32),
                                  self._pj[pos:pos + 1].astype(np.int32), np.array([new_len]), False, self.n_threads)
        return num + float(g["num"][0])

    def _change_pairs(self, changes):
        n = self.n_nodes
        if isinstance(changes, pd.DataFrame) or (np.asarray(changes).ndim == 2 and np.asarray(changes).shape[1] == 3
                                                 and np.asarray(changes).shape != (n, n)):
            table = changes.to_numpy() if isinstance(changes, pd.DataFrame) else np.asarray(changes)
            index = pd.Index(self.ids)
            fi, fj = index.get_indexer(table[:, 0]), index.get_indexer(table[:, 1])
            if np.any(fi < 0) or np.any(fj < 0) or np.any(fi == fj):
                raise ValueError("changes must join two different nodes of the graph")
            v = table[:, 2].astype(np.float64)
            return np.minimum(fi, fj), np.maximum(fi, fj), v
        m = np.asarray(changes, dtype=np.float64)
        if m.shape != (n, n):
            raise ValueError("changes must be a (from, to, value) table or an n × n matrix")
        i, j = np.triu_indices(n, 1)
        current = np.full(len(i), np.nan)
        key = self._pi.astype(np.int64) * n + self._pj
        current[np.searchsorted(i.astype(np.int64) * n + j, key)] = self._value
        v = m[i, j]
        keep = ~((v == current) | (np.isnan(v) & np.isnan(current)))
        return i[keep], j[keep], v[keep]

    # ---------------------------------------------------------- centrality
    def centrality(self, distance=None, *, probability: float | None = None, binary: bool = True) -> pd.DataFrame:
        """Radial and medial centrality of every node, as Makurhini's ``MK_RMCentrality`` (igraph).

        Parameters
        ----------
        distance
            Distance (or distances) of the links (``binary``) or of ``probability``
            (default: the median distance).
        probability
            Dispersal probability at ``distance``; giving it makes the graph
            weighted, as in Makurhini (None: exp(-d / distance)).
        binary
            ``True``: the graph of links, pairs closer than ``distance`` (up to
            it with ``convention="conefor"``). ``False``: every pair, weighted by
            its probability p, with 1 / p as the length of the edge and the
            weight of strength and eigenvector centrality.

        Returns
        -------
        pandas.DataFrame
            ``id``, ``distance``, ``degree`` (binary) or ``strength`` (the sum of
            1 / p), ``eigen`` (eigenvector centrality, the largest 1),
            ``close`` (closeness: 1 / the sum of the distances to the nodes it
            reaches; weighted, igraph's normalised closeness, times their
            number), ``BWC`` (betweenness; as igraph, paths equal within 1e-10
            share it) and ``cluster`` (component). With python-igraph installed
            also ``memb_rw`` (walktrap communities, weighted by p); Makurhini's
            Louvain communities are random and left out.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> distances = np.array([[0, 800, 1500, 5000], [800, 0, 600, 4000],
        ...                       [1500, 600, 0, 3000], [5000, 4000, 3000, 0]])
        >>> graph = ls.HabitatGraph([100, 50, 20, 80], distances, landscape_area=10_000)
        >>> table = graph.centrality(1000)
        >>> table.columns.tolist()
        ['id', 'distance', 'degree', 'eigen', 'close', 'BWC', 'cluster']
        """
        if self.kind != "distance":
            raise ValueError("centrality needs distances between the nodes (kind='distance')")
        if probability is not None:
            binary = False  # as MK_RMCentrality
        frames = [self._centrality_frame(d, probability, binary) for d in _as_list(distance)]
        return pd.concat(frames, ignore_index=True)

    def _centrality_frame(self, distance, probability, binary) -> pd.DataFrame:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n = self.n_nodes
        t = float(self._defaults(distance, None, None)[0])
        v = self._value
        ok = ~np.isnan(v)
        if binary:
            keep = ok & ((v <= t) if self.convention == "conefor" else (v < t))
            weight = np.ones(int(keep.sum()))
            length = weight
        else:
            with np.errstate(divide="ignore"):
                p = np.exp(-(1 / t) * v) if probability is None else np.exp((v * np.log(probability)) / t)
            keep = ok & (p > 0)
            weight = p[keep]
            length = 1.0 / weight
        i, j = self._pi[keep], self._pj[keep]
        c = _core.conn_centrality(np.ascontiguousarray(i, dtype=np.int32), np.ascontiguousarray(j, dtype=np.int32),
                                  np.ascontiguousarray(length, dtype=np.float64), n, binary, self.n_threads)
        A = coo_matrix((np.concatenate([length, length]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                       shape=(n, n)).tocsr()
        sums, reached = c["closeness_sum"], c["reached"]
        with np.errstate(divide="ignore", invalid="ignore"):
            close = np.where(reached > 1, 1.0 / sums, np.nan)
            if not binary:
                close = np.where(reached > 1, (reached - 1) / sums, np.nan)
        frame = pd.DataFrame({"id": self.ids, "distance": t,
                              "degree" if binary else "strength": np.asarray(A.sum(axis=1)).ravel(),
                              "eigen": _eigenvector(A), "close": close, "BWC": c["betweenness"],
                              "cluster": connected_components(A, directed=False)[1] + 1})
        try:
            import igraph as ig
        except ImportError:
            return frame
        graph = ig.Graph(n=n, edges=list(zip(i.tolist(), j.tolist())))
        communities = graph.community_walktrap(weights=None if binary else list(weight))
        frame["memb_rw"] = np.asarray(communities.as_clustering().membership) + 1
        return frame


def _eigenvector(A) -> np.ndarray:
    """igraph's eigenvector centrality: the leading eigenvector, its largest entry 1."""
    n = A.shape[0]
    if A.nnz == 0:
        return np.ones(n)
    if n <= 2000:
        values, vectors = np.linalg.eigh(A.toarray())
        vec = vectors[:, -1]
    else:
        from scipy.sparse.linalg import eigsh

        vec = eigsh(A, k=1, which="LA")[1][:, 0]
    vec = vec / vec[np.argmax(np.abs(vec))]
    vec[np.abs(vec) < 1e-15] = 0.0
    return vec
