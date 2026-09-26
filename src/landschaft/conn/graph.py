# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat connectivity graphs: Conefor Sensinode's indices and Makurhini's node importances."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .. import _core

CONVENTIONS = ("makurhini", "conefor")
KINDS = ("distance", "probability", "link")
METRICS = ("A", "NC", "NL", "H", "CCP", "LCP", "IIC", "F", "AWF", "PC")
LINK_METRICS = ("NC", "NL", "H", "CCP", "LCP", "IIC")
PROBABILITY_METRICS = ("F", "AWF", "PC")

# Makurhini stores the hop count of unconnected pairs as this number, so the
# overall IIC counts a_i a_j / (1 + 16443701) for each of them.
MAKURHINI_UNCONNECTED = 16443701.0


def _metric_list(metrics, kind: str) -> list[str]:
    if metrics is None:
        chosen = list(METRICS)
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
    n_threads
        OpenMP threads (default: all).
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
        n_threads: int | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
        a = np.asarray(attributes, dtype=np.float64).ravel()
        if a.size < 2:
            raise ValueError("a habitat graph needs at least two nodes")
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
        self._pi, self._pj, self._value, self._dense = self._pairs(connections)

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
                    habitat=None, directions: int = 8, nodata=None, band=None, **kwargs) -> "HabitatGraph":
        """Nodes from a raster of patch ids (or the patches of ``habitat`` classes).

        The attributes come from :func:`~landschaft.conn.node_table` (the patch
        areas in ``area_unit`` by default) and the connections from
        :func:`~landschaft.conn.distance_table` (``distance="centroid"`` or
        ``"edge"``, pairs up to ``max_distance`` in ``distance_unit``), as
        Makurhini's ``MK_dPCIIC`` does with raster nodes. Other arguments go
        to :class:`HabitatGraph`; :meth:`to_raster` paints node values on the
        patches.
        """
        from .nodes import distance_table, node_table, read_patches

        patches = read_patches(nodes, habitat=habitat, directions=directions, nodata=nodata, band=band)
        threads = kwargs.get("n_threads")
        table = node_table(patches, attribute=attribute, weighted=weighted, area_unit=area_unit, n_threads=threads)
        pairs = distance_table(patches, distance=distance, max_distance=max_distance, distance_unit=distance_unit,
                               n_threads=threads)
        graph = cls(table["attribute"].to_numpy(), pairs, ids=table["id"].to_numpy(), **kwargs)
        graph._patches = patches
        return graph

    @classmethod
    def from_vector(cls, nodes, *, id: str | None = None, attribute=None, weighted: bool = False,
                    area_unit: str = "m2", distance: str = "centroid", max_distance: float | None = None,
                    distance_unit: str = "m", **kwargs) -> "HabitatGraph":
        """Nodes from polygons or points (GeoDataFrame), as Makurhini's ``MK_dPCIIC``.

        ``id`` names the column of node ids, ``attribute`` a column (default:
        the area in ``area_unit``); distances are ``"centroid"`` or ``"edge"``
        (GEOS), pairs up to ``max_distance``. :meth:`to_geodataframe` joins
        node values to the geometries.
        """
        from .nodes import distance_table, node_table

        threads = kwargs.get("n_threads")
        table = node_table(nodes, attribute=attribute, weighted=weighted, area_unit=area_unit, id=id,
                           n_threads=threads)
        pairs = distance_table(nodes, distance=distance, id=id, max_distance=max_distance,
                               distance_unit=distance_unit, n_threads=threads)
        graph = cls(table["attribute"].to_numpy(), pairs, ids=table["id"].to_numpy(), **kwargs)
        graph._vector = nodes
        return graph

    def to_raster(self, values, name: str | None = None):
        """Node values painted on the patches of :meth:`from_raster` (NaN elsewhere).

        ``values``: one value per node (in node order), a Series indexed by
        node id, or a :meth:`node_importance` table and a column ``name``.
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
                result = result.rio.write_crs(da.rio.crs).rio.write_nodata(np.nan)
            except Exception:
                pass
            return result
        return out

    def to_geodataframe(self, values, name: str | None = None):
        """Node values joined to the geometries of :meth:`from_vector`.

        ``values``: a :meth:`node_importance` table (all its columns; with
        several distances, one block per distance), a Series indexed by node
        id, or one value per node with a ``name``.
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
        """Write Conefor's input files: ``id attribute`` and ``id id value`` lines (connected pairs only)."""
        with open(node_file, "w", encoding="ascii") as f:
            for i, a in zip(self.ids, self.attributes):
                f.write(f"{i}\t{float(a)!r}\n")
        keep = ~np.isnan(self._value)
        with open(connection_file, "w", encoding="ascii") as f:
            for i, j, v in zip(self.ids[self._pi[keep]], self.ids[self._pj[keep]], self._value[keep]):
                f.write(f"{i}\t{j}\t{float(v)!r}\n")

    @property
    def n_nodes(self) -> int:
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

    def _run(self, metrics, distance, probability, pij_min, importance):
        distance, probability, pij_min = self._defaults(distance, probability, pij_min)
        want_links = any(m in LINK_METRICS for m in metrics)
        want_prob = any(m in PROBABILITY_METRICS for m in metrics)
        if self.kind == "probability" and want_links and distance is None:
            raise ValueError("links between probabilities need a minimum probability (distance=)")
        conefor = self.convention == "conefor"
        expand = np.inf
        if conefor and pij_min > 0:
            expand = float(-np.log(pij_min))
        threshold = np.nan if distance is None else float(distance)
        if want_prob and self.kind == "distance" and not threshold > 0:
            raise ValueError("the distance of the probability must be positive")
        r = _core.conn_analyse(self._pi, self._pj, self._value, self.attributes, self.kind, threshold, probability,
                               pij_min, expand, conefor, want_links, want_prob, importance, self.n_threads)
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
        """
        chosen = _metric_list(metrics, self.kind)
        rows = [self._overall_row(chosen, d, probability, pij_min) for d in _as_list(distance)]
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
        """
        chosen = _metric_list(metrics, self.kind)
        frames = [self._importance_frame(chosen, d, probability, pij_min) for d in _as_list(distance)]
        return pd.concat(frames, ignore_index=True)

    def _importance_frame(self, chosen, distance, probability, pij_min) -> pd.DataFrame:
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
        return pd.DataFrame(cols)

    def components(self, distance=None) -> pd.Series:
        """Component (1, 2, ...) of every node in the graph of links at ``distance``."""
        _, r = self._run(["NC"], distance, None, None, importance=False)
        return pd.Series(r["component"] + 1, index=pd.Index(self.ids, name="id"), name="component")
