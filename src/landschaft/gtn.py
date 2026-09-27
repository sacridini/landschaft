# SPDX-License-Identifier: GPL-3.0-or-later
"""Graph Theory Network (GTN) & Connectivity Analysis on MSPA maps.

This module reimplements GuidosToolbox (GTB) connectivity tools:
- :func:`gtn_components`: Connected components of MSPA Cores and Bridges, Equivalent
  Connected Area (ECA), and Degree of Coherence (DOC).
- :func:`gtn_importance`: Quantify the importance of links (Bridges/corridors) and
  nodes (Cores/stepping stones) using delta-PC (probability of connectivity loss).
- :func:`conefor_inputs`: Export node and link topology for external modeling in
  Conefor Sensinode.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
Saura, S. & Pascual-Hortal, L. (2007). A new habitat availability index to integrate
    connectivity in landscape conservation planning: Comparison with existing indices
    and application to a case study. Landscape and Urban Planning, 83(2-3), 91-103.
Saura, S. & Rubio, L. (2010). A common currency for the different ways in which
    patches and links contribute to habitat availability and connectivity in the
    landscape. Ecography, 33(3), 523-537.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import xarray as xr

from . import _core
from .io import GTN_COMPONENTS_COLORMAP, GTN_IMPORTANCE_COLORMAP

__all__ = [
    "gtn_components",
    "gtn_importance",
    "conefor_inputs",
    "GTNComponentsResult",
    "GTNImportanceResult",
    "CORE_CLASSES",
    "BRIDGE_CLASSES",
]

#: MSPA class codes for Core and Bridge features
CORE_CLASSES: tuple[int, ...] = (17, 117)
BRIDGE_CLASSES: tuple[int, ...] = (33, 35, 37, 133, 135, 137)
MISSING_CLASS: int = 129

# GTB visualization palette codes for components
_GTB_NW_COLORS = (1, 5, 9, 17, 33, 65, 80, 150, 176)
_GTB_ISOLATED_CORE_COLOR = 103
_GTB_BACKGROUND_COLOR = 0
_GTB_MISSING_COLOR = 129

# GTB visualization palette codes for importance
_GTB_CORE_BASE_COLOR = 176
_GTB_BRIDGE_BASE_COLOR = 103
_GTB_LINK_HOT_10 = 195
_GTB_LINK_HOT_5 = 194
_GTB_LINK_HOT_1 = 192
_GTB_NODE_HOT_10 = 211
_GTB_NODE_HOT_5 = 210
_GTB_NODE_HOT_1 = 208


def _get_pixel_area_ha(data: Any, pixel_size: float | tuple[float, float] | None = None) -> float | None:
    """Return pixel area in hectares if pixel size is provided or resolvable from .rio (projected)."""
    if pixel_size is not None:
        if isinstance(pixel_size, (int, float)):
            dx = dy = float(pixel_size)
        else:
            dx, dy = float(pixel_size[0]), float(pixel_size[1])
        return abs(dx * dy) / 10000.0

    if isinstance(data, xr.DataArray) and hasattr(data, "rio"):
        try:
            res = data.rio.resolution()
            if res is not None:
                dx, dy = abs(res[0]), abs(res[1])
                # Only treat as projected meters if dx >= 0.1 (degrees in EPSG:4326 are ~0.001)
                if dx >= 0.1 and dy >= 0.1:
                    return (dx * dy) / 10000.0
        except Exception:
            pass
    return None


def _wrap_as_dataarray(
    arr: np.ndarray,
    reference: Any,
    name: str,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray | np.ndarray:
    """Wrap numpy array into xarray.DataArray if reference is one, preserving spatial metadata."""
    if not isinstance(reference, xr.DataArray):
        return arr

    spatial_dims = ("y", "x") if ("y" in reference.dims and "x" in reference.dims) else (reference.dims[-2], reference.dims[-1])
    da = xr.DataArray(
        arr,
        coords={d: reference.coords[d] for d in spatial_dims if d in reference.coords},
        dims=spatial_dims,
        name=name,
    )
    if hasattr(reference, "rio"):
        if reference.rio.crs is not None:
            da.rio.write_crs(reference.rio.crs, inplace=True)
        if reference.rio.transform(recalc=False) is not None:
            da.rio.write_transform(reference.rio.transform(), inplace=True)
        if reference.rio.nodata is not None:
            da.rio.write_nodata(reference.rio.nodata, inplace=True)
    if attrs:
        da.attrs.update(attrs)
    return da


@dataclass
class GTNComponentsResult:
    """Result of :func:`gtn_components`.

    Attributes
    ----------
    components : xarray.DataArray or numpy.ndarray
        Component id of every Core and Bridge pixel (0 elsewhere).
    colored_map : xarray.DataArray or numpy.ndarray
        Map for display: connected networks and isolated cores in their colours.
    stats : pandas.DataFrame
        One row per component: ``component_id``, ``core_pixels``,
        ``bridge_pixels``, ``total_pixels``, ``has_bridge`` and the areas.
    eca, eca_pixels
        Equivalent Connected Area, in hectares (when the pixel size is known)
        and in pixels.
    eca_max, eca_max_pixels
        ECA if all Core pixels formed one component.
    doc : float
        Degree of coherence, ECA / ECA_max in %.
    num_components, num_isolated_cores, num_connected_components
        Number of components, of components without bridges and of networks.
    num_nodes, num_links
        Number of Core patches and Bridge links.
    pixel_area_ha : float or None
        Area of a pixel in hectares.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> morph = ls.mspa(forest)
    >>> network = ls.gtn_components(morph)
    >>> network.num_components
    418
    >>> print(network.summary())  # doctest: +ELLIPSIS
    Network components: 418
    ...
    """

    components: xr.DataArray | np.ndarray
    colored_map: xr.DataArray | np.ndarray
    stats: pd.DataFrame
    eca: float
    eca_pixels: float
    eca_max: float
    eca_max_pixels: int
    doc: float
    num_components: int
    num_isolated_cores: int
    num_connected_components: int
    num_nodes: int
    num_links: int
    pixel_area_ha: float | None = None

    def __repr__(self) -> str:
        unit = "ha" if self.pixel_area_ha is not None else "pixels"
        return (
            f"<GTNComponentsResult: {self.num_components} components "
            f"({self.num_connected_components} networks, {self.num_isolated_cores} isolated), "
            f"ECA={self.eca:.2f} {unit}, DOC={self.doc:.2f}%>"
        )

    def summary(self) -> str:
        """Text summary, as GuidosToolbox's log.

        Returns
        -------
        str
        """
        unit = "ha" if self.pixel_area_ha is not None else "pixels"
        buf = io.StringIO()
        buf.write(f"Network components: {self.num_components}\n")
        buf.write(f"Connected networks: {self.num_connected_components}\n")
        buf.write(f"Isolated core components: {self.num_isolated_cores}\n")
        buf.write(f"Total core nodes: {self.num_nodes}\n")
        buf.write(f"Total bridge links: {self.num_links}\n")
        buf.write(f"ECA: {self.eca:.2f} {unit} ({self.eca_pixels:.1f} pixels)\n")
        buf.write(f"ECA_max: {self.eca_max:.2f} {unit} ({self.eca_max_pixels} pixels)\n")
        buf.write(f"DOC (Degree of Coherence): {self.doc:.2f}%\n")
        return buf.getvalue()


def gtn_components(
    data: xr.DataArray | np.ndarray,
    *,
    pixel_size: float | tuple[float, float] | None = None,
) -> GTNComponentsResult:
    """Components of the habitat network formed by MSPA Cores and Bridges.

    The network of GuidosToolbox's ``nw_components``: Core patches are the nodes
    and Bridges the links. Pixels of Core or Bridge are grouped into components
    (8-connected); only Core pixels count as habitat area. For the whole map:

    - ECA (Equivalent Connected Area) = sqrt(sum of the squared core areas of the
      components);
    - ECA_max, the ECA if all the core area formed one component;
    - DOC (degree of coherence) = ECA / ECA_max in %.

    Parameters
    ----------
    data
        MSPA map (:func:`mspa` classes), DataArray or array.
    pixel_size
        Pixel size in metres (e.g. 25 or ``(25, 25)``) for the areas in hectares;
        read from a georeferenced DataArray when omitted.

    Returns
    -------
    GTNComponentsResult

    See Also
    --------
    gtn_importance : importance of every link and stepping-stone node.
    HabitatGraph : networks of patches with distance-based links.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> morph = ls.mspa(forest)
    >>> network = ls.gtn_components(morph)
    >>> network.num_components, round(network.doc, 2)
    (418, 56.71)
    >>> network.stats.head(3)
       component_id  core_pixels  bridge_pixels  total_pixels  has_bridge
    0             1          199             21           220        True
    1             2          103              0           103       False
    2             3           12             10            22        True
    """
    arr = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    if arr.ndim != 2:
        raise ValueError(f"gtn_components requires a 2-D array, got shape {arr.shape}")

    core = np.isin(arr, CORE_CLASSES)
    bridge = np.isin(arr, BRIDGE_CLASSES)
    missing = arr == MISSING_CLASS

    s8 = np.ones((3, 3), dtype=bool)

    # 1. Label connected components of Core OR Bridge (8-connectivity)
    net_mask = core | bridge
    lbl_comp, nr_comp = ndimage.label(net_mask, structure=s8)

    # Label nodes (cores) and links (bridges) separately
    lbl_nodes, nr_nodes = ndimage.label(core, structure=s8)
    lbl_links, nr_links = ndimage.label(bridge, structure=s8)

    if nr_comp == 0:
        empty_stats = pd.DataFrame(
            columns=["component_id", "core_pixels", "bridge_pixels", "total_pixels", "has_bridge"]
        )
        return GTNComponentsResult(
            components=_wrap_as_dataarray(lbl_comp.astype(np.uint32), data, "components"),
            colored_map=_wrap_as_dataarray(np.zeros_like(arr, dtype=np.uint8), data, "colored_map"),
            stats=empty_stats,
            eca=0.0,
            eca_pixels=0.0,
            eca_max=0.0,
            eca_max_pixels=0,
            doc=0.0,
            num_components=0,
            num_isolated_cores=0,
            num_connected_components=0,
            num_nodes=0,
            num_links=0,
            pixel_area_ha=_get_pixel_area_ha(data, pixel_size),
        )

    # 2. Compute Core areas per component
    comp_core = lbl_comp * core
    core_areas = np.bincount(comp_core.ravel(), minlength=nr_comp + 1)[1:]

    # Bridge counts per component
    comp_bridge = lbl_comp * bridge
    bridge_counts = np.bincount(comp_bridge.ravel(), minlength=nr_comp + 1)[1:]
    has_bridge = bridge_counts > 0

    num_connected = int(has_bridge.sum())
    num_isolated = int((~has_bridge).sum())

    # 3. Overall connectivity metrics
    pcnum = np.sum(core_areas.astype(np.float64) ** 2)
    eca_px = float(np.sqrt(pcnum))
    eca_max_px = int(np.sum(core_areas))
    doc = float((eca_px / eca_max_px) * 100.0) if eca_max_px > 0 else 0.0

    px_ha = _get_pixel_area_ha(data, pixel_size)
    if px_ha is not None:
        eca_val = eca_px * px_ha
        eca_max_val = eca_max_px * px_ha
    else:
        eca_val = eca_px
        eca_max_val = float(eca_max_px)

    # 4. Generate visual colored map matching GTB
    colored = np.zeros_like(arr, dtype=np.uint8)
    for c_id in range(1, nr_comp + 1):
        mask_c = lbl_comp == c_id
        if has_bridge[c_id - 1]:
            color_idx = (c_id - 1) % len(_GTB_NW_COLORS)
            colored[mask_c] = _GTB_NW_COLORS[color_idx]
        else:
            colored[mask_c] = _GTB_ISOLATED_CORE_COLOR

    if missing.any():
        colored[missing] = _GTB_MISSING_COLOR

    # 5. Statistics DataFrame
    stats_data = {
        "component_id": np.arange(1, nr_comp + 1),
        "core_pixels": core_areas,
        "bridge_pixels": bridge_counts,
        "total_pixels": core_areas + bridge_counts,
        "has_bridge": has_bridge,
    }
    if px_ha is not None:
        stats_data["core_ha"] = core_areas * px_ha
        stats_data["total_ha"] = (core_areas + bridge_counts) * px_ha

    stats_df = pd.DataFrame(stats_data)

    comp_attrs = {
        "landschaft_product": "GTN_COMPONENTS_DATA",
        "num_components": nr_comp,
        "eca": eca_val,
        "doc": doc,
    }
    colored_attrs = {
        "landschaft_product": "GTN_COMPONENTS",
        "colormap": "gtn_components",
        "num_components": nr_comp,
        "eca": eca_val,
        "doc": doc,
    }

    return GTNComponentsResult(
        components=_wrap_as_dataarray(lbl_comp.astype(np.uint32), data, "gtn_components", comp_attrs),
        colored_map=_wrap_as_dataarray(colored, data, "gtn_colored", colored_attrs),
        stats=stats_df,
        eca=eca_val,
        eca_pixels=eca_px,
        eca_max=eca_max_val,
        eca_max_pixels=eca_max_px,
        doc=doc,
        num_components=nr_comp,
        num_isolated_cores=num_isolated,
        num_connected_components=num_connected,
        num_nodes=nr_nodes,
        num_links=nr_links,
        pixel_area_ha=px_ha,
    )


@dataclass
class GTNImportanceResult:
    """Result of :func:`gtn_importance`.

    Attributes
    ----------
    importance_map : xarray.DataArray or numpy.ndarray
        dPC of every link (Bridge, positive) and stepping-stone node (Core,
        negative) on their pixels.
    node_link_ids : xarray.DataArray or numpy.ndarray
        Link ids (positive) and node ids (negative) on their pixels.
    colored_map : xarray.DataArray or numpy.ndarray
        Importance classes for display (top 1 %, 5 %, 10 %...).
    links : pandas.DataFrame
        One row per link: ``link_id``, ``component_id``, ``pixels``, ``dpc``,
        ``connected_nodes``, ``normalized_importance``.
    nodes : pandas.DataFrame
        One row per stepping-stone node (degree 2 or more): ``node_id``,
        ``component_id``, ``pixels``, ``dpc``, ``degree``,
        ``normalized_importance``.
    max_link_dpc, max_node_dpc : float
        Largest dPC of a link and of a node.
    num_components, num_links, num_nodes : int
        Counts of the network.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> morph = ls.mspa(forest)
    >>> importance = ls.gtn_importance(morph)
    >>> round(importance.max_link_dpc, 2)
    19.87
    """

    importance_map: xr.DataArray | np.ndarray
    node_link_ids: xr.DataArray | np.ndarray
    colored_map: xr.DataArray | np.ndarray
    links: pd.DataFrame
    nodes: pd.DataFrame
    max_link_dpc: float
    max_node_dpc: float
    num_components: int
    num_links: int
    num_nodes: int

    def __repr__(self) -> str:
        return (
            f"<GTNImportanceResult: {len(self.links)} links (max dPC={self.max_link_dpc:.4f}%), "
            f"{len(self.nodes)} evaluated stepping-stone nodes (max dPC={self.max_node_dpc:.4f}%)>"
        )

    def summary_text(self, title: str = "landschaft Connectivity Importance") -> str:
        """Text report, as GuidosToolbox's ``_cs.txt``.

        Parameters
        ----------
        title
            First line of the report.

        Returns
        -------
        str
        """
        buf = io.StringIO()
        buf.write(f"{title}: Connectivity Importance(Node_max={self.max_node_dpc:.2f}, Link_max={self.max_link_dpc:.2f})\n")
        buf.write("(NOTE: node ID & importance in the actual image data are\n")
        buf.write("NEGATIVE to distinguish them from link ID & importance)\n")
        buf.write("=" * 68 + "\n")
        buf.write(f"Total number of components/links/nodes: {self.num_components} / {self.num_links} / {self.num_nodes}\n")
        buf.write("=" * 68 + "\n")

        for c_id in sorted(self.links["component_id"].unique()):
            c_links = self.links[self.links["component_id"] == c_id]
            c_nodes = self.nodes[self.nodes["component_id"] == c_id] if not self.nodes.empty else pd.DataFrame()

            buf.write(f"                   Component: {c_id}\n")
            buf.write("=" * 64 + "\n")
            buf.write(f"# of links/nodes: {len(c_links)} / {len(c_nodes)}\n")

            if not c_links.empty:
                buf.write("=" * 64 + "\n")
                buf.write("     Link ID        pixels          dPC      normalized.importance\n")
                buf.write("-" * 64 + "\n")
                for _, row in c_links.iterrows():
                    buf.write(
                        f"{int(row['link_id']):>12}  {int(row['pixels']):>12}    "
                        f"{row['dpc']:>10.6f}      {row['normalized_importance']:>10.4f}\n"
                    )

            if not c_nodes.empty:
                buf.write("=" * 64 + "\n")
                buf.write("     Node ID        pixels          dPC      normalized.importance\n")
                buf.write("-" * 64 + "\n")
                for _, row in c_nodes.iterrows():
                    buf.write(
                        f"{-int(row['node_id']):>12}  {int(row['pixels']):>12}    "
                        f"{row['dpc']:>10.6f}      {row['normalized_importance']:>10.4f}\n"
                    )
            buf.write("=" * 64 + "\n")
        return buf.getvalue()


def _extract_topology(
    lbl_links: np.ndarray,
    lbl_nodes: np.ndarray,
) -> dict[int, set[int]]:
    """Find which core nodes each bridge touches in 8-connectivity via fast C++ kernel."""
    contacts = _core.extract_node_link_contacts(
        np.ascontiguousarray(lbl_nodes, dtype=np.int64),
        np.ascontiguousarray(lbl_links, dtype=np.int64),
    )
    bridge_to_nodes: dict[int, set[int]] = defaultdict(set)
    for lid, nid in contacts:
        bridge_to_nodes[int(lid)].add(int(nid))
    return bridge_to_nodes


def gtn_importance(
    data: xr.DataArray | np.ndarray,
    *,
    components: GTNComponentsResult | None = None,
) -> GTNImportanceResult:
    """Importance of every link (Bridge) and stepping-stone node (Core) of the MSPA network.

    GuidosToolbox's ``nw_importance``: the importance of a link is the loss of
    connectivity (dPC: the drop of the numerator of PC, in %) when it is cut;
    the importance of a node with two or more links, its contribution to
    connecting the others. Maps show links positive and nodes negative, and
    classify the top 1 %, 5 % and 10 %.

    Parameters
    ----------
    data
        MSPA map (:func:`mspa` classes), DataArray or array.
    components
        A :func:`gtn_components` result of the same map (computed when omitted).

    Returns
    -------
    GTNImportanceResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> morph = ls.mspa(forest)
    >>> importance = ls.gtn_importance(morph)
    >>> importance.links.columns.tolist()
    ['link_id', 'component_id', 'pixels', 'dpc', 'connected_nodes', 'normalized_importance']
    >>> importance.nodes.head(2)
       node_id  component_id  pixels       dpc  degree  normalized_importance
    0        6             4     129  0.000011       2               0.000054
    1        7             4      30  0.000007       2               0.000037
    """
    arr = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    if arr.ndim != 2:
        raise ValueError(f"gtn_importance requires a 2-D array, got shape {arr.shape}")

    if components is None:
        components = gtn_components(data)

    core = np.isin(arr, CORE_CLASSES)
    bridge = np.isin(arr, BRIDGE_CLASSES)
    missing = arr == MISSING_CLASS

    s8 = np.ones((3, 3), dtype=bool)

    lbl_comp = np.asarray(components.components.values if isinstance(components.components, xr.DataArray) else components.components)
    nr_comp = components.num_components

    lbl_nodes, nr_nodes = ndimage.label(core, structure=s8)
    lbl_links, nr_links = ndimage.label(bridge, structure=s8)

    h, w = arr.shape

    # Node areas (pixel count)
    node_areas = np.bincount(lbl_nodes.ravel()) if nr_nodes > 0 else np.zeros(1, dtype=int)
    link_areas = np.bincount(lbl_links.ravel()) if nr_links > 0 else np.zeros(1, dtype=int)

    # Component core areas and global PCnum
    comp_core = lbl_comp * core
    h_comp_area = np.bincount(comp_core.ravel(), minlength=nr_comp + 1)
    pcnum = np.sum(h_comp_area[1:].astype(np.float64) ** 2)

    importance_map = np.zeros((h, w), dtype=np.float32)
    node_link_ids = np.zeros((h, w), dtype=np.int32)
    colored = np.zeros((h, w), dtype=np.uint8)

    if nr_nodes == 0 or nr_links == 0 or pcnum == 0:
        empty_links = pd.DataFrame(
            columns=["link_id", "component_id", "pixels", "dpc", "normalized_importance", "connected_nodes"]
        )
        empty_nodes = pd.DataFrame(
            columns=["node_id", "component_id", "pixels", "dpc", "normalized_importance", "degree"]
        )
        return GTNImportanceResult(
            importance_map=_wrap_as_dataarray(importance_map, data, "importance_map"),
            node_link_ids=_wrap_as_dataarray(node_link_ids, data, "node_link_ids"),
            colored_map=_wrap_as_dataarray(colored, data, "colored_map"),
            links=empty_links,
            nodes=empty_nodes,
            max_link_dpc=0.0,
            max_node_dpc=0.0,
            num_components=nr_comp,
            num_links=nr_links,
            num_nodes=nr_nodes,
        )

    # Base colors: Cores = 176, Bridges = 103
    colored[core] = _GTB_CORE_BASE_COLOR
    colored[bridge] = _GTB_BRIDGE_BASE_COLOR
    if missing.any():
        colored[missing] = _GTB_MISSING_COLOR

    # Node/link IDs: positive for links, negative for nodes
    node_link_ids[lbl_links > 0] = lbl_links[lbl_links > 0]
    node_link_ids[lbl_nodes > 0] = -lbl_nodes[lbl_nodes > 0]

    # Extract topology
    bridge_to_nodes = _extract_topology(lbl_links, lbl_nodes)

    link_records: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []

    # Process each component
    for idc in range(1, nr_comp + 1):
        maskcomp = lbl_comp == idc
        sub_lids = np.unique(lbl_links[maskcomp & bridge])
        sub_lids = sub_lids[sub_lids > 0]

        # If component has no links, skip
        if len(sub_lids) == 0:
            continue

        sub_nids = np.unique(lbl_nodes[maskcomp & core])
        sub_nids = sub_nids[sub_nids > 0]

        hh = pcnum - (h_comp_area[idc].astype(np.float64) ** 2)
        node_idx_map = {n: i for i, n in enumerate(sub_nids)}
        n_count = len(sub_nids)

        # 1. Importance of links in this component
        for l_id in sub_lids:
            rows, cols = [], []
            for other_lid in sub_lids:
                if other_lid == l_id:
                    continue
                nds = list(bridge_to_nodes[other_lid])
                for i in range(len(nds)):
                    for j in range(i + 1, len(nds)):
                        u, v = node_idx_map[nds[i]], node_idx_map[nds[j]]
                        rows.extend([u, v])
                        cols.extend([v, u])

            adj = sp.csr_matrix(
                (np.ones(len(rows), dtype=bool), (rows, cols)),
                shape=(n_count, n_count),
            )
            _, labels = connected_components(adj, directed=False)
            graph_comp_areas = np.bincount(labels, weights=[node_areas[n] for n in sub_nids])
            pcnew = np.sum(graph_comp_areas.astype(np.float64) ** 2) + hh
            dpc = float((pcnum - pcnew) / pcnum * 100.0)

            mask_link = lbl_links == l_id
            importance_map[mask_link] = np.float32(dpc)

            link_records.append({
                "link_id": int(l_id),
                "component_id": int(idc),
                "pixels": int(link_areas[l_id]),
                "dpc": dpc,
                "connected_nodes": sorted(bridge_to_nodes[l_id]),
            })

        # 2. Importance of nodes in this component
        node_adj: dict[int, set[int]] = defaultdict(set)
        for lid in sub_lids:
            nds = list(bridge_to_nodes[lid])
            for i in range(len(nds)):
                for j in range(len(nds)):
                    if i != j:
                        node_adj[nds[i]].add(nds[j])

        # GTB evaluates only stepping-stone nodes with degree >= 2
        nidsc = [n for n in sub_nids if len(node_adj[n]) >= 2]

        for n_id in nidsc:
            rem_nodes = [n for n in sub_nids if n != n_id]
            rem_map = {n: i for i, n in enumerate(rem_nodes)}
            rows, cols = [], []
            for lid in sub_lids:
                nds = [n for n in bridge_to_nodes[lid] if n != n_id]
                for i in range(len(nds)):
                    for j in range(i + 1, len(nds)):
                        u, v = rem_map[nds[i]], rem_map[nds[j]]
                        rows.extend([u, v])
                        cols.extend([v, u])

            adj_rem = sp.csr_matrix(
                (np.ones(len(rows), dtype=bool), (rows, cols)),
                shape=(len(rem_nodes), len(rem_nodes)),
            )
            _, labels = connected_components(adj_rem, directed=False)
            graph_comp_areas = np.bincount(labels, weights=[node_areas[n] for n in rem_nodes])
            pcnew1 = np.sum(graph_comp_areas.astype(np.float64) ** 2)
            rem_core_area = np.sum([node_areas[n] for n in rem_nodes])
            pcnew2 = float(rem_core_area) ** 2

            dpc = float((pcnew2 - pcnew1) / pcnum * 100.0)

            mask_node = lbl_nodes == n_id
            # GTB assigns negative dPC to nodes in the map
            importance_map[mask_node] = np.float32(-dpc)

            node_records.append({
                "node_id": int(n_id),
                "component_id": int(idc),
                "pixels": int(node_areas[n_id]),
                "dpc": dpc,
                "degree": len(node_adj[n_id]),
            })

    # Normalized importance & DataFrames
    links_df = pd.DataFrame(link_records)
    nodes_df = pd.DataFrame(node_records)

    max_link = float(links_df["dpc"].max()) if not links_df.empty else 0.0
    max_node = float(nodes_df["dpc"].max()) if not nodes_df.empty else 0.0

    if not links_df.empty:
        links_df["normalized_importance"] = (
            (links_df["dpc"] / max_link * 100.0) if max_link > 0 else 0.0
        )
    else:
        links_df["normalized_importance"] = []

    if not nodes_df.empty:
        nodes_df["normalized_importance"] = (
            (nodes_df["dpc"] / max_node * 100.0) if max_node > 0 else 0.0
        )
    else:
        nodes_df["normalized_importance"] = []

    # 3. Hot 1%, 5%, 10% visualization classes
    if max_link > 0:
        htop10 = max_link * 0.90
        htop5 = max_link * 0.95
        htop1 = max_link * 0.99
        colored[(importance_map >= htop10) & bridge] = _GTB_LINK_HOT_10
        colored[(importance_map >= htop5) & bridge] = _GTB_LINK_HOT_5
        colored[(importance_map >= htop1) & bridge] = _GTB_LINK_HOT_1

    if max_node > 0:
        # Note: in importance_map, nodes are negative (-dpc)
        # GTB: mx_core = min(importance); htop = mx_core * 0.9; qc10 = where(importance <= htop)
        neg_max = -max_node
        ntop10 = neg_max * 0.90
        ntop5 = neg_max * 0.95
        ntop1 = neg_max * 0.99
        colored[(importance_map <= ntop10) & core] = _GTB_NODE_HOT_10
        colored[(importance_map <= ntop5) & core] = _GTB_NODE_HOT_5
        colored[(importance_map <= ntop1) & core] = _GTB_NODE_HOT_1

    imp_attrs = {
        "landschaft_product": "GTN_IMPORTANCE_MAP",
        "max_link_dpc": max_link,
        "max_node_dpc": max_node,
    }
    id_attrs = {
        "landschaft_product": "GTN_NODE_LINK_IDS",
    }
    colored_attrs = {
        "landschaft_product": "GTN_IMPORTANCE",
        "colormap": "gtn_importance",
        "max_link_dpc": max_link,
        "max_node_dpc": max_node,
    }

    return GTNImportanceResult(
        importance_map=_wrap_as_dataarray(importance_map, data, "importance_map", imp_attrs),
        node_link_ids=_wrap_as_dataarray(node_link_ids, data, "node_link_ids", id_attrs),
        colored_map=_wrap_as_dataarray(colored, data, "colored_map", colored_attrs),
        links=links_df,
        nodes=nodes_df,
        max_link_dpc=max_link,
        max_node_dpc=max_node,
        num_components=nr_comp,
        num_links=nr_links,
        num_nodes=nr_nodes,
    )


def conefor_inputs(
    data: xr.DataArray | np.ndarray,
    output_dir: str | Path | None = None,
    prefix: str = "mspa",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes and links of the MSPA network as Conefor Sensinode input files.

    GuidosToolbox's ``nw_cs22`` / ``batch_mspaci``: nodes are the Core patches
    (negative ids, area in pixels) and the Bridges (positive ids, area 0); links
    join each Bridge to the nodes it touches, at topological distance 1.

    Parameters
    ----------
    data
        MSPA map (:func:`mspa` classes), DataArray or array.
    output_dir
        Folder where ``<prefix>_nodes_mspa.txt`` and ``<prefix>_links_mspa.txt``
        are written (nothing is written when omitted).
    prefix
        Prefix of the file names.

    Returns
    -------
    nodes, links : pandas.DataFrame
        ``id`` and ``pixels`` per node; ``link_id``, ``node_id`` and
        ``distance`` per link.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> morph = ls.mspa(forest)
    >>> nodes, links = ls.conefor_inputs(morph, output_dir=".")
    >>> nodes.head(3)
       id  pixels
    0  -1     194
    1  -2     103
    2  -3      10
    >>> links.head(3)
       link_id  node_id  distance
    0        1      -12         1
    1        2       -3         1
    2        2       -4         1
    """
    arr = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    if arr.ndim != 2:
        raise ValueError(f"conefor_inputs requires a 2-D array, got shape {arr.shape}")

    core = np.isin(arr, CORE_CLASSES)
    bridge = np.isin(arr, BRIDGE_CLASSES)

    s8 = np.ones((3, 3), dtype=bool)
    lbl_nodes, nr_nodes = ndimage.label(core, structure=s8)
    lbl_links, nr_links = ndimage.label(bridge, structure=s8)

    node_areas = np.bincount(lbl_nodes.ravel()) if nr_nodes > 0 else np.zeros(1, dtype=int)
    bridge_to_nodes = _extract_topology(lbl_links, lbl_nodes)

    # 1. Build Nodes DataFrame: -1..-N with areas, +1..+L with area 0
    node_ids: list[int] = []
    pixels: list[int] = []

    for i in range(1, nr_nodes + 1):
        node_ids.append(-i)
        pixels.append(int(node_areas[i]))

    for j in range(1, nr_links + 1):
        node_ids.append(j)
        pixels.append(0)

    nodes_df = pd.DataFrame({"id": node_ids, "pixels": pixels})

    # 2. Build Links DataFrame: bridge_id, -node_id, distance (1)
    link_col: list[int] = []
    node_col: list[int] = []
    dist_col: list[int] = []

    for b_id in range(1, nr_links + 1):
        connected_cores = sorted(bridge_to_nodes[b_id])
        for c_id in connected_cores:
            link_col.append(b_id)
            node_col.append(-c_id)
            dist_col.append(1)

    links_df = pd.DataFrame({"link_id": link_col, "node_id": node_col, "distance": dist_col})

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        nodes_file = out_path / f"{prefix}_nodes_mspa.txt"
        with open(nodes_file, "w", encoding="utf-8") as f:
            for _, row in nodes_df.iterrows():
                f.write(f"{int(row['id']):>10} {int(row['pixels']):>12}\n")

        links_file = out_path / f"{prefix}_links_mspa.txt"
        with open(links_file, "w", encoding="utf-8") as f:
            for _, row in links_df.iterrows():
                f.write(f"{int(row['link_id']):>8} {int(row['node_id']):>8} {int(row['distance']):>6}\n")

    return nodes_df, links_df
