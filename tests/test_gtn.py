# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Graph Theory Network (GTN) & Connectivity Analysis in landschaft.

Includes an accuracy verification against the original GuidosToolbox (GTB)
IDL implementation logic from guidostoolbox.pro (lines 12542-12880).
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
import xarray as xr

import landschaft as ls
from landschaft.gtn import BRIDGE_CLASSES, CORE_CLASSES, _extract_topology


# --- Reference GTB Simulation Function ----------------------------------------

def gtb_reference_gtn(mspa_arr: np.ndarray, eew: int = 1) -> dict:
    """Independent step-by-step reproduction of GTB's nw_components & nw_importance logic.

    Follows guidostoolbox.pro lines 12565-12850 directly using raster-based labeling.
    """
    sz_y, sz_x = mspa_arr.shape
    eew2 = eew * 2
    # GTB line 12573: ext = bytarr(sz(0) + eew2, sz(1) + eew2)
    ext = np.zeros((sz_y + eew2, sz_x + eew2), dtype=np.uint8)
    ext[eew : eew + sz_y, eew : eew + sz_x] = mspa_arr

    # GTB lines 12578-12579
    corex = np.isin(ext, CORE_CLASSES)
    bridgex = np.isin(ext, BRIDGE_CLASSES)
    image0 = (corex.astype(np.uint8) * 1) + (bridgex.astype(np.uint8) * 2)

    s8 = np.ones((3, 3), dtype=bool)

    # GTB line 12588: lbl_comp = label_region(image0, /all, /ulong)
    lbl_comp, nr_comp = ndimage.label(image0 > 0, structure=s8)

    # GTB line 12592: h_comp_area = histogram(lbl_comp * (image0 EQ 1b), /l64)
    comp_core = lbl_comp * (image0 == 1)
    h_comp_area = np.bincount(comp_core.ravel(), minlength=nr_comp + 1)

    # GTB line 12595: pcnum = total(h_comp_area(1:*)^2, /double)
    pcnum = float(np.sum(h_comp_area[1:].astype(np.float64) ** 2))
    eca = float(np.sqrt(pcnum))
    eca_max = float(np.sum(h_comp_area[1:]))
    doc = float((eca / eca_max) * 100.0) if eca_max > 0 else 0.0

    # GTB lines 12690-12700: nodes and links
    lbl_links, nr_links = ndimage.label(image0 == 2, structure=s8)
    lbl_nodes, nr_nodes = ndimage.label(image0 == 1, structure=s8)

    # GTB lines 12710-12735: bridge to node topology via dilation
    dilated_nodes = ndimage.maximum_filter(lbl_nodes, size=3, mode="constant", cval=0)
    bridge_edges = (lbl_links > 0) & (dilated_nodes > 0)
    edge_ys, edge_xs = np.nonzero(bridge_edges)

    bridge_to_nodes = defaultdict(set)
    for y, x in zip(edge_ys, edge_xs):
        b_id = int(lbl_links[y, x])
        y_min, y_max = max(0, y - 1), min(lbl_links.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(lbl_links.shape[1], x + 2)
        cores_near = np.unique(lbl_nodes[y_min:y_max, x_min:x_max])
        for c_id in cores_near:
            if c_id > 0:
                bridge_to_nodes[b_id].add(int(c_id))

    importance = np.zeros_like(image0, dtype=np.float32)
    link_dpc_dict: dict[int, float] = {}
    node_dpc_dict: dict[int, float] = {}

    # GTB lines 12750-12840: loop over components
    for idc in range(1, nr_comp + 1):
        qc = lbl_comp == idc
        lids = np.unique(lbl_links[qc & (image0 == 2)])
        lids = lids[lids > 0]
        if len(lids) == 0:
            continue

        nids = np.unique(lbl_nodes[qc & (image0 == 1)])
        nids = nids[nids > 0]

        # Connection table to find nodes with degree >= 2
        node_adj = defaultdict(set)
        for lid in lids:
            nds = list(bridge_to_nodes[lid])
            for i in range(len(nds)):
                for j in range(len(nds)):
                    if i != j:
                        node_adj[nds[i]].add(nds[j])
        nidsc = [n for n in nids if len(node_adj[n]) >= 2]

        ys, xs = np.nonzero(qc)
        x1, x2 = max(0, xs.min() - 1), min(ext.shape[1], xs.max() + 2)
        y1, y2 = max(0, ys.min() - 1), min(ext.shape[0], ys.max() + 2)

        maskcomp = lbl_comp[y1:y2, x1:x2] == idc
        subim = image0[y1:y2, x1:x2] * maskcomp
        subl0 = subim != 2  # cores only
        sub_links = lbl_links[y1:y2, x1:x2]
        sub_nodes = lbl_nodes[y1:y2, x1:x2]

        hh = pcnum - float(h_comp_area[idc]) ** 2

        # GTB lines 12805-12815: importance of links (raster method)
        for lid in lids:
            tmp = subim.copy()
            tmp[sub_links == lid] = 0
            lbl_tmp, _ = ndimage.label(tmp > 0, structure=s8)
            hca = np.bincount((lbl_tmp * subl0).ravel())
            pcnew = float(np.sum(hca[1:].astype(np.float64) ** 2)) + hh
            imp = float((pcnum - pcnew) / pcnum * 100.0)
            importance[lbl_links == lid] = imp
            link_dpc_dict[int(lid)] = imp

        # GTB lines 12820-12835: importance of nodes (raster method)
        tmp2 = maskcomp * subl0
        for nid in nidsc:
            tmp = subim.copy()
            tmp[sub_nodes == nid] = 0
            lbl_tmp, _ = ndimage.label(tmp > 0, structure=s8)
            hca = np.bincount((lbl_tmp * subl0).ravel())
            pcnew1 = float(np.sum(hca[1:].astype(np.float64) ** 2))

            tmp_isolated = tmp2.copy()
            tmp_isolated[sub_nodes == nid] = 0
            hca_isolated = np.bincount(tmp_isolated.ravel())
            pcnew2 = float(np.sum(hca_isolated[1:].astype(np.float64) ** 2))

            imp = float((pcnew2 - pcnew1) / pcnum * 100.0)
            importance[lbl_nodes == nid] = -imp
            node_dpc_dict[int(nid)] = imp

    # Crop out padding
    lbl_comp_cropped = lbl_comp[eew : eew + sz_y, eew : eew + sz_x]
    importance_cropped = importance[eew : eew + sz_y, eew : eew + sz_x]

    return {
        "nr_comp": nr_comp,
        "nr_nodes": nr_nodes,
        "nr_links": nr_links,
        "eca": eca,
        "eca_max": eca_max,
        "doc": doc,
        "lbl_comp": lbl_comp_cropped,
        "importance": importance_cropped,
        "link_dpcs": link_dpc_dict,
        "node_dpcs": node_dpc_dict,
    }


# --- Unit & Accuracy Tests ---------------------------------------------------

@pytest.fixture(scope="module")
def input_mspa():
    """Load test input.tif and compute MSPA."""
    da = ls.load_raster(ls.example_data("input"))
    return ls.mspa(da, edge_width=1, connectivity=8)


def test_gtn_accuracy_against_gtb_reference(input_mspa):
    """Verify bit-exact agreement between landschaft GTN and GTB's IDL reference."""
    ref = gtb_reference_gtn(input_mspa.values, eew=1)

    comp_res = ls.gtn_components(input_mspa)
    imp_res = ls.gtn_importance(input_mspa, components=comp_res)

    # 1. Global component metrics
    assert comp_res.num_components == ref["nr_comp"]
    assert comp_res.num_nodes == ref["nr_nodes"]
    assert comp_res.num_links == ref["nr_links"]
    assert pytest.approx(comp_res.eca_pixels, rel=1e-7) == ref["eca"]
    assert pytest.approx(comp_res.eca_max_pixels, rel=1e-7) == ref["eca_max"]
    assert pytest.approx(comp_res.doc, rel=1e-7) == ref["doc"]

    # 2. Components raster matching
    comp_arr = np.asarray(comp_res.components.values)
    assert (comp_arr > 0).sum() == (ref["lbl_comp"] > 0).sum()
    assert comp_arr.shape == input_mspa.shape

    # 3. Link dPCs matching 100%
    for _, row in imp_res.links.iterrows():
        lid = int(row["link_id"])
        expected_dpc = ref["link_dpcs"][lid]
        assert pytest.approx(row["dpc"], abs=1e-5) == expected_dpc

    # 4. Node dPCs matching 100%
    for _, row in imp_res.nodes.iterrows():
        nid = int(row["node_id"])
        expected_dpc = ref["node_dpcs"][nid]
        assert pytest.approx(row["dpc"], abs=1e-5) == expected_dpc

    # 5. Importance map raster matching
    imp_map = np.asarray(imp_res.importance_map.values)
    diff = np.abs(imp_map - ref["importance"])
    assert diff.max() < 1e-4


def test_gtn_synthetic_line_graph():
    """Test GTN on a synthetic 3-core line graph: [Core1] --Bridge1-- [Core2] --Bridge2-- [Core3]."""
    # 20x20 image
    arr = np.zeros((20, 20), dtype=np.uint8)
    # Core 1 (area 4)
    arr[2:4, 2:4] = 17
    # Bridge 1
    arr[3, 4:7] = 33
    # Core 2 (stepping stone, area 4)
    arr[2:4, 7:9] = 17
    # Bridge 2
    arr[3, 9:12] = 33
    # Core 3 (area 4)
    arr[2:4, 12:14] = 17

    comp = ls.gtn_components(arr)
    assert comp.num_components == 1
    assert comp.num_nodes == 3
    assert comp.num_links == 2
    assert comp.num_connected_components == 1
    assert comp.num_isolated_cores == 0

    # Total core area = 4 + 4 + 4 = 12
    # In single component: ECA = 12, DOC = 100%
    assert comp.eca_pixels == 12.0
    assert comp.eca_max_pixels == 12
    assert comp.doc == 100.0

    imp = ls.gtn_importance(arr, components=comp)
    assert len(imp.links) == 2
    assert len(imp.nodes) == 1  # only Core 2 has degree >= 2

    # If Bridge 1 is removed: Core 1 (4) is separated from Core 2+3 (8)
    # PC_orig = 12^2 = 144
    # PC_new = 4^2 + 8^2 = 16 + 64 = 80
    # dPC = (144 - 80) / 144 * 100 = 64 / 144 * 100 = 44.4444%
    assert pytest.approx(imp.links.iloc[0]["dpc"], rel=1e-4) == (64 / 144 * 100.0)
    assert pytest.approx(imp.links.iloc[1]["dpc"], rel=1e-4) == (64 / 144 * 100.0)

    # Core 2 is a stepping stone connecting Core 1 and Core 3:
    # If Core 2 is removed: remaining connected = Core 1 (4) and Core 3 (4), separate -> pcnew1 = 4^2 + 4^2 = 32
    # Remaining isolated without links = Core 1 (4) + Core 3 (4) -> total = 8 -> pcnew2 = 8^2 = 64
    # dPC = (64 - 32) / 144 * 100 = 32 / 144 * 100 = 22.2222%
    assert pytest.approx(imp.nodes.iloc[0]["dpc"], rel=1e-4) == (32 / 144 * 100.0)


def test_gtn_synthetic_redundant_cycle():
    """Test GTN on a 3-core cycle where removing one bridge does not disconnect the graph."""
    arr = np.zeros((25, 25), dtype=np.uint8)
    # Core 1
    arr[3:6, 3:6] = 17
    # Core 2
    arr[3:6, 18:21] = 17
    # Core 3
    arr[18:21, 10:13] = 17

    # Bridge 1-2 (top)
    arr[4, 6:18] = 33
    # Bridge 1-3 (left)
    arr[6:19, 4] = 33
    arr[19, 4:10] = 33
    # Bridge 2-3 (right)
    arr[6:19, 20] = 33
    arr[19, 13:20] = 33

    comp = ls.gtn_components(arr)
    assert comp.num_components == 1
    assert comp.num_nodes == 3
    assert comp.num_links == 3

    imp = ls.gtn_importance(arr, components=comp)
    # Because there are 2 alternate paths between all pairs of nodes,
    # severing any single link leaves all 3 nodes connected!
    # So link dPC for all 3 links must be exactly 0.0%
    for _, row in imp.links.iterrows():
        assert row["dpc"] == 0.0



def test_gtn_isolated_cores_only():
    """Test GTN on an array with cores but NO bridges."""
    arr = np.zeros((30, 30), dtype=np.uint8)
    # 3 isolated cores
    arr[2:5, 2:5] = 17    # area 9
    arr[10:14, 10:14] = 17  # area 16
    arr[20:25, 20:25] = 17  # area 25

    comp = ls.gtn_components(arr)
    assert comp.num_components == 3
    assert comp.num_connected_components == 0
    assert comp.num_isolated_cores == 3

    # ECA = sqrt(9^2 + 16^2 + 25^2) = sqrt(81 + 256 + 625) = sqrt(962) = 31.016
    # ECA_max = 9 + 16 + 25 = 50
    # DOC = 31.016 / 50 * 100 = 62.03%
    expected_eca = np.sqrt(81 + 256 + 625)
    assert pytest.approx(comp.eca_pixels, rel=1e-5) == expected_eca
    assert comp.eca_max_pixels == 50
    assert pytest.approx(comp.doc, rel=1e-5) == (expected_eca / 50.0 * 100.0)

    # In importance, no links to evaluate
    imp = ls.gtn_importance(arr, components=comp)
    assert imp.links.empty
    assert imp.nodes.empty
    assert imp.max_link_dpc == 0.0
    assert imp.max_node_dpc == 0.0


def test_gtn_empty_landscape():
    """Test GTN on an empty/background-only array."""
    arr = np.zeros((20, 20), dtype=np.uint8)
    comp = ls.gtn_components(arr)
    assert comp.num_components == 0
    assert comp.eca == 0.0
    assert comp.doc == 0.0

    imp = ls.gtn_importance(arr, components=comp)
    assert imp.links.empty
    assert imp.nodes.empty


def test_conefor_export(input_mspa):
    """Test ConeforInputs generation and text file export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        nodes_df, links_df = ls.conefor_inputs(input_mspa, output_dir=tmp_path, prefix="test")

        assert not nodes_df.empty
        assert not links_df.empty
        assert "id" in nodes_df.columns
        assert "pixels" in nodes_df.columns
        assert "link_id" in links_df.columns
        assert "node_id" in links_df.columns

        # Core node IDs are negative
        core_nodes = nodes_df[nodes_df["id"] < 0]
        assert (core_nodes["pixels"] > 0).all()

        # Bridge node IDs are positive with area 0
        bridge_nodes = nodes_df[nodes_df["id"] > 0]
        assert (bridge_nodes["pixels"] == 0).all()

        # Check files on disk
        nodes_file = tmp_path / "test_nodes_mspa.txt"
        links_file = tmp_path / "test_links_mspa.txt"
        assert nodes_file.exists()
        assert links_file.exists()
        assert nodes_file.stat().st_size > 0
        assert links_file.stat().st_size > 0


def test_gtn_write_raster_integration(input_mspa):
    """Verify write_raster properly saves GTN products with embedded colormaps."""
    comp = ls.gtn_components(input_mspa)
    imp = ls.gtn_importance(input_mspa, components=comp)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_comp = tmp_path / "components_colored.tif"
        out_imp = tmp_path / "importance_colored.tif"

        p1 = ls.write_raster(comp.colored_map, out_comp)
        p2 = ls.write_raster(imp.colored_map, out_imp)

        assert p1.exists()
        assert p2.exists()

        # Verify rasterio can read back with colormaps
        import rasterio
        with rasterio.open(p1) as src:
            cmap = src.colormap(1)
            assert cmap is not None
            assert 103 in cmap  # isolated core color
            assert 1 in cmap    # component 1 color
            assert "GTB_NW" in src.tags().get("TIFFTAG_IMAGEDESCRIPTION", "")

        with rasterio.open(p2) as src:
            cmap = src.colormap(1)
            assert cmap is not None
            assert 176 in cmap  # core base
            assert 103 in cmap  # bridge base
            assert "GTB_CS" in src.tags().get("TIFFTAG_IMAGEDESCRIPTION", "")
