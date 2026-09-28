# SPDX-License-Identifier: GPL-3.0-or-later
"""Hexagonal and H3 grids, the native format and tree conversions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA


def _site_bd():
    case = BASEDATA_CASES["site_counts"]
    kw = {k: v for k, v in case["landschaft"].items() if v is not None}
    return ls.bio.BaseData.from_table(DATA / case["file"], **kw)


# -- hexagons ------------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_hex_centres_are_nearest_centres(orientation):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-50, 50, 2000), rng.uniform(-50, 50, 2000)
    size = 3.0
    cx, cy = ls.bio.hex_centres(x, y, size, orientation)
    # the chosen centre is the nearest of all lattice centres around the point
    d_chosen = np.hypot(x - cx, y - cy)
    assert np.all(d_chosen <= size + 1e-9)
    # compare with the six neighbouring centres
    w = size * math.sqrt(3)
    if orientation == "pointy":
        offsets = [(w, 0), (-w, 0), (w / 2, 1.5 * size), (-w / 2, 1.5 * size), (w / 2, -1.5 * size),
                   (-w / 2, -1.5 * size)]
    else:
        offsets = [(0, w), (0, -w), (1.5 * size, w / 2), (1.5 * size, -w / 2), (-1.5 * size, w / 2),
                   (-1.5 * size, -w / 2)]
    for ox, oy in offsets:
        assert np.all(d_chosen <= np.hypot(x - cx - ox, y - cy - oy) + 1e-9)


def test_hex_basedata_and_geometry():
    pytest.importorskip("geopandas")
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"sp": rng.choice(list("abcde"), 500), "x": rng.uniform(0, 100, 500),
                       "y": rng.uniform(0, 100, 500)})
    bd = ls.bio.BaseData.from_hex(df, "x", "y", "sp", 10.0)
    assert int(bd.counts.sum()) == 500
    gdf = bd.to_geodataframe()
    assert len(gdf) == bd.n_groups
    np.testing.assert_allclose(gdf.geometry.area, 1.5 * math.sqrt(3) * 100, rtol=1e-12)
    # every record falls in its group's hexagon
    from shapely.geometry import Point

    geoms = dict(zip(gdf.index, gdf.geometry))
    cx, cy = ls.bio.hex_centres(df["x"], df["y"], 10.0)
    from landschaft.bio._names import join_element, perl_number

    for px, py, gx, gy in zip(df["x"], df["y"], cx, cy):
        assert geoms[join_element([perl_number(gx), perl_number(gy)])].buffer(1e-9).contains(Point(px, py))
    res = ls.bio.spatial_analysis(bd, "sp_circle (radius => 18)", ["richness"])
    assert res.table["RICHNESS_ALL"].max() <= 5


def test_h3_basedata():
    h3 = pytest.importorskip("h3")
    pytest.importorskip("geopandas")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"sp": rng.choice(list("abc"), 300), "lat": rng.uniform(-34, -33, 300),
                       "lon": rng.uniform(150, 151, 300)})
    bd = ls.bio.BaseData.from_h3(df, "lat", "lon", "sp", 6)
    cells = ls.bio.h3_cells(df["lat"], df["lon"], 6)
    assert sorted(set(cells)) == sorted(bd.grid["cells"])
    assert int(bd.counts.sum()) == 300
    # each group's coordinates are its cell's centre
    for (x, y), cell in zip(bd.coords, bd.grid["cells"]):
        lat, lng = h3.cell_to_latlng(cell)
        assert abs(x - lng) < 1e-9 and abs(y - lat) < 1e-9
    gdf = bd.to_geodataframe()
    assert gdf.crs.to_epsg() == 4326 and len(gdf) == bd.n_groups


# -- native format ---------------------------------------------------------------------------

def test_basedata_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    bd = _site_bd()
    bd.set_group_properties(ls.bio.read_properties(DATA / "gp_props.csv", [0], {"num_mean": 3}))
    path = tmp_path / "sites.lsbio"
    bd.save(path)
    back = ls.bio.BaseData.load(path)
    assert back.groups == bd.groups and back.labels == bd.labels
    assert (back.counts != bd.counts).nnz == 0
    np.testing.assert_array_equal(back.coords, bd.coords)
    assert back.cell_sizes == bd.cell_sizes
    pd.testing.assert_frame_equal(back.group_properties, bd.group_properties, check_dtype=False)
    a = ls.bio.spatial_analysis(bd, "sp_circle (radius => 100000)", ["endemism_whole"])
    b = ls.bio.spatial_analysis(back, "sp_circle (radius => 100000)", ["endemism_whole"])
    pd.testing.assert_frame_equal(a.table, b.table)


def test_session_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    bd = _site_bd()
    tree = ls.bio.read_tree(DATA / "tree_species.nex")
    res = ls.bio.spatial_analysis(bd, "sp_circle (radius => 100000)", ["richness", "pe", "pe_lists"], tree=tree)
    cl = ls.bio.cluster(bd, prng_seed=4, calculations=["richness"])
    mx = ls.bio.read_matrix(DATA / "Example_matrix.txt", [1, 2])
    path = tmp_path / "study.lsbio"
    ls.bio.save_session(path, sites=bd, tree=tree, mx=mx, pe=res, clusters=cl)
    s = ls.bio.load_session(path)
    assert s["tree"].to_newick() == tree.to_newick()
    np.testing.assert_array_equal(s["mx"].values, mx.values)
    pd.testing.assert_frame_equal(s["pe"].table, res.table)
    pd.testing.assert_frame_equal(s["pe"].lists["PE_WTLIST"].reset_index(drop=True),
                                  res.lists["PE_WTLIST"].reset_index(drop=True), check_dtype=False)
    assert s["clusters"].tree.to_newick() == cl.tree.to_newick()
    assert list(s["clusters"].nodes["children"]) == list(cl.nodes["children"])


# -- trees -----------------------------------------------------------------------------------

@pytest.mark.parametrize("library", ["biopython", "dendropy", "ete"])
def test_tree_round_trips(library):
    tree = ls.bio.read_tree(DATA / "Example_tree.nex")
    if library == "biopython":
        pytest.importorskip("Bio")
        back = ls.bio.Tree.from_biopython(tree.to_biopython())
    elif library == "dendropy":
        pytest.importorskip("dendropy")
        back = ls.bio.Tree.from_dendropy(tree.to_dendropy())
    else:
        pytest.importorskip("ete3")
        back = ls.bio.Tree.from_ete(tree.to_ete())
    assert back.to_newick() == tree.to_newick()
    # the same analyses on the converted tree
    bd = ls.bio.BaseData.from_table(DATA / "Example_site_data.csv", group_columns=[3, 4], label_columns=[1, 2],
                                    cell_sizes=[100000, 100000])
    a = ls.bio.spatial_analysis(bd, "sp_self_only()", ["pd", "pe"], tree=tree)
    b = ls.bio.spatial_analysis(bd, "sp_self_only()", ["pd", "pe"], tree=back)
    pd.testing.assert_frame_equal(a.table, b.table)
