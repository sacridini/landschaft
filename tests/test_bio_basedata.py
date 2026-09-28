# SPDX-License-Identifier: GPL-3.0-or-later
"""BaseData, trees and matrices against the original Biodiverse (Perl)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, MATRIX_CASES, NEWICK_CASES, RASTER_CASES, REFERENCE, TREE_FILES, VECTOR_CASES

pytestmark = pytest.mark.reference


def _load(name: str) -> dict:
    return json.loads((REFERENCE / f"{name}.json").read_text(encoding="utf-8"))


def _check_basedata(bd: ls.bio.BaseData, ref: dict, text_axes=()) -> None:
    assert bd.groups == sorted(ref["groups"])
    assert bd.labels == sorted(ref["labels"])
    for g, info in ref["groups"].items():
        i = bd.group_index(g)
        for k, c in enumerate(info["coord"]):
            if k in text_axes:
                assert bd.group_axes(g)[k] == c
            else:
                assert bd.coords[i, k] == c, (g, k)
        assert bd.labels_in_group(g) == {lb: float(v) for lb, v in info["labels"].items()}, g
    rng = bd.label_range
    sc = bd.label_sample_count
    for lb, info in ref["labels"].items():
        assert rng[lb] == info["range"], lb
        assert sc[lb] == info["sample_count"], lb


@pytest.mark.parametrize("name", list(BASEDATA_CASES))
def test_basedata_import_matches_biodiverse(name):
    case = BASEDATA_CASES[name]
    ref = _load("basedata")[name]
    kw = {k: v for k, v in case["landschaft"].items() if v is not None}
    bd = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    text_axes = [k for k, c in enumerate(kw["cell_sizes"]) if c < 0]
    _check_basedata(bd, ref, text_axes)
    assert list(bd.cell_sizes) == [float(c) for c in ref["cell_sizes"]]


@pytest.mark.parametrize("name", list(RASTER_CASES))
def test_raster_import_matches_biodiverse(name):
    case = RASTER_CASES[name]
    ref = _load("rasters")[name]
    bd = ls.bio.BaseData.from_raster(
        [DATA / f for f in case["files"]],
        labels_as_bands=bool(case["labels_as_bands"]),
        cell_sizes=case["cell_sizes"],
        cell_origins=case.get("cell_origins"),
    )
    _check_basedata(bd, ref)


def _check_tree(t: ls.bio.Tree, ref: dict) -> None:
    assert sorted(t.names) == sorted(ref["nodes"])
    for name, info in ref["nodes"].items():
        i = t.index(name)
        assert t.length[i] == info["length"], name
        p = t.parent[i]
        assert (t.names[p] if p >= 0 else None) == info["parent"], name
        assert [t.names[c] for c in t.children[i]] == info["children"], name
        assert bool(info["terminal"]) == (not t.children[i])
    assert t.total_length == ref["total_length"]


@pytest.mark.parametrize("fname", TREE_FILES)
def test_tree_files_match_biodiverse(fname):
    refs = _load("trees")["files"][fname]
    trees = ls.bio.read_trees(DATA / fname)
    assert [t.name for t in trees] == [r["name"] for r in refs]
    for t, r in zip(trees, refs):
        _check_tree(t, r)


@pytest.mark.parametrize("name", list(NEWICK_CASES))
def test_newick_matches_biodiverse(name):
    refs = _load("trees")["newick"][name]
    trees = ls.bio.read_trees(NEWICK_CASES[name])
    assert len(trees) == len(refs)
    for t, r in zip(trees, refs):
        _check_tree(t, r)


@pytest.mark.parametrize("name", list(MATRIX_CASES))
def test_matrix_matches_biodiverse(name):
    case = MATRIX_CASES[name]
    ref = _load("matrices")[name]
    m = ls.bio.read_matrix(DATA / case["file"], **case["landschaft"])
    assert m.elements == ref["elements"]
    n_ref = 0
    for a, row in ref["pairs"].items():
        for b, v in row.items():
            assert m.get(a, b) == v, (a, b)
            n_ref += 1
    # no pair beyond the reference ones
    assert int(np.count_nonzero(~np.isnan(m.values))) == n_ref


# -- landschaft-only behaviour --------------------------------------------------------

def test_snap_coordinates_matches_formula():
    x = np.array([-0.05, 0.0, 0.049999999999, 0.05, 0.1, 0.15, 1.25, -1.25])
    got = ls.bio.snap_coordinates(x, 0.1)
    assert np.allclose(got, [-0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 1.25, -1.25])


def test_from_dataframe_long_and_wide_agree():
    long = pd.DataFrame({"sp": ["a", "b", "a", "c"], "x": [1.0, 1.2, 5.5, 5.1], "y": [0.1, 0.2, 0.3, 0.4], "n": [2, 1, 3, 1]})
    bd1 = ls.bio.BaseData.from_dataframe(long, ["x", "y"], "sp", "n", cell_sizes=[2, 2])
    wide = long.pivot_table(index=["x", "y"], columns="sp", values="n", fill_value=0).reset_index()
    bd2 = ls.bio.BaseData.from_dataframe(wide, ["x", "y"], label_columns=["a", "b", "c"], cell_sizes=[2, 2])
    assert bd1.groups == bd2.groups == ["1:1", "5:1"]
    assert (bd1.counts != bd2.counts).nnz == 0
    assert bd1.labels_in_group("1:1") == {"a": 2.0, "b": 1.0}


def test_from_sparse_merges_sites():
    m = sp.csr_matrix(np.array([[1, 0, 2], [0, 4, 1], [3, 0, 0]]))
    bd = ls.bio.BaseData.from_sparse(m, [[0.2, 0.2], [0.7, 0.4], [1.5, 0.5]], ["x", "y", "z"], cell_sizes=[1, 1])
    assert bd.groups == ["0.5:0.5", "1.5:0.5"]
    assert bd.labels_in_group("0.5:0.5") == {"x": 1.0, "y": 4.0, "z": 3.0}
    assert bd.label_range.to_dict() == {"x": 2, "y": 1, "z": 1}


def test_from_xarray_threshold_and_to_raster_roundtrip():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(1)
    z = rng.random((3, 4, 5))
    da = xr.DataArray(z, dims=("sp", "y", "x"), coords={"sp": ["a", "b", "c"], "y": 35.0 - 10 * np.arange(4), "x": 5.0 + 10 * np.arange(5)})
    bd = ls.bio.BaseData.from_xarray(da, "sp", threshold=0.5)
    assert bd.cell_sizes == (10.0, 10.0)
    rich = bd.to_raster(bd.group_richness)
    expected = (z >= 0.5).sum(axis=0).astype(float)
    expected[expected == 0] = np.nan  # groups without labels are not created
    got = rich.values
    # align on the full grid: to_raster spans the groups' extent
    full = np.full((4, 5), np.nan)
    rows = ((35.0 - rich.y.values) / 10).astype(int)
    cols = ((rich.x.values - 5.0) / 10).astype(int)
    full[np.ix_(rows, cols)] = got
    np.testing.assert_array_equal(full, expected)


def test_points_from_geodataframe(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"sp": ["a", "a", "b"]}, geometry=[Point(10, 10), Point(12, 15), Point(35, 5)], crs="EPSG:3857")
    path = tmp_path / "pts.parquet"
    gdf.to_parquet(path)
    bd = ls.bio.BaseData.from_points(path, "sp", cell_sizes=[20, 20])
    assert bd.groups == ["10:10", "30:10"]
    assert bd.labels_in_group("10:10") == {"a": 2.0}
    assert bd.crs == gdf.crs
    cells = bd.to_geodataframe(bd.group_richness)
    assert cells.geometry.iloc[0].bounds == (0.0, 0.0, 20.0, 20.0)


def test_tree_helpers():
    t = ls.bio.read_tree("((a:1,b:2)ab:1,c:3)root:0.5;")
    assert t.terminal_names == ["a", "b", "c"]
    assert t.total_length == 7.5
    assert [t.names[i] for i in t.postorder()] == ["a", "b", "ab", "c", "root"]
    assert ls.bio.read_tree(t.to_newick()).names == t.names


@pytest.mark.parametrize("name", list(VECTOR_CASES))
def test_vector_import_matches_biodiverse(name):
    case = VECTOR_CASES[name]
    ref = _load("vectors")[name]
    bd = ls.bio.BaseData.from_vector(
        DATA / case["files"][0], case["labels"], case["cell_sizes"], case.get("cell_origins"),
        count_columns=case.get("counts"), measure=case.get("measure"),
    )
    if case.get("measure"):
        # areas/lengths of the pieces: GEOS vs our shapely calls, to rounding
        assert bd.groups == sorted(ref["groups"])
        for g, info in ref["groups"].items():
            got = bd.labels_in_group(g)
            assert got.keys() == info["labels"].keys()
            for lb, v in info["labels"].items():
                assert got[lb] == pytest.approx(v, rel=1e-12), (g, lb)
    else:
        _check_basedata(bd, ref)
