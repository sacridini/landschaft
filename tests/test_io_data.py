# SPDX-License-Identifier: GPL-3.0-or-later
"""Loading and saving everything: grids (like=, match_grid), vectors, tables, rasterize, save."""

from __future__ import annotations

import dataclasses
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)


def grid(values, transform=TRANSFORM, crs=CRS, nodata=None) -> xr.DataArray:
    values = np.asarray(values)
    h, w = values.shape[-2:]
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    dims = ("y", "x") if values.ndim == 2 else ("band", "y", "x")
    coords = {"y": y, "x": x} if values.ndim == 2 else {"band": np.arange(1, values.shape[0] + 1), "y": y, "x": x}
    da = xr.DataArray(values, coords=coords, dims=dims).rio.write_crs(crs).rio.write_transform(transform)
    return da.rio.write_nodata(nodata) if nodata is not None else da


@pytest.fixture
def ref():
    return grid(np.arange(60 * 80, dtype=np.float32).reshape(60, 80))


# --- grids ---------------------------------------------------------------------


def test_load_raster_like_reprojects_and_takes_the_coordinates(tmp_path, ref):
    coarse = ref.rio.reproject("EPSG:4326")[::2, ::2]
    path = ls.write_raster(coarse, tmp_path / "coarse.tif")
    da = ls.load_raster(path, like=ref)
    assert da.shape == ref.shape and da.rio.crs == ref.rio.crs
    assert (da.x.values == ref.x.values).all() and (da.y.values == ref.y.values).all()
    assert da.rio.transform().almost_equals(ref.rio.transform())
    assert float(np.nanmean(np.abs(da.values - ref.values))) < 150  # bilinear of a smooth ramp
    total = da + ref                                                  # aligns cell by cell
    assert total.shape == ref.shape
    via_path = ls.load_raster(path, like=ls.write_raster(ref, tmp_path / "ref.tif"))
    np.testing.assert_array_equal(via_path.values, da.values)


def test_like_takes_nearest_classes_for_integers(tmp_path, ref):
    classes = grid((np.arange(30 * 40).reshape(30, 40) % 5 + 1).astype(np.uint8),
                   transform=from_origin(500000.0, 4500000.0, 60.0, 60.0))
    path = ls.write_raster(classes, tmp_path / "classes.tif")
    da = ls.load_raster(path, like=ref)
    assert da.dtype == np.uint8 and set(np.unique(da.values)) <= {1, 2, 3, 4, 5}
    np.testing.assert_array_equal(da.values[::2, ::2], classes.values)
    mode = ls.load_raster(path, like=ref, resampling="mode")
    assert mode.shape == ref.shape
    with pytest.raises(ValueError, match="unknown resampling"):
        ls.load_raster(path, like=ref, resampling="magic")


def test_like_on_the_same_grid_keeps_values_and_copies_coordinates(tmp_path, ref):
    path = ls.write_raster(ref, tmp_path / "same.tif")
    da = ls.load_raster(path, like=ref)
    np.testing.assert_array_equal(da.values, ref.values)
    assert da.x is not ref.x and (da.x.values == ref.x.values).all()


def test_like_with_a_stack_and_outside_cells(tmp_path, ref):
    stack = grid(np.ones((2, 20, 30), np.float32))
    path = ls.write_raster(stack, tmp_path / "stack.tif", band_names=["a", "b"])
    da = ls.load_raster(path, like=ref)
    assert da.dims == ("band", "y", "x") and da.band.values.tolist() == ["a", "b"]
    assert np.isnan(da.values[:, 30:, :]).all() and (da.values[:, :20, :30] == 1).all()


def test_match_grid_in_memory_and_errors(ref):
    pytest.importorskip("dask")
    coarse = ref.coarsen(x=2, y=2).mean().chunk({"y": 10})
    back = ls.match_grid(coarse, ref)
    assert back.shape == ref.shape and (back.x.values == ref.x.values).all()
    with pytest.raises(ValueError, match="georeferenced DataArray"):
        ls.match_grid(np.zeros((3, 3)), ref)


# --- vectors -------------------------------------------------------------------


@pytest.fixture
def reserves():
    return gpd.GeoDataFrame({"id": [1, 2], "name": ["north", "south"]}, crs=CRS,
                            geometry=[box(500000, 4498800, 500600, 4499400), box(501000, 4498500, 501600, 4499100)])


def test_vector_roundtrips(tmp_path, reserves):
    for name in ("r.gpkg", "r.geojson", "r.fgb", "r.shp", "sub/r.parquet"):
        path = ls.write_vector(reserves, tmp_path / name)
        back = ls.load_vector(path)
        assert len(back) == 2 and back.crs == reserves.crs and sorted(back["name"]) == ["north", "south"]
    wkt = ls.write_vector(reserves, tmp_path / "r.csv")
    assert "POLYGON" in wkt.read_text()
    ls.write_vector(reserves.geometry, tmp_path / "geoms.gpkg")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        ls.write_vector(pd.DataFrame({"a": [1]}), tmp_path / "x.gpkg")


def test_load_vector_layers_columns_and_like(tmp_path, reserves, ref):
    path = tmp_path / "two.gpkg"
    reserves.to_file(path, layer="reserves")
    reserves.iloc[:1].to_file(path, layer="one")
    assert len(ls.load_vector(path, layer="one")) == 1
    assert list(ls.load_vector(path, layer="reserves", columns=["name"]).columns) == ["name", "geometry"]
    geo = ls.write_vector(reserves.to_crs("EPSG:4326"), tmp_path / "wgs.gpkg")
    on_grid = ls.load_vector(geo, like=ref)
    assert on_grid.crs == ref.rio.crs
    np.testing.assert_allclose(on_grid.total_bounds, reserves.total_bounds, atol=1e-3)


def test_load_vector_from_coordinate_tables(tmp_path, ref):
    pd.DataFrame({"sp": ["a", "b"], "decimalLongitude": [15.0, 15.01], "decimalLatitude": [40.6, 40.61]}).to_csv(
        tmp_path / "gbif.csv", index=False)
    pts = ls.load_vector(tmp_path / "gbif.csv")
    assert pts.crs.to_epsg() == 4326 and pts.geometry.x.tolist() == [15.0, 15.01]
    assert ls.load_vector(tmp_path / "gbif.csv", like=ref).crs == ref.rio.crs
    pd.DataFrame({"E": [500100.0], "N": [4499900.0]}).to_csv(tmp_path / "utm.csv", sep=";", index=False)
    utm = ls.load_vector(tmp_path / "utm.csv", x="E", y="N", crs=CRS)
    assert utm.crs == ref.rio.crs and utm.geometry.x.tolist() == [500100.0]
    with pytest.raises(ValueError, match="pass x="):
        ls.load_vector(tmp_path / "utm.csv")


# --- rasterize -----------------------------------------------------------------


def test_rasterize_mask_field_and_counts(tmp_path, reserves, ref):
    mask = ls.rasterize(reserves, like=ref)
    assert mask.dtype == np.uint8 and mask.shape == ref.shape and mask.rio.crs == ref.rio.crs
    assert int(mask.sum()) == 2 * 20 * 20  # two 600 m squares of 30 m cells
    assert (mask.x.values == ref.x.values).all()
    ids = ls.rasterize(reserves, like=ref, field="id")
    assert set(np.unique(ids.values)) == {0, 1, 2} and ids.name == "id"
    overlap = pd.concat([reserves, reserves.iloc[:1]])
    counts = ls.rasterize(overlap, like=ref, merge="add")
    assert int(counts.max()) == 2
    nan_fill = ls.rasterize(reserves, like=ref, field="id", fill=np.nan)
    assert nan_fill.dtype == np.float32 and np.isnan(nan_fill.rio.nodata)
    small = [Point(500098, 4499888).buffer(3)]  # inside a cell, away from its centre (500105, 4499895)
    assert int(ls.rasterize(small, like=ref, all_touched=True).sum()) == 1
    assert int(ls.rasterize(small, like=ref).sum()) == 0


def test_rasterize_reprojects_and_reads_paths(tmp_path, reserves, ref):
    path = ls.write_vector(reserves.to_crs("EPSG:4326"), tmp_path / "wgs.gpkg")
    ref_path = ls.write_raster(ref, tmp_path / "ref.tif")
    mask = ls.rasterize(path, like=ref_path)
    assert abs(int(mask.sum()) - 800) <= 8
    empty = ls.rasterize(reserves.iloc[:0], like=ref)
    assert int(empty.sum()) == 0
    with pytest.raises(ValueError, match="no column"):
        ls.rasterize(reserves, like=ref, field="area")


# --- tables --------------------------------------------------------------------


def test_load_table_detects_separators_and_decimal_commas(tmp_path):
    (tmp_path / "br.csv").write_text("especie;peso;area\nonca;2,5;1.200\nanta;1;30\n", encoding="utf-8")
    t = ls.load_table(tmp_path / "br.csv")
    assert t["peso"].tolist() == [2.5, 1.0]
    (tmp_path / "tab.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert ls.load_table(tmp_path / "tab.tsv").columns.tolist() == ["a", "b"]
    (tmp_path / "plain.csv").write_text("a,b\n1.5,2\n", encoding="utf-8")
    assert ls.load_table(tmp_path / "plain.csv")["a"].tolist() == [1.5]
    assert ls.load_table(tmp_path / "plain.csv", columns=["b"]).columns.tolist() == ["b"]
    with pytest.raises(ValueError, match="cannot tell the table format"):
        ls.load_table(tmp_path / "x.unknown")


def test_table_roundtrips(tmp_path, reserves):
    df = pd.DataFrame({"cls": [1, 2], "area": [1.5, 2.5]})
    names = ["t.csv", "t.csv.gz", "t.tsv", "t.json", "t.parquet", "t.feather"]
    if importlib.util.find_spec("openpyxl"):
        names.append("t.xlsx")
    for name in names:
        back = ls.load_table(ls.write_table(df, tmp_path / "sub" / name))
        pd.testing.assert_frame_equal(back.reset_index(drop=True), df, check_dtype=False)
    s = pd.Series([0.5, 0.7], index=pd.Index(["pland", "ed"], name="metric"), name="value")
    assert ls.load_table(ls.write_table(s, tmp_path / "s.csv")).columns.tolist() == ["metric", "value"]
    filtered = df[df["cls"] > 1]
    assert ls.load_table(ls.write_table(filtered, tmp_path / "f.csv")).columns.tolist() == ["cls", "area"]
    attrs = ls.load_table(ls.write_table(reserves, tmp_path / "r.gpkg"))
    assert "geometry" not in attrs.columns and attrs["name"].tolist() == ["north", "south"]


# --- save ----------------------------------------------------------------------


def test_save_single_objects(tmp_path, ref, reserves):
    assert ls.save(ref, tmp_path / "ramp").name == "ramp.tif"
    with rasterio.open(tmp_path / "ramp.tif") as src:
        assert src.crs == ref.rio.crs
    out = ls.save(np.ones((60, 80), np.uint8), tmp_path / "ones", like=ref)
    with rasterio.open(out) as src:
        assert src.transform == TRANSFORM
    assert ls.save(pd.DataFrame({"a": [1]}), tmp_path / "t").name == "t.csv"
    assert ls.save(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet").suffix == ".parquet"
    assert ls.save(reserves, tmp_path / "r").name == "r.gpkg"
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        ls.save(np.arange(5), tmp_path / "line")


@dataclasses.dataclass
class _Result(ls._save.Saveable):
    hab: xr.DataArray
    labels: np.ndarray
    matrix: np.ndarray
    table: pd.DataFrame
    per_class: dict
    maps: dict
    n: int
    ratio: float
    note: str
    nothing: None = None


def test_result_folder(tmp_path, ref):
    labels = np.arange(60 * 80, dtype=np.int64).reshape(60, 80) % 7
    res = _Result(hab=ref, labels=labels, matrix=np.eye(3), table=pd.DataFrame({"x": [1, 2]}),
                  per_class={1: 0.5, 2: np.float64(np.nan)}, maps={"a": labels.astype(np.float32), "b": None},
                  n=np.int64(3), ratio=0.25, note="ok")
    folder = res.save(tmp_path / "result")
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["hab.tif", "labels.tif", "maps_a.tif", "matrix.csv", "summary.json", "table.csv"]
    with rasterio.open(folder / "labels.tif") as src:  # numpy map georeferenced like the DataArray
        assert src.crs == ref.rio.crs and src.transform == TRANSFORM and src.dtypes[0] == "uint8"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary == {"result": "_Result", "per_class": {"1": 0.5, "2": None}, "n": 3, "ratio": 0.25, "note": "ok"}
    parquet = ls.save(res, tmp_path / "result_pq", table_format="parquet")
    assert (parquet / "table.parquet").exists()


def test_tool_results_save(tmp_path):
    forest = ls.load_raster(ls.example_data("input"))[:300, :300]
    folder = ls.fos(forest).save(tmp_path / "fos")
    assert {"map.tif", "summary.json"} <= {p.name for p in folder.iterdir()}
    with rasterio.open(folder / "map.tif") as src:
        assert src.crs == forest.rio.crs
    assert json.loads((folder / "summary.json").read_text())["result"] == "FOSResult"
    status = ls.save(ls.restoration_status(forest), tmp_path / "status")
    assert json.loads((status / "summary.json").read_text())["nr_comp"] > 0
    both = ls.save({"fos": ls.fos(forest), "mspa": ls.mspa(forest)}, tmp_path / "both")
    assert (both / "mspa.tif").exists() and (both / "fos" / "map.tif").exists()


def test_results_with_their_own_format(tmp_path):
    res = ls.prioritize.zonation(np.random.default_rng(0).random((2, 10, 12)))
    out = ls.save(res, tmp_path / "zonation")
    assert (out / "rankmap.tif").exists() and (out / "feature_curves.csv").exists()
    tree = ls.bio.read_tree(ls.example_data("tree"))
    nwk = ls.save(tree, tmp_path / "tree")
    assert nwk.suffix == ".nwk" and nwk.read_text().strip().endswith(";")


def test_biodiverse_results_save(tmp_path):
    bd = ls.bio.BaseData.from_table(ls.example_data("sites"), group_columns=["x", "y"],
                                    label_columns=["genus", "species"], cell_sizes=[100000, 100000])
    sp = ls.bio.spatial_analysis(bd, ["sp_self_only()"], calculations=["calc_richness"])
    folder = sp.save(tmp_path / "spatial")
    assert (folder / "table.csv").exists()
    assert json.loads((folder / "summary.json").read_text())["calculations"] == list(sp.calculations)


# --- example data and an end-to-end scenario ----------------------------------------


def test_example_data_covers_every_module():
    for name in ("input", "clc", "gscinput", "normforest", "orforest", "orelevation", "maungawhau", "europe",
                 "gen_dist", "pop_coord", "sites", "tree"):
        assert ls.example_data(name).exists(), name
    assert ls.example_data("orelevation") == ls.surface.example_data("orelevation")
    with pytest.raises(ValueError, match="maungawhau"):
        ls.example_data("nope")


def test_a_scenario_from_files_in_a_few_lines(tmp_path, ref):
    # species on one grid, a cost layer on another, reserves as polygons in another CRS
    rng = np.random.default_rng(1)
    sdm = grid(rng.random((3, 60, 80)).astype(np.float32))
    ls.write_raster(sdm, tmp_path / "sdm.tif", band_names=["sp1", "sp2", "sp3"])
    ls.write_raster(ref.rio.reproject("EPSG:4326"), tmp_path / "hfi.tif")
    pas = gpd.GeoDataFrame(geometry=[box(500300, 4498500, 501200, 4499400)], crs=CRS).to_crs("EPSG:4326")
    ls.write_vector(pas, tmp_path / "pas.gpkg")

    species = ls.load_raster(tmp_path / "sdm.tif")
    hfi = ls.load_raster(tmp_path / "hfi.tif", like=species)
    reserves = ls.rasterize(ls.load_vector(tmp_path / "pas.gpkg"), like=species)
    cost = (1 + hfi / float(hfi.max())).fillna(1)
    res = ls.prioritize.zonation(species, cost=cost, hierarchic_mask=1 + reserves)
    folder = res.save(tmp_path / "out")
    rank = ls.load_raster(folder / "rankmap.tif")
    assert rank.shape == (60, 80) and float(rank.where(reserves == 1).min()) > 0.8
