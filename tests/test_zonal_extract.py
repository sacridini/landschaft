# SPDX-License-Identifier: GPL-3.0-or-later
"""clip, zonal_stats, extract, load_raster(clip=, list + like=) and reclassify with a table,
checked against NumPy and GDAL's rasterizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio.features
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, Polygon, box  # noqa: E402

CRS = "EPSG:32633"
T = from_origin(500000.0, 4500000.0, 30.0, 30.0)


def grid(values, transform=T, crs=CRS, nodata=None, names=None) -> xr.DataArray:
    values = np.asarray(values)
    h, w = values.shape[-2:]
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    if values.ndim == 2:
        da = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"), name="v")
    else:
        da = xr.DataArray(values, coords={"band": names or list(range(1, values.shape[0] + 1)), "y": y, "x": x},
                          dims=("band", "y", "x"), name="v")
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    return da.rio.write_nodata(nodata) if nodata is not None else da


def random_polygons(rng, n, x0=500000.0, y0=4500000.0, size=3000.0):
    out = []
    while len(out) < n:
        cx, cy = x0 + rng.uniform(0, size), y0 - rng.uniform(0, size)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 8))
        radii = rng.uniform(20, 500, 8)
        g = Polygon(zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles))).buffer(0)
        if not g.is_empty:
            out.append(g)
    return out


def reference(values, mask, nodata=None):
    v = values[mask].astype(np.float64)
    ok = ~np.isnan(v)
    if nodata is not None:
        ok &= v != nodata
    return v[ok], int((~ok).sum())


@pytest.fixture
def raster():
    rng = np.random.default_rng(3)
    v = rng.random((100, 120)).astype(np.float32) * 100
    v[::9, ::7] = np.nan
    return grid(v)


# --- zonal_stats -------------------------------------------------------------------


@pytest.mark.parametrize("all_touched", [False, True])
def test_zonal_stats_match_numpy_and_gdal(raster, all_touched):
    rng = np.random.default_rng(1)
    polys = random_polygons(rng, 60) + [box(499000, 4497000, 500500, 4499000)]  # partly outside
    zones = gpd.GeoDataFrame({"id": range(len(polys))}, geometry=polys, crs=CRS)
    stats = ["count", "nodata", "sum", "mean", "min", "max", "range", "std", "var", "median", "p10", "p95.5"]
    out = ls.zonal_stats(raster, zones, stats, all_touched=all_touched)
    assert list(out.columns[:2]) == ["id", "geometry"] and len(out) == len(polys)
    values = raster.values
    for i, g in enumerate(polys):
        mask = rasterio.features.geometry_mask([g], out_shape=values.shape, transform=T, invert=True,
                                               all_touched=all_touched)
        v, missing = reference(values, mask)
        row = out.iloc[i]
        assert row["count"] == v.size and row["nodata"] == missing
        if v.size == 0:
            assert np.isnan(row["mean"])
            continue
        expected = [v.sum(), v.mean(), v.min(), v.max(), np.ptp(v), v.std(), v.var(), np.median(v),
                    np.percentile(v, 10), np.percentile(v, 95.5)]
        np.testing.assert_allclose(row[stats[2:]].to_numpy(float), expected, rtol=1e-9, atol=1e-9)


def test_overlapping_and_empty_zones(raster):
    a = box(500300, 4497500, 501500, 4499600)
    zones = gpd.GeoSeries([a, a.buffer(-200), box(0, 0, 10, 10)], crs=CRS)  # nested, outside
    out = ls.zonal_stats(raster, zones, ["count", "mean"])
    assert out["count"].iloc[0] > out["count"].iloc[1] > 0
    assert out["count"].iloc[2] == 0 and np.isnan(out["mean"].iloc[2])
    assert out["count"].dtype == np.int64


def test_majority_minority_unique_on_classes():
    v = np.zeros((50, 50), np.uint8)
    v[:, :20] = 3
    v[:, 20:45] = 7
    v[:, 45:] = 255  # NoData
    v[0, 0] = 9
    da = grid(v, nodata=255)
    out = ls.zonal_stats(da, box(500000, 4498500, 501500, 4500000), ["majority", "minority", "unique", "nodata"])
    assert out[["majority", "minority", "unique", "nodata"]].iloc[0].tolist() == [7, 9, 3, 250]


def test_zonal_stats_of_a_stack_and_zones_in_another_crs(raster):
    stack = xr.concat([raster, raster * 2], dim="band").assign_coords(band=["a", "b"])
    zone = gpd.GeoDataFrame(geometry=[box(500300, 4497500, 501500, 4499600)], crs=CRS).to_crs("EPSG:4326")
    out = ls.zonal_stats(stack, zone, ["mean", "max"])
    assert {"a_mean", "a_max", "b_mean", "b_max"} <= set(out.columns)
    assert out["b_mean"].iloc[0] == pytest.approx(2 * out["a_mean"].iloc[0], rel=1e-6)


def test_zones_from_a_raster_of_labels(raster):
    labels = np.repeat(np.arange(10, dtype=np.int32), 10)[:, None] * np.ones((1, 120), np.int32)
    labels[:, :5] = -1
    zones = grid(labels, nodata=-1)
    out = ls.zonal_stats(raster, zones, ["count", "mean", "median"])
    assert out["zone"].tolist() == list(range(10))
    for z in range(10):
        v, _ = reference(raster.values, labels == z)
        assert out["count"].iloc[z] == v.size
        assert out["mean"].iloc[z] == pytest.approx(v.mean(), rel=1e-12)
        assert out["median"].iloc[z] == pytest.approx(np.median(v), rel=1e-12)
    coarse = grid(labels[::2, ::2], transform=from_origin(500000.0, 4500000.0, 60.0, 60.0), nodata=-1)
    assert ls.zonal_stats(raster, coarse, "count")["zone"].tolist() == list(range(10))  # put on the grid


def test_zonal_stats_types_and_threads():
    rng = np.random.default_rng(5)
    zone = [box(500000, 4497000, 503000, 4500000)]  # one large zone: the blocked path
    base = rng.integers(0, 200, (100, 100))
    results = {}
    for dt in ("uint8", "int16", "int32", "int64", "float32", "float64"):
        results[dt] = ls.zonal_stats(grid(base.astype(dt)), zone, ["sum", "mean", "std", "p50"])
    first = results["uint8"].drop(columns="geometry")
    for out in results.values():
        pd.testing.assert_frame_equal(out.drop(columns="geometry"), first)
    big = grid(rng.random((800, 800)).astype(np.float32), transform=from_origin(500000.0, 4524000.0, 30.0, 30.0))
    zone = [box(500000, 4500000, 524000, 4524000)]
    one = ls.zonal_stats(big, zone, ["sum", "mean", "std"], n_threads=1)
    many = ls.zonal_stats(big, zone, ["sum", "mean", "std"], n_threads=8)
    assert one.drop(columns="geometry").equals(many.drop(columns="geometry"))  # bit for bit
    assert one["mean"].iloc[0] == pytest.approx(float(big.values.astype(np.float64).mean()), rel=1e-12)


def test_zonal_stats_errors(raster):
    with pytest.raises(ValueError, match="unknown statistic"):
        ls.zonal_stats(raster, box(500000, 4499000, 500300, 4500000), "average")
    with pytest.raises(ValueError, match="polygons"):
        ls.zonal_stats(raster, gpd.GeoSeries([Point(500100, 4499900)], crs=CRS))
    with pytest.raises(ValueError, match="already have"):
        ls.zonal_stats(raster, gpd.GeoDataFrame({"mean": [1]}, geometry=[box(500000, 4499000, 500300, 4500000)],
                                                crs=CRS), "mean")
    with pytest.raises(TypeError, match="georeferenced DataArray"):
        ls.zonal_stats(np.zeros((3, 3)), box(0, 0, 1, 1))


# --- extract -------------------------------------------------------------------------


def test_extract_nearest_and_bilinear(raster):
    rng = np.random.default_rng(2)
    x = rng.uniform(499900, 503700, 500)  # some outside
    y = rng.uniform(4496900, 4500100, 500)
    out = ls.extract(raster, (x, y))
    col = np.floor((x - T.c) / T.a).astype(int)
    row = np.floor((y - T.f) / T.e).astype(int)
    inside = (col >= 0) & (col < 120) & (row >= 0) & (row < 100)
    expected = np.full(x.size, np.nan)
    expected[inside] = raster.values[row[inside], col[inside]]
    np.testing.assert_array_equal(out["v"].to_numpy(), expected)
    # bilinear, away from the edges and no data
    xb = rng.uniform(500100, 503400, 200)
    yb = rng.uniform(4497100, 4499900, 200)
    full = grid(np.add.outer(np.arange(100.0) * 3, np.arange(120.0)))  # a plane: bilinear is exact
    got = ls.extract(full, np.c_[xb, yb], method="bilinear")["v"].to_numpy()
    fx, fy = (xb - T.c) / T.a - 0.5, (yb - T.f) / T.e - 0.5
    np.testing.assert_allclose(got, fy * 3 + fx, atol=1e-9)


def test_extract_points_in_another_crs_stacks_and_nodata():
    v = np.arange(12 * 10, dtype=np.int16).reshape(12, 10)
    v[2, 3] = -9
    stack = grid(np.stack([v, v * 2]), nodata=-9, names=["a", "b"])
    pts = gpd.GeoDataFrame({"name": ["p", "q"]}, crs=CRS,
                           geometry=[Point(500000 + 3.5 * 30, 4500000 - 2.5 * 30), Point(500000 + 15, 4500000 - 15)])
    out = ls.extract(stack, pts.to_crs("EPSG:4326"))
    assert np.isnan(out["a"].iloc[0]) and out[["a", "b"]].iloc[1].tolist() == [0.0, 0.0]
    assert out.crs.to_epsg() == 32633 and out["name"].tolist() == ["p", "q"]
    with pytest.raises(ValueError, match="takes points"):
        ls.extract(stack, gpd.GeoSeries([box(0, 0, 1, 1)], crs=CRS))


def test_extract_from_a_lazy_raster_reads_only_blocks_with_points(raster):
    pytest.importorskip("dask")
    lazy = raster.chunk({"y": 17, "x": -1})
    rng = np.random.default_rng(4)
    pts = np.c_[rng.uniform(500000, 503600, 300), rng.uniform(4497000, 4500000, 300)]
    for method in ("nearest", "bilinear"):
        a = ls.extract(raster, pts, method=method)["v"].to_numpy()
        b = ls.extract(lazy, pts, method=method)["v"].to_numpy()
        np.testing.assert_array_equal(a, b)


def test_extract_with_buffer_is_zonal_stats_of_circles(raster):
    pts = gpd.GeoDataFrame({"site": [1, 2]}, geometry=[Point(501000, 4498000), Point(502500, 4499000)], crs=CRS)
    got = ls.extract(raster, pts, buffer=150, stats=["mean", "count"])
    ref = ls.zonal_stats(raster, pts.buffer(150), ["mean", "count"])
    np.testing.assert_array_equal(got["mean"].to_numpy(), ref["mean"].to_numpy())
    assert got.geometry.geom_type.tolist() == ["Point", "Point"]
    plain = ls.extract(raster, np.array([[501000, 4498000]]), buffer=150)
    assert list(plain.columns) == ["x", "y", "mean"]


# --- clip --------------------------------------------------------------------------


def test_clip_crops_and_masks(raster):
    circle = Point(501500, 4498500).buffer(600)
    out = ls.clip(raster, circle)
    assert out.shape == (40, 40)
    tr = out.rio.transform()
    assert (tr.c, tr.f) == (500900.0, 4499100.0)
    expected = rasterio.features.geometry_mask([circle], out_shape=raster.shape, transform=T, invert=True)
    assert int(out.notnull().sum()) == int((expected & raster.notnull().values).sum())
    np.testing.assert_array_equal(out.x.values, raster.x.values[30:70])
    touched = ls.clip(raster, circle, all_touched=True)
    assert int(touched.notnull().sum()) > int(out.notnull().sum())
    kept = ls.clip(raster, circle, crop=False)
    assert kept.shape == raster.shape
    hole = ls.clip(raster, circle, invert=True)
    assert hole.shape == raster.shape and int(hole.notnull().sum()) + int(kept.notnull().sum()) == int(
        raster.notnull().sum())


def test_clip_class_maps_keep_their_type(tmp_path):
    classes = grid(np.full((50, 50), 7, np.uint8))
    out = ls.clip(classes, box(500300, 4498800, 500900, 4499700))
    assert out.dtype == np.uint8 and out.rio.nodata == 255 and set(np.unique(out.values)) == {7}
    with_nd = grid(np.full((50, 50), 7, np.int16), nodata=-1)
    masked = ls.clip(with_nd, Point(500750, 4499250).buffer(300))
    assert masked.dtype == np.int16 and set(np.unique(masked.values)) == {-1, 7}
    back = ls.load_raster(ls.write_raster(masked, tmp_path / "c.tif"))
    assert back.rio.nodata == -1 and back.rio.transform() == masked.rio.transform()


def test_clip_regions(raster):
    assert ls.clip(raster, (500300, 4497000, 501200, 4499100)).shape == (70, 30)
    other = gpd.GeoDataFrame(geometry=[box(500300, 4497000, 501200, 4499100)], crs=CRS).to_crs("EPSG:4326")
    assert abs(ls.clip(raster, other).shape[1] - 30) <= 1
    with pytest.raises(ValueError, match="does not overlap"):
        ls.clip(raster, box(0, 0, 10, 10))
    with pytest.raises(TypeError, match="load_vector"):
        ls.clip(raster, "area.gpkg")


def test_clip_is_lazy_on_dask(raster):
    pytest.importorskip("dask")
    lazy = ls.clip(raster.chunk({"y": 25}), Point(501500, 4498500).buffer(600))
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, ls.clip(raster, Point(501500, 4498500).buffer(600)).values)


# --- load_raster: clip= and stacks on a grid ----------------------------------------------


def test_load_raster_clip_reads_the_window(tmp_path, raster):
    path = ls.write_raster(raster, tmp_path / "r.tif")
    region = Point(501500, 4498500).buffer(600)
    a = ls.load_raster(path, clip=region)
    b = ls.clip(ls.load_raster(path), region)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.rio.transform() == b.rio.transform()
    assert ls.load_raster(path, clip=(500300, 4497000, 501200, 4499100)).shape == (70, 30)


def test_load_raster_list_like_stacks_any_grids(tmp_path, raster):
    ls.write_raster(raster, tmp_path / "sp_a.tif")
    ls.write_raster(raster.rio.reproject("EPSG:4326"), tmp_path / "sp_b.tif")
    ls.write_raster(raster[::2, ::2], tmp_path / "sp_c.tif")
    stack = ls.load_raster(tmp_path, like=raster)
    assert stack.dims == ("band", "y", "x") and stack.band.values.tolist() == ["sp_a", "sp_b", "sp_c"]
    assert stack.shape == (3, 100, 120) and (stack.x.values == raster.x.values).all()
    np.testing.assert_array_equal(stack.sel(band="sp_a").values, raster.values)
    small = ls.load_raster(tmp_path, like=raster, clip=(500300, 4497000, 501200, 4499100))
    assert small.shape == (3, 100, 120) and int(small.notnull().sum()) < int(stack.notnull().sum())
    with pytest.raises(ls.InvalidInputError, match="share the same grid"):
        ls.load_raster(tmp_path, stack=True)
    with pytest.raises(ValueError, match="needs like="):
        ls.load_raster(tmp_path, clip=(0, 0, 1, 1))


# --- reclassify with a table ----------------------------------------------------------


@pytest.fixture
def classes():
    v = np.array([[1, 2, 3], [23, 24, 255]], np.uint8)
    return grid(v, nodata=255)


def test_reclassify_pairs_dicts_frames(classes):
    d = ls.reclassify(classes, {1: 100, 2: 200, 23: 5})
    assert d.dtype == np.uint8 and d.values.tolist() == [[100, 200, 3], [5, 24, 255]] and d.rio.nodata == 255
    big = ls.reclassify(classes, {1: 1000}, others=1)
    assert big.dtype == np.uint16 and big.values.tolist() == [[1000, 1, 1], [1, 1, 255]]
    assert big.rio.nodata == 255  # the input's NoData, which the type holds
    frame = pd.DataFrame({"classe": [1, 23], "resistencia": [1.5, 9.0]})
    f = ls.reclassify(classes, frame, others="nodata")
    assert f.dtype == np.float32 and np.isnan(f.values[0, 1]) and f.values[0, 0] == 1.5 and np.isnan(f.values[1, 2])
    pairs = ls.reclassify(classes, [(1, 7), (2, 8)])
    assert pairs.values[0].tolist() == [7, 8, 3]
    with pytest.raises(ValueError, match="twice"):
        ls.reclassify(classes, [(1, 7), (1, 8)])


def test_reclassify_ranges_first_wins_and_nan():
    v = np.array([[0.0, 3.9, 4.0], [10.0, 99.0, np.nan]])
    out = ls.reclassify(v, [(0, 4, 1), (4, 10, 2), (10, np.inf, 3), (0, 100, 9)])
    np.testing.assert_array_equal(out, [[1, 1, 2], [3, 3, np.nan]])
    ints = ls.reclassify(v, [(0, 4, 1), (4, np.inf, 2)], others=0, dtype="uint8")
    assert ints.dtype == np.uint8 and ints.tolist() == [[1, 1, 2], [2, 2, 255]]
    mixed = ls.reclassify(v, {(0, 4): 1, 99.0: 5})
    np.testing.assert_array_equal(mixed, [[1, 1, 4], [10, 5, np.nan]])


def test_reclassify_lookup_table_and_general_paths_agree():
    rng = np.random.default_rng(0)
    table = {int(k): int(k) * 3 + 1 for k in rng.choice(200, 50, replace=False)}
    small = rng.integers(0, 200, (60, 70)).astype(np.int16)  # lookup table over the type
    wide = small.astype(np.int32)  # searchsorted
    np.testing.assert_array_equal(ls.reclassify(small, table, others=0), ls.reclassify(wide, table, others=0))


def test_reclassify_lazy_and_habitat_mode(classes):
    pytest.importorskip("dask")
    lazy = ls.reclassify(classes.chunk({"y": 1}), {1: 100})
    assert lazy.chunks is not None and lazy.values.tolist() == [[100, 2, 3], [23, 24, 255]]
    habitat = ls.reclassify(classes, [23, 24])  # a list of values: the habitat of a habitat map
    assert habitat.values.tolist() == [[1, 1, 1], [2, 2, 0]]
    assert ls.reclassify(classes, foreground=23).values.tolist() == [[1, 1, 1], [2, 1, 0]]
    with pytest.raises(ValueError, match="habitat maps"):
        ls.reclassify(classes, {1: 2}, missing=[3])
    with pytest.raises(TypeError, match="needs a table"):
        ls.reclassify(classes)


def test_rasterize_matches_gdal_on_edges_through_cell_centres():
    # polygons with vertices on cell centres and edges, in both orientations, with holes: GDAL's
    # rasterizer takes every ring clockwise, which decides the cells of horizontal edges on centres
    import shapely

    rng = np.random.default_rng(1)
    like = grid(np.zeros((10, 10)), transform=from_origin(0, 100, 10, 10))
    t = from_origin(0, 100, 10, 10)
    for _ in range(400):
        if rng.random() < 0.5:
            g = Polygon(rng.integers(0, 21, (rng.integers(3, 7), 2)) * 5.0).buffer(0)
        else:
            x0, y0 = rng.integers(0, 10, 2) * 5.0
            g = box(x0, y0, x0 + 5 + rng.integers(0, 12) * 5.0, y0 + 5 + rng.integers(0, 12) * 5.0)
        if rng.random() < 0.5:
            g = shapely.reverse(g)
        if rng.random() < 0.2:
            g = g.difference(box(40, 40, 60, 60))
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        for touched in (False, True):
            ours = ls.rasterize(gpd.GeoSeries([g], crs=CRS), like, all_touched=touched).values
            ref = rasterio.features.rasterize([(g, 1)], out_shape=(10, 10), transform=t, dtype="uint8",
                                              all_touched=touched)
            np.testing.assert_array_equal(ours, ref, err_msg=f"{g.wkt} all_touched={touched}")


def test_rasterize_points_lines_and_order_match_gdal():
    from shapely.geometry import LineString

    rng = np.random.default_rng(2)
    t = from_origin(500000, 4500000, 30, 30)
    like = grid(np.zeros((60, 80)), transform=t)
    pts = gpd.GeoDataFrame({"v": np.arange(1, 3001)}, crs=CRS,
                           geometry=gpd.points_from_xy(rng.uniform(499900, 502500, 3000),
                                                       rng.uniform(4498100, 4500100, 3000)))
    for merge, alg in (("replace", rasterio.enums.MergeAlg.replace), ("add", rasterio.enums.MergeAlg.add)):
        ours = ls.rasterize(pts, like, field="v", merge=merge, dtype="int32").values
        ref = rasterio.features.rasterize(zip(pts.geometry, pts.v), out_shape=(60, 80), transform=t, merge_alg=alg,
                                          dtype="int32")
        np.testing.assert_array_equal(ours, ref)
    mixed = gpd.GeoDataFrame({"v": [1, 2, 3]}, crs=CRS, geometry=[
        box(500300, 4498500, 501500, 4499700), LineString([(500000, 4498200), (502400, 4500000)]),
        Point(500900, 4499100)])
    ref = rasterio.features.rasterize(zip(mixed.geometry, mixed.v), out_shape=(60, 80), transform=t, dtype="uint8")
    np.testing.assert_array_equal(ls.rasterize(mixed, like, field="v").values, ref)
