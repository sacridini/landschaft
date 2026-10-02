# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.reproject: rasters against GDAL's warp (exact transformations: GDAL warping the whole
destination at once, bit for bit, for every method, type and odd projection; the default
lattice of coordinates: close to them), the same bits in memory, lazily, from a file and with
any number of threads; references (like=), vectors."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.warp import reproject as gdal_reproject

import landschaft as ls
from landschaft import _gdaltransform
from landschaft import _reproject as R

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import box

SRC_CRS = "EPSG:32633"
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)
H, W = 300, 400
METHODS = ("nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "min", "max", "med", "q1",
           "q3", "sum", "rms")
# GDAL's coordinates come from the GDAL rasterio loads; without it (pyproj's PROJ) the last bits differ.
needs_gdal = pytest.mark.skipif(_gdaltransform.open_library() is None, reason="rasterio's GDAL library not found")


def grid(values, transform=TRANSFORM, crs=SRC_CRS, nodata=None) -> xr.DataArray:
    values = np.asarray(values)
    h, w = values.shape[-2:]
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    dims = ("y", "x") if values.ndim == 2 else ("band", "y", "x")
    coords = {"y": y, "x": x} if values.ndim == 2 else {"band": [f"b{i}" for i in range(values.shape[0])], "y": y, "x": x}
    da = xr.DataArray(values, coords=coords, dims=dims, name="layer").rio.write_crs(crs).rio.write_transform(transform)
    return da.rio.write_nodata(nodata, encoded=False) if nodata is not None else da


@pytest.fixture
def surface():
    """Continuous values with a hole of NaN."""
    v = np.random.default_rng(0).random((H, W), dtype=np.float32)
    v[100:140, 150:220] = np.nan
    return grid(v)


@pytest.fixture
def classes():
    """Classes 1-6 in uint8, 0 = NoData."""
    v = np.random.default_rng(1).integers(1, 7, (H, W)).astype(np.uint8)
    v[:20, :] = 0
    return grid(v, nodata=0)


def gdal(da: xr.DataArray, out: xr.DataArray, method: str, src_nodata, dst_nodata, tolerance=0.0,
         **options) -> np.ndarray:
    """GDAL's warp of the whole destination in one chunk."""
    ref = np.full(out.shape, dst_nodata, dtype=out.dtype)
    gdal_reproject(da.values, ref, src_transform=da.rio.transform(), src_crs=CRS.from_user_input(da.rio.crs),
                   src_nodata=src_nodata, dst_transform=out.rio.transform(), dst_crs=CRS.from_user_input(out.rio.crs),
                   dst_nodata=dst_nodata, resampling=Resampling[method], tolerance=tolerance, warp_mem_limit=4096,
                   **options)
    return ref


def same(a, b) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.floating):
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def assert_ulps(a, b, ulps: int = 1) -> None:
    """Equal, NaN where NaN, else within ``ulps`` units in the last place (the blocks of a lazy
    warp see their source cells from another origin)."""
    a, b = np.asarray(a), np.asarray(b)
    assert (np.isnan(a) == np.isnan(b)).all()
    ok = ~np.isnan(a)
    np.testing.assert_array_max_ulp(a[ok], b[ok], maxulp=ulps)


# --- the grid ----------------------------------------------------------------------


def test_grid_is_gdals_suggestion(surface):
    out = ls.reproject(surface, "EPSG:4326")
    tr, w, h = calculate_default_transform(SRC_CRS, "EPSG:4326", W, H, *array_bounds(H, W, TRANSFORM))
    assert out.shape == (h, w) and out.rio.transform().almost_equals(tr) and out.rio.crs == CRS.from_epsg(4326)
    np.testing.assert_allclose(out.x.values, tr.c + tr.a * (np.arange(w) + 0.5))
    np.testing.assert_allclose(out.y.values, tr.f + tr.e * (np.arange(h) + 0.5))
    with_res = ls.reproject(surface, 3035, res=(50, 40))
    assert with_res.rio.resolution() == (50.0, -40.0) and with_res.rio.crs == CRS.from_epsg(3035)


def test_res_alone_resamples_in_the_same_crs(surface):
    coarse = ls.reproject(surface, res=90, resampling="average")
    assert coarse.rio.crs == surface.rio.crs and coarse.shape == (100, 134)
    assert coarse.rio.resolution() == (90.0, -90.0)


def test_same_grid_is_returned_as_it_is(surface):
    out = ls.reproject(surface, SRC_CRS)
    assert out.shape == surface.shape and same(out.values, surface.values).all()


# --- values: GDAL's ----------------------------------------------------------------


@needs_gdal
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target,res", [("EPSG:4326", None), ("EPSG:3035", 100), ("EPSG:3035", 12)])
def test_exact_transformations_give_gdals_whole_warp(surface, method, target, res):
    out = ls.reproject(surface, target, res=res, resampling=method, tolerance=0)
    assert out.dtype == np.float32 and np.isnan(out.rio.nodata)
    ref = gdal(surface, out, method, np.nan, np.nan)
    if method == "sum":  # GDAL's own warp in blocks, which see their source cells from other origins
        assert_ulps(out.values, ref)
    else:
        assert same(out.values, ref).all()


def typed(dtype, nodata, seed=3):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info is None:
        v = rng.normal(100, 30, (H // 2, W // 2)).astype(dtype)
    else:
        v = rng.integers(max(info.min, -1000), min(int(info.max), 1000) + 1, (H // 2, W // 2)).astype(dtype)
        v[0, :5] = info.max  # the destination NoData of a type without one: moved one step (AvoidNoData)
    if nodata is not None:
        v[rng.random(v.shape) < 0.05] = nodata
    return grid(v, nodata=nodata)


@needs_gdal
@pytest.mark.parametrize("dtype,nodata", [(np.uint8, 0), (np.uint8, None), (np.int8, -128), (np.int16, -9999),
                                          (np.uint16, None), (np.int32, -1), (np.uint32, None), (np.int64, -1),
                                          (np.float64, -9999.0)])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "lanczos", "average", "mode", "min", "med", "rms"])
def test_every_type_matches_gdal(dtype, nodata, method):
    if dtype == np.int8 and tuple(int(v) for v in rasterio.__gdal_version__.split(".")[:3]) < (3, 12, 3):
        pytest.skip("GDAL < 3.12.3 warps Int8 as Int16 (values past -128 do not avoid the NoData)")
    da = typed(dtype, nodata)
    out = ls.reproject(da, "EPSG:3035", res=45, resampling=method, tolerance=0)
    src_nodata, dst_nodata, _ = R._nodata(da)
    assert out.dtype == np.dtype(dtype) and out.rio.nodata == dst_nodata
    assert same(out.values, gdal(da, out, method, src_nodata, dst_nodata)).all()


@needs_gdal
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic_spline", "average", "mode", "q3"])
def test_bands_count_as_nodata_where_all_are(surface, method):
    """GDAL's unified NoData: a cell is NoData when every band is."""
    v = np.stack([surface.values, surface.values * 2, surface.values + 1])
    v[0, :30] = np.nan
    v[1, 20:60] = np.nan
    stack = grid(v)
    out = ls.reproject(stack, "EPSG:4326", resampling=method, tolerance=0)
    assert same(out.values, gdal(stack, out, method, np.nan, np.nan)).all()


@needs_gdal
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "lanczos", "average", "mode", "max", "med"])
@pytest.mark.parametrize("proj", ["+proj=ortho +lat_0=40 +lon_0=15 +datum=WGS84",
                                  "+proj=ortho +lat_0=0 +lon_0=170 +datum=WGS84", "EPSG:3031"])
def test_a_world_in_odd_projections_matches_gdal(method, proj):
    """A global geographic source: points off the globe, the antimeridian, a pole; NoData holes
    (cells at the edge of the projection's domain take a neighbour, as in GDAL). GDAL misses a
    pole in the destination when the source is geographic (it does not reproject it): GDAL with
    that fixed samples the source window on a grid, as SAMPLE_GRID=YES does."""
    t = from_origin(-180, 90, 1.0, 1.0)
    v = np.random.default_rng(0).random((180, 360), dtype=np.float32)
    v[60:75, 200:260] = np.nan
    v[:, 100:103] = np.nan
    world = grid(v, transform=t, crs="EPSG:4326")
    side, cell = (280, 50000) if "ortho" in proj else (320, 25000)
    half = side * cell / 2
    like = grid(np.zeros((side, side), np.float32), transform=from_origin(-half, half, cell, cell),
                crs=CRS.from_user_input(proj))
    out = ls.reproject(world, like=like, resampling=method, tolerance=0)
    points = R._Points(t, CRS.from_epsg(4326), (180, 360), like.rio.transform(), CRS.from_user_input(proj),
                       (side, side))
    pole = any(0 <= x <= side and 0 <= y <= side for x, y in points.special_points())
    assert pole == (proj != "+proj=ortho +lat_0=0 +lon_0=170 +datum=WGS84")  # that one's poles are on its far side
    options = {"SAMPLE_GRID": "YES"} if pole else {}
    assert same(out.values, gdal(world, out, method, np.nan, np.nan, **options)).all()
    approx = ls.reproject(world, like=like, resampling=method)
    lazy = ls.reproject(world.chunk({"y": 50, "x": 70}), like=like, resampling=method).compute()
    assert same(lazy.values, approx.values).all()


@needs_gdal
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average", "mode"])
def test_a_geographic_source_matches_gdal(surface, method):
    lonlat = ls.reproject(surface, "EPSG:4326", tolerance=0)
    for target, res in (("EPSG:32633", None), ("EPSG:3857", None), ("EPSG:4326", 0.0007)):
        out = ls.reproject(lonlat, target, res=res, resampling=method, tolerance=0)
        assert same(out.values, gdal(lonlat, out, method, np.nan, np.nan)).all()


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average", "mode"])
def test_the_same_crs_is_exact_at_any_tolerance(surface, method):
    """No reprojection: the coordinates are the geotransforms', GDAL's even with the default
    tolerance; a whole number of cells apart, GDAL takes the nearest cell whatever the method."""
    out = ls.reproject(surface, res=(41, 37), resampling=method)
    assert same(out.values, gdal(surface, out, method, np.nan, np.nan)).all()
    shifted = grid(np.zeros((150, 200), np.float32),
                   transform=from_origin(500000.0 + 30 * 7, 4500000.0 - 30 * 5, 30, 30))
    moved = ls.reproject(surface, like=shifted, resampling=method)
    assert same(moved.values, surface.values[5:155, 7:207]).all()
    assert same(moved.values, gdal(surface, moved, method, np.nan, np.nan)).all()


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "lanczos", "average", "mode", "min", "med", "rms"])
def test_the_default_tolerance_stays_close_to_exact(surface, method):
    approx = ls.reproject(surface, "EPSG:3035", res=20, resampling=method)
    exact = ls.reproject(surface, "EPSG:3035", res=20, resampling=method, tolerance=0)
    a, b = approx.values, exact.values
    assert (np.isnan(a) != np.isnan(b)).mean() < 1e-3
    ok = ~np.isnan(a) & ~np.isnan(b)
    if method in ("nearest", "mode", "min", "med"):  # values of the source: a few cells pick another
        assert (a[ok] != b[ok]).mean() < 1e-3
    else:
        assert np.abs(a[ok] - b[ok]).max() < 1e-3


def test_the_lattice_is_exact_at_its_nodes_and_close_between():
    t = from_origin(-180, -50, 0.1, 0.1)
    for src_crs, src_t, shape, dst_crs, res in (("EPSG:32633", TRANSFORM, (H, W), "EPSG:3035", 7.0),
                                                ("EPSG:4326", t, (400, 3600), "EPSG:3031", 5000.0)):
        s_crs, d_crs = CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs)
        dst_t, dw, dh = calculate_default_transform(s_crs, d_crs, shape[1], shape[0], *array_bounds(*shape, src_t),
                                                    resolution=res)
        warps = [R._Warp(shape, src_t, s_crs, dst_t, (dh, dw), d_crs, "bilinear", np.nan, np.nan,
                         np.dtype(np.float32), tol, 1, 2) for tol in (0.125, 0.0)]
        approx, exact = (w.sample_coords(0, 0, dh, dw, 2) for w in warps)
        assert (approx[::16, ::16] == exact[::16, ::16]).all()  # the nodes: exact transformations
        err = np.abs(approx - exact).sum(-1)
        inside = ((exact[..., 0] > 0) & (exact[..., 0] < shape[1]) & (exact[..., 1] > 0)
                  & (exact[..., 1] < shape[0]))
        assert err[inside].max() < (1e-3 if dst_crs == "EPSG:3035" else 0.25)  # checked at the midpoints only
        levels = np.bincount(warps[0].lattice(0, 0, dh, dw, 2).level, minlength=5)
        if dst_crs == "EPSG:3031":  # finer cells near the pole only
            assert levels[1:].sum() > 0 and levels[0] > 5 * levels[1:].sum()
        else:
            assert levels[1:].sum() == 0


def test_without_gdals_library_pyproj_gives_the_coordinates(surface, monkeypatch):
    monkeypatch.setattr(_gdaltransform, "_searched", True)
    monkeypatch.setattr(_gdaltransform, "_lib", None)
    for tolerance in (0.125, 0.0):
        out = ls.reproject(surface, "EPSG:4326", tolerance=tolerance)
        lazy = ls.reproject(surface.chunk({"y": 70, "x": 90}), "EPSG:4326", tolerance=tolerance).compute()
        assert same(out.values, lazy.values).all()
        ref = gdal(surface, out, "bilinear", np.nan, np.nan)
        assert (np.isnan(out.values) == np.isnan(ref)).mean() > 0.999
        np.testing.assert_allclose(out.values, ref, atol=1e-4)


def test_classes_keep_their_type(classes):
    out = ls.reproject(classes, "EPSG:4326")
    assert out.dtype == np.uint8 and out.rio.nodata == 0
    assert set(np.unique(out.values)) <= set(range(7))
    assert same(out.values, ls.reproject(classes, "EPSG:4326", tolerance=0).values).mean() > 0.99


def test_auto_takes_nearest_for_integers_bilinear_for_floats(surface, classes):
    assert same(ls.reproject(classes, 4326).values, ls.reproject(classes, 4326, resampling="nearest").values).all()
    assert same(ls.reproject(surface, 4326).values, ls.reproject(surface, 4326, resampling="bilinear").values).all()


def test_integers_without_nodata_and_booleans():
    v = np.random.default_rng(2).integers(-50, 50, (H, W)).astype(np.int16)
    out = ls.reproject(grid(v), 3035)
    assert out.dtype == np.int16 and out.rio.nodata == np.iinfo(np.int16).min
    assert (out.values == np.iinfo(np.int16).min).any()  # the corners outside the source
    flags = ls.reproject(grid(v > 0), 3035)
    assert flags.dtype == np.uint8 and flags.rio.nodata == 255 and set(np.unique(flags.values)) <= {0, 1, 255}


def test_bands_and_other_dimensions_are_kept(surface):
    stack = grid(np.stack([surface.values, surface.values * 2, surface.values + 1]))
    out = ls.reproject(stack, 4326)
    assert out.dims == ("band", "y", "x") and out.band.values.tolist() == ["b0", "b1", "b2"]
    one = ls.reproject(surface, 4326)
    for k in range(3):
        alone = ls.reproject(stack.isel(band=k), 4326)
        assert same(out.values[k], alone.values).all()
    assert same(out.values[0], one.values).all()
    times = stack.rename(band="time").assign_coords(time=[2000, 2010, 2020])
    assert ls.reproject(times, 4326).time.values.tolist() == [2000, 2010, 2020]
    assert out.name == "layer"


# --- the same values in blocks, lazily, with any number of threads ------------------


@pytest.fixture
def small_blocks(monkeypatch):
    """Blocks of 128 cells: many of them, some outside the source."""
    from landschaft import _reproject

    monkeypatch.setattr(_reproject, "_BLOCK", 128)


@pytest.mark.parametrize("method", ["nearest", "mode", "bilinear", "cubic", "average", "lanczos"])
@pytest.mark.parametrize("tolerance", [0.125, 0.0])
def test_lazy_gives_the_bits_in_memory(surface, classes, small_blocks, method, tolerance):
    pytest.importorskip("dask")
    for da in (surface, classes) if method in ("nearest", "mode") else (surface,):
        full = ls.reproject(da, "EPSG:3035", res=20, resampling=method, tolerance=tolerance)
        for chunks in ({"y": 70, "x": 90}, {"y": 300, "x": 25}):
            lazy = ls.reproject(da.chunk(chunks), "EPSG:3035", res=20, resampling=method, tolerance=tolerance)
            assert hasattr(lazy.data, "dask") and lazy.data.numblocks[-1] > 1
            lazy = lazy.compute()
            assert lazy.rio.transform().almost_equals(full.rio.transform())
            assert same(lazy.values, full.values).all()


def test_threads_do_not_change_results(surface, small_blocks):
    one = ls.reproject(surface, 3035, res=10, resampling="cubic", n_threads=1)
    many = ls.reproject(surface, 3035, res=10, resampling="cubic", n_threads=4)
    assert same(one.values, many.values).all()


def test_paths_are_read_by_window(tmp_path, surface, classes, small_blocks):
    path = ls.write_raster(classes, tmp_path / "classes.tif")
    out = ls.reproject(path, "EPSG:4326")
    assert not hasattr(out.data, "dask") and out.name == "classes"
    assert same(out.values, ls.reproject(classes, "EPSG:4326").values).all()
    fpath = ls.write_raster(surface, tmp_path / "surface.tif")
    assert same(ls.reproject(fpath, 3035, res=50).values, ls.reproject(surface, 3035, res=50).values).all()


# --- references --------------------------------------------------------------------


def test_like_a_raster_takes_its_grid(surface, tmp_path):
    lonlat = ls.reproject(surface, "EPSG:4326")
    back = ls.reproject(lonlat, like=surface)
    assert back.shape == surface.shape and back.rio.crs == surface.rio.crs
    assert (back.x.values == surface.x.values).all() and (back.y.values == surface.y.values).all()
    assert same(back.values, ls.match_grid(lonlat, surface).values).all()
    via_path = ls.reproject(lonlat, like=ls.write_raster(surface, tmp_path / "ref.tif"))
    assert same(via_path.values, back.values).all()
    np.testing.assert_allclose(np.nanmean(back.values), np.nanmean(surface.values), rtol=0.01)


def test_crs_or_a_vector_reference_take_only_the_crs(surface, tmp_path):
    lonlat = ls.reproject(surface, "EPSG:4326")
    points = gpd.GeoDataFrame(geometry=[box(14.9, 40.6, 15.1, 40.7)], crs="EPSG:4326")
    by_vector = ls.reproject(surface, like=points)
    by_crs_of_raster = ls.reproject(surface, crs=lonlat)       # the CRS of the raster, not its grid
    by_path = ls.reproject(surface, crs=ls.write_vector(points, tmp_path / "p.gpkg"))
    for out in (by_vector, by_crs_of_raster, by_path):
        assert out.rio.crs == CRS.from_epsg(4326) and same(out.values, lonlat.values).all()
    coarse = ls.reproject(surface, like=points, res=0.001)
    assert coarse.rio.resolution() == (0.001, -0.001)


def test_match_grid_and_load_raster_like_use_the_same_warp(surface, tmp_path):
    lonlat = ls.reproject(surface, "EPSG:4326")
    path = ls.write_raster(lonlat, tmp_path / "lonlat.tif")
    on_grid = ls.load_raster(path, like=surface)
    assert same(on_grid.values, ls.reproject(lonlat, like=surface).values).all()


# --- vectors -----------------------------------------------------------------------


@pytest.fixture
def parks():
    return gpd.GeoDataFrame({"name": ["north", "south"], "area": [1.0, 2.0]}, crs=SRC_CRS,
                            geometry=[box(500000, 4490000, 520000, 4500000), box(505000, 4470000, 530000, 4485000)])


def test_vectors_match_geopandas_and_keep_attributes(parks, surface):
    out = ls.reproject(parks, "EPSG:4326")
    assert list(out.columns) == ["name", "area", "geometry"] and out.crs == CRS.from_epsg(4326)
    assert out.geometry.geom_equals_exact(parks.to_crs(4326).geometry, 0).all()
    series = ls.reproject(parks.geometry, 4326)
    assert isinstance(series, gpd.GeoSeries) and series.crs == CRS.from_epsg(4326)
    lonlat = ls.reproject(surface, 4326)
    assert ls.reproject(parks, like=lonlat).crs == CRS.from_epsg(4326)
    assert ls.reproject(parks, crs=lonlat).crs == CRS.from_epsg(4326)


def test_densify_follows_the_curvature(parks):
    plain = ls.reproject(parks, "EPSG:4326")
    dense = ls.reproject(parks, "EPSG:4326", densify=500)
    assert (shapely.get_num_coordinates(dense.geometry.values)
            > shapely.get_num_coordinates(plain.geometry.values)).all()
    expected = gpd.GeoSeries(shapely.segmentize(parks.geometry.values, 500), crs=SRC_CRS).to_crs(4326)
    assert dense.geometry.geom_equals_exact(expected, 0).all()


def test_vector_paths(parks, tmp_path):
    path = ls.write_vector(parks, tmp_path / "parks.gpkg")
    out = ls.reproject(path, 3035)
    assert out.crs == CRS.from_epsg(3035) and out["name"].tolist() == ["north", "south"]


# --- errors ------------------------------------------------------------------------


def test_errors(surface, parks):
    with pytest.raises(ValueError, match="crs=, like= or res="):
        ls.reproject(surface)
    with pytest.raises(ValueError, match="not both"):
        ls.reproject(surface, 4326, like=surface)
    with pytest.raises(ValueError, match="raster like="):
        ls.reproject(surface, like=surface, res=10)
    with pytest.raises(ValueError, match="unknown resampling"):
        ls.reproject(surface, 4326, resampling="smooth")
    with pytest.raises(ValueError, match="tolerance"):
        ls.reproject(surface, 4326, tolerance=-1)
    with pytest.raises(ValueError, match="cannot read a CRS"):
        ls.reproject(surface, "not a crs")
    with pytest.raises(ValueError, match="georeferenced"):
        ls.reproject(xr.DataArray(np.zeros((3, 3)), dims=("y", "x")), 4326)
    with pytest.raises(ValueError, match="densify= is for vectors"):
        ls.reproject(surface, 4326, densify=10)
    with pytest.raises(ValueError, match="res= is for rasters"):
        ls.reproject(parks, 4326, res=10)
    with pytest.raises(ValueError, match="crs= or like="):
        ls.reproject(parks)
    with pytest.raises(ValueError, match="with a CRS"):
        ls.reproject(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), 4326)
    with pytest.raises(ValueError, match="positive"):
        ls.reproject(parks, 4326, densify=0)
