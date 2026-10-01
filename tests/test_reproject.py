# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.reproject: rasters against GDAL's warp (one block: GDAL's own call; exact transformations:
GDAL warping the whole destination at once), the same bits in memory, lazily, from a file and
with any number of threads; references (like=), vectors."""

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

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import box

SRC_CRS = "EPSG:32633"
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)
H, W = 300, 400
METHODS = ("nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "min", "max", "med", "q1",
           "q3", "sum", "rms")


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


def gdal(da: xr.DataArray, out: xr.DataArray, method: str, src_nodata, dst_nodata, tolerance=0.0) -> np.ndarray:
    """GDAL's warp of the whole destination in one chunk."""
    ref = np.full(out.shape, dst_nodata, dtype=out.dtype)
    gdal_reproject(da.values, ref, src_transform=da.rio.transform(), src_crs=CRS.from_user_input(da.rio.crs),
                   src_nodata=src_nodata, dst_transform=out.rio.transform(), dst_crs=CRS.from_user_input(out.rio.crs),
                   dst_nodata=dst_nodata, resampling=Resampling[method], tolerance=tolerance, warp_mem_limit=4096)
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


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target,res", [("EPSG:4326", None), ("EPSG:3035", 100)])
def test_one_block_is_gdals_own_warp(surface, method, target, res):
    out = ls.reproject(surface, target, res=res, resampling=method)
    assert out.dtype == np.float32 and np.isnan(out.rio.nodata) and max(out.shape) <= 1024
    assert same(out.values, gdal(surface, out, method, np.nan, np.nan, tolerance=0.125)).all()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target,res", [("EPSG:4326", None), ("EPSG:3035", 100), ("EPSG:3035", 12)])
def test_exact_transformations_give_gdals_whole_warp(surface, method, target, res):
    out = ls.reproject(surface, target, res=res, resampling=method, tolerance=0)
    ref = gdal(surface, out, method, np.nan, np.nan)
    if method in ("nearest", "mode", "min", "max", "med", "q1", "q3"):  # values of the source
        assert same(out.values, ref).all()
    else:  # the blocks see their source cells from other origins than GDAL's chunks
        assert_ulps(out.values, ref)


@pytest.mark.parametrize("method", ["nearest", "mode", "min", "max"])
def test_classes_match_gdal_and_keep_their_type(classes, method):
    out = ls.reproject(classes, "EPSG:4326", resampling=method)
    assert out.dtype == np.uint8 and out.rio.nodata == 0
    assert same(out.values, gdal(classes, out, method, 0, 0, tolerance=0.125)).all()
    assert set(np.unique(out.values)) <= set(range(7))
    exact = ls.reproject(classes, "EPSG:4326", resampling=method, tolerance=0)
    assert same(exact.values, gdal(classes, exact, method, 0, 0)).all()
    assert same(out.values, exact.values).mean() > 0.9  # the approximation moves few cells


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
