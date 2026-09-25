# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import guidospy as gp
from guidospy import InvalidInputError

TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)
CRS = "EPSG:32633"


def write_tif(path, data, *, count=None, dtype=None, nodata=None):
    data = np.asarray(data)
    bands = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=count or bands.shape[0],
        dtype=dtype or bands.dtype, crs=CRS, transform=TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(bands.astype(dtype or bands.dtype))
    return path


def mspa_map(shape=(30, 40)) -> np.ndarray:
    a = np.ones(shape, np.uint8)
    a[5:25, 5:30] = 2
    a[0, :] = 0
    return a


def test_load_raster_returns_georeferenced_2d(tmp_path):
    path = write_tif(tmp_path / "in.tif", mspa_map())
    da = gp.load_raster(path)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x") and da.dtype == np.uint8
    assert da.rio.crs.to_string() == CRS
    assert da.rio.transform() == TRANSFORM
    np.testing.assert_array_equal(da.values, mspa_map())


def test_load_raster_lazy_with_chunks(tmp_path):
    pytest.importorskip("dask")
    path = write_tif(tmp_path / "in.tif", mspa_map())
    da = gp.load_raster(path, chunks={"x": 16, "y": 16})
    assert da.chunks is not None
    np.testing.assert_array_equal(da.values, mspa_map())


@pytest.mark.parametrize(
    "data, dtype, message",
    [
        (np.stack([mspa_map(), mspa_map()]), None, "2 bands"),
        (mspa_map(), "float32", "not uint8"),
        (mspa_map() * 3, None, "larger than 2"),
        (np.ones((30, 40), np.uint8), None, "no foreground"),
    ],
)
def test_load_raster_rejects_non_mspa_input(tmp_path, data, dtype, message):
    path = write_tif(tmp_path / "bad.tif", data, dtype=dtype)
    with pytest.raises(InvalidInputError, match=message):
        gp.load_raster(path)


def test_load_raster_band_selection_and_no_validation(tmp_path):
    path = write_tif(tmp_path / "two.tif", np.stack([mspa_map() * 3, mspa_map()]))
    np.testing.assert_array_equal(gp.load_raster(path, band=2).values, mspa_map())
    assert gp.load_raster(path, band=1, validate=None).max() == 6


def test_write_raster_roundtrip_keeps_georeference(tmp_path):
    da = gp.load_raster(write_tif(tmp_path / "in.tif", mspa_map()))
    out = gp.write_raster(da, tmp_path / "out.tif", colormap=None)
    with rasterio.open(out) as src:
        assert src.crs.to_string() == CRS and src.transform == TRANSFORM
        np.testing.assert_array_equal(src.read(1), mspa_map())


def test_write_mspa_result_has_colours_and_parameters(tmp_path):
    da = gp.load_raster(write_tif(tmp_path / "in.tif", mspa_map()))
    result = gp.mspa(da, edge_width=2, connectivity=4, transition=False)
    out = gp.write_raster(result, tmp_path / "mspa.tif")
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(1), result.values)
        cmap = src.colormap(1)
        assert cmap[17][:3] == (0, 200, 0)  # core: green
        assert cmap[129][:3] == (255, 255, 255)  # missing: white
        assert cmap[35][:3] == (0, 0, 0)  # transition off: bridge-in-edge shown as edge
        tags = src.tags()
        assert "4_2_0_1" in tags["TIFFTAG_IMAGEDESCRIPTION"]
        assert tags["mspa_edge_width"] == "2"
        assert src.crs.to_string() == CRS


def test_write_numpy_with_explicit_georeference(tmp_path):
    out = gp.write_raster(mspa_map(), tmp_path / "np.tif", crs=CRS, transform=TRANSFORM, colormap="mspa")
    with rasterio.open(out) as src:
        assert src.crs.to_string() == CRS and src.transform == TRANSFORM
        assert src.colormap(1)[3][:3] == (0, 0, 0)


def test_mspa_colormap_transition_colours():
    on, off = gp.mspa_colormap(True), gp.mspa_colormap(False)
    assert on[35][:3] == (255, 0, 0) and off[35][:3] == (0, 0, 0)
    assert on[169][:3] == (255, 255, 0) and off[169][:3] == (0, 0, 255)
    assert len(on) == 256


def test_reclassify_categorical_map():
    lc = np.array([[23, 24, 1], [2, 48, 25]], np.uint8)
    out = gp.reclassify(lc, foreground=[23, 24, 25], missing=48)
    np.testing.assert_array_equal(out, [[2, 2, 1], [1, 0, 2]])


def test_reclassify_with_explicit_background_and_function():
    frac = np.array([[0.9, 0.1, np.nan], [0.5, 0.2, 0.0]])
    out = gp.reclassify(frac, foreground=lambda a: a >= 0.5)
    np.testing.assert_array_equal(out, [[2, 1, 0], [2, 1, 1]])
    lc = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(gp.reclassify(lc, foreground=1, background=2), [[2, 1, 0]])


def test_reclassify_dataarray_uses_nodata(tmp_path):
    lc = np.array([[23, 255, 1], [2, 23, 3]], np.uint8)
    da = gp.load_raster(write_tif(tmp_path / "lc.tif", lc, nodata=255), validate=None)
    out = gp.reclassify(da, foreground=23)
    assert isinstance(out, xr.DataArray) and out.dtype == np.uint8
    np.testing.assert_array_equal(out.values, [[2, 0, 1], [1, 2, 1]])
    assert out.rio.crs.to_string() == CRS
    gp.validate_mspa_input(out)
