# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls
from landschaft import InvalidInputError

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
    da = ls.load_raster(path)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x") and da.dtype == np.uint8
    assert da.rio.crs.to_string() == CRS
    assert da.rio.transform() == TRANSFORM
    np.testing.assert_array_equal(da.values, mspa_map())


def test_load_raster_lazy_with_chunks(tmp_path):
    pytest.importorskip("dask")
    path = write_tif(tmp_path / "in.tif", mspa_map())
    da = ls.load_raster(path, chunks={"x": 16, "y": 16})
    assert da.chunks is not None
    np.testing.assert_array_equal(da.values, mspa_map())


@pytest.mark.parametrize(
    "data, dtype, message",
    [
        (np.stack([mspa_map(), mspa_map()]), None, "single-band"),
        (mspa_map(), "float32", "not uint8"),
        (mspa_map() * 3, None, "larger than 2"),
        (np.ones((30, 40), np.uint8), None, "no foreground"),
    ],
)
def test_load_raster_rejects_non_mspa_input(tmp_path, data, dtype, message):
    path = write_tif(tmp_path / "bad.tif", data, dtype=dtype)
    with pytest.raises(InvalidInputError, match=message):
        ls.load_raster(path, validate="mspa")
    ls.load_raster(path)  # no checks by default


def test_load_raster_band_selection_and_no_validation(tmp_path):
    path = write_tif(tmp_path / "two.tif", np.stack([mspa_map() * 3, mspa_map()]))
    np.testing.assert_array_equal(ls.load_raster(path, band=2, validate="mspa").values, mspa_map())
    assert ls.load_raster(path, band=1).max() == 6


def test_write_raster_roundtrip_keeps_georeference(tmp_path):
    da = ls.load_raster(write_tif(tmp_path / "in.tif", mspa_map()))
    out = ls.write_raster(da, tmp_path / "out.tif", colormap=None)
    with rasterio.open(out) as src:
        assert src.crs.to_string() == CRS and src.transform == TRANSFORM
        np.testing.assert_array_equal(src.read(1), mspa_map())


def test_write_mspa_result_has_colours_and_parameters(tmp_path):
    da = ls.load_raster(write_tif(tmp_path / "in.tif", mspa_map()))
    result = ls.mspa(da, edge_width=2, connectivity=4, transition=False)
    out = ls.write_raster(result, tmp_path / "mspa.tif")
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
    out = ls.write_raster(mspa_map(), tmp_path / "np.tif", crs=CRS, transform=TRANSFORM, colormap="mspa")
    with rasterio.open(out) as src:
        assert src.crs.to_string() == CRS and src.transform == TRANSFORM
        assert src.colormap(1)[3][:3] == (0, 0, 0)


def test_mspa_colormap_transition_colours():
    on, off = ls.mspa_colormap(True), ls.mspa_colormap(False)
    assert on[35][:3] == (255, 0, 0) and off[35][:3] == (0, 0, 0)
    assert on[169][:3] == (255, 255, 0) and off[169][:3] == (0, 0, 255)
    assert len(on) == 256


def test_reclassify_categorical_map():
    lc = np.array([[23, 24, 1], [2, 48, 25]], np.uint8)
    out = ls.reclassify(lc, foreground=[23, 24, 25], missing=48)
    np.testing.assert_array_equal(out, [[2, 2, 1], [1, 0, 2]])


def test_reclassify_with_explicit_background_and_function():
    frac = np.array([[0.9, 0.1, np.nan], [0.5, 0.2, 0.0]])
    out = ls.reclassify(frac, foreground=lambda a: a >= 0.5)
    np.testing.assert_array_equal(out, [[2, 1, 0], [2, 1, 1]])
    lc = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(ls.reclassify(lc, foreground=1, background=2), [[2, 1, 0]])


def test_reclassify_dataarray_uses_nodata(tmp_path):
    lc = np.array([[23, 255, 1], [2, 23, 3]], np.uint8)
    da = ls.load_raster(write_tif(tmp_path / "lc.tif", lc, nodata=255))
    out = ls.reclassify(da, foreground=23)
    assert isinstance(out, xr.DataArray) and out.dtype == np.uint8
    np.testing.assert_array_equal(out.values, [[2, 0, 1], [1, 2, 1]])
    assert out.rio.crs.to_string() == CRS
    ls.validate_mspa_input(out)


# --- generic reading -----------------------------------------------------------


def named_stack(path, values, names, nodata=None, dtype="float32"):
    write_tif(path, values, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "r+") as dst:
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)
    return path


def test_load_raster_reads_every_band_with_names(tmp_path):
    values = np.stack([mspa_map(), mspa_map() * 2, mspa_map() * 3]).astype("float32")
    path = named_stack(tmp_path / "sdm.tif", values, ["sp_a", "sp_b", "sp_c"])
    da = ls.load_raster(path)
    assert da.dims == ("band", "y", "x") and da.band.values.tolist() == ["sp_a", "sp_b", "sp_c"]
    assert da.rio.crs.to_string() == CRS and da.rio.transform() == TRANSFORM
    np.testing.assert_array_equal(da.sel(band="sp_b").values, values[1])
    one = ls.load_raster(path, band=3)
    assert one.dims == ("y", "x") and one.attrs["long_name"] == "sp_c"
    sub = ls.load_raster(path, band=[3, 1])
    assert sub.band.values.tolist() == ["sp_c", "sp_a"]
    assert ls.load_raster(path, band=[2]).dims == ("band", "y", "x")
    with pytest.raises(InvalidInputError, match="band 4 not found"):
        ls.load_raster(path, band=4)


def test_load_raster_band_numbers_without_descriptions(tmp_path):
    path = write_tif(tmp_path / "two.tif", np.stack([mspa_map(), mspa_map()]))
    assert ls.load_raster(path).band.values.tolist() == [1, 2]


def test_load_raster_masks_float_nodata_and_keeps_integer_codes(tmp_path):
    f = np.arange(12, dtype="float32").reshape(3, 4)
    f[0, 0] = -9999
    da = ls.load_raster(write_tif(tmp_path / "f.tif", f, nodata=-9999))
    assert da.dtype == np.float32 and np.isnan(da.values[0, 0]) and np.isnan(da.rio.nodata)
    assert np.nanmin(da.values) == 1
    raw = ls.load_raster(tmp_path / "f.tif", masked=False)
    assert raw.values[0, 0] == -9999 and raw.rio.nodata == -9999

    i = np.array([[1, 2], [255, 3]], np.uint8)
    da = ls.load_raster(write_tif(tmp_path / "i.tif", i, nodata=255))
    assert da.dtype == np.uint8 and da.rio.nodata == 255
    np.testing.assert_array_equal(da.values, i)
    m = ls.load_raster(tmp_path / "i.tif", masked=True)
    assert m.dtype == np.float32 and np.isnan(m.values[1, 0]) and m.values[1, 1] == 3


def test_load_raster_float32_nodata_compared_in_band_type(tmp_path):
    nd = float(np.finfo(np.float32).min)  # the -3.4e38 many tools write
    f = np.array([[nd, 1.5]], np.float32)
    da = ls.load_raster(write_tif(tmp_path / "f.tif", f, nodata=nd))
    assert np.isnan(da.values[0, 0]) and da.values[0, 1] == 1.5


def test_load_raster_lazy_masking(tmp_path):
    pytest.importorskip("dask")
    f = np.ones((40, 30), np.float32)
    f[5] = -1
    da = ls.load_raster(write_tif(tmp_path / "f.tif", f, nodata=-1), chunks={"y": 10})
    assert da.chunks is not None
    assert int(da.isnull().sum()) == 30


def test_collection_masks_like_single_rasters(tmp_path):
    for k in range(2):
        f = np.full((4, 5), k, np.float32)
        f[0, 0] = -1
        write_tif(tmp_path / f"t{k}.tif", f, nodata=-1)
    coll = ls.load_raster(tmp_path)
    assert np.isnan(coll.load(1).values[0, 0])
    stack = coll.to_stack()
    assert stack.dims == ("file", "y", "x") and int(stack.isnull().sum()) == 2


def test_single_map_tools_reject_stacks_clearly(tmp_path):
    path = write_tif(tmp_path / "two.tif", np.stack([mspa_map(), mspa_map()]))
    with pytest.raises(InvalidInputError, match="2 bands, a single band is required"):
        ls.fos(path)


def test_tools_read_their_inputs_from_files_without_mspa_checks(tmp_path):
    # FOS maps (values up to 105) and land-cover classes used to be rejected when read from a file
    fos_map = ls.fos(mspa_map()).map
    a = ls.write_raster(fos_map, tmp_path / "fos1.tif", crs=CRS, transform=TRANSFORM)
    assert ls.fos_change(a, a) is not None
    classes = (np.arange(30 * 40).reshape(30, 40) % 7).astype(np.uint8) + 1
    c = ls.write_raster(classes, tmp_path / "classes.tif", crs=CRS, transform=TRANSFORM)
    assert ls.parcellation(c) is not None


# --- generic writing -----------------------------------------------------------


def test_write_raster_roundtrip_of_a_stack(tmp_path):
    values = np.stack([mspa_map(), mspa_map() * 2]).astype("float32")
    values[:, 0, 0] = np.nan
    da = ls.load_raster(named_stack(tmp_path / "in.tif", np.nan_to_num(values, nan=-1), ["a", "b"], nodata=-1))
    out = ls.write_raster(da, tmp_path / "sub" / "out.tif")  # creates the folder
    with rasterio.open(out) as src:
        assert src.count == 2 and src.descriptions == ("a", "b") and np.isnan(src.nodata)
        assert src.crs.to_string() == CRS and src.transform == TRANSFORM
        assert src.profile["compress"] == "lzw"
    back = ls.load_raster(out)
    np.testing.assert_array_equal(back.values, values)
    assert back.band.values.tolist() == ["a", "b"]


def test_write_raster_nan_and_explicit_nodata(tmp_path):
    f = np.array([[np.nan, 1.0], [2.0, 3.0]])
    with rasterio.open(ls.write_raster(f, tmp_path / "nan.tif")) as src:
        assert np.isnan(src.nodata) and src.dtypes[0] == "float64"
    with rasterio.open(ls.write_raster(f, tmp_path / "nd.tif", nodata=-9999, dtype="float32")) as src:
        assert src.nodata == -9999 and src.read(1)[0, 0] == -9999 and src.dtypes[0] == "float32"
    with rasterio.open(ls.write_raster(f, tmp_path / "int.tif", dtype="uint8")) as src:
        assert src.nodata == 255 and src.read(1).tolist() == [[255, 1], [2, 3]]
    with rasterio.open(ls.write_raster(np.ones((2, 2)), tmp_path / "plain.tif")) as src:
        assert src.nodata is None


def test_write_raster_storable_types(tmp_path):
    with rasterio.open(ls.write_raster(np.eye(3, dtype=bool), tmp_path / "b.tif")) as src:
        assert src.dtypes[0] == "uint8" and src.read(1).sum() == 3
    labels = np.arange(12, dtype=np.int64).reshape(3, 4) * 100
    with rasterio.open(ls.write_raster(labels, tmp_path / "l.tif")) as src:
        assert src.dtypes[0] == "uint16"
    with rasterio.open(ls.write_raster(-labels, tmp_path / "n.tif")) as src:
        assert src.dtypes[0] == "int16"


def test_write_raster_keeps_integer_nodata(tmp_path):
    i = np.array([[1, 2], [255, 3]], np.uint8)
    da = ls.load_raster(write_tif(tmp_path / "i.tif", i, nodata=255))
    with rasterio.open(ls.write_raster(da, tmp_path / "o.tif")) as src:
        assert src.nodata == 255 and src.read(1).tolist() == i.tolist()


def test_write_raster_like_georeferences_numpy(tmp_path):
    ref = ls.load_raster(write_tif(tmp_path / "ref.tif", mspa_map()))
    for like in (ref, tmp_path / "ref.tif"):
        out = ls.write_raster(np.zeros(ref.shape, np.float32), tmp_path / "z.tif", like=like)
        with rasterio.open(out) as src:
            assert src.crs.to_string() == CRS and src.transform == TRANSFORM
    with pytest.raises(ValueError, match="like has"):
        ls.write_raster(np.zeros((2, 2)), tmp_path / "bad.tif", like=ref)


def test_write_raster_formats_from_extension(tmp_path):
    f = np.array([[np.nan, 1.5], [2.0, 3.0]], np.float32)
    asc = ls.write_raster(f, tmp_path / "g.asc", nodata=-9999, crs=CRS, transform=TRANSFORM)
    with rasterio.open(asc) as src:
        assert src.driver == "AAIGrid" and src.read(1)[0, 0] == -9999
    cog = ls.write_raster(np.ones((600, 600), np.uint8), tmp_path / "c.tif", driver="COG", crs=CRS,
                          transform=TRANSFORM)
    with rasterio.open(cog) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
    with pytest.raises(ValueError, match="cannot tell the raster format"):
        ls.write_raster(f, tmp_path / "g.unknown_ext")


def test_write_raster_dask_by_blocks(tmp_path):
    dask_array = pytest.importorskip("dask.array")
    values = dask_array.from_array(np.arange(600 * 50, dtype=np.float32).reshape(600, 50), chunks=(128, 50))
    values = dask_array.where(values % 7 == 0, np.nan, values)
    out = ls.write_raster(values, tmp_path / "d.tif")
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(1), values.compute())
        assert np.isnan(src.nodata)


def test_write_raster_colormap_only_where_it_fits(tmp_path):
    with pytest.raises(ValueError, match="one-band uint8 or uint16"):
        ls.write_raster(np.zeros((2, 2), np.float32), tmp_path / "x.tif", colormap="mspa")


def test_write_raster_colour_tables_take_hex_names_and_tuples(tmp_path):
    pytest.importorskip("matplotlib")
    out = ls.write_raster(np.array([[0, 1], [2, 3]], np.uint8), tmp_path / "c.tif",
                          colormap={0: "#d9d9d9", 1: "forestgreen", 2: (1, 2, 3), 3: "#11223380"})
    with rasterio.open(out) as src:
        cmap = src.colormap(1)
    # GeoTIFF colour tables keep red, green and blue (GDAL writes no alpha)
    assert [cmap[i][:3] for i in range(4)] == [(217, 217, 217), (34, 139, 34), (1, 2, 3), (17, 34, 51)]
    assert ls.io._rgba("#11223380") == (17, 34, 51, 128)
    with pytest.raises(ValueError, match="r, g, b"):
        ls.write_raster(np.zeros((2, 2), np.uint8), tmp_path / "x.tif", colormap={0: (1, 2)})
