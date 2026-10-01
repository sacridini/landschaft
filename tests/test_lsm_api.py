# SPDX-License-Identifier: GPL-3.0-or-later
"""The rest of landscapemetrics' API: ``full_name``, ``sample_lsm(return_raster
= TRUE)`` and the public helpers ``construct_buffer``, ``pad_raster``,
``unpad_raster``, ``raster_to_points`` and ``get_unique_values``.

``tests/data/reference/lsm_api`` holds landscapemetrics' results for stored
maps (``make_reference_lsm.py api``); all must be the same bits (except the
metrics that are not bitwise anyway, e.g. ``circle``).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from conftest import DATA
from test_lsm_reference import NOT_BITWISE, RTOL
from test_lsm_utils import assert_same, matrix, read

REF_DIR = DATA / "reference" / "lsm_api"
MANIFEST = json.loads((REF_DIR / "manifest.json").read_text())
CASES = sorted(MANIFEST["files"])


def tif(case: str):
    return DATA / "lsm" / MANIFEST["files"][case]["input"]


def points(case: str) -> np.ndarray:
    return pd.read_csv(REF_DIR / case / "points.csv", float_precision="round_trip")[["x", "y"]].to_numpy()


def size(case: str) -> float:
    return MANIFEST["files"][case]["size"]


def columns(case: str, name: str) -> list[str]:
    return (REF_DIR / case / f"{name}_columns.txt").read_text().split()


def extent_of(da) -> tuple[float, ...]:
    t = da.rio.transform()
    ny, nx = da.shape
    return t.c, t.c + t.a * nx, t.f + t.e * ny, t.f


def assert_raster(da, case: str, name: str, exact: bool = True) -> None:
    ext = read(REF_DIR / case, f"{name}_extent")
    ny, nx = int(ext["nrow"][0]), int(ext["ncol"][0])
    assert da.shape == (ny, nx), f"{case} {name}: shape {da.shape}, R {(ny, nx)}"
    assert_same(np.asarray(da.values), matrix(REF_DIR / case, name, (ny, nx)), f"{case} {name}")
    ref = tuple(float(ext[c][0]) for c in ("xmin", "xmax", "ymin", "ymax"))
    ours = extent_of(da)
    # the geotransform holds terra's xmin and ymax; xmax and ymin come back
    # from it (xmin + ncol * xres) to within an ulp or so
    if exact:
        assert (ours[0], ours[3]) == (ref[0], ref[3]), f"{case} {name} extent {ours} != {ref}"
    atol = 8 * np.spacing(max(abs(v) for v in ref))
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=atol, err_msg=f"{case} {name} extent")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# -- full_name -----------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_full_name(case):
    ref = read(REF_DIR / case, "full_name")
    what = sorted(set(ref["function_name"]))
    ours = ls.landscape_metrics([tif(case)], what=what, full_name=True)
    assert list(ours.columns) == columns(case, "full_name")
    for col in ("layer", "level", "class", "id", "metric", "name", "type", "function_name"):
        a = ours[col].astype(object).where(ours[col].notna(), None).tolist()
        b = ref[col].astype(object).where(ref[col].notna(), None).tolist()
        assert a == b, f"{case}: column {col} differs"
    a, b = ours["value"].to_numpy(float), ref["value"].to_numpy(float)
    np.testing.assert_allclose(a, b, rtol=RTOL, equal_nan=True)
    bitwise = ~ours["metric"].astype(str).str.startswith(NOT_BITWISE).to_numpy()
    assert_same(a[bitwise], b[bitwise], f"{case} full_name values")


def test_full_name_everywhere():
    path = tif("augusta")
    single = ls.landscape_metrics(path, what=["lsm_p_area", "lsm_l_shdi"], full_name=True)
    assert list(single.columns) == ["level", "class", "id", "metric", "value", "name", "type", "function_name"]
    assert set(single["function_name"]) == {"lsm_p_area", "lsm_l_shdi"}
    assert (single.loc[single["metric"] == "shdi", "name"] == "shannon's diversity index").all()
    method = ls.Landscape(path).metrics(what=["lsm_p_area", "lsm_l_shdi"], full_name=True)
    pd.testing.assert_frame_equal(single, method)
    # the FRAGSTATS extras have their own descriptions
    extra = ls.landscape_metrics(path, what=["lsm_c_area_am", "lsm_l_perim_mn"], full_name=True)
    assert extra["function_name"].notna().all() and extra["name"].notna().all()
    assert set(extra["function_name"]) == {"lsm_c_area_am", "lsm_l_perim_mn"}
    xy = points("augusta")
    ex = ls.extract_metrics(path, xy, what="lsm_p_area", full_name=True)
    assert list(ex.columns) == ["level", "class", "id", "metric", "value", "name", "type", "function_name",
                                "extract_id"]
    sc = ls.scale_sample(path, xy, size=[60, 120], what="lsm_l_ta", full_name=True)
    assert list(sc.columns[:8]) == ["level", "class", "id", "metric", "value", "name", "type", "function_name"]
    assert (sc["function_name"] == "lsm_l_ta").all()
    off = ls.landscape_metrics(path, what="lsm_l_ta")
    assert "name" not in off.columns


# -- sample_lsm(return_raster = TRUE) ----------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", ["square", "circle"])
def test_sample_rasters(case, shape):
    s = ls.sample_metrics(tif(case), points(case), shape=shape, size=size(case), what="lsm_l_ta",
                          return_raster=True, verbose=False)
    # R: layer first, and a layer column even for one raster
    assert list(s.columns) == [c for c in columns(case, f"sample_{shape}") if c != "layer"]
    # terra::crop(mask = TRUE) moves the extent of the raster it crops by a
    # few ulps (removeWindow), so only the first plot sees the intact raster
    for plot_id, da in zip(s["plot_id"], s["raster_sample_plots"]):
        assert_raster(da, case, f"sample_{shape}_{plot_id}", exact=plot_id == 1)


@pytest.mark.reference
def test_sample_full_name_columns():
    case = "augusta"
    s = ls.sample_metrics(tif(case), points(case), size=size(case), what="lsm_l_ta", full_name=True,
                          verbose=False)
    ref = [c for c in columns(case, "sample_full_name") if c != "layer"]
    # landscapemetrics drops column 9 (function_name) and keeps an unasked raster column
    assert ref == ["level", "class", "id", "metric", "value", "name", "type", "plot_id", "percentage_inside",
                   "raster_sample_plots"]
    assert list(s.columns) == ["level", "class", "id", "metric", "value", "name", "type", "function_name",
                               "plot_id", "percentage_inside"]


def test_sample_rasters_all_classes_and_outside():
    path = tif("augusta")
    xy = np.vstack([points("augusta"), [[0.0, 0.0]]])
    s = ls.sample_metrics(path, xy, size=100, what=["lsm_c_ca", "lsm_l_ta"], return_raster=True, all_classes=True)
    by_plot = s.groupby("plot_id")["raster_sample_plots"].first()
    assert by_plot.loc[6] is None  # outside the raster
    for pid in range(1, 6):
        rasters = s.loc[s["plot_id"] == pid, "raster_sample_plots"]
        assert all(r is rasters.iloc[0] for r in rasters)
        assert rasters.iloc[0].rio.crs is not None
    # the cells of the plot raster are the ones measured
    area = s[(s["metric"] == "ta") & (s["plot_id"] == 1)]["value"].iloc[0]
    da = by_plot.loc[1]
    assert np.count_nonzero(~np.isnan(da.values)) * 30 * 30 / 10000 == area


def test_sample_rasters_numpy():
    data = np.arange(100).reshape(10, 10) % 3
    s = ls.sample_metrics(data, [[5.0, 5.0]], size=2, what="lsm_l_ta", return_raster=True)
    da = s["raster_sample_plots"].iloc[0]
    assert da.shape == (4, 4)
    np.testing.assert_array_equal(da.values, data[3:7, 3:7])
    assert extent_of(da) == (3.0, 7.0, 3.0, 7.0)


# -- pad_raster / unpad_raster -----------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", [c for c in CASES if "pad_false.csv" in MANIFEST["files"][c]["outputs"]])
def test_pad_unpad(case):
    for g in (False, True):
        assert_raster(ls.pad_raster(tif(case), -999, 2, global_=g), case, f"pad_{str(g).lower()}")
    padded = ls.pad_raster(tif(case), 7, 3)
    assert_raster(ls.unpad_raster(padded, 3), case, "unpad")


def test_pad_unpad_arrays():
    data = np.ma.masked_equal(np.array([[1, 2, 3], [4, 0, 6]]), 0)
    p = ls.pad_raster(data, -1, 1)
    assert isinstance(p, np.ndarray) and p.shape == (4, 5)
    assert np.isnan(p[2, 2]) and (p[0] == -1).all() and (p[:, -1] == -1).all()
    assert (ls.pad_raster(data, -1, 1, global_=True)[2, 2]) == -1
    np.testing.assert_array_equal(ls.unpad_raster(p, 1), np.ma.filled(data.astype(float), np.nan))
    # non-square cells (landscapemetrics fails there): each axis grows by its cell size
    da = ls.load_raster(tif("synthetic_na"))
    pd_ = ls.pad_raster(da, 0, 2)
    t0, t1 = da.rio.transform(), pd_.rio.transform()
    assert (t1.c, t1.f) == (t0.c - 2 * t0.a, t0.f - 2 * t0.e)
    assert pd_.shape == (da.shape[0] + 4, da.shape[1] + 4)
    back = ls.unpad_raster(pd_, 2)
    assert back.rio.transform() == t0
    several = ls.pad_raster([data, data], 0, 1)
    assert isinstance(several, list) and len(several) == 2
    with pytest.raises(ValueError):
        ls.unpad_raster(np.ones((2, 2)), 1)


# -- raster_to_points ------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("return_na", [True, False])
def test_raster_to_points(case, return_na):
    ref = read(REF_DIR / case, f"points_{str(return_na).lower()}")
    ours = ls.raster_to_points(tif(case), return_NA=return_na)
    assert list(ours.columns) == ["layer", "x", "y", "z"]
    assert_same(ours.to_numpy(float), ref[["layer", "x", "y", "z"]].to_numpy(float), f"{case} points")


def test_raster_to_points_layers():
    a = np.array([[1, 2], [np.nan, 4]])
    df = ls.raster_to_points([a, a], return_NA=False)
    assert df["layer"].tolist() == [1, 1, 1, 2, 2, 2]
    assert df[["x", "y"]].to_numpy().tolist()[:3] == [[0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]


# -- get_unique_values -----------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_unique_values(case):
    ref = read(REF_DIR / case, "unique")["value"].to_numpy()
    ours = ls.get_unique_values(tif(case))
    assert list(ours) == ["layer_1"]
    np.testing.assert_array_equal(ours["layer_1"], ref)
    np.testing.assert_array_equal(ls.get_unique_values(tif(case), simplify=True), ref)


def test_unique_values_paths():
    rng = np.random.default_rng(3)
    for hi in (5, 1 << 30):  # presence table and sorted chunks
        data = rng.integers(-hi, hi, size=(700, 400)).astype(np.int32)
        mask = rng.random(data.shape) < 0.1
        expected = np.unique(data[~mask])
        np.testing.assert_array_equal(ls.get_unique_values(np.ma.array(data, mask=mask), simplify=True), expected)
    assert ls.get_unique_values(np.array([3.7, 1.0, np.nan, 3.0]), simplify=True).tolist() == [1, 3]
    assert ls.get_unique_values(np.full((3, 3), np.nan), simplify=True).size == 0
    out = ls.get_unique_values([np.array([[2, 1]]), np.array([[5, 5]])])
    assert {k: v.tolist() for k, v in out.items()} == {"layer_1": [1, 2], "layer_2": [5]}
    with pytest.warns(UserWarning):
        ls.get_unique_values([np.array([[2, 1]]), np.array([[5, 5]])], simplify=True)


# -- construct_buffer ------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", ["square", "circle"])
def test_construct_buffer(case, shape):
    ref = read(REF_DIR / case, f"buffer_{shape}")
    ours = ls.construct_buffer(points(case), shape, size(case), return_vec=False)
    assert_same(ours, ref[["x", "y", "id"]].to_numpy(float), f"{case} {shape} buffer")
    vec = read(REF_DIR / case, f"buffer_{shape}_vector")
    gdf = ls.construct_buffer(points(case), shape, size(case))
    assert gdf["plot_id"].tolist() == sorted(set(vec["geom"]))
    for pid, poly in zip(gdf["plot_id"], gdf.geometry):
        ring = np.asarray(poly.exterior.coords)
        r = vec.loc[vec["geom"] == pid, ["x", "y"]].to_numpy()
        assert_same(ring, r, f"{case} {shape} polygon {pid}")


def test_construct_buffer_is_sample_plot():
    xy = points("augusta")
    gdf = ls.construct_buffer(xy, "circle", 200)
    path = tif("augusta")
    a = ls.sample_metrics(path, xy, size=200, shape="circle", what="lsm_l_ta")
    b = ls.sample_metrics(path, gdf.geometry, what="lsm_l_ta")
    np.testing.assert_array_equal(a["value"], b["value"])
    with pytest.raises(ValueError):
        ls.construct_buffer(xy, "hexagon", 1)


def test_misspelt_name():
    # landscapemetrics spells siei's name "simspon's"; both spellings select it
    a = ls.list_metrics(name="simspon's evenness index")
    b = ls.list_metrics(name="simpson's evenness index")
    assert a["metric"].tolist() == b["metric"].tolist() == ["siei"]
    assert "siei" not in ls.list_metrics(level="landscape", name="-simpson's evenness index")["metric"].tolist()
