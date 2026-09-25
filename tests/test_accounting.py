# SPDX-License-Identifier: GPL-3.0-or-later
"""Accounting: unit tests and agreement with GuidosToolbox's IDL code.

* ``reference``: digests stored in tests/data/reference/idl_reference.json,
  produced by tools/make_reference_idl.py running GTB's code in GDL.
* ``live_reference``: runs GTB's code in GDL now on fresh random inputs
  (GDL from ``LANDSCHAFT_GDL_EXE``, PATH or its default install).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import scipy.ndimage as ndi
import xarray as xr

import landschaft as ls
import scenes
from conftest import mspa_input
from gtb_reference import find_gdl, run_gtb_accounting, run_gtb_accounting_scaled_classes

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


# --- unit tests -------------------------------------------------------------------


def test_simple_scene_maps_and_stats():
    # objects of 1, 4 and 30 pixels; missing, special backgrounds
    a = mspa_input((10, 12), foreground=[np.s_[0, 0], np.s_[2:4, 2:4], np.s_[5:10, 6:12]])
    a[0, 11] = 0
    a[1, 11] = 3
    a[2, 11] = 4
    r = ls.accounting(a, (2, 10))

    # ids are numbered in raster order from the bottom row, as in GTB
    assert r.ids[9, 6] == 1 and r.ids[3, 2] == 2 and r.ids[0, 0] == 3
    assert r.object_area.tolist() == [30, 4, 1]
    assert r.area[9, 11] == 30 and r.area[2, 3] == 4 and r.area[0, 0] == 1
    assert r.classes[0, 0] == 103 and r.classes[2, 2] == 33 and r.classes[5, 6] == 65
    assert r.classes[1, 1] == 0 and r.ids[1, 1] == 0 and r.area[1, 1] == 0
    for (y, x), code, value in [((0, 11), 129, -1), ((1, 11), 105, -3), ((2, 11), 176, -4)]:
        assert r.classes[y, x] == code and r.ids[y, x] == value and r.area[y, x] == value

    s = r.stats
    assert s["objects"].tolist() == [1, 1, 1]
    assert s["area"].tolist() == [1, 4, 30]
    assert s["range"].tolist() == ["1 - 2", "3 - 10", "11 -> "]
    assert s["color"].tolist() == ["black", "red", "yellow"]
    np.testing.assert_allclose(s["pct_area"], [100 / 35, 400 / 35, 3000 / 35])
    assert r.n_objects == 3 and r.foreground_area == 35
    assert r.median_area == 4 and r.mean_area == pytest.approx(35 / 3)
    assert r.std_area == pytest.approx(np.std([1, 4, 30], ddof=1))


def test_matches_scipy_labelling():
    rng = np.random.default_rng(3)
    a = np.where(rng.random((80, 90)) < 0.5, 2, 1).astype(np.uint8)
    for conn, structure in [(8, np.ones((3, 3))), (4, None)]:
        r = ls.accounting(a, connectivity=conn)
        labels, n = ndi.label(np.flipud(a == 2), structure=structure)
        labels = np.flipud(labels)
        assert r.n_objects == n
        np.testing.assert_array_equal(np.where(a == 2, r.ids, 0), labels)
        np.testing.assert_array_equal(r.object_area, np.bincount(labels.ravel())[1:])


def test_connectivity():
    a = mspa_input((5, 5), foreground=[np.s_[1, 1], np.s_[2, 2], np.s_[3, 3]])
    assert ls.accounting(a).n_objects == 1
    assert ls.accounting(a, connectivity=4).n_objects == 3
    with pytest.raises(ValueError):
        ls.accounting(a, connectivity=6)


def test_objects_touching_the_border():
    a = np.full((6, 6), 2, np.uint8)
    a[2:4, :] = 1
    r = ls.accounting(a)
    assert r.object_area.tolist() == [12, 12]
    assert r.ids[5, 0] == 1 and r.ids[0, 0] == 2


def test_thresholds():
    a = mspa_input((20, 20), foreground=[np.s_[0, 0], np.s_[2:4, 2:4], np.s_[10:20, 10:20]])
    r = ls.accounting(a, (5,))
    assert r.thresholds == (5,)
    assert r.stats["objects"].tolist() == [2, 1]
    assert set(np.unique(r.classes)) == {0, 103, 33}
    # repeated values are dropped as in GTB
    assert ls.accounting(a, (4, 4, 50, 50)).thresholds == (4, 50)
    for bad in [(), (0, 5), (5, 3), (1, 2, 3, 4, 5, 6), "big"]:
        with pytest.raises(ValueError):
            ls.accounting(a, bad)


def test_auto_thresholds_follow_gtb_formula():
    a = scenes.make("blobs", 101)
    r = ls.accounting(a, "auto")
    labels, _ = ndi.label(a == 2, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    lo, hi = np.float32(areas.min()), np.float32(areas.max() - areas.min())
    expected = [int(lo + np.float32(f) * hi) for f in (0.001, 0.01, 0.1, 0.5, 0.9)]
    assert r.thresholds == tuple(dict.fromkeys(expected))
    # the proposal uses 8-connected objects even for 4-connectivity
    assert ls.accounting(a, "auto", connectivity=4).thresholds == r.thresholds


def test_highlight_largest():
    a = mspa_input((12, 12), foreground=[np.s_[0, 0:3], np.s_[2, 0:3], np.s_[4, 0:2], np.s_[6, 0], np.s_[8:12, 8:12]])
    r = ls.accounting(a, highlight_largest=True)
    # areas 16, 3, 3, 2, 1: the two objects of 3 pixels share rank 2
    assert [sorted(r.object_area[np.array(ids) - 1].tolist()) for ids in r.largest] == [[16], [3, 3], [2]]
    assert r.highlighted
    assert (r.classes[a == 2] == 80).sum() == 16 + 3 + 3 + 2
    assert r.classes[6, 0] == 103
    assert "Three largest object IDs" in r.summary()

    two = mspa_input((5, 5), foreground=[np.s_[0, 0], np.s_[3:5, 3:5]])
    r2 = ls.accounting(two, highlight_largest=True)  # GTB needs at least 3 objects
    assert not r2.highlighted and 80 not in r2.classes
    assert "Largest object:     4" in r2.summary()


def test_median_is_upper_middle_value():
    a = mspa_input((3, 12), foreground=[np.s_[0, 0], np.s_[0, 2:4], np.s_[0, 5:8], np.s_[0, 9:12]])
    a[2, :] = 2  # areas 1, 2, 3, 3 and 12
    a[1, :] = 1
    r = ls.accounting(a, connectivity=4)
    assert sorted(r.object_area.tolist()) == [1, 2, 3, 3, 12]
    b = a.copy()
    b[2, :] = 1  # areas 1, 2, 3, 3: IDL MEDIAN without /EVEN returns 3
    assert ls.accounting(b, connectivity=4).median_area == 3


def test_single_object():
    r = ls.accounting(mspa_input((4, 4), foreground=[np.s_[1:3, 1:3]]))
    assert r.n_objects == 1 and r.median_area == 4 and np.isnan(r.std_area)


def test_invalid_inputs():
    with pytest.raises(ls.InvalidInputError, match="no foreground"):
        ls.accounting(np.ones((5, 5), np.uint8))
    with pytest.raises(ls.InvalidInputError, match="no foreground"):
        ls.accounting(np.array([[1, 3], [4, 0]], np.uint8))
    with pytest.raises(ls.InvalidInputError, match="larger than 4"):
        ls.accounting(np.array([[2, 5]], np.uint8))
    with pytest.raises(ls.InvalidInputError, match="uint8"):
        ls.accounting(np.array([[2, 1]], np.int16))
    with pytest.raises(ls.InvalidInputError, match="2-D"):
        ls.accounting(np.full((2, 3, 3), 2, np.uint8))


def test_xarray_and_geotiff(tmp_path):
    da = ls.load_raster(ls.example_data("input"))
    r = ls.accounting(da, highlight_largest=True)
    for out in (r.classes, r.ids, r.area):
        assert isinstance(out, xr.DataArray)
        assert out.rio.crs == da.rio.crs and out.rio.transform() == da.rio.transform()
    assert r.classes.dtype == np.uint8 and r.ids.dtype == np.int32 and r.area.dtype == np.int32
    assert r.classes.attrs["landschaft_product"] == "ACCOUNTING"

    path = ls.write_raster(r.classes, tmp_path / "acc.tif")
    with rasterio.open(path) as src:
        assert src.colormap(1)[80] == (255, 0, 255, 255)
        assert src.colormap(1)[176] == (191, 191, 255, 255)
        assert "GTB_ACC" in src.tags()["TIFFTAG_IMAGEDESCRIPTION"]
        np.testing.assert_array_equal(src.read(1), r.classes.values)
    path = ls.write_raster(r.ids, tmp_path / "ids.tif")
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), r.ids.values)

    from_path = ls.accounting(ls.example_data("input"))
    np.testing.assert_array_equal(from_path.ids.values, r.ids.values)


def test_summary_and_stats_layout():
    r = ls.accounting(ls.example_data("input"))
    assert isinstance(r.stats, pd.DataFrame)
    assert list(r.stats.columns) == ["size_class", "range", "color", "code", "objects", "area",
                                     "pct_objects", "pct_area"]
    assert r.stats["objects"].sum() == r.n_objects
    assert r.stats["area"].sum() == r.foreground_area
    text = r.summary()
    assert "Size class 6: [20001 -> ] pixels; color: green" in text
    assert "pixel_to_hectare: 0.0625" in text


def test_threads_do_not_change_the_result():
    a = scenes.make("salt", 5)
    one = ls.accounting(a, n_threads=1)
    many = ls.accounting(a, n_threads=0)
    np.testing.assert_array_equal(one.ids, many.ids)
    np.testing.assert_array_equal(one.classes, many.classes)


# --- agreement with GTB's IDL code ------------------------------------------------


def _reference_cases(kind: str):
    if not REFERENCE.exists():
        return []
    return [c for c in json.loads(REFERENCE.read_text())["accounting"] if kind in c]


def _inputs() -> dict[str, np.ndarray]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import accounting_inputs

    return accounting_inputs()


@pytest.fixture(scope="module")
def reference_inputs():
    return _inputs()


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def assert_matches_gtb(r: ls.AccountingResult, stats: dict[str, str]) -> None:
    """Every number of GTB's report; integers exactly, GTB's single-precision floats to 1e-5."""
    n_classes = int(stats["nr_cat"])
    assert r.thresholds == tuple(int(v) for v in stats["cat"].split(","))
    assert r.stats["objects"].tolist() == [int(v) for v in stats["cat_id"].split(",")]
    assert r.stats["area"].tolist() == [int(v) for v in stats["cat_area"].split(",")]
    assert len(r.stats) == n_classes + 1
    np.testing.assert_allclose(r.stats["pct_objects"], _floats(stats["pct_objects"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r.stats["pct_area"], _floats(stats["pct_area"]), rtol=1e-5, atol=1e-5)
    assert r.n_objects == int(stats["tot_ids"]) == int(stats["allid"])
    assert r.foreground_area == int(stats["allarea"])
    assert r.median_area == int(stats["median"])
    assert r.mean_area == pytest.approx(float(stats["mean"]), rel=1e-5)
    if r.n_objects > 1:
        # IDL/GDL MOMENT works in single precision; landschaft uses double.
        assert r.std_area == pytest.approx(float(stats["sdev"]), rel=1e-4)
    assert int(r.object_area.max()) == int(stats["oba_max"])
    if r.highlighted:
        for rank in range(3):
            expected = [int(v) for v in stats[f"id_max{rank + 1}"].split(",")]
            assert list(r.largest[rank]) == ([] if expected == [0] else expected)


@pytest.mark.reference
@pytest.mark.parametrize(
    "case", _reference_cases("sha256"),
    ids=lambda c: f"{c['input']}-c{c['connectivity']}-{'_'.join(map(str, c['thresholds']))}-h{int(c['highlight_largest'])}",
)
def test_accounting_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"], "reference input changed: rerun tools/make_reference_idl.py"
    r = ls.accounting(data, case["thresholds"], connectivity=case["connectivity"],
                      highlight_largest=case["highlight_largest"])
    for name in ("classes", "ids", "area"):
        assert digest(getattr(r, name)) == case["sha256"][name], f"{name} differs from GTB"
    assert_matches_gtb(r, case["stats"])


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference_cases("scaled_classes"), ids=lambda c: c["input"])
def test_auto_thresholds_match_stored_gtb_proposal(reference_inputs, case):
    data = reference_inputs[case["input"]]
    expected = tuple(dict.fromkeys(case["scaled_classes"]))
    assert ls.accounting(data, "auto").thresholds == expected


@pytest.fixture(scope="module")
def gdl():
    exe = find_gdl()
    if exe is None:
        pytest.skip("GDL not found: set LANDSCHAFT_GDL_EXE to run GTB's IDL code live")
    return exe


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(6))
def test_accounting_live_against_gtb(gdl, seed):
    rng = np.random.default_rng(1000 + seed)
    kind = list(scenes.SCENES)[seed % len(scenes.SCENES)]
    data = scenes.make(kind, int(rng.integers(1, 10_000))).copy()
    bg = data == 1
    data[bg & (rng.random(data.shape) < 0.05)] = int(rng.choice([3, 4]))
    n = int(rng.integers(1, 6))
    thresholds = tuple(sorted(int(v) for v in rng.integers(1, 400, n)))
    conn = int(rng.choice([4, 8]))
    highlight = bool(rng.integers(0, 2))

    ref = run_gtb_accounting(gdl, data, thresholds, connectivity=conn, highlight_largest=highlight)
    r = ls.accounting(data, thresholds, connectivity=conn, highlight_largest=highlight)
    for name in ("classes", "ids", "area"):
        np.testing.assert_array_equal(np.asarray(getattr(r, name)), ref[name], err_msg=name)
    assert_matches_gtb(r, ref["stats"])
    assert ls.accounting(data, "auto").thresholds == tuple(dict.fromkeys(run_gtb_accounting_scaled_classes(gdl, data)))
