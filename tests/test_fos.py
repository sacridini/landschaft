# SPDX-License-Identifier: GPL-3.0-or-later
"""FOS (Fixed Observation Scale): unit tests and agreement with GuidosToolbox (GDL + spatcon/grayspatcon)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
import scenes
from gtb_reference import find_gdl, find_grayspatcon, find_spatcon, run_gtb_fos

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def binary_scene() -> np.ndarray:
    a = np.ones((40, 50), np.uint8)
    a[5:30, 5:30] = 2
    a[35, 5:45] = 2
    a[:, 45:] = 0
    a[2, 2] = 3
    a[3, 3] = 4
    return a


def test_fad_interior_and_codes():
    a = binary_scene()
    r = ls.fos(a, 7, "FAD", 6)
    m = np.asarray(r.map)
    assert m[17, 17] == 100  # deep inside the square
    assert m[5, 5] < m[17, 17]
    assert (m[:, 45:] == 102).all() and m[0, 0] == 101 and m[2, 2] == 105 and m[3, 3] == 106
    assert r.fg_area == int((a == 2).sum()) and r.reporting_area == int((a != 0).sum())
    assert r.n_patches == 2
    np.testing.assert_allclose(r.stats["percent"].sum(), 100.0)
    np.testing.assert_allclose(r.histogram.sum(), 100.0, rtol=1e-5)
    assert r.average == pytest.approx(np.asarray(r.map)[a == 2].mean())
    assert r.avcon == pytest.approx(r.average * r.fg_area / r.reporting_area)


def test_five_classes_merge_interior_and_intact():
    a = binary_scene()
    r6, r5 = ls.fos(a, 7, "FAD", 6), ls.fos(a, 7, "FAD", 5)
    np.testing.assert_array_equal(np.asarray(r6.map), np.asarray(r5.map))
    assert list(r5.stats["class"]) == ["Rare", "Patchy", "Transitional", "Dominant", "Interior"]
    assert r5.stats["percent"].iloc[4] == pytest.approx(r6.stats["percent"].iloc[4:].sum())


def test_fed_and_fac_are_lower_at_edges():
    a = binary_scene()
    for method in ("FED", "FAC"):
        m = np.asarray(ls.fos(a, 7, method).map)
        assert m[17, 17] == 100 and m[5, 17] < 100, method


def test_grayscale_threshold_and_codes():
    g = np.zeros((30, 30), np.uint8)
    g[5:25, 5:25] = 80
    g[10, 10] = 40
    g[0, :] = 255
    g[29, 0], g[29, 1] = 103, 104
    r = ls.fos(g, 5, "FAD")
    assert r.grayscale and r.gray_threshold == 30
    m = np.asarray(r.map)
    assert (m[0] == 102).all() and m[29, 0] == 105 and m[29, 1] == 106 and m[2, 2] == 101
    assert m[15, 15] == 80  # FAD of a grayscale map is the mean cover
    r50 = ls.fos(g, 5, "FAD", gray_threshold=50)
    assert np.asarray(r50.map)[10, 10] == 101 and r50.fg_area == r.fg_area - 1


def test_invalid_input():
    a = binary_scene()
    with pytest.raises(ValueError, match="method"):
        ls.fos(a, 7, "XYZ")
    with pytest.raises(ValueError, match="classes"):
        ls.fos(a, 7, classes=4)
    with pytest.raises(ValueError, match="window_size"):
        ls.fos(a, 8)
    with pytest.raises(ValueError, match="binary maps only"):
        ls.fos(np.full((20, 20), 50, np.uint8) | (np.arange(20) < 5).astype(np.uint8), 5, "FAC")
    with pytest.raises(ls.InvalidInputError, match="no background"):
        ls.fos(np.full((20, 20), 60, np.uint8), 5)
    g = np.full((20, 20), 60, np.uint8)
    g[0, 0], g[1, 1] = 0, 150
    with pytest.raises(ls.InvalidInputError, match="0-100"):
        ls.fos(g, 5)


def test_xarray_colormap_summary_and_batch(tmp_path):
    da = ls.load_raster(ls.example_data("input"))
    r = ls.fos(da, 13, "FED", 5)
    assert isinstance(r.map, xr.DataArray) and r.map.attrs["fos_classes"] == 5
    path = ls.write_raster(r.map, tmp_path / "fos.tif")
    import rasterio

    with rasterio.open(path) as src:
        assert src.colormap(1)[100][:3] == (0, 175, 0)  # fadcolors5: intact = interior
    text = r.summary()
    assert "FOS-type selected: FED_5class" in text and "Window size [pixels]: 13" in text
    out = ls.batch("fos", [ls.example_data("input")], tmp_path / "b", window_size=7, method="FAC", n_workers=1)
    assert out[0].ok and any(p.suffix == ".tif" for p in out[0].outputs)


# --- agreement with GTB ------------------------------------------------------------


def _reference() -> list[dict]:
    return json.loads(REFERENCE.read_text()).get("fos", []) if REFERENCE.exists() else []


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import fos_inputs

    return fos_inputs()


def assert_matches(r: ls.FOSResult, stats: dict[str, str]) -> None:
    assert r.fg_area == int(stats["fgarea"])
    assert r.reporting_area == int(stats["n_ruarea"])
    assert r.n_patches == int(stats["obj_last"])
    for attr, key in [("average_patch_size", "aps"), ("eca", "ECA"), ("coherence", "COH"),
                      ("coherence_ru", "COH_ru"), ("fg_area_pct", "fareaperc")]:
        assert getattr(r, attr) == pytest.approx(float(stats[key]), rel=1e-12), key
    # GTB computes these in single precision
    assert r.average == pytest.approx(float(stats["fad_av"]), rel=1e-6)
    assert r.avcon == pytest.approx(float(stats["fadru_av"]), rel=1e-6)
    classes = [float(v) for v in stats["classes"].split(",")]
    shares = list(r.stats["percent"])
    if r.classes == 6:
        np.testing.assert_allclose(shares, classes, rtol=1e-6, atol=1e-9)
    else:
        np.testing.assert_allclose(shares[:4], classes[:4], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(r.histogram, [float(v) for v in stats["hist2"].split(",")], rtol=1e-6)


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference(),
                         ids=lambda c: f"{c['input']}-{c['fosclass']}-w{c['window']}-c{c['connectivity']}")
def test_fos_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"]
    method, classes = case["fosclass"].split("_")
    r = ls.fos(data, case["window"], method, int(classes[0]), connectivity=case["connectivity"],
               gray_threshold=case["gray_threshold"])
    assert digest(r.map) == case["sha256"], "FOS map differs from GTB"
    assert_matches(r, case["stats"])


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(4))
def test_fos_live_against_gtb(seed):
    gdl, spatcon, gsc = find_gdl(), find_spatcon(), find_grayspatcon()
    if gdl is None or spatcon is None or gsc is None:
        pytest.skip("GDL, spatcon or grayspatcon not found")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import gray_scene, with_special_background

    rng = np.random.default_rng(1300 + seed)
    kind = list(scenes.SCENES)[seed % len(scenes.SCENES)]
    binary = with_special_background(scenes.make(kind, int(rng.integers(1, 10_000))), seed)
    gray = gray_scene(int(rng.integers(1, 10_000)))
    window = int(rng.choice([3, 5, 9, 15]))
    conn = int(rng.choice([4, 8]))
    for data, methods in ((binary, ("FAD", "FED", "FAC")), (gray, ("FAD", "FED"))):
        for method in methods:
            classes = int(rng.choice([5, 6]))
            grayt = int(rng.integers(10, 90))
            ref = run_gtb_fos(gdl, spatcon, gsc, data, f"{method}_{classes}class", window, conn, grayt)
            r = ls.fos(data, window, method, classes, connectivity=conn, gray_threshold=grayt)
            np.testing.assert_array_equal(np.asarray(r.map), ref["map"], err_msg=f"{method}_{classes}class")
            assert_matches(r, ref["stats"])
