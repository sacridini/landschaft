# SPDX-License-Identifier: GPL-3.0-or-later
"""FAD-MS and FAD Multiscale change: unit tests and agreement with GuidosToolbox.

The references run GTB's own code (tests/gtb_idl: FAD-MS and the FAD change
statements of guidostoolbox.pro) in GDL with GTB's spatcon executable; see
tools/make_reference_idl.py. ``live_reference`` tests need GDL and spatcon.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
import scenes
from gtb_reference import FAD_SCALES, find_gdl, find_spatcon, run_gtb_fad_change, run_gtb_fad_mscale

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"
KEYS = [*FAD_SCALES, "mscale"]


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def fad_map(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8).reshape(1, -1)


# --- unit tests -------------------------------------------------------------------


@pytest.fixture
def single_scale_change(monkeypatch):
    """The change of two FAD maps that stand for every observation scale."""
    from landschaft import fad_change

    def run(a, b):
        maps = {"A": {k: a for k in KEYS}, "B": {k: b for k in KEYS}}
        monkeypatch.setattr(fad_change, "_fad_ms_maps", lambda src, *_: maps[src])
        return ls.fad_multiscale_change("A", "B").scales["mscale"]

    return run


def test_class_boundaries_and_codes(single_scale_change):
    a = fad_map([9, 10, 39, 40, 59, 60, 89, 90, 99, 100, 101, 105, 106, 102, 50, 200, 106])
    b = fad_map([10, 9, 40, 39, 60, 59, 90, 89, 100, 99, 50, 50, 50, 50, 102, 50, 106])
    sc = single_scale_change(a, b)
    m = sc.matrix.values
    # rare<->patchy, patchy<->transitional, ... at every class boundary
    for lo in range(1, 6):
        assert m[lo, lo + 1] == 1 and m[lo + 1, lo] == 1
    assert m[0, 3] == 3 and m[0, 0] == 1  # 101, 105 and 106 are background; 50 is transitional
    assert m.sum() == 14  # missing at either time (102) and values >= 150 are left out
    assert sc.fg_area_a == 11 and sc.fg_area_b == 15 and sc.net_change == 4
    assert sc.forcom == 10
    assert sc.same_class == 0 and sc.changed == 10
    assert sc.frag_decrease_pct == pytest.approx(50.0) and sc.frag_increase_pct == pytest.approx(50.0)
    np.testing.assert_allclose(np.diag(sc.percent.values), 0.0)
    assert sc.percent.values.sum() == pytest.approx(100.0)


def test_direction_of_fragmentation_change(single_scale_change):
    sc = single_scale_change(fad_map([5, 5, 50, 100]), fad_map([50, 100, 100, 50]))
    # three pixels move to less fragmented classes (above the diagonal), one to a more fragmented one
    assert sc.frag_decrease_pct == pytest.approx(75.0)
    assert sc.frag_increase_pct == pytest.approx(25.0)
    assert sc.percent.loc["A1-Rare", "B6-Intact"] == pytest.approx(25.0)
    assert sc.percent.loc["A6-Intact", "B3-Transitional"] == pytest.approx(25.0)


def test_identical_times():
    data = scenes.blobs(3, shape=(260, 270))
    res = ls.fad_multiscale_change(data, data)
    for sc in res.scales.values():
        assert sc.changed == 0 and np.isnan(sc.frag_decrease_pct)
        assert sc.net_change == 0 and sc.fg_area_a == sc.forcom
        off = sc.matrix.values - np.diag(np.diag(sc.matrix.values))
        assert not off.any()
    np.testing.assert_allclose(res.averages["delta_fad_av"], 0.0)
    assert list(res.scales) == KEYS
    assert "average over 7x7-13x13-27x27-81x81-243x243" in res.summary()
    assert "Net foreground cover change (A->B) [pixels]: 0" in res.summary(27)


def test_inputs_and_checks(tmp_path):
    a = scenes.blobs(5, shape=(260, 260))
    b = scenes.blobs(6, shape=(260, 260))
    from_maps = ls.fad_multiscale_change(a, b)
    from_results = ls.fad_multiscale_change(ls.fad_multiscale(a), ls.fad_multiscale(b))
    for k in KEYS:
        np.testing.assert_array_equal(from_maps.scales[k].matrix, from_results.scales[k].matrix)

    # GTB output files: <name>_fad_mscale.tif next to <name>_fad_<window>.tif
    for name, data in (("a", a), ("b", b)):
        res = ls.fad_multiscale(data)
        for w in FAD_SCALES:
            ls.write_raster(res.single_scales[w], tmp_path / f"{name}_fad_{w}.tif", colormap=None)
        ls.write_raster(res.multiscale_map, tmp_path / f"{name}_fad_mscale.tif", colormap=None)
    from_files = ls.fad_multiscale_change(tmp_path / "a_fad_mscale.tif", tmp_path / "b_fad_mscale.tif")
    for k in KEYS:
        np.testing.assert_array_equal(from_files.scales[k].matrix, from_maps.scales[k].matrix)

    with pytest.raises(ValueError, match="different dimensions"):
        ls.fad_multiscale_change(a, a[:, :255])
    special = a.copy()
    special[special == 1][:10] = 4
    special[0, :5] = 4
    with pytest.raises(ValueError, match="non-fragmenting"):
        ls.fad_multiscale_change(special, b)
    with pytest.raises(ValueError, match="window sizes"):
        ls.fad_multiscale_change(ls.fad_multiscale(a, (7, 13)), b)
    small = ls.fad_multiscale_change(ls.fad_multiscale(a, (7, 13)), ls.fad_multiscale(b, (7, 13)),
                                     window_sizes=(7, 13))
    assert list(small.scales) == [7, 13, "mscale"]


def test_fad_multiscale_special_background_codes():
    data = scenes.blobs(8, shape=(60, 70))
    data[data == 1][:0] = 3
    data[0:3, 0:3] = 3
    data[5:8, 5:8] = 4
    res = ls.fad_multiscale(data, (7, 13))
    for m in (*res.single_scales.values(), res.multiscale_map):
        m = np.asarray(m)
        assert (m[0:3, 0:3] == 105).all() and (m[5:8, 5:8] == 106).all()
    assert "avcon" in res.stats.columns


# --- agreement with GTB -------------------------------------------------------------


def _reference(section: str) -> list[dict]:
    if not REFERENCE.exists():
        return []
    return json.loads(REFERENCE.read_text()).get(section, [])


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import fad_inputs

    return fad_inputs()


def assert_change_matches(res, change, perc, sums, delta_fad_av, delta_avcon):
    for i, key in enumerate(KEYS):
        sc = res.scales[key]
        np.testing.assert_array_equal(sc.matrix.values, np.asarray(change[i]), err_msg=f"scale {key}")
        got = [sc.frag_decrease_pct, sc.frag_increase_pct, sc.fg_area_a, sc.fg_area_b, sc.net_change,
               sc.forcom, sc.same_class, sc.changed]
        expected = list(sums[i])
        if sc.changed == 0:
            # 0 changed pixels: perc is 0/0 = NaN and IDL's TOTAL keeps the NaN; GDL's TOTAL
            # (unlike IDL's) skips NaN and reports 0. landschaft follows IDL.
            assert np.isnan(got[0]) and np.isnan(got[1])
            got[:2] = expected[:2]
        np.testing.assert_allclose(np.nan_to_num(got, nan=-1.0), expected, rtol=1e-12, err_msg=f"scale {key}")
        np.testing.assert_allclose(np.nan_to_num(sc.percent.values, nan=-1.0), np.asarray(perc[i]), rtol=1e-12,
                                   atol=1e-12, err_msg=f"scale {key}")
    # GTB keeps FAD_AV in single precision
    np.testing.assert_allclose(res.averages["delta_fad_av"], delta_fad_av, atol=5e-5)
    np.testing.assert_allclose(res.averages["delta_avcon"], delta_avcon, atol=5e-5)


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference("fad_mscale"), ids=lambda c: c["input"])
def test_fad_multiscale_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"], "reference input changed: rerun tools/make_reference_idl.py"
    res = ls.fad_multiscale(data)
    maps = {**{str(w): res.single_scales[w] for w in FAD_SCALES}, "mscale": res.multiscale_map}
    for key, sha in case["sha256"].items():
        assert digest(maps[key]) == sha, f"FAD map at scale {key} differs from GTB"
    fad_av = [float(np.asarray(maps[str(k)])[np.asarray(maps[str(k)]) <= 100].mean()) for k in KEYS]
    np.testing.assert_allclose(fad_av, case["fad_av"], rtol=1e-5)
    np.testing.assert_allclose(res.stats["average_fad"], np.round(case["fad_av"], 2), atol=0.0051)
    np.testing.assert_allclose(res.stats["avcon"], np.round(case["fadru_av"], 2), atol=0.0051)


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference("fad_change"), ids=lambda c: f"{c['a']}-{c['b']}")
def test_fad_change_matches_stored_gtb_outputs(reference_inputs, case):
    res = ls.fad_multiscale_change(reference_inputs[case["a"]], reference_inputs[case["b"]])
    assert_change_matches(res, case["change"], case["perc"], case["sums"], case["delta_fad_av"],
                          case["delta_avcon"])


@pytest.fixture(scope="module")
def gtb_tools():
    gdl, spatcon = find_gdl(), find_spatcon()
    if gdl is None or spatcon is None:
        pytest.skip("GDL and GTB's spatcon are needed to run GTB's code live "
                    "(LANDSCHAFT_GDL_EXE, LANDSCHAFT_SPATCON_EXE)")
    return gdl, spatcon


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(3))
def test_fad_change_live_against_gtb(gtb_tools, seed):
    rng = np.random.default_rng(500 + seed)
    shape = (int(rng.integers(250, 330)), int(rng.integers(250, 330)))
    a = scenes.blobs(int(rng.integers(1, 1000)), shape=shape, scale=int(rng.integers(4, 20)))
    b = a.copy()
    flip = scenes.blobs(int(rng.integers(1, 1000)), shape=shape, scale=8, cover=0.15, missing=0) == 2
    b[flip & (b > 0)] = 3 - b[flip & (b > 0)]  # swap foreground/background in some patches
    if seed == 2:
        a[(a == 1) & (rng.random(shape) < 0.1)] = 4
        b[(b == 1) & (rng.random(shape) < 0.1)] = 4
        a[(a == 1) & (rng.random(shape) < 0.1)] = 3

    gdl, spatcon = gtb_tools
    ms = run_gtb_fad_mscale(gdl, spatcon, a)
    res_a = ls.fad_multiscale(a)
    for w in FAD_SCALES:
        np.testing.assert_array_equal(np.asarray(res_a.single_scales[w]), ms["maps"][w], err_msg=f"scale {w}")
    np.testing.assert_array_equal(np.asarray(res_a.multiscale_map), ms["maps"]["mscale"])

    ref = run_gtb_fad_change(gdl, spatcon, a, b)
    res = ls.fad_multiscale_change(res_a, b)
    assert_change_matches(res, ref["change"], np.nan_to_num(ref["perc"], nan=-1.0),
                          np.nan_to_num(ref["sums"], nan=-1.0), ref["delta_fad_av"], ref["delta_avcon"])
