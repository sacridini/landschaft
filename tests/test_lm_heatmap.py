# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape Mosaic heatmap and heatmap change: unit tests and agreement with GuidosToolbox.

The references run GTB's spatcon (rule 7) and the heatmap / delta_lm
statements of guidostoolbox.pro in GDL (tests/gtb_idl); see
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
from gtb_reference import find_gdl, find_spatcon, run_gtb_lm_delta, run_gtb_lm_heatmap
from landschaft.lm_heatmap import LM103_CODES

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def float32_hex(a) -> str:
    return np.asarray(a, dtype="<f4").tobytes().hex()


@pytest.fixture(scope="module")
def landcover() -> np.ndarray:
    rng = np.random.default_rng(9)
    return rng.choice([0, 1, 2, 3], size=(80, 90), p=[0.05, 0.4, 0.4, 0.15]).astype(np.uint8)


# --- unit tests -------------------------------------------------------------------


def test_heatmap_values(landcover):
    h = ls.landscape_mosaic_heatmap(landcover, 7)
    assert h.values.dtype == np.float32 and h.values.shape == (256,)
    im103 = np.asarray(ls.landscape_mosaic(landcover, 7, output_classes=103))
    counts = np.bincount(im103.ravel(), minlength=256)
    np.testing.assert_allclose(h.values, counts / counts[1:].sum() * 100, rtol=1e-6)
    assert h.values[list(LM103_CODES)].sum() == pytest.approx(100.0, abs=1e-3)
    assert set(np.nonzero(counts[1:])[0] + 1) <= set(LM103_CODES)
    same = ls.landscape_mosaic_heatmap(im103, is_lm103=True, window="7")
    np.testing.assert_array_equal(same.values, h.values)

    t = h.table
    assert len(t) == 103 and t.index.name == "class_code"
    assert t.loc[170, "recode_class_19"] == 17


def test_triangle_layout(landcover):
    h = ls.landscape_mosaic_heatmap(landcover, 7)
    tri = h.triangle
    assert tri.shape == (12, 21)
    assert np.isfinite(tri).sum() == 103
    assert tri[0, 10] == h.values[170]  # forest corner at the top
    assert tri[11, 0] == h.values[190] and tri[11, 20] == h.values[180]
    assert np.isnan(tri[0, 0]) and np.isnan(tri[11, 10])


def test_csv_layout(landcover, tmp_path):
    h = ls.landscape_mosaic_heatmap(landcover, 7)
    lines = h.csv_text().splitlines()
    assert len(lines) == 13
    assert lines[0] == "Landscape Mosaic using Window size (7x7)" + ", " * 20 + ","
    assert all(line.count(",") == 21 for line in lines[1:])
    assert ls.landscape_mosaic_heatmap(landcover, 7, window="multiscale").csv_text().startswith(
        "Landscape Mosaic (multiscale),")
    assert h.to_csv(tmp_path / "heatmap.csv").read_text() == h.csv_text()


def test_idl_cell_format():
    from landschaft.lm_heatmap import _idl_cell

    assert _idl_cell(np.float32(10.869387)) == "      10.869"
    assert _idl_cell(np.float32(0.0)) == "      0.000"
    assert _idl_cell(np.float32(0.0254)) == "    0.025"
    assert _idl_cell(np.float32(-3.14159)) == "     -3.142"
    assert _idl_cell(np.float32(-0.0003)) == "      0.000"  # rounds to 0, no sign
    assert _idl_cell(np.float32(100.0)) == "      100.000"


def test_change(landcover):
    other = landcover.copy()
    other[other == 1] = 2
    ha = ls.landscape_mosaic_heatmap(landcover, 7)
    hb = ls.landscape_mosaic_heatmap(other, 7)
    d = ls.landscape_mosaic_change(ha, hb)
    np.testing.assert_array_equal(d.values, hb.values - ha.values)
    assert d.values[list(LM103_CODES)].sum() == pytest.approx(0.0, abs=1e-3)
    assert list(d.table.columns) == ["percent_a", "percent_b", "change", "recode_class_19"]
    assert d.csv_text().splitlines()[0].startswith("Delta Landscape Mosaic: B(Window size: 7) - A(Window size: 7)")
    from_maps = ls.landscape_mosaic_change(landcover, other, 7)
    np.testing.assert_array_equal(from_maps.values, d.values)
    assert not ls.landscape_mosaic_change(ha, ha).values.any()


def test_lm_class_boundary_matches_spatcon():
    # 13x13 window, 9 missing, 12 agriculture, 112 forest (p = 0.7 exactly), 36 developed:
    # spatcon compares the float proportion with the double constant 0.70, so the
    # window is *not* ">= 70% forest" (class 94, not 93).
    data = np.array([0] * 9 + [1] * 12 + [2] * 112 + [3] * 36, dtype=np.uint8).reshape(13, 13)
    assert data[6, 6] != 0
    assert np.asarray(ls.landscape_mosaic(data, 13, output_classes=103))[6, 6] == 94


def test_invalid():
    with pytest.raises(ValueError, match="no classified"):
        ls.landscape_mosaic_heatmap(np.zeros((5, 5), np.uint8), is_lm103=True)
    with pytest.raises(ValueError, match="uint8"):
        ls.landscape_mosaic_heatmap(np.zeros((5, 5), np.int32), is_lm103=True)


# --- agreement with GTB -------------------------------------------------------------


def _reference(section: str) -> list[dict]:
    if not REFERENCE.exists():
        return []
    return json.loads(REFERENCE.read_text()).get(section, [])


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import lm_inputs

    return lm_inputs()


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference("lm_heatmap"), ids=lambda c: f"{c['input']}-w{c['window']}")
def test_heatmap_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"], "reference input changed: rerun tools/make_reference_idl.py"
    im103 = ls.landscape_mosaic(data, case["window"], output_classes=103)
    assert digest(im103) == case["im103_sha256"], "103-class LM differs from GTB's spatcon"
    h = ls.landscape_mosaic_heatmap(im103, is_lm103=True, window=str(case["window"]))
    assert float32_hex(h.values) == case["hn"], "heatmap vector differs from GTB (bit for bit)"
    assert h.csv_text() == case["csv"]


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference("lm_delta"), ids=lambda c: f"{c['a'][0]}-{c['b'][0]}")
def test_heatmap_change_matches_stored_gtb_outputs(reference_inputs, case):
    (name_a, win_a), (name_b, win_b) = case["a"], case["b"]
    ha = ls.landscape_mosaic_heatmap(reference_inputs[name_a], win_a)
    hb = ls.landscape_mosaic_heatmap(reference_inputs[name_b], win_b)
    d = ls.landscape_mosaic_change(ha, hb)
    assert float32_hex(d.values) == case["hn"]
    assert d.csv_text() == case["csv"]


@pytest.fixture(scope="module")
def gtb_tools():
    gdl, spatcon = find_gdl(), find_spatcon()
    if gdl is None or spatcon is None:
        pytest.skip("GDL and GTB's spatcon are needed to run GTB's code live "
                    "(LANDSCHAFT_GDL_EXE, LANDSCHAFT_SPATCON_EXE)")
    return gdl, spatcon


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(4))
def test_heatmap_live_against_gtb(gtb_tools, seed):
    gdl, spatcon = gtb_tools
    rng = np.random.default_rng(700 + seed)
    maps, heatmaps = [], []
    for _ in range(2):
        p = rng.dirichlet([2, 2, 2]) * 0.95
        shape = (int(rng.integers(60, 200)), int(rng.integers(60, 200)))
        maps.append(rng.choice([0, 1, 2, 3], size=shape, p=[0.05, *p]).astype(np.uint8))
    window = int(rng.choice([5, 7, 13, 27]))
    refs = [run_gtb_lm_heatmap(gdl, spatcon, m, window) for m in maps]
    for m, ref in zip(maps, refs):
        im103 = np.asarray(ls.landscape_mosaic(m, window, output_classes=103))
        np.testing.assert_array_equal(im103, ref["im103"])
        h = ls.landscape_mosaic_heatmap(im103, is_lm103=True, window=str(window))
        np.testing.assert_array_equal(h.values.view(np.uint32), ref["hn"].view(np.uint32))
        np.testing.assert_array_equal(np.nan_to_num(h.triangle, nan=-1.0), ref["hmap"])
        assert h.csv_text() == ref["csv"]
        heatmaps.append(h)
    delta = run_gtb_lm_delta(gdl, refs[0]["hn"], refs[1]["hn"], str(window), str(window))
    d = ls.landscape_mosaic_change(*heatmaps)
    np.testing.assert_array_equal(d.values.view(np.uint32), delta["hn"].view(np.uint32))
    assert d.csv_text() == delta["csv"]
