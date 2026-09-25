# SPDX-License-Identifier: GPL-3.0-or-later
"""SPA2/3/5/6: unit tests and agreement with GuidosToolbox's IDL code (run in GDL)."""

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
from conftest import mspa_input
from gtb_reference import find_gdl, run_gtb_spa

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def scene():
    a = mspa_input((30, 40), foreground=[np.s_[3:18, 3:18], np.s_[25, 3:30], np.s_[24:27, 34:37]])
    a[9:12, 9:12] = 1  # a hole in the big object
    return a


def test_spa2():
    r = ls.spa(scene(), 2)
    m = np.asarray(r.map)
    assert set(np.unique(m)) == {0, 1, 17}
    assert (m[25, 5:28] == 1).all()  # a 1-pixel line is a small linear feature
    assert m[4, 4] == 17 and (m[24:27, 34:37] == 17).all()
    assert r.counts == {"foreground": 3, "contiguous": 2, "slf": 1}
    assert r.stats["area"].sum() == r.areas["foreground"]


def test_spa3_and_core_opening():
    r = ls.spa(scene(), 3)
    m = np.asarray(r.map)
    assert (m[10, 10] == 100) and (m[3, 3] == 1) and m[5, 5] == 17
    assert r.counts["core_opening"] == 1
    assert r.areas["core_opening"] == 9 and r.porosity > 0


def test_spa5_and_spa6():
    r5, r6 = ls.spa(scene(), 5), ls.spa(scene(), 6)
    m5, m6 = np.asarray(r5.map), np.asarray(r6.map)
    assert m5[3, 3] == 3  # outline of the core: edge
    assert m5[8, 10] == 5  # boundary of the core-opening: perforation
    assert (m5[25, 5:28] == 1).all() and (m6[25, 5:28] == 9).all()  # the line has no core: islet in SPA6
    assert m6[25, 35] == 17  # the 3x3 block has a core pixel
    assert list(r6.stats["class"]) == ["Core", "Islet", "Edge", "Perforation", "Margin"]
    np.testing.assert_allclose(r6.stats["pct_fg"].sum(), 100.0)


def test_missing_does_not_create_edges():
    a = np.full((20, 20), 2, np.uint8)
    a[:, 10:] = 0  # the right half is missing
    m = np.asarray(ls.spa(a, 5).map)
    assert (m[:, 10:] == 129).all()
    assert m[10, 9] == 17  # foreground next to missing data stays core


def test_invalid_and_xarray(tmp_path):
    with pytest.raises(ValueError, match="variant"):
        ls.spa(scene(), 4)
    with pytest.raises(ls.InvalidInputError):
        ls.spa(np.full((5, 5), 3, np.uint8))
    da = ls.load_raster(ls.example_data("input"))
    r = ls.spa(da, 6)
    assert isinstance(r.map, xr.DataArray) and r.map.attrs["landschaft_product"] == "SPA6"
    ls.write_raster(r.map, tmp_path / "spa6.tif")
    assert "SPA6: 8-connected Foreground" in r.summary()


# --- agreement with GTB ------------------------------------------------------------


def _reference() -> list[dict]:
    return json.loads(REFERENCE.read_text()).get("spa", []) if REFERENCE.exists() else []


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import spa_inputs

    return spa_inputs()


def assert_matches(r: ls.SPAResult, stats: dict[str, str]) -> None:
    keys = {"nr_FG": "foreground", "nr_Cores": "core", "nr_margin": "margin", "nr_holes": "core_opening",
            "nr_edge": "edge", "nr_perf": "perforation", "nr_islet": "islet", "nr_contiguous": "contiguous"}
    if r.variant == 2:
        keys = {"nr_FG": "foreground", "nr_Cores": "contiguous", "nr_margin": "slf"}
    for k, name in keys.items():
        if k in stats:
            assert r.counts[name] == int(stats[k]), k
    assert r.porosity == pytest.approx(float(stats["porosity"]), abs=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference(), ids=lambda c: f"{c['input']}-spa{c['variant']}")
def test_spa_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"]
    r = ls.spa(data, case["variant"])
    assert digest(r.map) == case["sha256"], "SPA map differs from GTB"
    assert_matches(r, case["stats"])


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(4))
def test_spa_live_against_gtb(seed):
    gdl = find_gdl()
    if gdl is None:
        pytest.skip("GDL not found")
    rng = np.random.default_rng(1200 + seed)
    kind = list(scenes.SCENES)[seed % len(scenes.SCENES)]
    data = scenes.make(kind, int(rng.integers(1, 10_000)))
    for variant in (2, 3, 5, 6):
        ref = run_gtb_spa(gdl, data, f"spa{variant}")
        r = ls.spa(data, variant)
        np.testing.assert_array_equal(np.asarray(r.map), ref["map"], err_msg=f"spa{variant}")
        assert_matches(r, ref["stats"])
