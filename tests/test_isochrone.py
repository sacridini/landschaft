# SPDX-License-Identifier: GPL-3.0-or-later
"""Restoration Planner: isochrone maps and Distance -> Resistance.

The isochrone references run GTB's own code (tests/gtb_idl/gtb_isochrone.pro,
the 'cost_map' event of guidostoolbox.pro) in GDL together with GTB's ``ggeo``
program; see tools/make_reference_idl.py. ``live_reference`` tests need GDL
and ggeo (``LANDSCHAFT_GGEO_EXE`` or a GTB desktop installation).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from gtb_reference import find_gdl, find_ggeo, run_gtb_isochrone
from landschaft import _core

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


# --- geodesic time (ggeo) ------------------------------------------------------------


def test_geodesic_time_steps():
    mask = np.full((7, 9), 10, np.uint8)
    ref = np.zeros((7, 9), np.uint8)
    ref[3, 4] = 1
    t = _core.compute_geodesic_time(mask, ref)
    assert t[3, 4] == 0
    assert t[3, 5] == 20 and t[2, 5] == 20  # diagonal steps cost the same (r[p] + r[q])
    assert t[3, 6] == 40 and t[1, 6] == 40
    assert (t[0] == 0).all() and (t[:, 0] == 0).all()  # the 1-pixel frame is not processed


def test_geodesic_time_barriers_and_zero_resistance():
    mask = np.full((7, 9), 10, np.uint8)
    ref = np.zeros((7, 9), np.uint8)
    ref[3, 1] = 1
    ref[1:6, 4] = 3  # a closed wall inside the frame
    t = _core.compute_geodesic_time(mask, ref)
    assert (t[1:6, 5:8] == np.iinfo(np.uint32).max).all()  # unreachable
    mask[3, 2:4] = 0  # resistance 0 costs nothing (as in ggeo)
    t0 = _core.compute_geodesic_time(mask, ref)
    assert t0[3, 3] == 10


# --- isochrone maps ------------------------------------------------------------------


def simple_case():
    r = np.full((30, 40), 50, np.uint8)
    r[:, 18:22] = 2  # a cheap foreground corridor
    m = np.zeros(r.shape, np.uint8)
    m[3:5, 3:5] = 1
    m[25, 35] = 2
    return r, m


def test_isochrone_a():
    r, m = simple_case()
    res = ls.isochrone_map(r, m)
    out = np.asarray(res.map)
    assert (out[m == 1] == 101).all()
    assert out.max() <= 101 and res.lcp is None
    cost = np.asarray(res.cost)
    assert cost[4, 4] == 0 and cost[4, 5] == 50  # (50 + 50) / 2
    assert out[29, 39] == 100 and cost[29, 39] == res.max_cost  # farthest pixel
    # cost grows away from A
    assert cost[10, 30] > cost[10, 10]


def test_isochrone_ab_path():
    r = np.full((30, 40), 10, np.uint8)
    r[12:18, :] = 100  # an expensive band ...
    r[12:18, 14:17] = 3  # ... with a cheap gap
    m = np.zeros(r.shape, np.uint8)
    m[3:5, 3:5] = 1
    m[26, 5] = 2
    res = ls.isochrone_map_ab(r, m)
    out, lcp = np.asarray(res.map), np.asarray(res.lcp)
    assert lcp.dtype == bool and lcp[m == 1].any() and lcp[m == 2].any()
    assert (out[lcp & (m == 0)] == 103).all() and (out[m > 0] == 101).all()
    assert lcp[12:18, 14:17].any() and not lcp[12:18, :14].any() and not lcp[12:18, 17:].any()
    # a 1-pixel wide path: no 2x2 block of path pixels outside the markers
    p = lcp & (m == 0)
    assert not (p[:-1, :-1] & p[1:, :-1] & p[:-1, 1:] & p[1:, 1:]).any()


def test_isochrone_missing_and_unreachable():
    r, m = simple_case()
    m[10, :] = 3
    m[10, 0] = 0  # a gap on the left
    m[20:23, 0:3] = 3
    m[21, 1] = 0  # an enclosed pixel: unreachable
    res = ls.isochrone_map(r, m)
    out = np.asarray(res.map)
    assert (out[10, 1:] == 102).all() and out[21, 1] == 104
    ab = ls.isochrone_map_ab(r, m)
    assert np.asarray(ab.map)[21, 1] == 104


def test_isochrone_checks():
    r, m = simple_case()
    with pytest.raises(ValueError, match="object A"):
        ls.isochrone_map(r, np.where(m == 1, 0, m).astype(np.uint8))
    two = m.copy()
    two[10, 10] = 1
    with pytest.raises(ValueError, match="only one object"):
        ls.isochrone_map(r, two)
    touch = m.copy()
    touch[5, 5] = 2
    touch[25, 35] = 0
    with pytest.raises(ValueError, match="touch"):
        ls.isochrone_map_ab(r, touch)
    walled = m.copy()
    walled[15, :] = 3
    with pytest.raises(ValueError, match="can not be connected"):
        ls.isochrone_map_ab(r, walled)
    with pytest.raises(ValueError, match="differ in size"):
        ls.isochrone_map(r[:-1], m)


def test_isochrone_xarray(tmp_path):
    r, m = simple_case()
    da = xr.DataArray(r, dims=("y", "x"), coords={"y": np.arange(30), "x": np.arange(40)})
    res = ls.isochrone_map_ab(da, m)
    assert isinstance(res.map, xr.DataArray) and res.map.attrs["landschaft_product"] == "ISOCHRONE_AB"
    ls.write_raster(res.map, tmp_path / "iso.tif")


# --- Distance -> Resistance --------------------------------------------------------


def test_distance_to_resistance():
    a = np.ones((20, 30), np.uint8)
    a[5:10, 5:10] = 2
    a[0, 29] = 0
    res = ls.distance_to_resistance(a)
    assert res.dtype == np.uint8
    assert (res[5:10, 5:10] == 2).all() and res[0, 29] == 0
    assert res[7, 10] == 3 and res[7, 11] == 4  # round(distance + 2)
    big = np.ones((10, 300), np.uint8)
    big[:, 0] = 2
    assert ls.distance_to_resistance(big)[:, 150:].max() == 100  # capped at 100
    with pytest.raises(ValueError, match="no foreground"):
        ls.distance_to_resistance(np.ones((5, 5), np.uint8))


# --- agreement with GTB ------------------------------------------------------------


def _reference(section: str) -> list[dict]:
    if not REFERENCE.exists():
        return []
    return json.loads(REFERENCE.read_text()).get(section, [])


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import isochrone_inputs

    return isochrone_inputs()


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference("isochrone"), ids=lambda c: f"{c['input']}-{c['mode']}")
def test_isochrone_matches_stored_gtb_outputs(reference_inputs, case):
    resist, marker = reference_inputs[case["input"]]
    assert digest(resist) == case["resist_sha256"] and digest(marker) == case["marker_sha256"]
    fn = ls.isochrone_map if case["mode"] == "cost_map_aa" else ls.isochrone_map_ab
    assert digest(fn(resist, marker).map) == case["sha256"], "isochrone map differs from GTB"


@pytest.fixture(scope="module")
def gtb_tools():
    gdl, ggeo = find_gdl(), find_ggeo()
    if gdl is None or ggeo is None:
        pytest.skip("GDL and GTB's ggeo are needed to run GTB's code live (LANDSCHAFT_GDL_EXE, LANDSCHAFT_GGEO_EXE)")
    return gdl, ggeo


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(6))
def test_isochrone_live_against_gtb(gtb_tools, seed):
    gdl, ggeo = gtb_tools
    rng = np.random.default_rng(900 + seed)
    ny, nx = (int(v) for v in rng.integers(25, 80, 2))
    r = rng.integers(3, 101, (ny, nx)).astype(np.uint8)
    if seed % 3 == 0:
        r[:] = int(rng.integers(3, 101))  # uniform: many ties
    r[rng.random(r.shape) < 0.2] = 2
    m = np.zeros_like(r)
    m[1:3, 1:3] = 1
    m[ny - 3, nx - 2] = 2
    if seed % 2:
        m[ny // 2, 3:nx] = 3
    for mode, fn in (("cost_map_aa", ls.isochrone_map), ("cost_map_ab", ls.isochrone_map_ab)):
        np.testing.assert_array_equal(np.asarray(fn(r, m).map), run_gtb_isochrone(gdl, ggeo, r, m, mode),
                                      err_msg=mode)
