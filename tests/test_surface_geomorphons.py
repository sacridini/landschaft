# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface.geomorphons against GRASS GIS's r.geomorphon 8.4 (tools/make_reference_geomorphons.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "surface"
DATA = np.load(REF / "geomorphons.npz")
META = json.loads((REF / "geomorphons.json").read_text())
CASES = list(META["cases"])
OUTPUTS = ("forms", "ternary", "positive", "negative", "intensity", "exposition", "range", "variance",
           "elongation", "azimuth", "extend", "width")
CORRECTED = ("positive", "negative", "elongation", "azimuth", "extend", "width")


def _run(name, **kw):
    c = META["cases"][name]
    z = DATA[f"{name}/dem"]
    ny, nx = z.shape
    ew, ns = c["cell"]
    x0, y0 = c["origin"]
    return ls.surface.geomorphons(z, search=c["search"], skip=c["skip"], flat=c["flat"], dist=c["dist"],
                                  units="map" if c.get("meters") else "cells", comparison=c["comparison"],
                                  extended=c.get("extended", False), measures=True,
                                  extent=(x0, x0 + nx * ew, y0 - ny * ns, y0), **kw)


def _same(a, b):
    return np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("name", CASES)
def test_bit_for_bit_with_r_geomorphon(name):
    """With r.geomorphon's errors emulated, every output is GRASS's, bit for bit."""
    g = _run(name, _grass_errors=True)
    valid = ~np.isnan(g.forms.values)
    assert valid.sum() > 0.5 * valid.size
    for k in OUTPUTS:
        ref = DATA[f"{name}/{k}"].astype(np.float64)
        assert not np.isnan(ref[valid & ~np.isnan(getattr(g, k).values)]).any(), k
        assert _same(getattr(g, k).values[valid], ref[valid]), k


@pytest.mark.parametrize("name", CASES)
def test_the_correct_form_changes_only_what_it_corrects(name):
    g, e = _run(name), _run(name, _grass_errors=True)
    extended_skip = META["cases"][name].get("extended") and META["cases"][name]["skip"] > 0
    for k in OUTPUTS:
        if k in CORRECTED or (k == "forms" and extended_skip):
            continue
        assert _same(getattr(g, k).values, getattr(e, k).values), k
    # the shape differs only where r.geomorphon has none (its first direction without a point: NaN)
    differ = ~np.isclose(g.width.values, e.width.values, equal_nan=True)
    assert np.isnan(e.width.values[differ]).all() and not np.isnan(g.width.values[differ]).any()
    # the binary patterns: as many bits as directions over the threshold (at most 8)
    pos = g.positive.values[~np.isnan(g.positive.values)].astype(int)
    assert pos.max() <= 255 and pos.min() >= 0


def test_peaks_pits_and_their_binary_patterns():
    yy, xx = np.mgrid[0:41, 0:41].astype(float)
    cone = 100 - 0.1 * ((yy - 20) ** 2 + (xx - 20) ** 2)   # a dome: steeper away from the top
    g = ls.surface.geomorphons(cone, search=5)
    assert ls.surface.GEOMORPHON_FORMS[int(g.forms[20, 20])] == "peak"
    assert (g.positive[20, 20], g.negative[20, 20], g.ternary[20, 20]) == (0, 255, 0)
    p = ls.surface.geomorphons(-cone, search=5)
    assert ls.surface.GEOMORPHON_FORMS[int(p.forms[20, 20])] == "pit"
    assert (p.positive[20, 20], p.negative[20, 20], p.ternary[20, 20]) == (255, 0, 6560)
    flat = ls.surface.geomorphons(np.full((20, 20), 5.0), search=3)
    assert set(np.unique(flat.forms.values[~np.isnan(flat.forms.values)])) == {1}


def test_lines_of_sight_reach_the_last_row():
    """A cliff on the last row is seen from the cells above it (r.geomorphon misses it)."""
    z = np.zeros((30, 30))
    z[-1, :] = 500.0
    g = ls.surface.geomorphons(z, search=6)
    e = ls.surface.geomorphons(z, search=6, _grass_errors=True)  # the emulation reads the last row too
    assert ls.surface.GEOMORPHON_FORMS[int(g.forms[25, 15])] != "flat"
    assert _same(g.forms.values, e.forms.values)


def test_borders_nodata_and_threads():
    z = DATA["holes/dem"]
    a = ls.surface.geomorphons(z, search=6, skip=1, measures=True, n_threads=1)
    b = ls.surface.geomorphons(z, search=6, skip=1, measures=True, n_threads=3)
    for k in OUTPUTS:
        assert _same(getattr(a, k).values, getattr(b, k).values), k
    f = a.forms.values
    assert np.isnan(f[:2]).all() and np.isnan(f[:, -2:]).all()   # the outer skip + 1 cells
    assert np.isnan(f[np.isnan(z)]).all()


def test_whitebox_tools_mostly_agrees():
    """WhiteboxTools' Geomorphons is another implementation (other distances and edges): a loose check."""
    g = _run("default")
    w = DATA["default/wbt_forms"]
    m = ~np.isnan(g.forms.values) & ~np.isnan(w)
    assert np.mean(w[m] == g.forms.values[m]) > 0.95


def test_arguments_and_result(tmp_path):
    z = DATA["hills/dem"] if "hills/dem" in DATA.files else DATA["map_units/dem"]
    with pytest.raises(ValueError, match="skip"):
        ls.surface.geomorphons(z, search=3, skip=3)
    with pytest.raises(ValueError, match="flatness"):
        ls.surface.geomorphons(z, flat=0)
    with pytest.raises(ValueError, match="units"):
        ls.surface.geomorphons(z, units="km")
    with pytest.raises(ValueError, match="comparison"):
        ls.surface.geomorphons(z, comparison="angle")
    with pytest.raises(ValueError, match="projected"):
        ls.surface.geomorphons(ls.surface.example_data("orelevation"))
    g = ls.surface.geomorphons(z, search=4)
    assert g.intensity is None and g.share()["share"].sum() == pytest.approx(1.0)
    folder = g.save(tmp_path / "gm")
    assert (folder / "forms.tif").is_file() and (folder / "ternary.tif").is_file()
