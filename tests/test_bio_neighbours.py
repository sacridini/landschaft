# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial conditions (neighbourhoods) against the original Biodiverse."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, NEIGHBOUR_CASES, NEIGHBOUR_DEFQ, REFERENCE


@pytest.fixture(scope="module")
def site_bd():
    case = BASEDATA_CASES["site_100k"]
    kw = {k: v for k, v in case["landschaft"].items() if v is not None}
    return ls.bio.BaseData.from_table(DATA / case["file"], **kw)


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "neighbours.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(NEIGHBOUR_CASES))
def test_neighbours_match_biodiverse(site_bd, reference, name):
    shapes = str(DATA / "site_polygons.shp").replace("\\", "/")
    conds = [c.replace("SHAPES", shapes) for c in NEIGHBOUR_CASES[name]]
    nb = ls.bio.neighbours(site_bd, conds, definition_query=NEIGHBOUR_DEFQ.get(name))
    ref = reference[name]
    assert [site_bd.groups[g] for g in nb.groups] == sorted(ref)
    for g, sets in ref.items():
        for i in range(len(conds)):
            assert nb.of(g, i, stored=True) == sets[str(i + 1)], (g, i)


def test_neighbours_threads_agree(site_bd):
    one = ls.bio.neighbours(site_bd, "sp_circle (radius => 400000)", n_threads=1)
    many = ls.bio.neighbours(site_bd, "sp_circle (radius => 400000)", n_threads=4)
    for (a, b), (c, d) in zip(one.sets, many.sets):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_grid_search_matches_brute_force():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"sp": ["a"] * 3000, "x": rng.uniform(0, 100, 3000), "y": rng.uniform(0, 100, 3000)})
    bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    grid = ls.bio.neighbours(bd, "$D <= 4.5")
    # "|| 0" hides the bound: every pair is evaluated
    brute = ls.bio.neighbours(bd, "$D <= 4.5 || 0")
    np.testing.assert_array_equal(grid.sets[0][0], brute.sets[0][0])
    np.testing.assert_array_equal(grid.sets[0][1], brute.sets[0][1])


def test_condition_errors():
    df = pd.DataFrame({"sp": ["a", "b"], "x": [0.5, 1.5], "y": [0.5, 0.5]})
    bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    with pytest.raises(ValueError, match="cannot parse"):
        ls.bio.SpatialCondition("$D <= ")
    with pytest.raises(ValueError, match="unknown variable"):
        ls.bio.neighbours(bd, "$foo < 3")
    with pytest.raises(NotImplementedError):
        ls.bio.neighbours(bd, "sp_in_label_range (label => 'a')")
    with pytest.raises(ValueError, match="Nothing passed"):
        ls.bio.neighbours(bd, "sp_self_only()", definition_query="$x > 10")
