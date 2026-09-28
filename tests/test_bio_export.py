# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial results written in Biodiverse's delimited text format, compared with Biodiverse's export."""

from __future__ import annotations

import json

import pytest

import landschaft as ls
from bio_cases import DATA, EXPORT_CASES, PHYLO_CASES, PROPERTY_CASES, REFERENCE, SPATIAL_CASES, phylo_calcs, spatial_calcs
from landschaft.bio._export import natsort
from test_bio_props import _run as _run_props
from test_bio_spatial import _basedata


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "export.json").read_text(encoding="utf-8"))


_RESULTS: dict = {}


def _result(case: dict):
    key = (case["from"], case["case"])
    if key not in _RESULTS:
        if case["from"] == "spatial":
            c = SPATIAL_CASES[case["case"]]
            res = ls.bio.spatial_analysis(_basedata(c["basedata"]), c["conditions"], spatial_calcs(c),
                                          definition_query=c.get("definition_query"))
        elif case["from"] == "phylo":
            c = PHYLO_CASES[case["case"]]
            res = ls.bio.spatial_analysis(_basedata(c["basedata"]), c["conditions"], phylo_calcs(c),
                                          tree=ls.bio.read_tree(DATA / c["tree"]))
        else:
            assert case["case"] in PROPERTY_CASES
            res = _run_props(case["case"])
        _RESULTS[key] = res
    return _RESULTS[key]


@pytest.mark.reference
@pytest.mark.parametrize("name", list(EXPORT_CASES))
def test_export_matches_biodiverse(reference, name, tmp_path):
    case = EXPORT_CASES[name]
    res = _result(case)
    args = case.get("args", {})
    for lst in case["lists"]:
        path = tmp_path / f"{lst}.csv"
        res.to_biodiverse_csv(path, lst, sep=args.get("sep_char", ","), quote=args.get("quote_char", '"'),
                              no_data_value=args.get("no_data_value"))
        got = path.read_text(encoding="utf-8")
        want = reference[name][lst]
        if got != want:
            g, w = got.splitlines(), want.splitlines()
            first = next((i for i, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
            pytest.fail(f"{lst} line {first}:\n got  {g[first] if first < len(g) else None}\n"
                        f" want {w[first] if first < len(w) else None}")


def test_natural_sort():
    keys = ["Q10", "Q2", "Q002", "a10b", "a9b", "10", "9.5", "1e5", "B", "_x", "x100000000000", "x99999999999"]
    # as Sort::Key::Natural::natsort sorts them (keys given in string order)
    assert natsort(keys) == ["1e5", "9.5", "10", "B", "Q002", "Q2", "Q10", "a9b", "a10b", "_x", "x99999999999",
                             "x100000000000"]
