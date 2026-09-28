# SPDX-License-Identifier: GPL-3.0-or-later
"""Phylogenetic indices against the original Biodiverse."""

from __future__ import annotations

import json

import numpy as np
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, PHYLO_CASES, REFERENCE, phylo_calcs
from test_bio_spatial import compare

_BD: dict = {}


def _basedata(name: str):
    if name not in _BD:
        case = BASEDATA_CASES[name]
        kw = {k: v for k, v in case["landschaft"].items() if v is not None}
        _BD[name] = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    return _BD[name]


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "phylo.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(PHYLO_CASES))
def test_phylo_matches_biodiverse(reference, name):
    case = PHYLO_CASES[name]
    bd = _basedata(case["basedata"])
    tree = ls.bio.read_tree(DATA / case["tree"])
    res = ls.bio.spatial_analysis(bd, case["conditions"], phylo_calcs(case), tree=tree)
    ref = reference[name]
    assert sorted(res.table.index) == sorted(ref)
    diffs: list = []
    compare(res, ref, diffs)
    assert not diffs, "\n".join(map(str, diffs[:20]))


def test_phylo_needs_tree():
    bd = _basedata("site_100k")
    with pytest.raises(ValueError, match="need a tree"):
        ls.bio.spatial_analysis(bd, "sp_self_only()", ["pd"])


def test_phylo_threads_agree():
    bd = _basedata("site_100k")
    tree = ls.bio.read_tree(DATA / "Example_tree.nex")
    calcs = ["pd", "pe", "pe_lists", "pd_clade_contributions", "phylo_aed_t", "phylo_abundance"]
    a = ls.bio.spatial_analysis(bd, "sp_circle (radius => 300000)", calcs, tree=tree, n_threads=1)
    b = ls.bio.spatial_analysis(bd, "sp_circle (radius => 300000)", calcs, tree=tree, n_threads=4)
    np.testing.assert_array_equal(a.table.to_numpy(), b.table.to_numpy())
    for k in a.lists:
        assert a.lists[k].equals(b.lists[k])
