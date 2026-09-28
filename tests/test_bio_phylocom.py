# SPDX-License-Identifier: GPL-3.0-or-later
"""PhyloCom indices (MPD, MNTD, NRI, NTI, net VPD) against the original Biodiverse."""

from __future__ import annotations

import json

import numpy as np
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, PHYLOCOM_CASES, REFERENCE, phylocom_calcs
from test_bio_spatial import compare

_BD: dict = {}


def _basedata(name: str):
    if name not in _BD:
        case = BASEDATA_CASES[name]
        kw = {k: v for k, v in case["landschaft"].items() if v is not None}
        _BD[name] = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    return _BD[name]


def _run(name: str, **kw):
    case = PHYLOCOM_CASES[name]
    bd = _basedata(case["basedata"])
    tree = ls.bio.read_tree(DATA / case["tree"])
    return ls.bio.spatial_analysis(bd, case["conditions"], phylocom_calcs(case), tree=tree,
                                   prng_seed=case.get("seed"), **kw)


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "phylocom.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(PHYLOCOM_CASES))
def test_phylocom_matches_biodiverse(reference, name):
    res = _run(name)
    ref = reference[name]
    assert sorted(res.table.index) == sorted(ref)
    diffs: list = []
    compare(res, ref, diffs)
    assert not diffs, "\n".join(map(str, diffs[:20]))


def test_mersenne_twister_shuffle():
    # Math::Random::MT::Auto->new(seed => 42)->shuffle(0 .. 30), twice
    got = ls._core.bio_mt_shuffle(42, 31, 2)
    assert got[0] == [18, 26, 27, 23, 6, 9, 17, 22, 4, 15, 8, 11, 7, 3, 5, 29, 12, 30, 25, 21, 19, 16, 20, 0, 14, 1,
                      10, 28, 2, 24, 13]
    assert got[1][:6] == [27, 24, 5, 15, 26, 20]


@pytest.mark.parametrize("name", ["sampled", "um_circle"])
def test_phylocom_threads_agree(name):
    a = _run(name, n_threads=1)
    b = _run(name, n_threads=4)
    np.testing.assert_array_equal(a.table.to_numpy(), b.table.to_numpy())


def test_sampled_nti_depends_on_seed():
    a = _run("sampled")
    case = PHYLOCOM_CASES["sampled"]
    tree = ls.bio.read_tree(DATA / case["tree"])
    b = ls.bio.spatial_analysis(_basedata(case["basedata"]), case["conditions"], ["nri_nti_expected_values"],
                                tree=tree, prng_seed=case["seed"] + 1)
    col = "PHYLO_NTI_SAMPLE_MEAN"
    assert not np.array_equal(a.table[col].to_numpy(), b.table[col].to_numpy(), equal_nan=True)
    # the pairwise expectation is exact
    np.testing.assert_array_equal(a.table["PHYLO_NRI_SAMPLE_MEAN"], b.table["PHYLO_NRI_SAMPLE_MEAN"])
