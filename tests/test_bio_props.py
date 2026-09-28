# SPDX-License-Identifier: GPL-3.0-or-later
"""Element property, numeric label and matrix indices against the original Biodiverse."""

from __future__ import annotations

import json

import numpy as np
import pytest

import landschaft as ls
from bio_cases import (
    BASEDATA_CASES,
    DATA,
    GROUP_PROPS,
    LABEL_PROPS,
    MATRIX_CASES,
    PROPERTY_CASES,
    REFERENCE,
    property_case_uses,
)
from test_bio_spatial import compare


def _basedata(case: dict):
    bdcase = BASEDATA_CASES[case["basedata"]]
    kw = {k: v for k, v in bdcase["landschaft"].items() if v is not None}
    bd = ls.bio.BaseData.from_table(DATA / bdcase["file"], **kw)
    lb, gp = property_case_uses(case)
    if lb:
        bd.set_label_properties(ls.bio.read_properties(DATA / LABEL_PROPS["file"], LABEL_PROPS["element_columns"],
                                                       LABEL_PROPS["properties"]))
    if gp:
        bd.set_group_properties(ls.bio.read_properties(DATA / GROUP_PROPS["file"], GROUP_PROPS["element_columns"],
                                                       GROUP_PROPS["properties"]))
    return bd


def _matrix(name: str):
    mc = MATRIX_CASES[name]
    return ls.bio.read_matrix(DATA / mc["file"], **mc["landschaft"])


def _run(name: str, **kw):
    case = PROPERTY_CASES[name]
    matrix = _matrix(case["matrix"]) if "matrix" in case else None
    return ls.bio.spatial_analysis(_basedata(case), case["conditions"], case["calcs"], matrix=matrix, **kw)


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "props.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(PROPERTY_CASES))
def test_props_match_biodiverse(reference, name):
    res = _run(name)
    ref = reference[name]
    assert sorted(res.table.index) == sorted(ref)
    diffs: list = []
    compare(res, ref, diffs)
    assert not diffs, "\n".join(map(str, diffs[:20]))


@pytest.mark.parametrize("name", ["props_circle", "numeric_two_sets", "matrix_two_sets"])
def test_props_threads_agree(name):
    a = _run(name, n_threads=1)
    b = _run(name, n_threads=4)
    np.testing.assert_array_equal(a.table.to_numpy(), b.table.to_numpy())
    for k in a.lists:
        assert a.lists[k].equals(b.lists[k])


def test_matrix_needed():
    case = PROPERTY_CASES["matrix_circle"]
    with pytest.raises(ValueError, match="need a matrix"):
        ls.bio.spatial_analysis(_basedata(case), case["conditions"], ["mx_rao_qe"])


def test_properties_as_dataframe_values():
    import pandas as pd

    case = PROPERTY_CASES["props_circle"]
    bd = _basedata(case)
    raw = ls.bio.read_properties(DATA / LABEL_PROPS["file"], LABEL_PROPS["element_columns"], LABEL_PROPS["properties"])
    # numbers given as floats print as Perl does, so "20.700" becomes "20.7"
    num = raw.apply(pd.to_numeric)
    bd2 = _basedata(case)
    bd2.set_label_properties(num)
    assert bd2.label_properties.loc["Genus:sp2", "PROP_A"] == "20.7"
    assert bd.label_properties.loc["Genus:sp2", "PROP_A"] == "20.700"
