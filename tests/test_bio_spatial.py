# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial analyses (index values) against the original Biodiverse."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, REFERENCE, SPATIAL_CASES, spatial_calcs

_BD_CACHE: dict = {}


def _basedata(name: str):
    if name not in _BD_CACHE:
        case = BASEDATA_CASES[name]
        kw = {k: v for k, v in case["landschaft"].items() if v is not None}
        _BD_CACHE[name] = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    return _BD_CACHE[name]


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "spatial.json").read_text(encoding="utf-8"))


def _same(a, b) -> bool:
    """Equal numbers (bit for bit), NaN matching None/undef."""
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return float(a) == float(b)


def compare(res, ref, diffs: list, limit: int = 20) -> None:
    for g, lists in ref.items():
        sr = lists.get("SPATIAL_RESULTS", {})
        for key, v in sr.items():
            got = res.table.at[g, key] if key in res.table.columns else "missing"
            if got == "missing" or not _same(got, v):
                diffs.append((g, key, v, got))
        for lname, lv in lists.items():
            if lname == "SPATIAL_RESULTS":
                continue
            if lname not in res.lists:
                diffs.append((g, lname, "list", "missing"))
                continue
            df = res.lists[lname]
            sub = df[df["group"] == g]
            if isinstance(lv, list):
                got = list(sub["key"])
                if got != lv:
                    diffs.append((g, lname, lv[:5], got[:5]))
            else:
                got = dict(zip(sub["key"], sub["value"]))
                if set(got) != set(lv):
                    diffs.append((g, lname, sorted(lv)[:5], sorted(got)[:5]))
                else:
                    for k, v in lv.items():
                        if not _same(got[k], v):
                            diffs.append((g, f"{lname}[{k}]", v, got[k]))
        if len(diffs) > limit:
            return


@pytest.mark.reference
@pytest.mark.parametrize("name", list(SPATIAL_CASES))
def test_spatial_matches_biodiverse(reference, name):
    case = SPATIAL_CASES[name]
    bd = _basedata(case["basedata"])
    res = ls.bio.spatial_analysis(bd, case["conditions"], spatial_calcs(case),
                                  definition_query=case.get("definition_query"))
    ref = reference[name]
    assert sorted(res.table.index) == sorted(ref)
    diffs: list = []
    compare(res, ref, diffs)
    assert not diffs, "\n".join(map(str, diffs[:20]))


def test_spatial_threads_agree():
    bd = _basedata("site_counts")
    calcs = spatial_calcs({"calcs": "one"})
    a = ls.bio.spatial_analysis(bd, "sp_circle (radius => 300000)", calcs, n_threads=1)
    b = ls.bio.spatial_analysis(bd, "sp_circle (radius => 300000)", calcs, n_threads=4)
    np.testing.assert_array_equal(a.table.to_numpy(), b.table.to_numpy())
    for k in a.lists:
        assert a.lists[k].equals(b.lists[k])


def test_unknown_calculation():
    bd = _basedata("site_100k")
    with pytest.raises(ValueError, match="unknown calculation"):
        ls.bio.spatial_analysis(bd, "sp_self_only()", ["no_such_index"])
    with pytest.raises(ValueError, match="neighbour sets"):
        ls.bio.spatial_analysis(bd, "sp_self_only()", ["sorenson"])
