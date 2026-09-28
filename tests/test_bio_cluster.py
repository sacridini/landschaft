# SPDX-License-Identifier: GPL-3.0-or-later
"""Cluster analyses against the original Biodiverse."""

from __future__ import annotations

import json
import math

import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, CLUSTER_CASES, DATA, REFERENCE

_BD: dict = {}


def _basedata(name: str):
    if name not in _BD:
        case = BASEDATA_CASES[name]
        kw = {k: v for k, v in case["landschaft"].items() if v is not None}
        _BD[name] = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    return _BD[name]


def _run(name: str, **kw):
    case = CLUSTER_CASES[name]
    a = case["args"]
    tree = ls.bio.read_tree(DATA / case["tree"]) if "tree" in case else None
    if case.get("region_grower"):
        return ls.bio.region_grower(
            _basedata(case["basedata"]), a.get("index", "RICHNESS_ALL"), a.get("spatial_conditions"),
            a.get("definition_query"), calculations=case.get("calcs"), tree=tree, prng_seed=a.get("prng_seed"), **kw)
    return ls.bio.cluster(
        _basedata(case["basedata"]), a.get("index", "SORENSON"), a.get("linkage_function", "link_average"),
        a.get("spatial_conditions"), a.get("definition_query"), calculations=case.get("calcs"), tree=tree,
        prng_seed=a.get("prng_seed"), tie_breaker=a.get("cluster_tie_breaker"), **kw)


def _same(a, b) -> bool:
    if b is None:
        return a is None or (isinstance(a, float) and math.isnan(a))
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return False
    return float(a) == float(b)


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "cluster.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(CLUSTER_CASES))
def test_cluster_matches_biodiverse(reference, name):
    res = _run(name)
    ref = reference[name]["nodes"]
    nodes = res.nodes
    assert sorted(nodes.index) == sorted(ref)
    diffs = []
    for nm, r in ref.items():
        row = nodes.loc[nm]
        if row["parent"] != r["parent"]:
            diffs.append((nm, "parent", r["parent"], row["parent"]))
        if list(row["children"]) != r["children"]:
            diffs.append((nm, "children", r["children"], list(row["children"])))
        if not _same(row["length"], r["length"]):
            diffs.append((nm, "length", r["length"], row["length"]))
        for k, v in r["values"].items():
            if k in nodes.columns and not _same(row[k], v):
                diffs.append((nm, k, v, row[k]))
        for k, v in r.get("lists", {}).get("SPATIAL_RESULTS", {}).items():
            got = res.table.at[nm, k] if k in res.table.columns else "missing"
            if got == "missing" or not _same(got, v):
                diffs.append((nm, k, v, got))
    assert not diffs, "\n".join(map(str, diffs[:20]))


@pytest.mark.parametrize("seed", [1, 2])
def test_cut_nests_subtrees(seed):
    bd = _basedata("site_100k")
    cl = ls.bio.cluster(bd, prng_seed=seed)
    terminals = {}  # node -> set of groups below it
    children = cl.nodes["children"].to_dict()

    def below(nd):
        if nd not in terminals:
            terminals[nd] = {nd} if not len(children[nd]) else set().union(*(below(c) for c in children[nd]))
        return terminals[nd]

    subtrees = {frozenset(below(nd)) for nd in cl.nodes.index}
    previous = None
    for k in (1, 2, 3, 5, 8, 20, bd.n_groups):
        cut = cl.cut(k)
        assert sorted(cut.index) == sorted(bd.groups)
        assert sorted(cut.unique()) == list(range(1, k + 1))
        for c in range(1, k + 1):
            assert frozenset(cut.index[cut == c]) in subtrees
        if previous is not None:  # a finer cut splits clusters of the coarser one
            assert (cut.groupby(previous).nunique() >= 1).all()
            assert (previous.groupby(cut).nunique() == 1).all()
        previous = cut
    with pytest.raises(ValueError):
        cl.cut(0)
