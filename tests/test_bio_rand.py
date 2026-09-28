# SPDX-License-Identifier: GPL-3.0-or-later
"""Randomisations, significance, z-scores and CANAPE against the original Biodiverse."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import landschaft as ls
from bio_cases import BASEDATA_CASES, DATA, RAND_CASES, REFERENCE

_BD: dict = {}


def _basedata(name: str):
    if name not in _BD:
        case = BASEDATA_CASES[name]
        kw = {k: v for k, v in case["landschaft"].items() if v is not None}
        _BD[name] = ls.bio.BaseData.from_table(DATA / case["file"], **kw)
    return _BD[name]


def _run(name: str, **kw):
    case = RAND_CASES[name]
    tree = ls.bio.read_tree(DATA / case["tree"]) if "tree" in case else None
    res = ls.bio.spatial_analysis(_basedata(case["basedata"]), case["conditions"], case["calcs"], tree=tree)
    return ls.bio.randomise(res, case["function"], case["iterations"], seed=case["seed"], name="rnd",
                            **case.get("args", {}), **kw)


def _biodiverse_lists(rnd) -> dict:
    """Our results as Biodiverse's per-group lists (``rnd>>SPATIAL_RESULTS``...)."""
    out: dict = {}
    comp = rnd.comparison()
    pr = rnd.p_rank()
    z = rnd.z_scores()
    for g in comp.index:
        d = out.setdefault(g, {})
        d["rnd>>SPATIAL_RESULTS"] = comp.loc[g].to_dict()
        d["rnd>>p_rank>>SPATIAL_RESULTS"] = pr.loc[g].to_dict()
        d["rnd>>z_scores>>SPATIAL_RESULTS"] = z.loc[g].to_dict()
    for central, diff, lname in ((False, False, "CANAPE"), (False, True, "CANAPE_DIFF"),
                                 (True, False, "CANAPE_CENTRAL"), (True, True, "CANAPE_DIFF_CENTRAL")):
        try:
            cn = rnd.canape(central=central, diff=diff)
        except ValueError:
            continue
        for g in cn.index:
            out[g][f"rnd>>{lname}>>"] = cn.loc[g].to_dict()
    for lname in rnd._lists:
        for kind, frame in (("", rnd.comparison(lname)), ("p_rank>>", rnd.p_rank(lname)),
                            ("z_scores>>", rnd.z_scores(lname))):
            for g, sub in frame.groupby("group"):
                if kind:
                    vals = dict(zip(sub["key"], sub["value"]))
                else:
                    vals = {}
                    for col in ("C", "P", "Q", "SUMX", "SUMXX", "T"):
                        vals.update({f"{col}_{k}": v for k, v in zip(sub["key"], sub[col])})
                out.setdefault(g, {})[f"rnd>>{kind}{lname}"] = vals
    return out


def _same(a, b) -> bool:
    if b is None or (isinstance(b, str) and b == "NaN"):
        return a is None or (isinstance(a, float) and math.isnan(a))
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return False
    return float(a) == float(b)


@pytest.fixture(scope="module")
def reference():
    return json.loads((REFERENCE / "rand.json").read_text(encoding="utf-8"))


@pytest.mark.reference
@pytest.mark.parametrize("name", list(RAND_CASES))
def test_randomisation_matches_biodiverse(reference, name):
    rnd = _run(name)
    ours = _biodiverse_lists(rnd)
    ref = reference[name]
    diffs = []
    for g, lists in ref.items():
        for lname, vals in lists.items():
            if not lname.startswith("rnd>>"):
                continue
            got = ours.get(g, {}).get(lname)
            if got is None:
                diffs.append((g, lname, "missing"))
                continue
            for k, v in vals.items():
                if k.endswith("RECYCLED_SET"):
                    continue  # Biodiverse's internal flag (not a result)
                if not _same(got.get(k), v):
                    diffs.append((g, lname, k, v, got.get(k)))
            # keys we have that Biodiverse does not must be undefined
            for k, v in got.items():
                if k not in vals and not _same(v, None):
                    diffs.append((g, lname, k, "extra", v))
    assert not diffs, "\n".join(map(str, diffs[:20]))


def test_randomisation_continues():
    a = _run("csr_circle")
    case = RAND_CASES["csr_circle"]
    tree = ls.bio.read_tree(DATA / case["tree"])
    res = ls.bio.spatial_analysis(_basedata(case["basedata"]), case["conditions"], case["calcs"], tree=tree)
    b = ls.bio.randomise(res, case["function"], 4, seed=case["seed"], name="rnd")
    b.run(case["iterations"] - 4)
    assert b.iterations == a.iterations
    np.testing.assert_array_equal(a.comparison().to_numpy(), b.comparison().to_numpy())
