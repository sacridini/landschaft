# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat graphs against Makurhini (R) and Conefor Sensinode 2.2.

The references come from ``tools/make_reference_conn.py``: Makurhini's
``MK_dPCIIC`` (its rounded output and the unrounded values of its internal
steps) and Conefor's own index code (``tools/conefor``), which computes in
single precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "conn"


def _scenes() -> dict:
    data = np.load(REF / "scenes.npz")
    out = {}
    for name in sorted({k.split("__")[0] for k in data.files}):
        ids, attr, upper = data[f"{name}__ids"], data[f"{name}__attr"], data[f"{name}__upper"]
        n = len(ids)
        D = np.zeros((n, n))
        i, j = np.triu_indices(n, 1)
        D[i, j] = upper
        D[j, i] = upper
        out[name] = (ids, attr, D)
    return out


SCENES = _scenes()
MAKURHINI = pd.read_csv(REF / "makurhini.csv.gz", float_precision="round_trip", keep_default_na=False,
                        dtype={"probability": str, "pij_min": str, "la": str, "id": str})
CONEFOR = pd.read_csv(REF / "conefor.csv.gz", keep_default_na=False, dtype={"id": str})
MK_KEYS = ["scene", "metric", "distance", "probability", "pij_min", "la"]
MK_CASES = sorted(MAKURHINI.groupby(MK_KEYS).groups)
CF_KEYS = ["scene", "distance", "probability", "la"]
CF_CASES = sorted(CONEFOR.groupby(CF_KEYS).groups)


def _makurhini_run(case):
    scene, metric, d, p, pij, la = case
    ids, attr, D = SCENES[scene]
    g = ls.HabitatGraph(attr, D, ids=ids, landscape_area=float(la) if la else None)
    # "" is NULL in R: no probability cut.
    kw = {"distance": float(d), "probability": float(p) if p else None, "pij_min": float(pij) if pij else 0.0}
    ov = g.overall([metric], **kw).iloc[0]
    imp = g.node_importance([metric], **kw)
    imp.index = imp["id"].astype(str)
    ref = MAKURHINI.groupby(MK_KEYS).get_group(case)
    return ov, imp, ref


def _r_fractions(ov, metric):
    """MK_dPCIIC's overall fractions: rounded to 5 digits, the rest to 100 shared out."""
    names = [f"{metric}intra(%)", f"{metric}direct(%)", f"{metric}step(%)"]
    values = np.round([ov[n] for n in names], 5)
    total = values.sum()
    if total < 100:
        values = values + (100 - total) / 3
    return dict(zip(names, values))


@pytest.mark.parametrize("case", MK_CASES, ids=lambda c: "-".join(str(x) for x in c if x != ""))
def test_makurhini_unrounded(case):
    ov, imp, ref = _makurhini_run(case)
    for name, sub in ref[ref["level"] == "node_raw"].groupby("name"):
        np.testing.assert_allclose(imp.loc[sub["id"], name], sub["value"], rtol=1e-11, atol=1e-12, err_msg=name)
    for _, row in ref[ref["level"] == "overall_raw"].iterrows():
        assert ov[row["name"]] == pytest.approx(row["value"], rel=1e-13, abs=1e-12), row["name"]


@pytest.mark.parametrize("case", MK_CASES, ids=lambda c: "-".join(str(x) for x in c if x != ""))
def test_makurhini_output(case):
    ov, imp, ref = _makurhini_run(case)
    metric = case[1]
    for name, sub in ref[ref["level"] == "node"].groupby("name"):
        ours = imp.loc[sub["id"], name].to_numpy()
        if name.endswith("connector"):  # rounded to 20 digits: unrounded
            np.testing.assert_allclose(ours, sub["value"], rtol=1e-11, atol=1e-12, err_msg=name)
        else:  # rounded to 7 digits
            np.testing.assert_allclose(np.round(ours, 7), sub["value"], rtol=0, atol=1e-12, err_msg=name)
    fractions = _r_fractions(ov, metric)
    for _, row in ref[ref["level"] == "overall"].iterrows():
        ours = fractions.get(row["name"], ov[row["name"]])
        assert ours == pytest.approx(row["value"], rel=1e-13, abs=1e-9), row["name"]


@pytest.mark.parametrize("case", CF_CASES, ids=lambda c: "-".join(str(x) for x in c))
def test_conefor(case):
    scene, d, p, la = case
    ids, attr, D = SCENES[scene]
    g = ls.HabitatGraph(attr, D, ids=ids, landscape_area=float(la) if la else None, convention="conefor")
    ov = g.overall(distance=float(d), probability=float(p)).iloc[0]
    imp = g.node_importance(distance=float(d), probability=float(p))
    imp.index = imp["id"].astype(str)
    ref = CONEFOR.groupby(CF_KEYS).get_group(case)
    # Conefor computes in float: agreement to single precision.
    for _, row in ref[ref["level"] == "overall"].iterrows():
        assert ov[row["name"]] == pytest.approx(float(row["value"]), rel=1e-5), row["name"]
    for name, sub in ref[ref["level"] == "node"].groupby("name"):
        values = sub["value"].to_numpy(float)
        np.testing.assert_allclose(imp.loc[sub["id"], name], values, rtol=1e-3,
                                   atol=2e-5 * np.abs(values).max() + 1e-6, err_msg=name)
