# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics against the stored outputs of the R package landscapemetrics.

The references (``tests/data/reference/lsm``) hold every value of
``calculate_lsm`` for each input raster (``tests/data/lsm``) and option set;
``tools/make_reference_lsm.py`` regenerates them. The live tests run R now,
on random landscapes (Rscript and landscapemetrics required).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import lsm_reference
import lsm_scenes
from conftest import DATA, same_bits

REF_DIR = DATA / "reference" / "lsm"
MANIFEST = json.loads((REF_DIR / "manifest.json").read_text())

# calculate_lsm builds te, ed, iji and contag from the neighbourhood matrix of
# the complexity metrics; the lsm_l_te/lsm_*_iji/lsm_l_contag functions (and
# landschaft) always use 4 neighbours. With neighbourhood = 8 these are
# compared with the 4-neighbour reference instead.
NEIGHBOURHOOD_QUIRK = {"te", "ed", "iji", "contag"}

RTOL = 1e-10
ATOL = 1e-14

# Metrics whose values agree with landscapemetrics within RTOL but not always
# to the last bit: circle (landscapemetrics shuffles the points of Welzl's
# algorithm at random), gyrate (R sums the coordinates and distances of the
# cells one by one in extended precision) and enn (R's pruned search can
# settle on an equally near cell pair whose distance rounds differently when
# the cell sides differ). Every other value must be the same bits: sums, means
# and standard deviations use R's 80-bit accumulation, log and sqrt R's x87
# functions, pafrac the QR decomposition of R's lm().
NOT_BITWISE = ("circle", "gyrate", "enn")
# Units in the last place allowed without the x87 functions (conftest.BITWISE): the
# metrics add up many logarithms and square roots and some cancel (sd, cv, contag);
# the live runs compare with R on the same machine, which on ARM64 has no 80-bit sums.
ULPS = 128


def compare(ours: pd.DataFrame, ref: pd.DataFrame, label: str) -> None:
    keys = ["level", "class", "id", "metric"]
    merged = ours.merge(ref, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    missing = merged[merged["_merge"] != "both"]
    assert missing.empty, f"{label}: rows only on one side:\n{missing.head(10)}"
    a = merged["value"].to_numpy(float)
    b = merged["value_ref"].to_numpy(float)
    ok = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert ok.all(), f"{label}: values differ:\n{merged.loc[~ok, keys + ['value', 'value_ref']].head(10)}"
    bitwise = ~merged["metric"].astype(str).str.startswith(NOT_BITWISE).to_numpy()
    bad = bitwise & ~same_bits(a, b, ulps=ULPS)
    assert not bad.any(), f"{label}: not the same bits:\n{merged.loc[bad, keys + ['value', 'value_ref']].head(10)}"


def implemented(ref: pd.DataFrame) -> pd.DataFrame:
    available = set(ls.list_metrics()["function_name"])
    prefix = ref["level"].map({"patch": "lsm_p_", "class": "lsm_c_", "landscape": "lsm_l_"})
    return ref[(prefix + ref["metric"]).isin(available)]


@pytest.mark.reference
@pytest.mark.parametrize("fname", sorted(MANIFEST["files"]))
def test_matches_landscapemetrics(fname):
    info = MANIFEST["files"][fname]
    options = info["options"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # geographic CRS of podlasie
        ours = ls.Landscape(DATA / "lsm" / info["input"], **options).metrics()
    ref = implemented(lsm_reference.read_reference(REF_DIR / fname))
    if not ref["metric"].str.startswith("circle").any():  # non-square cells
        ours = ours[~ours["metric"].str.startswith("circle")]
    if options.get("neighbourhood") == 8:
        quirk = ref["metric"].isin(NEIGHBOURHOOD_QUIRK)
        default = implemented(lsm_reference.read_reference(REF_DIR / fname.replace("__complexity", "__default")))
        ref = pd.concat([ref[~quirk], default[default["metric"].isin(NEIGHBOURHOOD_QUIRK)]])
    compare(ours, ref, fname)


def test_every_catalogue_metric_is_referenced():
    ref = lsm_reference.read_reference(REF_DIR / "lm_landscape__default.csv.gz")
    prefix = ref["level"].map({"patch": "lsm_p_", "class": "lsm_c_", "landscape": "lsm_l_"})
    assert set(ls.list_metrics()["function_name"]) <= set(prefix + ref["metric"])


# -- live comparison ---------------------------------------------------------


@pytest.fixture(scope="module")
def rscript():
    exe = lsm_reference.find_rscript()
    if exe is None:
        pytest.skip("Rscript not found (set LANDSCHAFT_RSCRIPT)")
    return exe


LIVE_CASES = [
    ("clustered", dict(shape=(40, 55), classes=[1, 2, 3, 4, 5], seed=1, sigma=1.5, na_fraction=0.1), (10.0, 10.0), {}),
    ("clustered", dict(shape=(33, 47), classes=[3, 9, 12], seed=2, sigma=3.0), (25.0, 25.0), {"directions": 4}),
    ("salt", dict(shape=(20, 26), classes=[0, 1, 2, 3], seed=3, na_fraction=0.1), (2.0, 2.0),
     {"count_boundary": True, "neighbourhood": 8, "ordered": True, "base": "log10"}),
    ("clustered", dict(shape=(50, 30), classes=[-3, 4, 70, 71], seed=4, sigma=1.0, na_fraction=0.05), (30.0, 15.0),
     {"directions": 4, "count_boundary": True, "cell_center": True}),
    ("salt", dict(shape=(15, 60), classes=[1, 2], seed=5), (1.0, 1.0), {"ordered": False, "base": "log"}),
    ("clustered", dict(shape=(45, 45), classes=[1, 2, 3], seed=6, sigma=4.0, na_fraction=0.08), (5.0, 5.0),
     {"edge_depth": 0, "consider_boundary": True}),
    ("clustered", dict(shape=(60, 50), classes=[1, 2], seed=7, sigma=5.0), (20.0, 20.0),
     {"edge_depth": 4, "directions": 4}),
    ("sparse", dict(shape=(80, 70), classes=[2, 3, 4], seed=8, n_blobs=30, background=1), (0.7, 0.7), {}),
    ("sparse", dict(shape=(50, 90), classes=[1, 2], seed=9, n_blobs=25, na_fraction=0.03), (5.0, 5.0),
     {"directions": 4}),
]


@pytest.mark.live_reference
@pytest.mark.parametrize("kind, params, res, options", LIVE_CASES)
def test_live_landscapemetrics(rscript, kind, params, res, options):
    params = dict(params)
    shape = params.pop("shape")
    classes = params.pop("classes")
    make = {"clustered": lsm_scenes.clustered, "salt": lsm_scenes.salt_and_pepper, "sparse": lsm_scenes.sparse}[kind]
    data = make(shape, classes, **params)
    ref = implemented(lsm_reference.run_array(rscript, data, res, **options))
    if options.get("neighbourhood") == 8:
        ref = ref[~ref["metric"].isin(NEIGHBOURHOOD_QUIRK)]
    ours = ls.Landscape(data, res=res, **options).metrics()
    if options.get("neighbourhood") == 8:
        ours = ours[~ours["metric"].isin(NEIGHBOURHOOD_QUIRK)]
    if res[0] != res[1]:
        ours = ours[~ours["metric"].str.startswith("circle")]
    compare(ours, ref, f"{kind} {options}")
