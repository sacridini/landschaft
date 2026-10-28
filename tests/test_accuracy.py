# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.accuracy against numpy's sampling and R's mapaccuracy (tests/data/reference/accuracy/accuracy.json,
tools/make_reference_accuracy.py, the same file as Terraflux's): the points drawn as numpy draws them, to the bit;
olofsson and stehman2014 to 1e-12 (R sums in long double); the published example of Olofsson et al. (2014) and its
sample size."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "accuracy" / "accuracy.json"


@pytest.fixture(scope="module")
def ref():
    return json.loads(REF.read_text(encoding="utf-8"))


def _nums(v):
    return np.array([np.nan if x is None else float(x) for x in v])


def _raster(r, key="map"):
    ny, nx = r["shape"]
    t = r["transform"]
    tr = Affine(t[1], t[2], t[0], t[4], t[5], t[3])
    da = xr.DataArray(_nums(r[key]).reshape(ny, nx), dims=("y", "x"),
                      coords={"y": tr.f + tr.e * (np.arange(ny) + 0.5), "x": tr.c + tr.a * (np.arange(nx) + 0.5)})
    return da.rio.write_transform(tr).rio.write_crs("EPSG:31983").rio.write_nodata(np.nan)


def _close(what, got, want):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, what
    both = np.isnan(got) & np.isnan(want)
    ok = both | (np.abs(got - want) <= 1e-12 * np.maximum(np.abs(want), 1e-3))
    assert ok.all(), f"{what}: {got[~ok]} {want[~ok]}"


def _same_as_r(acc, r):
    t = acc.table
    _close("classes", t["class"], _nums(r["classes"]))
    for ours, theirs in [("user_accuracy", "UA"), ("user_se", "SEua"), ("producer_accuracy", "PA"),
                         ("producer_se", "SEpa"), ("area_share", "area"), ("area_share_se", "SEa")]:
        _close(ours, t[ours], _nums(r[theirs]))
    _close("overall", acc.overall["overall_accuracy"], _nums(r["OA"]))
    _close("overall se", acc.overall["overall_se"], _nums(r["SEoa"]))
    _close("matrix", acc.matrix.to_numpy().ravel(), _nums(r["matrix"]))


@pytest.mark.external_reference
def test_the_sample_is_numpys(ref):
    m = _raster(ref)
    pts = ls.accuracy.sample(m, n={int(k): v for k, v in ref["sizes"].items()}, seed=ref["seed"])
    nx = ref["shape"][1]
    assert (pts["row"] * nx + pts["col"]).tolist() == ref["cells"]
    assert pts["stratum"].tolist() == ref["stratum_of"]
    p = pts.geometry.iloc[0]
    assert (p.x, p.y) == (500000.0 + 30.0 * (pts["col"][0] + 0.5), 8000000.0 - 30.0 * (pts["row"][0] + 0.5))
    assert pts.crs == "EPSG:31983"
    assert pts["reference"].isna().all() and pts["id"].tolist() == list(range(1, len(pts) + 1))
    # the sizes from a table draw the same
    sizes = pd.DataFrame({"stratum": [int(k) for k in ref["sizes"]], "n": list(ref["sizes"].values())})
    again = ls.accuracy.sample(m, sizes, seed=ref["seed"])
    assert again.geometry.equals(pts.geometry)


def _labelled(ref):
    nx = ref["shape"][1]
    cells = np.array(ref["cells"])
    row, col = cells // nx, cells % nx
    return gpd.GeoDataFrame({"truth": _nums(ref["reference"])},
                            geometry=gpd.points_from_xy(500000.0 + 30.0 * (col + 0.5), 8000000.0 - 30.0 * (row + 0.5)),
                            crs="EPSG:31983")


@pytest.mark.external_reference
def test_olofsson_and_stehman_are_mapaccuracys(ref):
    m = _raster(ref)
    acc = ls.accuracy.assess(m, _labelled(ref), reference="truth")
    assert acc.estimator == "olofsson"
    _same_as_r(acc, ref["r"]["olofsson"])
    t = acc.table
    total = acc.overall["cells"][0]
    _close("area cells", t["area_cells"], t["area_share"] * total)
    _close("area ha", t["area_ha"], t["area_cells"] * 0.09)
    assert t["samples"].tolist() == [60, 50, 40, 30, 0]
    _close("ci", t["user_ci"], t["user_se"] * ls.accuracy.qnorm(0.975))
    assert acc.counts.to_numpy().sum() == len(ref["cells"])

    acc = ls.accuracy.assess(m, _labelled(ref), reference="truth", estimator="stehman")
    _same_as_r(acc, ref["r"]["stehman_map"])

    acc = ls.accuracy.assess(m, _labelled(ref), reference="truth", strata=_raster(ref, "strata"))
    assert acc.estimator == "stehman"
    _same_as_r(acc, ref["r"]["stehman_strata"])

    # points in another CRS are put on the map's
    acc = ls.accuracy.assess(m, _labelled(ref).to_crs("EPSG:4326"), reference="truth")
    _same_as_r(acc, ref["r"]["olofsson"])


@pytest.mark.external_reference
def test_olofssons_example(ref):
    """Olofsson et al. (2014)'s example from its table of sample counts: the areas of section 5 (21 158 ± 6 158 ha
    of deforestation in cells of 30 m), and its sample size of 641."""
    p = ref["paper"]
    cells = _nums(p["cells"])
    width = 6_450_000 // 50
    data = np.full((4, width), np.nan)
    for k, n in enumerate(cells):
        data[k, : int(n) // 50] = k + 1
    samples = pd.DataFrame({"map": p["map"], "reference": p["reference"]})
    acc = ls.accuracy.assess(data, samples=samples, cell_area=900.0 * 50.0)
    _same_as_r(acc, ref["r"]["paper"])
    assert round(acc.table["area_ha"][0]) == 21158
    assert round(acc.table["area_ha_ci"][0]) == 6158
    assert abs(acc.overall["overall_accuracy"][0] - 0.95) < 0.005


def test_sample_size_and_allocation():
    w = [0.02, 0.015, 0.32, 0.645]
    sd = [(u * (1 - u)) ** 0.5 for u in (0.7, 0.6, 0.9, 0.95)]
    caps = [200_000, 150_000, 3_200_000, 6_450_000]
    assert ls.accuracy.allocate(641, w, sd, caps, "minimum", 75) == ([75, 75, 163, 328], False)
    assert ls.accuracy.allocate(641, w, sd, caps, "proportional")[0] == [13, 10, 205, 413]
    assert sum(ls.accuracy.allocate(641, w, sd, caps, "neyman")[0]) == 641
    assert ls.accuracy.allocate(10, w, sd, [2, 2, 100, 100], "equal")[0] == [2, 2, 3, 3]
    m = np.repeat([1.0, 2.0, 3.0, 4.0], [40, 30, 640, 1290]).reshape(1, -1)
    s = ls.accuracy.sample_size(m, user_accuracy={1: 0.7, 2: 0.6, 3: 0.9, 4: 0.95}, minimum=30)
    # 30 for the two rare classes, the 581 left in proportion to 640 and 1290 cells
    assert s.table["n"].tolist() == [30, 30, 193, 388] and s.n == 641
    assert ls.accuracy.qnorm(0.975) == pytest.approx(1.959963984540054, abs=1e-15)
    assert ls.accuracy.qnorm(0.995) == pytest.approx(2.5758293035489004, abs=1e-15)
    assert ls.accuracy.qnorm(0.05) == pytest.approx(-1.6448536269514729, abs=1e-15)
    with pytest.warns(UserWarning, match="minimum"):
        ls.accuracy.sample_size(m, n=50, minimum=30)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ls.accuracy.sample_size(m, user_accuracy=1.5)


def test_sample_then_assess_round_trip():
    m = np.repeat([1, 2, 3], [600, 300, 100]).reshape(20, 50)
    design = ls.accuracy.sample_size(m, standard_error=0.02, minimum=20)
    pts = ls.accuracy.sample(m, design, seed=3)
    assert pts.groupby("stratum").size().tolist() == design.table["n"].tolist()
    pts["reference"] = pts["stratum"]  # a perfect map
    acc = ls.accuracy.assess(m, pts)
    assert acc.overall["overall_accuracy"][0] == pytest.approx(1.0)
    assert acc.table["area_cells"].tolist() == pytest.approx([600, 300, 100])
    pts.loc[0, "reference"] = np.nan
    with pytest.warns(UserWarning, match="without a reference"):
        ls.accuracy.assess(m, pts)
    with pytest.raises(ValueError, match="olofsson takes"):
        ls.accuracy.assess(m, pts, strata=m, estimator="olofsson")
    with pytest.raises(ValueError, match="has 600 cells|has 100 cells"):
        ls.accuracy.sample(m, n=700)
