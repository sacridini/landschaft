# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.redd (VT0007) against Clark Labs' UDef-ARP (tests/data/reference/redd, tools/make_reference_redd.py, which runs
UDef-ARP's own functions): the negligible risk threshold, the benchmark and alternative vulnerability classes, the
relative frequency table and the fitted density, the allocation to a period's deforestation and to an expected area
(a region the fit lacks filled from its class), to the bit (the adjustment ratio to 1e-14: UDef-ARP reads its table
back from a CSV); the assessment grid's cells and agreement to float32's rounding (UDef-ARP sums in float32)."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "redd"


@pytest.fixture(scope="module")
def ref():
    z = dict(np.load(REF / "redd.npz"))
    meta = json.loads((REF / "redd.json").read_text(encoding="utf-8"))
    ny, nx = z["in/distance"].shape
    x0, w, _, y0, _, h = meta["transform"]
    coords = {"y": y0 + (np.arange(ny) + 0.5) * h, "x": x0 + (np.arange(nx) + 0.5) * w}

    def m(k):
        return xr.DataArray(z[k], dims=("y", "x"), coords=coords).rio.write_crs("EPSG:32722")

    return z, meta, m


def near(got, want, tol):
    assert abs(got - want) <= tol * max(abs(want), 1.0), (got, want)


def test_vt0007_is_udef_arps(ref):
    z, meta, m = ref
    nrt = ls.redd.nrt(m("in/distance"), m("in/cal"), mask=m("in/mask"))
    assert nrt == meta["nrt"]
    classes = ls.redd.benchmark(m("in/distance"), nrt, mask=m("in/mask"))
    np.testing.assert_array_equal(classes.values, z["benchmark"])
    alt = ls.redd.alternative(m("in/vulnerability"), m("in/forest"), mask=m("in/mask"))
    np.testing.assert_array_equal(alt.values, z["alternative"])

    f = ls.redd.fit(classes, m("in/cal"), subdivisions=m("in/regions"))
    t = f.table
    np.testing.assert_array_equal(t[["id", "deforested", "cells", "frequency"]].to_numpy(float), z["fit/table"])
    np.testing.assert_array_equal(f.density.values, z["fit/density"])

    # the prediction's table lacks a region (class 3 of subdivision 4): it is filled from the class's others
    lacking = t[t["id"] != meta["dropped"][0]]
    for name, kw in [("cnf", {"deforestation": m("in/cnf")}), ("vp", {"expected": meta["expected"]})]:
        a = ls.redd.allocate(classes, lacking, subdivisions=m("in/regions"), max_iterations=meta["max_iterations"],
                             **kw)
        np.testing.assert_array_equal(a.density.values, z[f"{name}/density"], err_msg=name)
        near(a.adjustment_ratio, meta["ar"][name], 1e-14)
        assert a.iterations == meta["iterations"][name] and a.filled == 1

    a = meta["assess"]
    s = ls.redd.assess(m("cnf/density"), m("in/cnf"), grid_area=a["grid_area"], mask=m("in/mask"))
    cells = z["assess/cells"]
    assert len(s.table) == len(cells)
    np.testing.assert_array_equal(s.table[["x", "y", "actual_ha"]].to_numpy(), cells[:, :3])
    np.testing.assert_allclose(s.table["predicted_ha"], cells[:, 3], rtol=1e-6)
    mt = s.metrics
    assert mt["samples"] == a["samples"]
    for ours, theirs in [("medae", "medae"), ("mae", "mae"), ("iou_percent", "iou"), ("agreement", "agreement"),
                         ("difference", "difference"), ("r2", "r2")]:
        near(mt[ours], a[theirs], 1e-5)
    near(mt["ols_slope"], a["ols"][0], 1e-5)
    near(mt["ols_intercept"], a["ols"][1], 1e-5)
    near(mt["theil_sen_slope"], a["theil_sen"][0], 1e-5)
    near(mt["theil_sen_intercept"], a["theil_sen"][1], 1e-5)
    r = s.residuals.values
    assert np.isnan(r).any() and set(np.unique(r[~np.isnan(r)])) <= set(s.table["residual_ha"])


def test_redd_errors(ref):
    z, meta, m = ref
    d = m("in/distance")
    with pytest.raises(ValueError, match="above the cell size"):
        ls.redd.benchmark(d, 20)
    classes = ls.redd.benchmark(d, meta["nrt"], mask=m("in/mask"))
    f = ls.redd.fit(classes, m("in/cal"))
    with pytest.raises(ValueError, match="one of them"):
        ls.redd.allocate(classes, f)
    with pytest.raises(ValueError, match="cannot take"):
        ls.redd.allocate(classes, f, expected=1e9, max_iterations=3)
    with pytest.raises(ValueError, match="grid"):
        ls.redd.nrt(z["in/distance"], z["in/cal"])
    a = ls.redd.allocate(classes, f, expected=100.0)
    assert abs(a.allocated_ha - 100.0) < 1e-2 and a.filled == 0
