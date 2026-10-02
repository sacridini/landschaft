# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.vif / select_predictors against usdm (vif, vifstep, vifcor)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("predictors"))
def test_matches_usdm(case):
    ref = sdm_reference.result("predictors", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        sdm_reference.assert_same(ours[key], ref[key], key)


def test_usdm_errors_are_fixed():
    t = sdm_cases.vif_table()
    # usdm fails when one variable would remain; here it stops
    out = ls.sdm.select_predictors(t[["a", "b"]], vif=None, cor=0.5)
    assert out.attrs["selection"]["excluded"] == ["b"] and list(out.columns) == ["a"]
    out = ls.sdm.select_predictors(t[["a", "b"]], vif=1.5)
    assert len(out.columns) == 1
    # every variable kept: usdm fails, here nothing is dropped
    out = ls.sdm.select_predictors(t, vif=2, keep=["a", "b", "c", "d"])
    assert out.attrs["selection"]["excluded"] == []
    out = ls.sdm.select_predictors(t, vif=None, cor=0.5, keep=["a", "b", "c", "d"])
    assert out.attrs["selection"]["excluded"] == []
    # names with spaces: the same choices as without
    plain = ls.sdm.select_predictors(t, vif=None, cor=0.7).attrs["selection"]["excluded"]
    spaced = ls.sdm.select_predictors(t.rename(columns={"b": "b 2"}), vif=None, cor=0.7)
    assert spaced.attrs["selection"]["excluded"] == [("b 2" if e == "b" else e) for e in plain]


def test_raster_sample_is_random_between_7000_and_10000_cells():
    stack = sdm_cases.planar_stack(90, 90, 9)
    a = ls.sdm.vif(stack, seed=1)["vif"].to_numpy()
    b = ls.sdm.vif(stack, seed=2)["vif"].to_numpy()
    assert not np.array_equal(a, b)  # terra took the first 5000 cells whatever the seed


def test_area_and_errors():
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    m = ls.sdm.accessible_area(occ, distance=300_000)
    out = ls.sdm.select_predictors(env, vif=10, area=m, seed=1)
    assert out.attrs["selection"]["cells"] < 5000
    with pytest.raises(ValueError, match="unknown predictors"):
        ls.sdm.select_predictors(env, keep="bio99", seed=1)
    with pytest.raises(ValueError, match="vif and / or a cor"):
        ls.sdm.select_predictors(env, vif=None)
