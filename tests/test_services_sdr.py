# SPDX-License-Identifier: GPL-3.0-or-later
"""Sediment delivery ratio (ls.services.sdr) against InVEST's outputs (tools/make_reference_invest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_services import CRS, _case, _raster

import landschaft as ls

CASES = ["sdr_mfd", "sdr_d8", "sdr_drainage"]
FIELDS = ["usle_tot", "sed_export", "sed_dep", "avoid_exp", "avoid_eros"]


def _polygons(rings):
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame({"ws_id": [1, 2]}, geometry=[Polygon(r) for r in rings], crs=CRS)


def _run(name, **kw):
    inputs, ref, meta = _case(name)
    nd = meta["input_nodata"]
    bio = pd.DataFrame({"lucode": inputs["codes"], "usle_c": inputs["usle_c"], "usle_p": inputs["usle_p"]})
    p = dict(meta["params"])
    if "method" in kw:
        p["flow_dir_algorithm"] = kw.pop("method")
    extra = {}
    if "drainage" in inputs:
        extra["drainage"] = _raster(inputs["drainage"], 255)
    s = ls.services.sdr(_raster(inputs["dem"], nd), _raster(inputs["erosivity"], nd), _raster(inputs["erodibility"], nd),
                        _raster(inputs["lulc"], meta["nodata"]), bio, _polygons(meta["watersheds"]),
                        threshold=p["threshold_flow_accumulation"], k=p["k_param"], ic_0=p["ic_0_param"],
                        sdr_max=p["sdr_max"], l_max=p["l_max"], method=p["flow_dir_algorithm"], **extra, **kw)
    return s, ref, meta


def _same(ours, ref, name):
    from conftest import assert_ucrt_close

    a = np.asarray(ours)
    assert a.dtype == ref.dtype, (name, a.dtype, ref.dtype)
    assert_ucrt_close(a, ref, err_msg=name)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.external_reference
def test_sdr_matches_invest(name):
    s, ref, meta = _run(name)
    inter = s.intermediate
    for key in ("pit_filled_dem", "slope", "slope_threshold", "flow_accumulation", "ls", "w", "w_threshold", "cp",
                "w_bar", "s_bar", "d_up", "ws_inverse", "d_dn", "e_prime", "flux"):
        _same(inter[key], ref[key], key)
    stream_key = "stream_and_drainage" if "stream_and_drainage" in ref else "stream"
    _same(s.stream, ref[stream_key], "stream")
    _same(s.rkls, ref["rkls"], "rkls")
    _same(s.usle, ref["usle"], "usle")
    _same(s.ic, ref["ic"], "ic")
    _same(s.sdr, ref["sdr_factor"], "sdr")
    _same(s.sed_export, ref["sed_export"], "sed_export")
    _same(s.sed_deposition, ref["sed_deposition"], "sed_deposition")
    _same(s.avoided_erosion, ref["avoided_erosion"], "avoided_erosion")
    _same(s.avoided_export, ref["avoided_export"], "avoided_export")
    _same(s.drains_to_stream, ref["what_drains_to_stream"], "what_drains_to_stream")
    ours = s.watersheds.set_index("ws_id")
    for row in meta["watershed_results"]:
        for f in FIELDS:  # InVEST adds the cells in float32 and writes 11 decimals
            assert ours.loc[row["ws_id"], f] == pytest.approx(row[f], rel=3e-6), (row["ws_id"], f)


def test_connectivity_index_is_the_sdr_ic():
    s, ref, meta = _run("sdr_mfd")
    inputs = _case("sdr_mfd")[0]
    dem = _raster(np.where(np.isnan(ref["masked_dem"]), np.nan, ref["masked_dem"]))
    w = s.intermediate["w"]
    ic = ls.hydro.connectivity_index(dem, w, threshold=meta["params"]["threshold_flow_accumulation"])
    np.testing.assert_array_equal(ic.values, ref["ic"])
    ic2 = ls.hydro.connectivity_index(dem, w, streams=s.stream.where(s.stream != 255))
    np.testing.assert_array_equal(ic2.values, ref["ic"])
    assert inputs["dem"].shape == ic.shape


def test_sdr_threads_and_checks():
    a, ref, meta = _run("sdr_mfd", n_threads=1)
    b, _, _ = _run("sdr_mfd", n_threads=4)
    np.testing.assert_array_equal(a.sed_deposition.values, b.sed_deposition.values)
    with pytest.raises(ValueError, match="method"):
        _run("sdr_mfd", method="dinf")
