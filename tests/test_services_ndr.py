# SPDX-License-Identifier: GPL-3.0-or-later
"""Nutrient delivery ratio (ls.services.ndr) against InVEST's outputs (tools/make_reference_invest_ndr.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_services import CRS, _case, _gt, _on_invest_grid, _raster

import landschaft as ls

CASES = ["ndr_mfd", "ndr_d8", "ndr_p_only", "ndr_tall", "ndr_wide", "ndr_grids"]
CHAIN = ["masked_dem", "masked_lulc", "masked_runoff_proxy", "filled_dem", "slope", "thresholded_slope",
         "flow_direction", "flow_accumulation", "s_accumulation", "runoff_proxy_index", "s_bar", "d_up",
         "s_factor_inverse", "d_dn", "dist_to_channel"]
BY_NUTRIENT = ["load_{x}", "modified_load_{x}", "surface_load_{x}", "eff_{x}", "crit_len_{x}"]
SUBSURFACE = ["sub_load_n", "sub_ndr_n"]


def _polygons(rings):
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame({"ws_id": [1, 2]}, geometry=[Polygon(r) for r in rings], crs=CRS)


def _table(inputs, meta):
    t = {"lucode": inputs["codes"], "load_type_n": meta["load_type_n"], "load_type_p": meta["load_type_p"]}
    for c in ("load_n", "eff_n", "crit_len_n", "proportion_subsurface_n", "load_p", "eff_p", "crit_len_p"):
        t[c] = inputs[c]
    return pd.DataFrame(t)


def _run(name, **kw):
    inputs, ref, meta = _case(name)
    nd = meta["input_nodata"]
    p = meta["params"]
    args = dict(threshold=p["threshold_flow_accumulation"], k=p["k_param"], nitrogen=p["calc_n"],
                phosphorus=p["calc_p"], method=p["flow_dir_algorithm"], runoff_proxy_mean=p["runoff_proxy_av"],
                subsurface_critical_length=p["subsurface_critical_length_n"],
                subsurface_efficiency=p["subsurface_eff_n"])
    args.update(kw)
    r = ls.services.ndr(_raster(inputs["dem"], nd, _gt(meta, "dem")),
                        _raster(inputs["lulc"], meta["nodata"], _gt(meta, "lulc")),
                        _raster(inputs["runoff"], nd, _gt(meta, "runoff")), _table(inputs, meta),
                        _polygons(meta["watersheds"]), **args)
    return r, ref, meta


def _same(ours, ref, name):
    from conftest import assert_ucrt_close

    a = np.asarray(ours)
    assert a.dtype == ref.dtype, (name, a.dtype, ref.dtype)
    assert_ucrt_close(a, ref, err_msg=name)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.external_reference
def test_ndr_matches_invest(name):
    r, ref, meta = _run(name)
    _on_invest_grid(r.ndr_n if r.ndr_n is not None else r.ndr_p, ref["ndr_n" if r.ndr_n is not None else "ndr_p"], meta)
    inter = r.intermediate
    for key in CHAIN:
        _same(inter[key], ref[key], key)
    _same(r.stream, ref["stream"], "stream")
    _same(r.drains_to_stream, ref["what_drains_to_stream"], "what_drains_to_stream")
    _same(r.ic, ref["ic_factor"], "ic_factor")
    nutrients = [x for x in "np" if meta["params"][f"calc_{x}"]]
    for x in nutrients:
        for key in BY_NUTRIENT:
            _same(inter[key.format(x=x)], ref[key.format(x=x)], key.format(x=x))
        _same(getattr(r, f"effective_retention_{x}"), ref[f"effective_retention_{x}"], f"effective_retention_{x}")
        _same(getattr(r, f"ndr_{x}"), ref[f"ndr_{x}"], f"ndr_{x}")
        _same(getattr(r, f"{x}_surface_export"), ref[f"{x}_surface_export"], f"{x}_surface_export")
    if "n" in nutrients:
        for key in SUBSURFACE:
            _same(inter[key], ref[key], key)
        _same(r.n_subsurface_export, ref["n_subsurface_export"], "n_subsurface_export")
        _same(r.n_total_export, ref["n_total_export"], "n_total_export")
    else:
        assert r.n_total_export is None and r.ndr_n is None
    ours = r.watersheds.set_index("ws_id")
    for row in meta["watershed_results"]:
        fields = [f for f in row if f != "ws_id"]
        assert fields == list(ours.columns[1:])
        for f in fields:  # InVEST adds the cells in float32 and writes 11 decimals
            assert ours.loc[row["ws_id"], f] == pytest.approx(row[f], rel=3e-6), (row["ws_id"], f)


@pytest.mark.external_reference
def test_ndr_subsurface_distance_in_metres():
    """InVEST takes the distance to the stream in cells for metres; only the subsurface part differs."""
    r, ref, meta = _run("ndr_subsurface_pixels")
    assert meta["pixels"]
    _same(r.n_surface_export, ref["n_surface_export"], "n_surface_export")
    _same(r.p_surface_export, ref["p_surface_export"], "p_surface_export")
    ours, theirs = r.intermediate["sub_ndr_n"].values, ref["sub_ndr_n"]
    d = r.intermediate["dist_to_channel"].values
    far = d > 0
    assert np.all(ours[far] < theirs[far])  # in metres the soil retains more: delivery is lower
    cell = meta["pixel"]
    l_sub, e_sub = meta["params"]["subsurface_critical_length_n"], meta["params"]["subsurface_eff_n"]
    np.testing.assert_allclose(ours[far], 1 - e_sub * (1 - np.exp(-5 * d[far] * cell / l_sub)), rtol=1e-14)
    np.testing.assert_allclose(theirs[far], 1 - e_sub * (1 - np.exp(-5 * d[far] / l_sub)), rtol=1e-14)


def test_ndr_threads_options_and_checks():
    a = _run("ndr_mfd", n_threads=1)[0]
    b = _run("ndr_mfd", n_threads=4)[0]
    np.testing.assert_array_equal(a.n_total_export.values, b.n_total_export.values)
    np.testing.assert_array_equal(a.effective_retention_p.values, b.effective_retention_p.values)
    ic = a.ic.values
    assert np.nanmin(ic) < a.ic_0 < np.nanmax(ic) and a.runoff_proxy_mean > 0
    # the mean of the runoff proxy given: the index scales by it
    c, _, _ = _run("ndr_mfd", runoff_proxy_mean=2 * a.runoff_proxy_mean)
    np.testing.assert_allclose(c.intermediate["runoff_proxy_index"].values,
                               a.intermediate["runoff_proxy_index"].values / 2, rtol=1e-6)
    n_only, _, _ = _run("ndr_mfd", phosphorus=False)
    assert n_only.p_surface_export is None
    assert list(n_only.watersheds.columns[2:]) == ["n_surface_load", "n_surface_export", "n_subsurface_export",
                                                    "n_total_export", "n_subsurface_load"]
    np.testing.assert_array_equal(n_only.n_total_export.values, a.n_total_export.values)
    assert "N export" in repr(a)
    with pytest.raises(ValueError, match="method"):
        _run("ndr_mfd", method="dinf")
    with pytest.raises(ValueError, match="nitrogen, phosphorus"):
        _run("ndr_mfd", nitrogen=False, phosphorus=False)
    with pytest.raises(ValueError, match="subsurface"):
        _run("ndr_mfd", subsurface_efficiency=None)


def test_ndr_table_checks():
    inputs, _, meta = _case("ndr_mfd")
    nd = meta["input_nodata"]
    args = (_raster(inputs["dem"], nd), _raster(inputs["lulc"], meta["nodata"]), _raster(inputs["runoff"], nd))
    ws = _polygons(meta["watersheds"])
    kw = dict(threshold=60, subsurface_critical_length=150.0, subsurface_efficiency=0.6)
    t = _table(inputs, meta)
    with pytest.raises(ValueError, match="load_type_n"):
        ls.services.ndr(*args, t.assign(load_type_n="fertilizer"), ws, **kw)
    with pytest.raises(ValueError, match="lacks the column"):
        ls.services.ndr(*args, t.drop(columns="load_type_p"), ws, **kw)
    with pytest.raises(ValueError, match="class"):
        ls.services.ndr(*args, t.iloc[1:], ws, **kw)
    # measured runoff: the load as given; application rate: times 1 - eff
    r = ls.services.ndr(*args, t.assign(load_type_n="measured-runoff"), ws, phosphorus=False, **kw)
    load, lulc = r.intermediate["load_n"].values, inputs["lulc"]
    for i, code in enumerate(inputs["codes"]):
        cells = (lulc == code) & ~np.isnan(load)
        assert np.all(load[cells] == np.float32(inputs["load_n"][i]))
