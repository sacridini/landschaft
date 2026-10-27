# SPDX-License-Identifier: GPL-3.0-or-later
"""Seasonal water yield (ls.services.seasonal_water_yield) against InVEST (tools/make_reference_invest_swy.py).

The ``*_patched`` references are InVEST with its bugs corrected (the tool's doc): the
maps there are compared bit for bit; ``swy_mfd`` is InVEST as released, for the parts
the bugs do not touch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_services import CRS, _case, _gt, _on_invest_grid, _raster

import landschaft as ls

PATCHED = ["swy_mfd_patched", "swy_d8_patched", "swy_float64_patched", "swy_grids_patched"]
USER = ["swy_user_recharge_patched", "swy_user_recharge_d8_patched", "swy_grids_recharge_patched"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _polygons(rings):
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame({"ws_id": list(range(1, len(rings) + 1))}, geometry=[Polygon(r) for r in rings], crs=CRS)


def _bio(inputs):
    cn, kc = inputs["cn"], inputs["kc"]
    d = {"lucode": inputs["codes"], **{c: cn[:, i] for i, c in enumerate(["cn_a", "cn_b", "cn_c", "cn_d"])},
         **{f"kc_{m + 1}": kc[:, m] for m in range(12)}}
    return pd.DataFrame(d)


def _run(name, **kw):
    inputs, ref, meta = _case(name)
    nd = meta["input_nodata"]
    p = meta["params"]
    args = dict(threshold=p["threshold"], method=p["flow"], beta=p["beta"], gamma=p["gamma"])
    if "recharge" in inputs:
        args["recharge"] = _raster(inputs["recharge"], nd, _gt(meta, "l"))
        bio = None
    else:
        bio = _bio(inputs)
        args["precipitation"] = [_raster(a, nd, _gt(meta, f"p{m + 1}")) for m, a in enumerate(inputs["precip"])]
        args["et0"] = [_raster(a, nd, _gt(meta, f"e{m + 1}")) for m, a in enumerate(inputs["et0"])]
        args["soil_group"] = _raster(inputs["soil"], 255, _gt(meta, "soil"))
        if "zones" in inputs:
            args["climate_zones"] = _raster(inputs["zones"], -1, _gt(meta, "cz"))
            args["climate_zone_table"] = pd.DataFrame({"cz_id": inputs["zone_ids"],
                                                       **{m: inputs["zone_events"][:, i] for i, m in enumerate(MONTHS)}})
        else:
            args["rain_events"] = inputs["events"]
        args["alpha"] = (pd.DataFrame({"month": range(1, 13), "alpha": inputs["alpha"]}) if "alpha" in inputs
                         else p["alpha"])
    args.update(kw)
    s = ls.services.seasonal_water_yield(_raster(inputs["dem"], nd, _gt(meta, "dem")),
                                         _raster(inputs["lulc"], 0, _gt(meta, "lulc")), bio,
                                         _polygons(meta["watersheds"]), **args)
    return s, ref, meta


def _same(ours, ref, name):
    from conftest import assert_ucrt_close

    a = np.asarray(ours)
    assert a.dtype == ref.dtype, (name, a.dtype, ref.dtype)
    assert_ucrt_close(a, ref, err_msg=name)


def _check_sheds(s, meta):
    ours = s.watersheds
    for i, row in enumerate(meta["watershed_results"]):  # InVEST adds the cells in float32 and writes 11 decimals
        assert ours["qb"].iloc[i] == pytest.approx(row["qb"], rel=3e-6, abs=1e-9), (i, "qb")
        assert ours["vri_sum"].iloc[i] == pytest.approx(row["vri_sum"], rel=3e-6, abs=1e-9), (i, "vri_sum")
        assert ours["geom_id"].iloc[i] == row["geom_id"]


@pytest.mark.parametrize("name", PATCHED)
@pytest.mark.external_reference
def test_swy_matches_invest(name):
    s, ref, meta = _run(name)
    _on_invest_grid(s.vri, ref["vri"], meta)
    inter = s.intermediate
    for key in ("pit_filled_dem", "flow_dir", "cn", "si"):
        _same(inter[key], ref[key], key)
    _same(inter["flow_accumulation"], ref["flow_accum"], "flow_accum")
    for m in range(12):
        _same(inter["quickflow_monthly"].values[m], ref[f"qf_{m + 1}"], f"qf_{m + 1}")
        _same(inter["kc"].values[m], ref[f"kc_{m + 1}"], f"kc_{m + 1}")
        _same(inter["n_events"].values[m], ref[f"n_events{m + 1}"], f"n_events{m + 1}")
    for ours, key in ((s.stream, "stream"), (s.quickflow, "qf"), (s.local_recharge, "l"), (s.l_avail, "l_avail"),
                      (s.l_sum_avail, "l_sum_avail"), (s.l_sum, "l_sum"), (s.baseflow, "b"), (s.b_sum, "b_sum"),
                      (s.vri, "vri"), (s.aet, "aet"), (s.precipitation, "annual_precip")):
        _same(ours, ref[key], key)
    _check_sheds(s, meta)
    got = s.monthly.set_index(["geom_id", "month"])
    for row in meta["monthly"]:
        for f in ("quickflow", "baseflow", "precipitation"):
            assert got.loc[(row["geom_id"], row["month"]), f] == pytest.approx(row[f], rel=3e-6, abs=1e-6), (row, f)


@pytest.mark.parametrize("name", USER)
@pytest.mark.external_reference
def test_swy_user_recharge_matches_invest(name):
    s, ref, meta = _run(name)
    _on_invest_grid(s.vri, ref["vri"], meta)
    for ours, key in ((s.stream, "stream"), (s.local_recharge, "l"), (s.l_avail, "l_avail"), (s.l_sum, "l_sum"),
                      (s.baseflow, "b"), (s.b_sum, "b_sum"), (s.vri, "vri")):
        _same(ours, ref[key], key)
    _same(s.intermediate["flow_dir"], ref["flow_dir"], "flow_dir")
    assert s.quickflow is None and s.l_sum_avail is None and s.monthly is None
    _check_sheds(s, meta)


@pytest.mark.external_reference
def test_swy_released_invest_where_its_bugs_do_not_reach():
    """InVEST as released: the same curve numbers, quickflow and routing; its L_sum_avail differs."""
    s, ref, _ = _run("swy_mfd")
    for key in ("si", "flow_dir"):
        _same(s.intermediate[key], ref[key], key)
    # InVEST writes the integer table's curve numbers as int16 (nodata 32767), here float64
    cn_ref = np.where(ref["cn"] == np.iinfo(np.int16).max, np.nan, ref["cn"]).astype(np.float64)
    np.testing.assert_array_equal(s.intermediate["cn"].values, cn_ref)
    for m in range(12):
        _same(s.intermediate["quickflow_monthly"].values[m], ref[f"qf_{m + 1}"], f"qf_{m + 1}")
    _same(s.quickflow, ref["qf"], "qf")
    _same(s.stream, ref["stream"], "stream")
    _same(s.precipitation, ref["annual_precip"], "annual_precip")
    # InVEST's mean of the upslope terms (by the raw weights) is not their sum by the shares of the flow:
    # less where several cells drain into one, more where a cell gets a small share of one cell's flow
    lsa, lsa_ref = s.l_sum_avail.values, ref["l_sum_avail"]
    ok = ~np.isnan(lsa_ref)
    assert np.nanmax(lsa[ok] - lsa_ref[ok]) > 100
    assert np.nanmin(lsa[ok] - lsa_ref[ok]) < -100


def test_swy_l_sum_avail_is_the_upslope_sum():
    """Eq. 103 on a D8 chain: L_sum_avail is the sum of L_avail + L_sum_avail of the cells draining in."""
    s, _, _ = _run("swy_d8_patched")
    d = s.intermediate["flow_dir"].values
    la, lsa = s.l_avail.values.astype(np.float64), s.l_sum_avail.values.astype(np.float64)
    ny, nx = d.shape
    row = [0, -1, -1, -1, 0, 1, 1, 1]
    col = [1, 1, 0, -1, -1, -1, 0, 1]
    expect = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            k = int(d[r, c])
            if k == 128 or np.isnan(lsa[r, c]):
                continue
            rr, cc = r + row[k], c + col[k]
            if 0 <= rr < ny and 0 <= cc < nx:
                expect[rr, cc] += la[r, c] + lsa[r, c]
    ok = ~np.isnan(lsa)
    np.testing.assert_allclose(lsa[ok], expect[ok], rtol=1e-5, atol=1e-2)


def test_swy_outlets_off_the_raster_keep_their_baseflow():
    """Eq. 108: a cell draining only off the raster has B_sum = L_sum (InVEST gives 0)."""
    s, _, _ = _run("swy_user_recharge_d8_patched")
    d = s.intermediate["flow_dir"].values
    ny, nx = d.shape
    row = [0, -1, -1, -1, 0, 1, 1, 1]
    col = [1, 1, 0, -1, -1, -1, 0, 1]
    seen = 0
    for r in range(ny):
        for c in range(nx):
            k = int(d[r, c])
            if k == 128:
                continue
            rr, cc = r + row[k], c + col[k]
            if not (0 <= rr < ny and 0 <= cc < nx):
                assert s.b_sum.values[r, c] == np.float32(s.l_sum.values[r, c])
                seen += 1
    assert seen > 0


def test_swy_fractional_curve_numbers_kept():
    s, _, _ = _run("swy_float64_patched")
    cn = s.intermediate["cn"].values
    assert np.any(cn[~np.isnan(cn)] % 1 != 0)
    assert cn.dtype == np.float64  # int16 land use and uint8 soil groups


def test_swy_threads_and_checks():
    a, _, _ = _run("swy_mfd_patched", n_threads=1)
    b, _, _ = _run("swy_mfd_patched", n_threads=4)
    np.testing.assert_array_equal(a.baseflow.values, b.baseflow.values)
    with pytest.raises(ValueError, match="method"):
        _run("swy_mfd_patched", method="dinf")
    with pytest.raises(ValueError, match="rain events or the climate zones"):
        _run("swy_mfd_patched", rain_events=None)
    with pytest.raises(ValueError, match="12"):
        _run("swy_mfd_patched", rain_events=[1, 2, 3])
    inputs = _case("swy_mfd_patched")[0]
    bad = inputs["soil"].copy()
    bad[0, 0] = 6
    with pytest.raises(ValueError, match="soil group"):
        _run("swy_mfd_patched", soil_group=_raster(bad, 255))


def test_swy_monthly_stack_and_save(tmp_path):
    inputs, _, meta = _case("swy_mfd_patched")
    nd = meta["input_nodata"]
    a, _, _ = _run("swy_mfd_patched")
    import xarray as xr

    stack = xr.concat([_raster(p, nd) for p in inputs["precip"]], dim="band")
    b, _, _ = _run("swy_mfd_patched", precipitation=stack)
    np.testing.assert_array_equal(a.baseflow.values, b.baseflow.values)
    folder = a.save(tmp_path / "swy")
    assert (folder / "baseflow.tif").is_file()
    assert "SeasonalWaterYield(" in repr(a)
