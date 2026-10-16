# SPDX-License-Identifier: GPL-3.0-or-later
"""Annual water yield (ls.services.water_yield) against InVEST's outputs (tools/make_reference_invest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_services import CRS, PIXEL, _case, _raster

import landschaft as ls

WY_FIELDS = ["precip_mn", "PET_mn", "AET_mn", "wyield_mn", "wyield_vol", "consum_vol", "consum_mn", "rsupply_vl"]


def _polygons(rings, field, ids):
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame({field: ids}, geometry=[Polygon(r) for r in rings], crs=CRS)


def _bio(inputs):
    return pd.DataFrame({"lucode": inputs["codes"], "lulc_veg": inputs["veg"], "root_depth": inputs["root"],
                         "kc": inputs["kc"]})


def _args(inputs, meta):
    nd = meta["input_nodata"]
    return (_raster(inputs["lulc"], meta["nodata"]), _bio(inputs),
            *[_raster(inputs[k], nd) for k in ("precip", "eto", "depth", "pawc")])


def _run_wy(name):
    inputs, ref, meta = _case(name)
    kw = dict(watersheds=_polygons(meta["watersheds"], "ws_id", [1, 2]))
    if "subwatersheds" in meta:
        kw["subwatersheds"] = _polygons(meta["subwatersheds"], "subws_id", [10, 11, 12, 13])
    if "demand" in inputs:
        kw["demand"] = pd.DataFrame({"lucode": inputs["codes"], "demand": inputs["demand"]})
    if "valuation" in meta:
        kw["valuation"] = pd.DataFrame(meta["valuation"], columns=["ws_id", "efficiency", "fraction", "height",
                                                                   "kw_price", "cost", "time_span", "discount"])
    wy = ls.services.water_yield(*_args(inputs, meta), seasonality=meta["z"], **kw)
    return wy, ref, meta


def _bits(ours, ref):
    a = np.asarray(ours)
    assert a.dtype == ref.dtype, (a.dtype, ref.dtype)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(ref))
    np.testing.assert_array_equal(a, ref)


@pytest.mark.parametrize("name", ["wy_basic", "wy_full", "wy_int_inputs", "wy_float64"])
def test_water_yield_matches_invest(name):
    wy, ref, meta = _run_wy(name)
    _bits(wy.fractp, ref["fractp"])
    _bits(wy.wyield, ref["wyield"])
    _bits(wy.aet, ref["aet"])
    _bits(wy.pet, ref["pet"])
    if "demand" in ref:
        _bits(wy.demand, ref["demand"])
    sheds = [("watersheds", "watershed_results", "ws_id")]
    if "subwatershed_results" in meta:
        sheds.append(("subwatersheds", "subwatershed_results", "subws_id"))
    for attr, key, id_name in sheds:
        ours = getattr(wy, attr).set_index(id_name)
        for row in meta[key]:
            for f in WY_FIELDS:
                if f in row:  # InVEST adds the cells in float32 and writes 11 decimals
                    assert ours.loc[row[id_name], f] == pytest.approx(row[f], rel=2e-6), (attr, row[id_name], f)
            if "rsupply_mn" in row:  # InVEST subtracts m3 per cell from mm: the correct form, from its own fields
                assert row["rsupply_mn"] == pytest.approx(row["wyield_mn"] - row["consum_mn"], rel=1e-9)
                expected = row["wyield_mn"] * PIXEL * PIXEL / 1000 - row["consum_mn"]
                assert ours.loc[row[id_name], "rsupply_mn"] == pytest.approx(expected, rel=2e-6)
    if "valuation" in meta:
        ours = wy.watersheds.set_index("ws_id")
        for row, v in zip(meta["watershed_results"], meta["valuation"]):
            assert ours.loc[row["ws_id"], "hp_energy"] == pytest.approx(row["hp_energy"], rel=2e-6)
            if v[7] == 0:  # InVEST: no discount gives a net present value of 0; it is T years of value
                assert row["hp_val"] == 0.0
                expected = (v[4] * row["hp_energy"] - v[5]) * v[6]
            else:
                expected = row["hp_val"]
            assert ours.loc[row["ws_id"], "hp_val"] == pytest.approx(expected, rel=2e-6)


def test_water_yield_threads_and_checks():
    inputs, ref, meta = _case("wy_full")
    args = _args(inputs, meta)
    a = ls.services.water_yield(*args, seasonality=7.5, n_threads=1)
    b = ls.services.water_yield(*args, seasonality=7.5, n_threads=3)
    np.testing.assert_array_equal(a.wyield.values, b.wyield.values)
    ws = _polygons(meta["watersheds"], "ws_id", [1, 2])
    with pytest.raises(ValueError, match="demand"):
        ls.services.water_yield(*args, seasonality=7.5, watersheds=ws, valuation=pd.DataFrame({"ws_id": [1, 2]}))
    with pytest.raises(ValueError, match="lulc_veg"):
        ls.services.water_yield(args[0], _bio(inputs).assign(lulc_veg=2), *args[2:], seasonality=7.5)
    with pytest.raises(ValueError, match="ws_id"):
        ls.services.water_yield(*args, seasonality=7.5, watersheds=_polygons(meta["watersheds"], "id", [1, 2]))
    with pytest.raises(ValueError, match="positive"):
        ls.services.water_yield(*args, seasonality=0)


def test_water_yield_saves(tmp_path):
    wy, ref, meta = _run_wy("wy_full")
    parts = ls.load_result(wy.save(tmp_path / "wy"))
    np.testing.assert_array_equal(parts["wyield"].values, wy.wyield.values)
