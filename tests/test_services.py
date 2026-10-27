# SPDX-License-Identifier: GPL-3.0-or-later
"""Ecosystem services (ls.services) against InVEST's outputs (tools/make_reference_invest.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import landschaft as ls
from landschaft import _core

REF = Path(__file__).parent / "data" / "reference" / "invest"
PIXEL, ORIGIN, CRS = 30.0, (500000.0, 8000000.0), "EPSG:31983"


def _case(name):
    z = np.load(REF / f"{name}.npz")
    meta = json.loads((REF / f"{name}.json").read_text())
    return {k[3:]: z[k] for k in z.files if k.startswith("in/")}, {k[4:]: z[k] for k in z.files if k.startswith("out/")}, meta


def _raster(a, nodata=None, gt=None):
    """``a`` on the cases' grid, or on the GDAL geotransform ``gt`` (an input of a ``*_grids`` case)."""
    x0, px, _, y0, _, py = gt or (ORIGIN[0], PIXEL, 0.0, ORIGIN[1], 0.0, -PIXEL)
    ny, nx = a.shape
    x = x0 + (np.arange(nx) + 0.5) * px
    y = y0 + (np.arange(ny) + 0.5) * py
    da = xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs(CRS).rio.write_transform(Affine(px, 0, x0, 0, py, y0))
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def _gt(meta, stem):
    """The grid InVEST read the input ``stem`` on (None: the cases' grid)."""
    return meta.get("grids", {}).get(stem)


def _on_invest_grid(da, ref, meta):
    """A result on the grid InVEST computed its outputs on (the aligned, clipped grid of a ``*_grids`` case)."""
    assert da.shape == ref.shape
    if "out_gt" in meta:
        assert da.rio.transform() == Affine.from_gdal(*meta["out_gt"])
        assert da.rio.crs.to_epsg() == 31983


def _same(ours, ref):
    """The same float32 bits (NaN where InVEST has nodata); off Windows within UCRT_RTOL."""
    from conftest import assert_ucrt_close

    assert_ucrt_close(np.asarray(ours, dtype=np.float32), ref.astype(np.float32))


def _ulps(ours, ref):
    a = np.asarray(ours, dtype=np.float32)
    b = ref.astype(np.float32)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    return int(np.max(np.abs(a[ok].view(np.int32).astype(np.int64) - b[ok].view(np.int32).astype(np.int64)),
                      initial=0))


# --------------------------------------------------------------------------- carbon

def _pools(inputs):
    p = inputs["pools"]
    return pd.DataFrame({"lucode": inputs["codes"], "c_above": p[:, 0], "c_below": p[:, 1], "c_soil": p[:, 2],
                         "c_dead": p[:, 3]})


@pytest.mark.parametrize("name", ["carbon_storage", "carbon_sequestration", "carbon_valuation",
                                  "carbon_valuation_flat", "carbon_minus_one_patched", "carbon_grids"])
def test_carbon_matches_invest(name):
    inputs, ref, meta = _case(name)
    kw = {}
    if "alt" in inputs:
        kw["future"] = _raster(inputs["alt"], meta["nodata"], _gt(meta, "alt"))
    v = meta["valuation"]
    if v:
        kw.update(price=v["price"], discount_rate=v["discount"], rate_change=v["change"], year=v["years"][0],
                  future_year=v["years"][1])
    c = ls.services.carbon(_raster(inputs["bas"], meta["nodata"]), _pools(inputs), **kw)
    _same(c.storage, ref["c_storage_bas"])
    for pool in ("c_above", "c_below", "c_soil", "c_dead"):
        _same(c.pools[pool], ref[f"{pool}_bas"])
    if "alt" in inputs:
        _on_invest_grid(c.storage_future, ref["c_storage_alt"], meta)
        _same(c.storage_future, ref["c_storage_alt"])
        _same(c.change, ref["c_change_bas_alt"])
        for pool in ("c_above", "c_below", "c_soil", "c_dead"):
            _same(c.pools_future[pool], ref[f"{pool}_alt"])
    if v:
        _same(c.npv, ref["npv_alt"])
    totals = dict(zip(c.totals["label"], c.totals["total"]))
    for row in meta["summary"]:
        assert totals[row["label"]] == pytest.approx(row["total"], rel=1e-12, abs=1e-9)
    if meta.get("patched"):
        assert np.any(c.change.values == -1.0)


def test_carbon_valuation_constant_and_errors():
    assert ls.services.valuation_constant(10.0, 0.0, 0.0, 2000, 2010) == 10.0
    with pytest.raises(ValueError, match="after"):
        ls.services.valuation_constant(10.0, 5.0, 0.0, 2010, 2010)
    lulc = _raster(np.array([[1, 2], [2, 9]]))
    pools = {1: {"c_above": 1, "c_below": 1, "c_soil": 1, "c_dead": 1},
             2: {"c_above": 2, "c_below": 2, "c_soil": 2, "c_dead": 2}}
    with pytest.raises(ValueError, match=r"\[9\]"):
        ls.services.carbon(lulc, pools)
    with pytest.raises(ValueError, match="future"):
        ls.services.carbon(_raster(np.array([[1, 2], [2, 1]])), pools, price=10.0)


def test_carbon_threads_give_the_same_bits():
    inputs, ref, meta = _case("carbon_valuation")
    args = (_raster(inputs["bas"], meta["nodata"]), _pools(inputs))
    kw = dict(future=_raster(inputs["alt"], meta["nodata"]), price=43.0, discount_rate=7.0, rate_change=2.5,
              year=2000, future_year=2030)
    a = ls.services.carbon(*args, **kw, n_threads=1)
    b = ls.services.carbon(*args, **kw, n_threads=3)
    np.testing.assert_array_equal(a.npv.values, b.npv.values)
    pd.testing.assert_frame_equal(a.totals, b.totals)


# --------------------------------------------------------------------------- habitat quality

def _hq_inputs(inputs, meta, scenario):
    tables = meta["threats"]
    maps = {}
    for t in tables:
        a = inputs[f"threat_{t['threat']}_{scenario}"].astype(np.float64)
        nd = meta["threat_nodata"].get(t["threat"])
        if nd is not None:
            a = np.where(a == nd, np.nan, a)
        maps[t["threat"]] = _raster(a, gt=_gt(meta, f"{t['threat']}_{scenario}"))
    return maps


def _sensitivity(inputs, meta):
    names = [t["threat"] for t in meta["threats"]]
    df = pd.DataFrame({"lucode": inputs["codes"], "habitat": inputs["habitat"],
                       **{n: inputs["sensitivity"][:, k] for k, n in enumerate(names)}})
    if "sensitivity_extra" in inputs:
        e = inputs["sensitivity_extra"]
        df.loc[len(df)] = [int(e[0]), *e[1:]]
    return df


def _run_hq(name, **extra):
    inputs, ref, meta = _case(name)
    threats = pd.DataFrame(meta["threats"])
    kw = {}
    if "fut" in inputs:
        kw.update(future=_raster(inputs["fut"], meta["nodata"], _gt(meta, "fut")),
                  future_threat_maps=_hq_inputs(inputs, meta, "fut"))
    if "bas" in inputs:
        kw["base"] = _raster(inputs["bas"], meta["nodata"], _gt(meta, "bas"))
    if meta["polygons"]:
        import geopandas as gpd
        from shapely.geometry import Polygon

        kw["access"] = gpd.GeoDataFrame({"ACCESS": [p["access"] for p in meta["polygons"]]},
                                        geometry=[Polygon(p["ring"]) for p in meta["polygons"]], crs=CRS)
    hq = ls.services.habitat_quality(_raster(inputs["cur"], meta["nodata"]), threats, _sensitivity(inputs, meta),
                                     _hq_inputs(inputs, meta, "cur"), half_saturation=meta["k"], **kw, **extra)
    return hq, ref, meta


@pytest.mark.parametrize("name", ["hq_current", "hq_scenarios", "hq_access", "hq_empty_threat_patched", "hq_grids"])
@pytest.mark.external_reference
def test_habitat_quality_matches_invest(name):
    hq, ref, meta = _run_hq(name)
    _on_invest_grid(hq.quality["current"], ref["quality_c"], meta)
    for sc, s in (("current", "cur"), ("future", "fut")):
        if f"deg_sum_{s[0]}" not in ref:
            continue
        for t in meta["threats"]:
            ours, theirs = hq.threats[sc][t["threat"]], ref[f"degradation_[THREAT]_{s}/{t['threat']}"]
            _same(ours, theirs)
        _same(hq.degradation[sc], ref[f"deg_sum_{s[0]}"])
        _same(hq.quality[sc], ref[f"quality_{s[0]}"])
    if "rarity_c" in ref:
        _same(hq.rarity["current"], ref["rarity_c"])
        _same(hq.rarity["future"], ref["rarity_f"])
        for key, sc in (("rarity_c_csv", "current"), ("rarity_f_csv", "future")):
            rows = hq.rarity_table[hq.rarity_table["scenario"] == sc]
            expected = np.array(meta["rarity"][key])
            np.testing.assert_array_equal(rows["lucode"].to_numpy(), expected[:, 0])
            np.testing.assert_array_equal(rows["rarity"].to_numpy(), expected[:, 1])


def test_linear_threats_are_bit_for_bit():
    hq, ref, meta = _run_hq("hq_current")
    lin = [t["threat"] for t in meta["threats"] if t["decay"] == "linear"]
    total = sum(hq.threats["current"][t].values for t in lin)
    assert np.nanmax(total) > 0
    for t in lin:
        _same(hq.threats["current"][t], ref[f"degradation_[THREAT]_cur/{t}"])


def test_a_threat_without_cells_degrades_nothing():
    """InVEST degrades every cell from a threat with no cell when max_dist exceeds the half perimeter."""
    hq, ref, meta = _run_hq("hq_empty_threat")
    assert np.nanmax(ref["degradation_[THREAT]_cur/urban"]) > 0          # InVEST's error
    assert np.nanmax(hq.threats["current"]["urban"].values) == 0


def test_distance_transform_is_exact():
    rng = np.random.default_rng(3)
    region = (rng.random((50, 61)) < 0.02).astype(np.uint8)
    d = _core.serv_distance_transform(region, 3)
    ys, xs = np.nonzero(region)
    yy, xx = np.mgrid[:50, :61]
    exact = np.sqrt(((yy[..., None] - ys) ** 2 + (xx[..., None] - xs) ** 2).min(axis=-1)).astype(np.float32)
    np.testing.assert_array_equal(d, exact)
    assert np.all(np.isinf(_core.serv_distance_transform(np.zeros((4, 5), np.uint8), 1)))


def test_base_degradation_is_the_current_model_on_the_base_map():
    inputs, ref, meta = _case("hq_scenarios")
    threats = pd.DataFrame(meta["threats"])
    sens = _sensitivity(inputs, meta)
    maps = _hq_inputs(inputs, meta, "cur")
    both = ls.services.habitat_quality(_raster(inputs["cur"], meta["nodata"]), threats, sens, maps,
                                       half_saturation=0.3, base=_raster(inputs["bas"], meta["nodata"]),
                                       base_threat_maps=maps)
    alone = ls.services.habitat_quality(_raster(inputs["bas"], meta["nodata"]), threats, sens, maps,
                                        half_saturation=0.3)
    np.testing.assert_array_equal(both.quality["base"].values, alone.quality["current"].values)


def test_habitat_quality_checks_its_inputs():
    lulc = _raster(np.array([[1, 2], [2, 1]]))
    threats = pd.DataFrame({"threat": ["r"], "max_dist": [60.0], "weight": [1.0], "decay": ["linear"]})
    sens = pd.DataFrame({"lucode": [1, 2], "habitat": [1.0, 0.5], "r": [1.0, 0.0]})
    road = _raster(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="from 0 to 1"):
        ls.services.habitat_quality(lulc, threats, sens, {"r": _raster(np.array([[0.0, 2.0], [0, 0]]))},
                                    half_saturation=0.1)
    with pytest.raises(ValueError, match="no current map"):
        ls.services.habitat_quality(lulc, threats, sens, {}, half_saturation=0.1)
    with pytest.raises(ValueError, match="decay"):
        ls.services.habitat_quality(lulc, threats.assign(decay="gaussian"), sens, {"r": road}, half_saturation=0.1)
    with pytest.raises(ValueError, match="lacks the column"):
        ls.services.habitat_quality(lulc, threats, sens.drop(columns="r"), {"r": road}, half_saturation=0.1)
    hq = ls.services.habitat_quality(lulc, threats, sens, {"r": road}, half_saturation=0.1, n_threads=1)
    assert hq.quality["current"].rio.crs is not None


def test_align_inputs_keeps_the_grid_and_marks_cells_outside():
    from landschaft.services._common import align_inputs

    inputs, ref, meta = _case("wy_grids")
    lulc = _raster(inputs["lulc"], meta["nodata"])
    depth = _raster(inputs["depth"], meta["input_nodata"])
    out = align_inputs({"lulc": (lulc, "near"), "depth": (depth, "near"), "none": (None, "near")}, reference="lulc")
    assert out["lulc"] is lulc and out["depth"] is depth and "none" not in out  # on the grid: no copy
    # a raster without nodata over less than the reference: no value outside it (GDAL writes 0 there)
    small = _raster(np.ones((10, 12), np.float32), gt=(ORIGIN[0] + 7.0, 20.0, 0.0, ORIGIN[1] - 3.0, 0.0, -20.0))
    out = align_inputs({"lulc": (lulc, "near"), "small": (small, "bilinear")}, reference="lulc", bbox="reference")
    a = out["small"].values
    assert a.shape == lulc.shape and out["small"].rio.transform() == lulc.rio.transform()
    assert np.isnan(a[-1, -1]) and a[2, 2] == 1.0
    far = _raster(np.ones((5, 5), np.float32), gt=(ORIGIN[0] + 1e6, 30.0, 0.0, ORIGIN[1], 0.0, -30.0))
    with pytest.raises(ValueError, match="do not overlap"):
        align_inputs({"lulc": (lulc, "near"), "far": (far, "near")}, reference="lulc")


def test_results_save_and_load(tmp_path):
    hq, ref, meta = _run_hq("hq_scenarios")
    parts = ls.load_result(hq.save(tmp_path / "hq"))
    np.testing.assert_array_equal(parts["quality_current"].values, hq.quality["current"].values)
    assert set(parts["rarity"]["scenario"]) == {"current", "future"}
