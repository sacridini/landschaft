# SPDX-License-Identifier: GPL-3.0-or-later
"""Crop pollination (ls.services.pollination) against InVEST's outputs
(tools/make_reference_invest_pollination.py)."""

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


def _raster(a, nodata=None):
    ny, nx = a.shape
    x = ORIGIN[0] + (np.arange(nx) + 0.5) * PIXEL
    y = ORIGIN[1] - (np.arange(ny) + 0.5) * PIXEL
    da = xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs(CRS).rio.write_transform(Affine(PIXEL, 0, ORIGIN[0], 0, -PIXEL, ORIGIN[1]))
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def _same(ours, ref, what=""):
    """The same bits in InVEST's type (float32 or float64), NaN where InVEST has nodata;
    off Windows within UCRT_RTOL (the kernels' exp is the UCRT's there)."""
    from conftest import assert_ucrt_close

    ref = np.asarray(ref)
    assert_ucrt_close(np.asarray(ours, dtype=ref.dtype), ref, err_msg=what)


def _farms_gdf(meta):
    import geopandas as gpd
    from shapely.geometry import Polygon

    rows = [{k: v for k, v in f.items() if k != "ring"} for f in meta["farms"]]
    return gpd.GeoDataFrame(rows, geometry=[Polygon(f["ring"]) for f in meta["farms"]], crs=CRS)


def _run(name, **kw):
    inputs, ref, meta = _case(name)
    lulc = _raster(inputs["lulc"], meta["nodata"])
    guilds = pd.DataFrame(meta["guild"])
    bio = pd.DataFrame(meta["bio"])
    farms = _farms_gdf(meta) if "farms" in meta else None
    res = ls.services.pollination(lulc, guilds, bio, farms=farms, intermediates=True, **kw)
    return res, ref, meta


def _ours(res, farms: bool) -> dict:
    """Our maps under InVEST's file names."""
    out = {}
    for sp, da in res.supply.items():
        out[f"pollinator_supply_{sp}"] = da
    for sp, d in res.abundance.items():
        for se, da in d.items():
            out[f"pollinator_abundance_{sp}_{se}"] = da
    for se, da in res.total_abundance.items():
        out[f"total_pollinator_abundance_{se}"] = da
    pre = "farm_" if farms else ""
    for s, da in res.nesting.items():
        out[f"{pre}nesting_substrate_index_{s}"] = da
    for s, da in res.floral.items():
        out[f"{pre}relative_floral_abundance_index_{s}"] = da
    for name, d in (("habitat_nesting_index", res.habitat_nesting),
                    ("local_foraging_effectiveness", res.foraging_effectiveness),
                    ("floral_resources", res.floral_resources), ("convolve_ps", res.convolved_supply),
                    ("half_saturation", res.half_saturation), ("farm_pollinator", res.farm_abundance)):
        for k, da in d.items():
            out[f"{name}_{k}"] = da
    for sp, d in res.foraged_flowers.items():
        for se, da in d.items():
            out[f"foraged_flowers_index_{sp}_{se}"] = da
    for k, v in (("farm_pollinators", res.farm_pollinators), ("total_pollinator_yield", res.total_yield),
                 ("wild_pollinator_yield", res.wild_yield), ("managed_pollinators", res.managed)):
        if v is not None:
            out[k] = v
    return out


@pytest.mark.parametrize("name", ["pollination_guilds", "pollination_farms", "pollination_blocks",
                                  "pollination_wide_kernel", "pollination_faint"])
def test_pollination_matches_invest(name):
    res, ref, meta = _run(name)
    ours = _ours(res, "farms" in meta)
    keys = [k for k in ref if not k.startswith("kernel_")]
    if "farms" in meta:  # InVEST's maps before the farms are burnt in
        keys = [k for k in keys if not k.startswith(("nesting_substrate_index_", "relative_floral_abundance_index_"))]
    assert keys
    for key in keys:
        assert key in ours, key
        _same(ours[key].values, ref[key], key)


@pytest.mark.parametrize("name", ["pollination_guilds", "pollination_farms"])
def test_pollination_kernels_match_invest(name):
    res, ref, meta = _run(name)
    alphas = dict(zip(meta["guild"]["species"], meta["guild"]["alpha"]))
    for sp, kernel in res.kernels.items():
        a = np.float64(alphas[sp]) / PIXEL
        _same(kernel, ref[f"kernel_{a:.6f}"], sp)


def test_pollination_farm_results_match_invest():
    res, _, meta = _run("pollination_farms")
    farms = res.farms
    for i, row in enumerate(meta["farm_results"]):
        for f in ("p_abund", "y_tot", "pdep_y_w", "y_wild"):
            want = row[f]
            got = farms[f].iloc[i]
            if want is None:
                assert np.isnan(got), (i, f)
            else:
                # InVEST sums the farm's cells in float32 and writes 11 decimals
                assert got == pytest.approx(want, rel=1e-6, abs=1e-10), (i, f)


def test_pollination_unpatched_clean_up_is_invests_bug():
    """InVEST as it is zeroes the block convolutions' values from -1e-13 to 2e-8 (within
    1e-8 of 1e-8) instead of within 1e-8 of 0: with faint flowers, convolutions of about
    1e-8 at the edges of the flowering patches, it loses those; ours (the documented
    clean-up) is InVEST's with that corrected, and differs from InVEST as it is only by
    values of that size."""
    res, ref, _ = _run("pollination_faint_unpatched")
    _, patched, _ = _case("pollination_faint")
    ours = _ours(res, False)
    differs = 0
    for key, r in ref.items():
        ok = ~np.isnan(r)
        np.testing.assert_array_equal(np.isnan(np.asarray(ours[key].values, np.float64)), ~ok, err_msg=key)
        d = np.abs(np.asarray(ours[key].values, np.float64)[ok] - r[ok].astype(np.float64))
        assert d.max(initial=0) < 1e-7, key
        differs += int(np.count_nonzero(patched[key][ok] != r[ok]))
    assert differs > 0


def test_pollination_threads_and_without_intermediates():
    inputs, _, meta = _case("pollination_farms")
    lulc = _raster(inputs["lulc"], meta["nodata"])
    args = (lulc, pd.DataFrame(meta["guild"]), pd.DataFrame(meta["bio"]))
    one = ls.services.pollination(*args, farms=_farms_gdf(meta), n_threads=1)
    many = ls.services.pollination(*args, farms=_farms_gdf(meta), n_threads=4, intermediates=True)
    assert not one.floral_resources and many.floral_resources
    for se in one.total_abundance:
        np.testing.assert_array_equal(one.total_abundance[se].values, many.total_abundance[se].values)
    np.testing.assert_array_equal(one.total_yield.values, many.total_yield.values)
    pd.testing.assert_frame_equal(pd.DataFrame(one.farms.drop(columns="geometry")),
                                  pd.DataFrame(many.farms.drop(columns="geometry")))


def test_convolution_kernel_sum_and_flat_signal():
    """A flat signal convolved with edges ignored is the flat value times the kernel's sum."""
    k = _core.serv_poll_kernel(3.0, 15.0)
    assert k.shape == (31, 31) and k.dtype == np.float32
    assert abs(float(k.sum(dtype=np.float64)) - 1.0) < 1e-6
    sig = np.full((300, 270), 0.25, np.float32)
    sig[40:60, 100:130] = np.nan
    out = _core.serv_poll_convolve(sig, k, 2)
    assert np.isnan(out[50, 110]) and not np.isnan(out[0, 0])
    np.testing.assert_allclose(out[~np.isnan(sig)], 0.25 * float(np.sum(k)), rtol=1e-6)


def test_pollination_validation():
    inputs, _, meta = _case("pollination_guilds")
    lulc = _raster(inputs["lulc"], meta["nodata"])
    guilds, bio = pd.DataFrame(meta["guild"]), pd.DataFrame(meta["bio"])
    with pytest.raises(ValueError, match="seasons"):
        ls.services.pollination(lulc, guilds, bio.drop(columns="floral_resources_fall_index"))
    with pytest.raises(ValueError, match="alpha"):
        ls.services.pollination(lulc, guilds.drop(columns="alpha"), bio)
    with pytest.raises(ValueError, match="lacks"):
        ls.services.pollination(lulc, guilds, bio[bio.lucode != 4])
    import geopandas as gpd
    from shapely.geometry import box

    farms = gpd.GeoDataFrame({"crop_type": ["a"], "season": ["winter"], "half_sat": [0.5], "p_dep": [1.0],
                              "p_managed": [0.0], "fr_fall": [1.0], "fr_spring": [1.0], "n_cavity": [1.0],
                              "n_ground": [1.0], "n_stem": [1.0]},
                             geometry=[box(500000, 7999000, 501000, 8000000)], crs=CRS)
    with pytest.raises(ValueError, match="winter"):
        ls.services.pollination(lulc, guilds, bio, farms=farms)


def test_pollination_save(tmp_path):
    res, _, _ = _run("pollination_farms")
    folder = res.save(tmp_path / "poll")
    names = {p.name for p in folder.iterdir()}
    assert "total_pollinator_abundance_spring.tif" in names and "farm_results.gpkg" in names
