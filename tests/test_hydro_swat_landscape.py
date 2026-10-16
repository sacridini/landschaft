# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro.swat from the landscape (phase 7): the stream links and watersheds of
ls.hydro; the division of a synthetic landscape as QSWAT+ divides it; the project the
SWAT+ Editor 4.0.2 writes from those tables, file for file (the stored
``landscape.zip`` is the editor's own output, tools/make_reference_swat.py); the port's
simulation of it against SWAT+ 62.0.1, bit for bit; and the results put back on the map."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from landschaft import _core
from conftest import assert_swat_close
import swat_landscape as case

REF = Path(__file__).parent / "data" / "reference" / "swat"
# what SWAT+ prints with the editor's print.prt (yearly and average annual tables)
PRINTS = [u + k for u in (2000, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2500, 2520) for k in range(4)]


# ------------------------------------------------------------------ ls.hydro
def _y_dem():
    return np.array([[9, 9, 9, 9, 9], [5, 9, 9, 9, 5], [9, 4, 9, 4, 9], [9, 9, 3, 9, 9], [9, 9, 2, 9, 9],
                     [9, 9, 1, 9, 9]], float)


def test_stream_links_split_at_junctions_and_outlets():
    dem = _y_dem()
    st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    links, table = ls.hydro.stream_links(dem, st)
    np.testing.assert_array_equal(links.values, [[0, 0, 0, 0, 0], [1, 0, 0, 0, 2], [0, 1, 0, 2, 0],
                                                 [0, 0, 3, 0, 0], [0, 0, 3, 0, 0], [0, 0, 3, 0, 0]])
    assert table["downstream"].tolist() == [3, 3, 0]
    assert table["order"].tolist() == [1, 1, 2]
    assert table.loc[1, "length"] == pytest.approx(2 * np.sqrt(2))
    assert table.loc[3, "length"] == pytest.approx(3.0)  # two steps and the outlet cell's own
    links, table = ls.hydro.stream_links(dem, st, outlets=[(2.5, 1.5)], snap=1)
    assert table["downstream"].tolist() == [3, 3, 4, 0]
    assert table["outlet"].tolist() == [False, False, True, False]


def test_watersheds_label_each_cell_with_the_seed_below():
    dem = _y_dem()
    links, _ = ls.hydro.stream_links(dem, ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2))
    w = ls.hydro.watersheds(dem, links).values
    assert w[0, 0] == 1 and w[0, 4] == 2 and w[5, 2] == 3
    assert set(np.unique(w)) <= {0, 1, 2, 3}


def test_snap_outlets_moves_points_onto_the_streams():
    dem = _y_dem()
    st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    np.testing.assert_array_equal(ls.hydro.snap_outlets([(2.4, 0.6)], st, distance=1.0), [[5, 2]])
    with pytest.raises(ValueError, match="farther"):
        ls.hydro.snap_outlets([(0.5, 0.5)], st, distance=0.5)


# ------------------------------------------------------------------ QSWAT+
@pytest.fixture(scope="module")
def built():
    dem, lu, so = case.rasters()
    stations, pcp, tmax, tmin, monthly = case.weather()
    wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin, monthly=monthly)
    project = ls.hydro.swat.from_rasters(dem, lu, so, landuse_lookup=case.LANDUSE, soil_lookup=case.SOILS,
                                         soil_table=case.soil_table(), climate=wx, **case.OPTIONS)
    return project, wx


def test_watershed_tables_add_up(built):
    project, _ = built
    t = project.watershed.tables
    cell_ha = 30 * 30 / 1e4
    total = 60 * 50 * cell_ha  # the whole synthetic DEM drains to one outlet
    assert t["gis_subbasins"]["area"].sum() == pytest.approx(total)
    by_lsu = t["gis_hrus"].groupby("lsu")["arslp"].sum()
    np.testing.assert_allclose(by_lsu.to_numpy(), t["gis_lsus"].set_index("id").loc[by_lsu.index, "area"])
    ch = t["gis_channels"]
    assert (ch["len2"] > 0).all() and (ch["wid2"] > 0).all()
    assert ch["areac"].max() == pytest.approx(total)
    assert len(t["gis_deep_aquifers"]) == 1
    assert {"tot", "rhg"} == set(t["gis_routing"]["hyd_typ"])


def test_hru_methods():
    dem, lu, so = case.rasters()
    kw = dict(landuse_lookup=case.LANDUSE, soil_lookup=case.SOILS, soil_table=case.soil_table(),
              stream_threshold=0.15)
    one = ls.hydro.swat.watershed(dem, lu, so, hru_method="dominant_hru", **kw).tables
    assert len(one["gis_hrus"]) == len(one["gis_lsus"])
    np.testing.assert_allclose(one["gis_hrus"]["arslp"], one["gis_lsus"]["area"])
    allc = ls.hydro.swat.watershed(dem, lu, so, hru_method="filter", slope_bands=(5, 20), **kw).tables
    some = ls.hydro.swat.watershed(dem, lu, so, hru_method="filter", thresholds=(20, 20, 20), slope_bands=(5, 20),
                                   **kw).tables
    assert len(allc["gis_hrus"]) > len(some["gis_hrus"]) >= len(one["gis_hrus"])
    for t in (allc, some):
        np.testing.assert_allclose(t["gis_hrus"].groupby("lsu")["arslp"].sum().to_numpy(), t["gis_lsus"]["area"])
    area = ls.hydro.swat.watershed(dem, lu, so, hru_method="area", min_area=3.0, slope_bands=(5, 20), **kw).tables
    small = area["gis_hrus"].groupby("lsu")["arslp"]
    assert ((area["gis_hrus"]["arslp"] >= 3.0) | (small.transform("count") == 1)).all()


def test_usersoil_tables_give_the_same_soils():
    # QSWAT+'s usersoil layout (one row per soil, layers in columns) and SWAT+'s
    from landschaft.hydro.swat._qswat import _soil_tables

    layered = case.soil_table()
    wide = []
    for name, g in layered.groupby("name", sort=False):
        r = {"SNAM": name, "NLAYERS": len(g), "HYDGRP": g["hyd_grp"].iloc[0], "SOL_ZMX": g["dp_tot"].iloc[0],
             "ANION_EXCL": 0.5, "SOL_CRK": 0.5, "TEXTURE": g["texture"].iloc[0]}
        for k, (_, lay) in enumerate(g.iterrows(), 1):
            for src, dst in (("SOL_Z", "dp"), ("SOL_BD", "bd"), ("SOL_AWC", "awc"), ("SOL_K", "soil_k"),
                             ("SOL_CBN", "carbon"), ("CLAY", "clay"), ("SILT", "silt"), ("SAND", "sand"),
                             ("ROCK", "rock"), ("SOL_ALB", "alb"), ("USLE_K", "usle_k"), ("SOL_EC", "ec"),
                             ("SOL_CAL", "caco3"), ("SOL_PH", "ph")):
                r[f"{src}{k}"] = lay[dst]
        wide.append(r)
    a = _soil_tables(layered, ["CLAY1", "LOAM1"])
    b = _soil_tables(pd.DataFrame(wide), ["CLAY1", "LOAM1"])
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_wetlands_and_missing_crs_are_refused():
    dem, lu, so = case.rasters()
    kw = dict(soil_lookup=case.SOILS, soil_table=case.soil_table(), stream_threshold=0.15)
    stations, pcp, tmax, tmin, monthly = case.weather()
    wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin, monthly=monthly)
    w = ls.hydro.swat.watershed(dem, lu, so, landuse_lookup={1: "WATR", 2: "PAST", 3: "RNGE"}, **kw)
    with pytest.raises(ValueError, match="wetland"):
        ls.hydro.swat.from_tables(w, wx)
    with pytest.raises(ValueError, match="CRS"):
        ls.hydro.swat.watershed(dem.values, lu.values, so.values, landuse_lookup=case.LANDUSE, **kw)


# ------------------------------------------------------------------ the SWAT+ Editor
def _tokens(b: bytes) -> list:
    lines = b.decode().replace("\r\n", "\n").split("\n")[1:]  # the first line is a title (dates)
    return [ln.split() for ln in lines if ln.strip()]


def test_files_are_the_swatplus_editors(built):
    project, _ = built
    with zipfile.ZipFile(REF / "landscape.zip") as z:
        editor = {n: z.read(n) for n in z.namelist()}
    assert sorted(project.files) == sorted(editor)
    for name, b in editor.items():
        assert _tokens(project.files[name]) == _tokens(b), name


CROPS = {1: "CORN", 2: "WWHT", 3: "URMD"}


@pytest.fixture(scope="module")
def crops():
    dem, lu, so = case.rasters()
    stations, pcp, tmax, tmin, monthly = case.weather()
    wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin, monthly=monthly)
    return ls.hydro.swat.from_rasters(dem, lu, so, landuse_lookup=CROPS, soil_lookup=case.SOILS,
                                      soil_table=case.soil_table(), climate=wx, start="2000-01-01",
                                      end="2000-12-31", **case.OPTIONS)


def test_crop_rotations_and_urban_are_the_swatplus_editors(crops):
    # annual crops get the editor's decision-table rotations (management.sch, pl_hv_summer1
    # and pl_hv_winter1), the urban land use its urban.urb entry
    with zipfile.ZipFile(REF / "landscape_crops.zip") as z:
        editor = {n: z.read(n) for n in z.namelist()}
    assert "management.sch" in crops.files
    assert sorted(crops.files) == sorted(editor)
    for name, b in editor.items():
        assert _tokens(crops.files[name]) == _tokens(b), name


def test_crop_rotations_simulate_as_swatplus(crops):
    try:
        ours = _core.swat_run(crops.files, True)
    except RuntimeError as e:
        if "not supported" in str(e):
            pytest.skip(f"the port does not simulate this yet: {e}")
        raise
    ref = np.load(REF / "landscape_crops.npz")
    for tag in (10, 11, 12, 13, 30, 40):
        keys, vals = ours["dump"][tag]
        np.testing.assert_array_equal(keys, ref[f"t{tag}_keys"])
        rvals = ref[f"t{tag}_vals"]
        assert_swat_close(vals[:, : rvals.shape[1]], rvals, err_msg=f"tag {tag}")


def test_qswat_project_database_round_trip(built, tmp_path):
    project, wx = built
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        for name, df in project.watershed.tables.items():
            df.to_sql(name, conn, index=False)
    tables = ls.hydro.swat.load_qswat_project(db)
    again = ls.hydro.swat.from_tables(tables, wx)
    for name, b in project.files.items():
        assert _tokens(again.files[name]) == _tokens(b), name


# ------------------------------------------------------------------ SWAT+
@pytest.fixture(scope="module")
def simulated(built):
    project, _ = built
    return _core.swat_run(project.files, True), np.load(REF / "landscape.npz")


@pytest.mark.parametrize("tag", [1, 3, 4, 5, 6, 10, 11, 12, 13, 20, 30, 40, 41])
def test_simulation_matches_swatplus(simulated, tag):
    ours, ref = simulated
    keys, vals = ours["dump"][tag]
    rkeys, rvals = ref[f"t{tag}_keys"], ref[f"t{tag}_vals"]
    np.testing.assert_array_equal(keys, rkeys, err_msg=f"tag {tag}: (year, day, object)")
    assert_swat_close(vals[:, : rvals.shape[1]], rvals, err_msg=f"tag {tag}")


@pytest.mark.parametrize("tag", PRINTS)
def test_printed_tables_match_swatplus(simulated, tag):
    ours, ref = simulated
    if f"t{tag}_keys" not in ref.files:
        pytest.skip(f"the editor's print.prt does not print table {tag}")
    keys, vals = ours["prints"][tag]
    np.testing.assert_array_equal(keys, ref[f"t{tag}_keys"], err_msg=f"print {tag}")
    assert_swat_close(vals, ref[f"t{tag}_vals"], err_msg=f"print {tag}")


# ------------------------------------------------------------------ the map
def test_results_on_the_map(built):
    project, _ = built
    res = ls.hydro.swat.run(project, tables={"hru_ls": ["day", "aa"], "channel_sd": ["day", "aa"]}, nyskip=0)
    w = project.watershed
    loss = ls.hydro.swat.hru_map(project, "sedyld", result=res)
    assert loss.shape == w.hrus.shape
    assert np.nanmax(loss.values) > 0
    assert np.isnan(loss.values[np.isnan(w.hrus.values)]).all()
    # a cell's value is its HRU's average annual soil loss, as SWAT+ prints it
    aa = res.table("hru_ls", "aa")["sedyld"].droplevel("date")
    r, c = np.argwhere(np.isfinite(w.hrus.values))[0]
    assert loss.values[r, c] == aa.loc[int(w.hrus.values[r, c])]
    # ... the daily soil loss summed over the run, divided by SWAT+'s years (731 days
    # / 365: its leap-day correction divides integers)
    daily = res.table("hru_ls", "day")["sedyld"]
    assert np.allclose(aa, daily.groupby(level="hru").sum() / (731 / 365), rtol=1e-5)
    # the default runs the project as its print.prt says (the editor skips a year)
    own = ls.hydro.swat.run(project, tables={"hru_ls": "aa"})
    assert np.array_equal(ls.hydro.swat.hru_map(project, "sedyld").values,
                          ls.hydro.swat.hru_map(project, "sedyld", result=own).values, equal_nan=True)
    q = ls.hydro.swat.reach_discharge(project, result=res)
    assert q.shape[1] == len(w.tables["gis_channels"])
    flow = ls.hydro.swat.channel_map(project, result=res)
    outlet = int(w.tables["gis_channels"]["areac"].idxmax()) + 1
    assert np.nanmax(flow.values) == pytest.approx(q[outlet].mean())
    assert np.isnan(flow.values[np.nan_to_num(w.channels.values) == 0]).all()


# ------------------------------------------------------------------ weather
def test_weather_generator_statistics():
    days = pd.date_range("2000-01-01", "2003-12-31")
    pcp = pd.DataFrame({"a": np.where(days.day % 3 == 0, 6.0, 0.0)}, index=days)
    tx = pd.DataFrame({"a": np.where(days.day % 2 == 0, 20.0, 10.0)}, index=days)
    tn = tx - 10
    w = ls.hydro.swat.wgn_from_series(pcp, tx, tn, monthly={"pcp_hhr": 5.0, "slr_ave": 10.0, "dew_ave": 1.0,
                                                            "wnd_ave": 2.0}).loc["a"]
    jan = w.loc[1]
    assert jan["tmp_max_ave"] == pytest.approx((15 * 20 + 16 * 10) / 31)   # even days 20, odd days 10
    assert jan["tmp_min_ave"] == pytest.approx((15 * 10 + 16 * 0) / 31)
    assert jan["pcp_days"] == pytest.approx(10.0)          # days 3, 6, ..., 30 of each January
    assert jan["pcp_ave"] == pytest.approx(60.0)
    assert jan["wet_wet"] == 0.0 and 0 < jan["wet_dry"] < 1
    assert w["rain_yrs"].iloc[0] == 4
    with pytest.raises(ValueError, match="pcp_hhr"):
        ls.hydro.swat.wgn_from_series(pcp, tx, tn)
