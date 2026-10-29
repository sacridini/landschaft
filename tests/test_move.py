# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.move against R's move2 0.5.0 (tests/move_cases.py, tools/make_reference_move.py)."""

import platform
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely import Point

import landschaft as ls
from landschaft import _core
from conftest import BITWISE, UCRT_LIBM

REF = Path(__file__).parent / "data" / "reference" / "move"
FISHERS = Path(__file__).parent / "data" / "move" / "fishers.csv.gz"

# R made the references on Windows with an Intel CPU, whose x87 unit gives the bits of sin, cos and atan2
# (conftest.BITWISE) and whose UCRT gives asin's: there the steps on the sphere and the ellipsoid must be
# R's. AMD's x87 microcode rounds a few of those results otherwise (4 of the fishers' 32904 distances),
# ARM has no x87 and other systems another libm, so elsewhere the positions behind a step agree to a few
# units in the last place of the Earth's radius: METRES (1.5e-9 m seen) on lengths, METRES / length on
# angles.
EXACT = UCRT_LIBM and BITWISE and "GenuineIntel" in platform.processor()
METRES = 1e-8


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "move2.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs.npz"))


@pytest.fixture(scope="module")
def fishers():
    return ls.load_tracks(FISHERS, sort=False)


def track_set(inputs, name, nan_empty=True):
    x, y, t = inputs[f"{name}/x"], inputs[f"{name}/y"], inputs[f"{name}/t"]
    crs = str(inputs[f"{name}/crs"]) or None
    geom = [Point() if (np.isnan(a) or np.isnan(b)) else Point(a, b) for a, b in zip(x, y)]
    df = {"time": pd.to_datetime((t * 1e9).astype(np.int64), utc=True), "track": inputs[f"{name}/track"]}
    for key in ("v1", "v2"):
        if f"{name}/{key}" in inputs:
            df[key] = inputs[f"{name}/{key}"]
    return ls.move.as_tracks(gpd.GeoDataFrame(df, geometry=geom, crs=crs), "time", "track", sort=False)


def same(got, want, err=""):
    np.testing.assert_array_equal(np.asarray(got, float), np.asarray(want, float), err_msg=err)


def same_libm(got, want, err="", atol=1e-14):
    """Equal where the references were made (EXACT); elsewhere within ulps, or ``atol`` (an array: per
    value)."""
    if EXACT:
        same(got, want, err)
    else:
        got, want = np.asarray(got, float), np.asarray(want, float)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=err)
        bad = ~(np.abs(got - want) <= 1e-14 * np.abs(want) + atol) & ~np.isnan(want)
        assert not bad.any(), f"{err}: {bad.sum()} values differ, at {np.flatnonzero(bad)[:5]}: "                               f"{got[bad][:5]} vs {want[bad][:5]}"


def same_angle(got, want, length, err="", turn=False, period=2 * np.pi):
    """Angles (``period`` per circle) as same_libm, off EXACT within METRES / ``length`` (the steps'
    lengths, in metres); a ``turn`` angle takes the shorter of the steps on either side. Zero-length
    steps have no direction to compare."""
    length = np.asarray(length, float)
    if turn:
        length = np.fmin(np.fmin(np.r_[np.nan, length[:-1]], length), np.r_[length[1:], np.nan])
    if EXACT:
        same(got, want, err)
        return
    got, want = np.asarray(got, float), np.asarray(want, float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=err)
    d = np.abs(np.remainder(got - want + period / 2, period) - period / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tol = 1e-14 * np.abs(want) + 2 * METRES / length * (period / (2 * np.pi))
    bad = ~(d <= tol) & ~np.isnan(want) & (length > 0)
    assert not bad.any(), f"{err}: {bad.sum()} values differ, at {np.flatnonzero(bad)[:5]}: "                           f"{got[bad][:5]} vs {want[bad][:5]}"


@pytest.mark.parametrize("name", ["ll", "utm", "plain"])
def test_steps_are_move2s(ref, inputs, name):
    tr = track_set(inputs, name)
    case = f"measures_{name}"
    d = ls.move.distance(tr)
    if name == "ll":
        same_libm(d, ref[f"{case}/distance"], "distance", METRES)
    else:
        same(d, ref[f"{case}/distance"], "distance")
    if f"{case}/azimuth" in ref:
        same_angle(ls.move.azimuth(tr), ref[f"{case}/azimuth"], d, "azimuth")
        same_angle(ls.move.turn_angle(tr), ref[f"{case}/turnangle"], d, "turn angle", turn=True)
    same(ls.move.time_lags(tr), ref[f"{case}/time_lags"], "time lags")
    if f"{case}/speed" in ref:
        if name == "ll":
            same_libm(ls.move.speed(tr), ref[f"{case}/speed"], "speed", METRES / ls.move.time_lags(tr))
        else:
            same(ls.move.speed(tr), ref[f"{case}/speed"], "speed")


def test_the_fishers_steps(ref, fishers):
    tr = fishers[~fishers.geometry.is_empty]
    d = ls.move.distance(tr)
    same_libm(d, ref["measures_fishers/distance"], "distance", METRES)
    same_angle(ls.move.azimuth(tr), ref["measures_fishers/azimuth"], d, "azimuth")
    same_angle(ls.move.turn_angle(tr), ref["measures_fishers/turnangle"], d, "turn angle", turn=True)
    # the times have milliseconds: R holds them as seconds since 1970 in doubles (to ~1e-7 s), ls.move's
    # lags are exact differences of nanoseconds
    np.testing.assert_allclose(ls.move.time_lags(tr), ref["measures_fishers/time_lags"], rtol=0, atol=5e-7)
    np.testing.assert_allclose(ls.move.speed(tr), ref["measures_fishers/speed"], rtol=1e-7)
    same(ls.move.speed(tr), ls.move.distance(tr) / ls.move.time_lags(tr))


def test_units(ref, inputs):
    tr = track_set(inputs, "ll")
    if EXACT:
        np.testing.assert_allclose(ls.move.distance(tr, "km"), ref["km_deg/km"], rtol=2.3e-16)
        np.testing.assert_allclose(ls.move.azimuth(tr, "deg"), ref["km_deg/deg"], rtol=2.3e-16)
    else:
        d = ls.move.distance(tr)
        same_libm(ls.move.distance(tr, "km"), ref["km_deg/km"], "km", METRES / 1000)
        same_angle(ls.move.azimuth(tr, "deg"), ref["km_deg/deg"], d, "deg", period=360.0)
    lag = ls.move.time_lags(tr)
    np.testing.assert_allclose(ls.move.time_lags(tr, "h"), lag / 3600)


@pytest.mark.parametrize("threads", [1, 3])
def test_threads_give_the_same_bits(inputs, threads):
    tr = track_set(inputs, "ll")
    big = pd.concat([tr] * 300, ignore_index=True)
    big["track"] = np.repeat(np.arange(300), len(tr)).astype(str) + np.tile(tr["track"].to_numpy(), 300)
    big = ls.move.as_tracks(big, "time", "track", sort=False)
    one = ls.move.turn_angle(big, n_threads=1)
    same(ls.move.turn_angle(big, n_threads=threads), one)
    same(ls.move.distance(big, n_threads=threads), ls.move.distance(big, n_threads=1))


def test_r_modulus():
    for x1, x2 in [(5.5, 2.0), (-5.5, 2.0), (7.0, -3.0), (1e-300, 2 * np.pi), (-np.pi, 2 * np.pi), (np.nan, 2.0)]:
        want = np.nan if np.isnan(x1) else x1 - np.floor(x1 / x2) * x2
        got = _core.move_r_fmod(x1, x2)
        assert (np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-15
    assert _core.move_r_fdiv(-0.5, 2.0) == -1 and _core.move_r_fdiv(5.0, 2.0) == 2


def test_unique_is_move2s(ref, inputs):
    tr = track_set(inputs, "dup")
    for c in ("first", "last"):
        same(ls.move.unique(tr, c), ref[f"unique/{c}"], c)
    with pytest.warns(UserWarning, match="remaining duplicates"):
        same(ls.move.unique(tr, "subsets"), ref["unique/subsets"], "subsets")
    with pytest.warns(UserWarning, match="remaining duplicates"):
        same(ls.move.unique(tr, "subsets_equal"), ref["unique/subsets_equal"], "subsets_equal")
    same(ls.move.unique(tr, "sample", seed=7), ref["unique/sample"], "sample")
    assert len(ls.move.filter_unique(tr, "first")) == int(ref["unique/first"].sum())


@pytest.mark.parametrize("key,args", [("hour", {}), ("last", {"criterion": "last"}), ("m15", {"unit": "15 mins"}),
                                      ("h2", {"unit": "2 hours"}), ("h5", {"unit": "5 hours"}),
                                      ("day", {"unit": "day"}), ("d3", {"unit": "3 days"}),
                                      ("week", {"unit": "week"}), ("month", {"unit": "month"}),
                                      ("random", {"unit": "day", "criterion": "random", "seed": 3})])
def test_per_interval_is_move2s(ref, fishers, key, args):
    same(ls.move.per_interval(fishers, **args), ref[f"per_interval/{key}"], key)


def test_movebank_visible(ref, fishers):
    same(ls.move.movebank_visible(fishers), ref["visible/visible"])
    assert len(ls.move.filter_movebank_visible(fishers)) == int(ref["visible/visible"].sum())


def coords(tr):
    g = tr.geometry
    x = np.array([p.x if not p.is_empty else np.nan for p in g])
    y = np.array([p.y if not p.is_empty else np.nan for p in g])
    return x, y


def test_interpolating_empty_locations(ref, inputs):
    x, y = coords(ls.move.interpolate(track_set(inputs, "gap")))
    same_libm(x, ref["interpolate_empty/x"])
    same_libm(y, ref["interpolate_empty/y"])


def test_interpolating_at_times(ref, inputs):
    tr = track_set(inputs, "gap")
    t = tr["time"]
    times = pd.date_range(t.min(), t.max(), freq="45min")
    out = ls.move.interpolate(tr, times, max_time_lag=7200)
    x, y = coords(out)
    same(out["time"].astype("int64") / 1e9, ref["interpolate_times/t"])
    same_libm(x, ref["interpolate_times/x"])
    same_libm(y, ref["interpolate_times/y"])
    out = ls.move.interpolate(tr, "1 hour", omit=True)
    x, y = coords(out)
    same(out["time"].astype("int64") / 1e9, ref["interpolate_hour_omit/t"])
    same_libm(x, ref["interpolate_hour_omit/x"])
    same_libm(y, ref["interpolate_hour_omit/y"])


def test_interpolating_without_crs(ref, inputs):
    tr = track_set(inputs, "gap_plain")
    t = tr["time"]
    out = ls.move.interpolate(tr, pd.date_range(t.min(), t.max(), freq="20min"))
    x, y = coords(out)
    same(out["time"].astype("int64") / 1e9, ref["interpolate_plain/t"])
    same(x, ref["interpolate_plain/x"])
    same(y, ref["interpolate_plain/y"])


def test_aeqd_centre(ref, fishers):
    for kind in ("centroid", "center"):
        p = dict(w.lstrip("+").split("=") for w in ls.move.aeqd_crs(fishers, kind).srs.split() if "=" in w)
        np.testing.assert_array_equal([float(p["lon_0"]), float(p["lat_0"])], ref[f"aeqd/{kind}"], err_msg=kind)


def test_brownian_is_move2s(ref):
    b = ls.move.simulate_brownian([0, 1, 2.5, 4, 8], [1, 2, 0, 0.5], 3, seed=11)
    same(b.geometry.x, ref["brownian/x1"])
    same(b.geometry.y, ref["brownian/y1"])
    d = ls.move.simulate_brownian(seed=5)
    same(d.geometry.x, ref["brownian/x2"])
    same(d.geometry.y, ref["brownian/y2"])


def test_loading_movebank(fishers):
    assert fishers.crs.to_epsg() == 4326
    assert len(fishers) == 47347 and fishers["individual-local-identifier"].nunique() == 8
    assert str(fishers["timestamp"].dt.tz) == "UTC"
    assert fishers.geometry.is_empty.sum() > 0
    sorted_ = ls.load_tracks(FISHERS)
    codes = pd.factorize(sorted_["individual-local-identifier"])[0]
    assert np.all(np.diff(codes) >= 0)


def test_lines_segments_and_track_data(fishers):
    tr = fishers[~fishers.geometry.is_empty]
    lines = ls.move.track_lines(tr)
    assert len(lines) == 8 and (lines.geom_type == "LineString").all()
    assert lines["n_locations"].sum() == len(tr)
    assert "individual-taxon-canonical-name" in lines.columns
    seg = ls.move.segments(tr)
    assert (seg.geom_type == "LineString").sum() == len(tr) - 8
    np.testing.assert_allclose(shapely.length(seg[seg.geom_type == "LineString"].to_numpy()),
                               np.hypot(*np.diff(np.c_[tr.geometry.x, tr.geometry.y], axis=0).T)[
                                   np.diff(pd.factorize(tr["individual-local-identifier"])[0]) == 0])
    td = ls.move.track_data(tr)
    assert len(td) == 8 and "tag-local-identifier" in td.columns


def test_errors(inputs):
    tr = track_set(inputs, "ll")
    with pytest.raises(ValueError, match="consecutive"):
        ls.move.distance(tr.iloc[np.r_[0:5, 200:205, 5:10]])
    with pytest.raises(ValueError, match="decrease"):
        ls.move.distance(tr.iloc[::-1])
    dup = track_set(inputs, "dup")
    with pytest.raises(ValueError, match="share their time"):
        ls.move.speed(dup)
    with pytest.raises(ValueError, match="empty"):
        ls.move.distance(track_set(inputs, "gap"))
    with pytest.raises(ValueError, match="which column"):
        ls.move.distance(gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)]))
