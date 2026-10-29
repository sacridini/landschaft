# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.move against R's amt 0.3.1 (tests/move_cases.py, group "amt"; tools/amt_fixed.R)."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import landschaft as ls
from move_cases import EXTRACT_RASTER
from test_move import FISHERS, same, same_libm, track_set

REF = Path(__file__).parent / "data" / "reference" / "move"


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "amt.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs.npz"))


@pytest.mark.parametrize("name", ["utm", "plain", "ll"])
def test_steps_are_amts(ref, inputs, name):
    s = ls.move.steps(track_set(inputs, name))
    case = f"amt_steps_{name}"
    same_libm(s["sl_"], ref[f"{case}/sl"], "sl_")
    same_libm(s["direction_p"], ref[f"{case}/dir"], "direction_p")
    same_libm(s["ta_"], ref[f"{case}/ta"], "ta_")
    same(s["x2_"], ref[f"{case}/x2"])
    same(s["y2_"], ref[f"{case}/y2"])
    same(s["dt_"].dt.total_seconds(), ref[f"{case}/dt"])


def test_steps_columns(inputs):
    tr = track_set(inputs, "utm")
    tr["speed"] = np.arange(len(tr), dtype=float)
    s = ls.move.steps(tr, keep_cols="both")
    assert list(s.columns) == ["track", "x1_", "x2_", "y1_", "y2_", "sl_", "direction_p", "ta_", "t1_", "t2_", "dt_",
                               "speed_start", "speed_end"]
    assert len(s) == len(tr) - 2  # two tracks
    same(s["speed_end"] - s["speed_start"], np.ones(len(s)))
    assert str(s["t1_"].dtype) == "datetime64[ns, UTC]"
    assert list(ls.move.steps(tr, keep_cols="end").columns)[-1] == "speed"


def test_resample_is_amts(ref, inputs):
    tr = track_set(inputs, "utm")
    t = lambda d: d["time"].astype("int64").to_numpy() / 1e9  # noqa: E731
    a = ls.move.resample(tr, 60, 15)
    same(t(a), ref["amt_resample/t_a"])
    same(a["burst_"], ref["amt_resample/burst_a"])
    b = ls.move.resample(tr, "15min", datetime.timedelta(minutes=2), start=2)
    same(t(b), ref["amt_resample/t_b"])
    same(b["burst_"], ref["amt_resample/burst_b"])
    f = ls.move.filter_min_n_burst(a, 4)
    same(t(f), ref["amt_resample/t_f"])
    same(f["burst_"], ref["amt_resample/burst_f"])
    s = ls.move.steps(a)
    same_libm(s["sl_"], ref["amt_resample/sl"])
    same_libm(s["direction_p"], ref["amt_resample/dir"])
    same_libm(s["ta_"], ref["amt_resample/ta"])
    same(s["burst_"], ref["amt_resample/sburst"])


def test_summaries_are_amts(ref, inputs):
    tr = track_set(inputs, "utm")
    same(ls.move.nsd(tr), ref["amt_summaries_utm/nsd"])
    same(ls.move.msd(tr), ref["amt_summaries_utm/msd"])
    same(ls.move.cum_dist(tr), ref["amt_summaries_utm/cum"])
    same(ls.move.tot_dist(tr), ref["amt_summaries_utm/tot"])
    same(ls.move.straightness(tr), ref["amt_summaries_utm/straight"])
    same_libm(ls.move.sinuosity(tr), ref["amt_summaries_utm/sinuosity"])
    same_libm(ls.move.tac(tr), ref["amt_summaries_utm/tac"])
    assert list(ls.move.tac(tr).index) == ["p1", "p2"]
    ll = track_set(inputs, "ll")
    same(ls.move.nsd(ll), ref["amt_summaries_ll/nsd"])
    same(ls.move.msd(ll), ref["amt_summaries_ll/msd"])
    same_libm(ls.move.cum_dist(ll), ref["amt_summaries_ll/cum"])
    same_libm(ls.move.tot_dist(ll), ref["amt_summaries_ll/tot"])


def test_fits_are_amts(ref, inputs):
    s = ls.move.steps(track_set(inputs, "utm"))
    got = []
    for _, one in s.groupby("track", sort=False):
        sl, ta = one["sl_"].to_numpy(), one["ta_"].to_numpy()
        g, v = ls.move.fit_distr(sl, "gamma"), ls.move.fit_distr(ta, "vonmises")
        e, ln = ls.move.fit_distr(sl[sl > 0], "exp"), ls.move.fit_distr(sl[sl > 0], "lnorm")
        h, u = ls.move.fit_distr(sl, "hnorm"), ls.move.fit_distr(ta, "unif")
        got += [g.params["shape"], g.params["scale"], *g.vcov.to_numpy().ravel(order="F"), v.params["kappa"],
                v.vcov.iloc[0, 0], e.params["rate"], ln.params["meanlog"], ln.params["sdlog"], h.params["sd"],
                u.params["min"], u.params["max"]]
    same_libm(got, ref["amt_fit/fits"])


def test_random_numbers_are_amts(ref):
    d = [("gamma", {"shape": 0.4, "scale": 120}), ("gamma", {"shape": 3.5, "scale": 20}), ("exp", {"rate": 0.02}),
         ("lnorm", {"meanlog": 3, "sdlog": 0.7}), ("hnorm", {"sd": 50}), ("unif", {}),
         ("vonmises", {"kappa": 0}), ("vonmises", {"kappa": 2.5})]
    got = [ls.move.random_numbers(ls.move.make_distribution(n, **p), 300, seed=17) for n, p in d]
    same_libm(np.concatenate(got), ref["amt_random_numbers/draws"])


def test_random_steps_are_amts(ref, inputs):
    s = ls.move.steps(track_set(inputs, "utm"))
    r = ls.move.random_steps(s, n_control=5, seed=42)
    for key, col in [("x2", "x2_"), ("y2", "y2_"), ("sl", "sl_"), ("ta", "ta_")]:
        same_libm(r[col], ref[f"amt_random_steps/{key}"], col)
    same(r["case_"], ref["amt_random_steps/case"])
    same(r["step_id_"], ref["amt_random_steps/id"])
    assert "direction_p" not in r.columns
    sl = r.attrs["landschaft.move"]["sl_distr"]
    assert set(sl) == {"p1", "p2"} and sl["p1"].name == "gamma"
    u = ls.move.random_steps(s, n_control=3, sl_distr=ls.move.make_distribution("exp", rate=0.01),
                             ta_distr=ls.move.make_distribution("unif"), seed=3)
    for key, col in [("ux2", "x2_"), ("uy2", "y2_"), ("usl", "sl_"), ("uta", "ta_")]:
        same_libm(u[col], ref[f"amt_random_steps/{key}"], col)


def test_random_steps_of_bursts_are_amts(ref, inputs):
    b = ls.move.resample(track_set(inputs, "utm"), 60, 15)
    with pytest.warns(UserWarning, match="fewer than 3 steps"):
        r = ls.move.random_steps(ls.move.steps(b), seed=7)
    for key, col in [("x2", "x2_"), ("y2", "y2_"), ("sl", "sl_"), ("ta", "ta_")]:
        same_libm(r[col], ref[f"amt_random_steps_bursts/{key}"], col)
    same(r["case_"], ref["amt_random_steps_bursts/case"])
    same(r["step_id_"], ref["amt_random_steps_bursts/id"])
    same(r["burst_"], ref["amt_random_steps_bursts/burst"])


def codes(c):
    c = pd.Categorical(c)
    return np.where(c.codes < 0, np.nan, c.codes + 1.0)


def test_time_of_day_is_amts(ref, inputs):
    tr = track_set(inputs, "ll")
    same(codes(ls.move.time_of_day(tr)), ref["amt_time_of_day/tod"])
    same(codes(ls.move.time_of_day(tr, include_crepuscule=True)), ref["amt_time_of_day/crep"])
    s = ls.move.steps(tr)
    same(codes(ls.move.time_of_day(s, include_crepuscule=True)), ref["amt_time_of_day/end"])
    same(codes(ls.move.time_of_day(s, where="start")), ref["amt_time_of_day/start"])
    assert np.isnan(ref["amt_time_of_day/tod"]).any()  # near the pole, days without a sunrise
    f = ls.load_tracks(FISHERS, sort=False)
    f = f[~f.geometry.is_empty]
    same(codes(ls.move.time_of_day(f, include_crepuscule=True)), ref["amt_time_of_day_fishers/crep"])


def test_extract_covariates_is_amts(ref, inputs):
    tr = track_set(inputs, "utm")
    p = EXTRACT_RASTER
    x0, y1 = tr.geometry.x.iloc[p["x_from"]], tr.geometry.y.iloc[p["y_from"]]
    v = np.arange(1, p["nrows"] * p["ncols"] + 1, dtype=float)
    v[v % p["na_every"] == 0] = np.nan
    res = p["res"]
    cover = xr.DataArray(v.reshape(p["nrows"], p["ncols"]), dims=("y", "x"), name="cover",
                         coords={"y": y1 - res * (np.arange(p["nrows"]) + 0.5),
                                 "x": x0 + res * (np.arange(p["ncols"]) + 0.5)})
    cover = cover.rio.write_crs("EPSG:32723").rio.write_transform(Affine(res, 0, x0, 0, -res, y1))
    got = ls.move.extract_covariates(tr, cover)["cover"]
    same(got, ref["amt_extract/track"])
    assert np.isnan(ref["amt_extract/track"]).any() and not np.isnan(ref["amt_extract/track"]).all()
    s = ls.move.steps(tr)
    both = ls.move.extract_covariates(s, cover, where="both")
    same(both["cover_start"], ref["amt_extract/both_start"])
    same(both["cover_end"], ref["amt_extract/both_end"])
    same(ls.move.extract_covariates(s, cover, where="start")["cover"], ref["amt_extract/start"])
    r = ls.move.random_steps(s, n_control=5, seed=42)
    same(ls.move.extract_covariates(r, {"cover": cover})["cover"], ref["amt_extract/random"])


def test_distributions():
    g = ls.move.make_distribution("gamma", shape=2, rate=0.5)
    assert g.params == {"shape": 2, "rate": 0.5}
    a = ls.move.random_numbers(g, 50, seed=1)
    b = ls.move.random_numbers(ls.move.make_distribution("gamma", shape=2, scale=2.0), 50, seed=1)
    same(a, b)
    with pytest.raises(ValueError, match="not implemented"):
        ls.move.make_distribution("weibull")
    with pytest.raises(ValueError, match="parameters of exp"):
        ls.move.make_distribution("exp", scale=3)
    t = ls.move.random_numbers(ls.move.make_distribution("vonmises", kappa=4), 1000, seed=2)
    assert t.min() > -np.pi and t.max() <= np.pi
    # exp: the maximum likelihood rate, where amt's BFGS fails (mean steps over 1000 m)
    sl = ls.move.random_numbers(ls.move.make_distribution("exp", rate=1 / 2500), 400, seed=5)
    assert np.isclose(ls.move.fit_distr(sl, "exp").params["rate"], 1 / np.mean(sl), rtol=1e-15)


def test_random_steps_refuses_lonlat(inputs):
    with pytest.raises(ValueError, match="projected"):
        ls.move.random_steps(ls.move.steps(track_set(inputs, "ll")))


def test_time_of_day_needs_a_crs(inputs):
    with pytest.raises(ValueError, match="no CRS"):
        ls.move.time_of_day(track_set(inputs, "plain"))
