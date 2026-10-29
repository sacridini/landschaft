# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases of ls.move against R's move2 (tools/make_reference_move.py, tools/move_reference.R).

Each track set is drawn here once and stored with the references
(tests/data/reference/move/inputs.npz), so the tests read the same inputs R
did on every platform. A case is R code run with ``x`` (the move2 object of
its track set) returning a named list of numbers; ``fishers`` is move2's
example (mt_read(mt_example()), also in tests/data/move/fishers.csv.gz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T0 = 1_600_000_000  # 2020-09-13 12:26:40 UTC


def _walk(rng, n, start, scales, lonlat):
    x, y = [start[0]], [start[1]]
    for i in range(n - 1):
        s = scales[i % len(scales)]
        dx, dy = rng.normal(0, s, 2)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    x, y = np.array(x), np.array(y)
    if lonlat:
        x = (x + 180) % 360 - 180
        y = np.clip(y, -89.9, 89.9)
    return x, y


def track_sets() -> dict[str, dict]:
    """name -> {x, y, t (seconds), track (ids), crs}."""
    rng = np.random.default_rng(20261029)
    sets = {}
    # lon / lat: steps from millimetres to hundreds of km, a repeated location, the antimeridian, near a pole
    xs, ys, ts, ks = [], [], [], []
    specs = [("a", 120, (-47.9, -15.8), [1e-6, 1e-4, 1e-3, 0.01, 0.1]),
             ("b", 90, (179.7, 10.0), [0.05, 0.2, 1.0]),
             ("c", 60, (20.0, 88.5), [0.01, 0.3, 2.0])]
    for name, n, start, scales in specs:
        x, y = _walk(rng, n, start, scales, True)
        x[10], y[10] = x[9], y[9]  # same location twice
        t = T0 + np.cumsum(rng.choice([30, 45, 600, 3600, 7200], n)).astype(np.float64)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ks += [name] * n
    sets["ll"] = {"x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts),
                  "track": np.array(ks), "crs": "EPSG:4326"}
    # projected (UTM 23S, metres) and without a CRS
    xs, ys, ts, ks = [], [], [], []
    for name, n in [("p1", 80), ("p2", 70)]:
        x, y = _walk(rng, n, (190000.0, 8250000.0), [0.5, 25.0, 400.0], False)
        x[5], y[5] = x[4], y[4]
        t = T0 + np.cumsum(rng.choice([20, 60, 900], n)).astype(np.float64)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ks += [name] * n
    base = {"x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts), "track": np.array(ks)}
    sets["utm"] = {**base, "crs": "EPSG:32723"}
    sets["plain"] = {**base, "x": base["x"] - 190000.0, "y": base["y"] - 8250000.0, "crs": None}
    # duplicated times: repeated records, records with less, records that differ
    t = T0 + np.array([0, 60, 60, 60, 120, 180, 180, 240, 300, 300, 300, 360], np.float64)
    sets["dup"] = {"x": np.array([0, 1, 1, 1, 2, 3, 3.5, 4, 5, 5, 5, 6], np.float64),
                   "y": np.zeros(12), "t": np.r_[t, t], "track": np.array(["u"] * 12 + ["v"] * 12),
                   "crs": None,
                   "v1": np.array([1, 2, 2, np.nan, 3, 4, 4, 5, 6, 6, np.nan, 7] * 2, np.float64),
                   "v2": np.array([1, 2, np.nan, 2, 3, 4, 5, 5, 6, 7, 7, 8] * 2, np.float64)}
    sets["dup"]["x"] = np.r_[sets["dup"]["x"], sets["dup"]["x"] + 10]
    sets["dup"]["y"] = np.r_[sets["dup"]["y"], sets["dup"]["y"]]
    # gaps to interpolate: lon / lat and plain, with records without a location (NaN)
    x, y = _walk(rng, 40, (-60.0, -3.0), [0.01, 0.5], True)
    t = T0 + np.cumsum(rng.choice([600, 1800, 5400], 40)).astype(np.float64)
    empty = np.zeros(40, bool)
    empty[[3, 4, 11, 20, 21, 22, 39]] = True
    x[empty], y[empty] = np.nan, np.nan
    sets["gap"] = {"x": x, "y": y, "t": t, "track": np.array(["g"] * 20 + ["h"] * 20), "crs": "EPSG:4326"}
    sets["gap_plain"] = {**sets["gap"], "x": x * 1000.0, "y": y * 1000.0, "crs": None}
    return sets


@dataclass(frozen=True)
class Case:
    data: str     # a track set or "fishers"
    r: str        # R code with x -> named list of numbers
    group: str = "move2"


_MEASURES = """list(
  distance = as.numeric(mt_distance(x)),
  azimuth = if (is.na(st_crs(x)) || st_is_longlat(x)) as.numeric(mt_azimuth(x)) else NULL,
  turnangle = if (is.na(st_crs(x)) || st_is_longlat(x)) as.numeric(mt_turnangle(x)) else NULL,
  time_lags = as.numeric(units::set_units(mt_time_lags(x), "s")),
  speed = if (all(as.numeric(mt_time_lags(x)) > 0, na.rm = TRUE)) as.numeric(mt_speed(x)) else NULL
)"""

CASES: dict[str, Case] = {
    "measures_ll": Case("ll", _MEASURES),
    "measures_utm": Case("utm", _MEASURES),
    "measures_plain": Case("plain", _MEASURES),
    "measures_fishers": Case("fishers", "x <- x[!st_is_empty(x), ]\n" + _MEASURES),
    "km_deg": Case("ll", """list(km = as.numeric(mt_distance(x, units = "km")),
                             deg = as.numeric(mt_azimuth(x, units = "degrees")))"""),
    "unique": Case("dup", """list(
  first = as.numeric(mt_unique(x, "first")), last = as.numeric(mt_unique(x, "last")),
  subsets = as.numeric(suppressWarnings(mt_unique(x, "subsets"))),
  subsets_equal = as.numeric(suppressWarnings(mt_unique(x, "subsets_equal"))),
  sample = { set.seed(7); as.numeric(mt_unique(x, "sample")) })"""),
    "per_interval": Case("fishers", """list(
  hour = as.numeric(mt_per_interval(x)), last = as.numeric(mt_per_interval(x, "last")),
  m15 = as.numeric(mt_per_interval(x, unit = "15 mins")), h2 = as.numeric(mt_per_interval(x, unit = "2 hours")),
  h5 = as.numeric(mt_per_interval(x, unit = "5 hours")), day = as.numeric(mt_per_interval(x, unit = "day")),
  d3 = as.numeric(mt_per_interval(x, unit = "3 days")), week = as.numeric(mt_per_interval(x, unit = "week")),
  month = as.numeric(mt_per_interval(x, unit = "month")),
  random = { set.seed(3); as.numeric(mt_per_interval(x, "random", unit = "day")) })"""),
    "visible": Case("fishers", "list(visible = as.numeric(mt_movebank_visible(x)))"),
    "interpolate_empty": Case("gap", """y <- mt_interpolate(x); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2])"""),
    "interpolate_times": Case("gap", """tt <- seq(min(mt_time(x)), max(mt_time(x)), by = "45 mins")
y <- mt_interpolate(x, tt, max_time_lag = units::as_units(2, "h")); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)), tr = as.numeric(factor(mt_track_id(y))))"""),
    "interpolate_hour_omit": Case("gap", """y <- mt_interpolate(x, "1 hour", omit = TRUE); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)))"""),
    "interpolate_plain": Case("gap_plain", """tt <- seq(min(mt_time(x)), max(mt_time(x)), by = "20 mins")
y <- mt_interpolate(x, tt); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)))"""),
    "aeqd": Case("fishers", """f <- function(p) as.numeric(regmatches(p, regexpr("(?<=lat_0=)[-0-9.]+", p, perl = TRUE)))
g <- function(p) as.numeric(regmatches(p, regexpr("(?<=lon_0=)[-0-9.]+", p, perl = TRUE)))
a <- mt_aeqd_crs_fixed(x)$proj4string; b <- mt_aeqd_crs_fixed(x, "center")$proj4string
list(centroid = c(g(a), f(a)), center = c(g(b), f(b)))"""),
    "brownian": Case("ll", """set.seed(11); b <- mt_sim_brownian_motion(t = c(0, 1, 2.5, 4, 8), sigma = c(1, 2, 0, 0.5), tracks = 3)
c1 <- st_coordinates(b)
set.seed(5); d <- mt_sim_brownian_motion()
c2 <- st_coordinates(d)
list(x1 = c1[, 1], y1 = c1[, 2], x2 = c2[, 1], y2 = c2[, 2])"""),
}
