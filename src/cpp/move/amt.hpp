// SPDX-License-Identifier: GPL-3.0-or-later
//
// amt 0.3.1's steps of tracks (ls.move.steps): the length, the absolute
// direction and the turn angle of each step, with R's arithmetic, and the
// summaries of a track built on them.
#pragma once

#include <cstdint>

namespace move {

using i64 = std::int64_t;

// For each location i whose next location i + 1 has the same `group` code
// (a track, or a burst of it): the step's length `sl` (sqrt(dx^2 + dy^2),
// R's x87 vector sqrt; s2's distance in longitude / latitude), its
// direction `dir` (amt's direction_abs with zero_dir "E", anticlockwise, in
// (-pi, pi]; NaN for a step of length 0) and the turn angle `ta` from the
// previous step of the group (direction_rel, in (-pi, pi]; NaN at the first
// step of a group). NaN at the last location of each group.
//
// In longitude / latitude amt takes geosphere::bearing (clockwise from
// north) as if it were the anticlockwise angle from east; here the bearing
// is turned into that angle (pi / 2 - bearing), the direction amt means.
void amt_steps(const double* x, const double* y, const i64* group, i64 n, bool lonlat, double* sl, double* dir,
               double* ta);

// tac: 1 / n * sum(diff(cos(ta))^2 + diff(sin(ta))^2, na.rm = TRUE) of the
// turn angles ta[1..n-1] of a track's n steps (amt's tac without its first
// step's angle).
double amt_tac(const double* ta, i64 n_steps);

// sinuosity (Benhamou 2004): 2 * (p * ((1 - mc^2 - ms^2) / ((1 - mc)^2 +
// ms^2) + cv^2))^-0.5 with p the mean step length, cv its coefficient of
// variation (R's sd), mc and ms the means of the cosines and sines of the
// turn angles. amt converts the angles, already in radians, from degrees
// (cos(ta * pi / 180)); here they are taken in radians.
double amt_sinuosity(const double* sl, const double* ta, i64 n_steps);

// msd: mean((x - mean(x))^2 + (y - mean(y))^2) with R's means.
double amt_msd(const double* x, const double* y, i64 n);

// track_resample's mk_reg for each group (a track) in turn: the burst of
// each location (1, 2...) or -1 for the locations left out, with locations
// `rate` seconds apart within `tolerance`, from the location `start`
// (0-based) of each group. `t`: seconds, increasing within a group.
void amt_resample(const double* t, const i64* group, i64 n, double rate, double tolerance, i64 start, i64* burst);

// The control steps of random_steps: from (x1, y1), the step of length sl
// and turn angle ta from the observed direction `angle` (x1 + sl * cos(ta +
// angle), ...), and the turn angle of that step from the previous step's
// start (x0, y0), as amt recomputes it (atan2, wrapped to (-pi, pi)).
void amt_controls(const double* x1, const double* y1, const double* x0, const double* y0, const double* angle,
                  const double* sl, const double* ta, i64 n, double* x2, double* y2, double* ta_out);

// suncalc 0.5.1's getSunlightTimes for the date whose noon (UTC) is
// `noon` seconds since 1970 at (lon, lat): sunrise, sunset, dawn and dusk
// in whole seconds since 1970 (lubridate::floor_date), NaN when the sun does
// not reach the altitude that day.
void sun_times(double lon, double lat, double noon, double* sunrise, double* sunset, double* dawn, double* dusk);

// R's sum() and mean() of doubles (long double accumulation).
double r_sum_of(const double* x, i64 n);
double r_mean_of(const double* x, i64 n);

}  // namespace move
