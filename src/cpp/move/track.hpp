// SPDX-License-Identifier: GPL-3.0-or-later
//
// Measures between the consecutive locations of tracks (ls.move), as move2's
// mt_distance, mt_azimuth and mt_turnangle: one value per location, the step
// to the next location of the same track, NaN at the last location of each
// track. `track` holds a code per location; the locations of a track are
// consecutive (ordered by track and time), so a step crosses tracks where the
// code changes.
#pragma once

#include <cstdint>

namespace move {

using i64 = std::int64_t;

enum class Coords { Planar = 0, LonLat = 1 };

// mt_distance: s2 (metres) in lon / lat, Euclidean in the CRS's units otherwise.
void distances(const double* x, const double* y, const i64* track, i64 n, Coords coords, int threads, double* out);

// mt_azimuth (radians): lwgeom's geodesic azimuth on the spheroid (a, 1 / f) in
// lon / lat, atan2(dx, dy) otherwise.
void azimuths(const double* x, const double* y, const i64* track, i64 n, Coords coords, double a,
              double inv_flattening, int threads, double* out);

// mt_turnangle from mt_azimuth's values: NaN first, then the change of
// azimuth wrapped to [-pi, pi) as move2 does it on values with units: the
// units package's %% (e1 - round(e1 / e2) * e2) and R's %/% (80-bit).
void turn_angles(const double* azimuth, i64 n, int threads, double* out);

// R's x %% y and x %/% y for doubles (arithmetic.c's myfmod / myfloor).
double r_fmod(double x1, double x2);
double r_fdiv(double x1, double x2);

// mt_sim_brownian_motion for one track: the cumulative sums (R's 80-bit
// cumsum) of start + rnorm(n - 1, 0, sqrt(sigma^2 * dt)) for x, then for y,
// drawn from `rng` (an nlm::RRng). sigma: one value or n - 1.
void brownian(void* rng, const double* t, i64 n, const double* sigma, i64 n_sigma, double x0, double y0,
              double* x, double* y);

}  // namespace move
