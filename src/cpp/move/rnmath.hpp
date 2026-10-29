// SPDX-License-Identifier: GPL-3.0-or-later
//
// R 4.4.2's nmath for ls.move's distributions of steps (amt's fit_distr and
// random_numbers): its own C code (src/cpp/move/vendor/rmath-4.4.2) compiled
// with the elementary functions of R's Windows build (mingw-w64's x87 log,
// exp, sin, cos, log1p, expm1; lsm/rmath.hpp), so the same numbers as R.
#pragma once

namespace nlm {
class RRng;
}

namespace move {

// dgamma(x, shape, scale, log)
double r_dgamma(double x, double shape, double scale, bool give_log);
// lgamma(x) (R's lgammafn) and gamma(x)
double r_lgammafn(double x);
double r_gammafn(double x);
// besselI(x, nu, expon.scaled = TRUE) for 0 <= nu <= 100 (NaN otherwise)
double r_bessel_i_scaled(double x, double nu);
// rgamma(1, shape, scale) drawn from `rng`: Ahrens & Dieter's GD (1982) for
// shape >= 1 and GS (1974) below, as R's rgamma.c
double r_rgamma(nlm::RRng& rng, double shape, double scale);

}  // namespace move
