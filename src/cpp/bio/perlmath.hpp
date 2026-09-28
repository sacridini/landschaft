// SPDX-License-Identifier: GPL-3.0-or-later
//
// Arithmetic as Strawberry Perl does it (the Biodiverse reference): doubles,
// mingw-w64's x87 libm (the functions of lsm/rmath.hpp, which R on Windows
// uses too) and Perl's own `**`, which raises to an integer power by
// squaring in doubles.
#pragma once

#include <cmath>
#include <cstdint>

#include "lsm/rmath.hpp"

namespace bio {

// Perl's pp_pow: integer exponents by repeated squaring in double (negative
// ones as 1 / x^n), anything else through the C library's pow.
inline double perl_pow(double x, double y) {
    if (std::isfinite(y) && y == std::floor(y) && std::fabs(y) < 9.2e18) {
        std::uint64_t n = static_cast<std::uint64_t>(std::fabs(y));
        double result = 1.0, base = x;
        while (n) {
            if (n & 1) result *= base;
            n >>= 1;
            if (n) base *= base;
        }
        return y < 0 ? 1.0 / result : result;
    }
    return lsm::r_pow(x, y);
}

inline double perl_sqrt(double x) { return lsm::r_sqrt(x); }
inline double perl_sin(double x) { return lsm::r_sin(x); }
inline double perl_cos(double x) { return lsm::r_cos(x); }
inline double perl_atan2(double y, double x) { return lsm::r_atan2(y, x); }
inline double perl_log(double x) { return lsm::r_log(x); }
inline double perl_exp(double x) { return lsm::r_exp(x); }

// POSIX::round: halves away from zero.
inline double posix_round(double x) { return std::round(x); }

}  // namespace bio
