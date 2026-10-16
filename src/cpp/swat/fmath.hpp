// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fortran arithmetic as the SWAT+ reference build does it (flang, UCRT libm): REAL is
// float, every literal a float literal, intrinsics on floats call the C float
// functions (expf, logf, powf, sinf, ...), x ** n with an integer n is flang-rt's
// exponentiation by squaring, x ** y with a real y is powf.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace swat {

// x ** n, n integer (flang-rt FPowI): result *= base on each set bit, base squared.
inline float fpowi(float base, int n) {
    if (n == 0) return 1.0f;
    const bool neg = n < 0;
    const bool min_power = n == std::numeric_limits<int>::min();
    unsigned e = min_power ? static_cast<unsigned>(std::numeric_limits<int>::max()) : static_cast<unsigned>(neg ? -n : n);
    float result = 1.0f;
    const float orig = base;
    for (;;) {
        if (e & 1u) result *= base;
        e >>= 1;
        if (e == 0) break;
        base *= base;
    }
    if (min_power) result *= orig;
    if (neg) result = 1.0f / result;
    return result;
}

inline double fpowi(double base, int n) {
    if (n == 0) return 1.0;
    const bool neg = n < 0;
    unsigned e = static_cast<unsigned>(neg ? -n : n);
    double result = 1.0;
    for (;;) {
        if (e & 1u) result *= base;
        e >>= 1;
        if (e == 0) break;
        base *= base;
    }
    if (neg) result = 1.0 / result;
    return result;
}

// x ** y, real y
inline float fpow(float x, float y) { return std::pow(x, y); }

// Fortran MAX/MIN as flang lowers them: the running result is replaced by a later
// argument only when that one is greater (smaller).
inline float fmax2(float a, float b) { return b > a ? b : a; }
inline float fmin2(float a, float b) { return b < a ? b : a; }
inline float fmax3(float a, float b, float c) { return fmax2(fmax2(a, b), c); }
inline float fmin3(float a, float b, float c) { return fmin2(fmin2(a, b), c); }

// INT(x) (truncation) and NINT(x) (round half away from zero) of a real.
inline int fint(float x) { return static_cast<int>(x); }
inline int fnint(float x) { return static_cast<int>(std::lround(x)); }

// SWAT+ utils exp_w: exp with the result 0 below -80 and exp(80) above 80.
inline float exp_w(float y) {
    if (y < -80.0f) return 0.0f;
    if (y > 80.0f) return std::exp(80.0f);
    return std::exp(y);
}

// aunif.f90: Park-Miller minimal standard generator; the seed is updated in place.
inline float aunif(int& x1) {
    const int x2 = x1 / 127773;
    x1 = 16807 * (x1 - x2 * 127773) - x2 * 2836;
    if (x1 < 0) x1 = x1 + 2147483647;
    return static_cast<float>(static_cast<double>(x1) * 4.656612875e-10);
}

}  // namespace swat
