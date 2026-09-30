// SPDX-License-Identifier: GPL-3.0-or-later
//
// vdt's fast_log and fast_exp (Piparo, Hauth, Innocente; LGPL-3, after Cephes)
// as Zonation 5 v2.4 runs them: its libvdt.dll is built with -Ofast, which folds
// constants and reassociates the operations, so the IEEE-strict source gives
// other bits. These are the operations of the compiled vector loops (read from
// the binary, identical to its scalar tails), checked bit for bit against the
// DLL on 1.6 million arguments. Zonation computes x^y as fast_exp(fast_log(x) * y).
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

namespace prio::vdt {

inline double bits_to_double(std::uint64_t u) {
    double d;
    std::memcpy(&d, &u, sizeof d);
    return d;
}

inline std::uint64_t double_to_bits(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

inline double fast_log(double x0) {
    const std::uint64_t n = double_to_bits(x0);
    const double e = static_cast<double>(static_cast<std::int32_t>(n >> 52) - 1023);
    const double m = bits_to_double((n & 0x800FFFFFFFFFFFFFULL) | 0x3FE0000000000000ULL);
    double fe, mp;
    if (m > 0.7071067811865476) {
        fe = e + 1.0;
        mp = m + (-0.0);
    } else {
        fe = e;
        mp = m + m;
    }
    const double x = mp + (-1.0);
    double px = x * 0.00010187566380458093 + 0.497494994976747;
    px = px * x + 4.705791198788817;
    px = px * x + 14.498922534161093;
    px = px * x + 17.936867850781983;
    px = px * x + 7.708387337558854;
    const double x2 = x * x;
    const double x3 = x2 * x;
    double qx = mp + 10.287358718916746;
    qx = qx * x + 45.227914583753225;
    qx = qx * x + 82.98752669127767;
    qx = qx * x + 71.15447506185639;
    qx = qx * x + 23.125162012676533;
    const double res = (x3 * px) / qx;
    double r = ((fe * 0.6931471805599453 + x) + x2 * (-0.5)) + res;
    if (1e307 < x0) r = std::numeric_limits<double>::infinity();
    if (x0 < 0) r = std::numeric_limits<double>::quiet_NaN();
    return r;
}

inline double fast_exp(double x0) {
    const double t = x0 * 1.4426950408889634 + 0.5;
    // cvttpd2dq, minus the sign of t (read through its conversion to float)
    std::int32_t n = (t > -2147483649.0 && t < 2147483648.0) ? static_cast<std::int32_t>(t)
                                                              : std::numeric_limits<std::int32_t>::min();
    const float tf = static_cast<float>(t);
    std::uint32_t tb;
    std::memcpy(&tb, &tf, sizeof tb);
    n = static_cast<std::int32_t>(static_cast<std::uint32_t>(n) - (tb >> 31));
    const double x = x0 + static_cast<double>(n) * -0.6931471805599453;
    const double xx = x * x;
    const double p = ((0.00012617719307481058 * xx + 0.030299440770744195) * xx + 1.0) * x;
    const double q = 2.0 + ((3.0019850513866446e-06 * xx + 0.002524483403496841) * xx + 0.22726554820815503) * xx;
    double y = (p + p) / (q - p) + 1.0;
    y *= bits_to_double((static_cast<std::uint64_t>(static_cast<std::uint32_t>(n)) << 52) + 0x3FF0000000000000ULL);
    if (708.0 < x0) y = std::numeric_limits<double>::infinity();
    if (!(-708.0 <= x0)) y = 0.0;
    return y;
}

// Zonation's fastPower: x^y.
inline double fast_pow(double x, double y) { return fast_exp(fast_log(x) * y); }

}  // namespace prio::vdt
