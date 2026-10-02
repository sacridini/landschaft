// SPDX-License-Identifier: GPL-3.0-or-later
//
// The elementary functions with the bits that R's Windows build gives them.
//
// R packages on Windows are compiled with Rtools' mingw-w64 gcc and link the
// mingw-w64 math library: its sin(), cos() and atan2() run fsin / fcos /
// fpatan on the x87 unit in extended precision. Verified for geosphere 1.6-8
// (GeographicLib C++ 2.7), terra 1.9-50 (GeographicLib C 1.52) and the GEOS
// 3.12.2 terra links: with the UCRT's functions 3063 of 95046 destPoint
// coordinates differ from R, with the x87 ones none do. They are lsm's
// (lsm/rmath.hpp: MASM on MSVC x64, inline assembly with gcc / clang on x86,
// the standard functions elsewhere). hypot is the UCRT's _hypot
// (lsm::ucrt_hypot), cbrt mingw-w64's Cephes cube root (cost::mingw_cbrt).
#pragma once

#include <cmath>

#include "cost/geodesic.hpp"
#include "lsm/rmath.hpp"

namespace sdm {
namespace geocircle_libm {

inline double sin(double x) { return lsm::r_sin(x); }
inline double cos(double x) { return lsm::r_cos(x); }
inline double atan2(double y, double x) { return lsm::r_atan2(y, x); }
inline double hypot(double x, double y) { return lsm::ucrt_hypot(x, y); }
// Three-argument hypot (GeographicLib's Math::hypot3 names it; the geodesics do not use it).
inline double hypot(double x, double y, double z) { return std::hypot(x, y, z); }
inline double cbrt(double x) { return cost::mingw_cbrt(x); }

}  // namespace geocircle_libm
}  // namespace sdm
