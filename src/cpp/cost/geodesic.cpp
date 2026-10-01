// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/geodesic.hpp"

#include <cfloat>
#include <cmath>

#include "lsm/rmath.hpp"

namespace cost {

// Cephes' cube root as mingw-w64 has it (math/cbrt.c): a polynomial first
// guess and two Newton steps.
double mingw_cbrt(double x) {
    if (!std::isfinite(x) || x == 0.0) return x;
    int sign = 1;
    if (!(x > 0)) {
        sign = -1;
        x = -x;
    }
    const double z = x;
    int e = 0;
    x = std::frexp(x, &e);
    x = (((-1.3466110473359520655053e-1 * x + 5.4664601366395524503440e-1) * x - 9.5438224771509446525043e-1) * x +
         1.1399983354717293273738e0) * x + 4.0238979564544752126924e-1;
    if (e >= 0) {
        int rem = e;
        e /= 3;
        rem -= 3 * e;
        if (rem == 1) x *= 1.2599210498948731647672;
        else if (rem == 2) x *= 1.5874010519681994747517;
    } else {
        e = -e;
        int rem = e;
        e /= 3;
        rem -= 3 * e;
        if (rem == 1) x *= 0.79370052598409973737585;
        else if (rem == 2) x *= 0.62996052494743658238361;
        e = -e;
    }
    x = std::ldexp(x, e);
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
    return sign < 0 ? -x : x;
}

namespace {

// mingw-w64's hypot calls the UCRT's _hypot (lsm::ucrt_hypot gives its bits on
// every platform).
inline double r_hypot(double x, double y) { return lsm::ucrt_hypot(x, y); }

}  // namespace

namespace geographiclib {

// GeographicLib's C code, unchanged, with R's (mingw-w64's) maths. remainder
// and remquo are exact operations; sqrt is the inline correctly rounded one.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4244 4456 4701 4702 4706)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define sin(x) ::lsm::r_sin(x)
#define cos(x) ::lsm::r_cos(x)
#define atan2(y, x) ::lsm::r_atan2(y, x)
#define hypot(x, y) ::cost::r_hypot(x, y)
#define cbrt(x) ::cost::mingw_cbrt(x)
#define sqrt(x) std::sqrt(x)
#define remainder(x, y) std::remainder(x, y)
#define remquo(x, y, q) std::remquo(x, y, q)
#define copysign(x, y) std::copysign(x, y)
#define fabs(x) std::fabs(x)
#define signbit(x) std::signbit(x)
#define atanh(x) std::atanh(x)
#define atan(x) std::atan(x)
#define nan(x) std::nan(x)
#define round(x) std::round(x)
#include "cost/geographiclib/geodesic.inc"
#undef sin
#undef cos
#undef atan2
#undef hypot
#undef cbrt
#undef sqrt
#undef remainder
#undef remquo
#undef copysign
#undef fabs
#undef signbit
#undef atanh
#undef atan
#undef nan
#undef round
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

}  // namespace geographiclib

Geodesic::Geodesic(double a, double f) {
    auto* g = new geographiclib::geod_geodesic;
    geographiclib::geod_init(g, a, f);
    g_ = g;
}

Geodesic::~Geodesic() { delete static_cast<geographiclib::geod_geodesic*>(g_); }

double Geodesic::distance(double lon1, double lat1, double lon2, double lat2) const {
    double s12 = 0, azi1 = 0, azi2 = 0;
    geographiclib::geod_inverse(static_cast<const geographiclib::geod_geodesic*>(g_), lat1, lon1, lat2, lon2, &s12,
                                &azi1, &azi2);
    return s12;
}

}  // namespace cost
