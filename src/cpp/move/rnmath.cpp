// SPDX-License-Identifier: GPL-3.0-or-later
//
// R 4.4.2's src/nmath files (GPL >= 2) in src/cpp/move/vendor/rmath-4.4.2,
// compiled here inside namespace move::rnmath with the shim nmath.h of that
// folder. Their unqualified log(), exp(), sin(), cos(), log1p(), expm1(), pow()
// and lgamma() find this namespace's functions first: those of R's Windows build,
// whose nmath calls mingw-w64's math library (x87, lsm/rmath.hpp; lgamma, used
// by stirlerr() for 1 <= n <= 5.25, is the UCRT's). sqrt() is the correctly
// rounded one gcc inlines. The random numbers come from the nlm::RRng given
// to r_rgamma(). The vendored sources are #included here and must not be
// compiled on their own.
#include "move/rnmath.hpp"

#include <cfloat>
#include <cmath>
#include <limits>

#include "lsm/rmath.hpp"
#include "nlm/rng.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4244 4456 4457 4458 4701 4702 4706 4996 4305)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

namespace move {
namespace rnmath {

inline double log(double x) { return lsm::r_log(x); }
inline double exp(double x) { return lsm::r_exp(x); }
inline double sin(double x) { return lsm::r_sin(x); }
inline double cos(double x) { return lsm::r_cos(x); }
inline double log1p(double x) { return lsm::r_log1p(x); }
inline double expm1(double x) { return lsm::r_expm1(x); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double pow(double x, double y) { return lsm::r_pow(x, y); }

// mingw-w64's lgamma() (math/lgamma.c: Cephes' lgam, Stephen L. Moshier,
// with its IBMPC constants), which stirlerr() calls; log and sin are the x87
// ones.
namespace cephes {
constexpr double A[] = {0x1.a985027336661p-11, -0x1.37fbdb580e943p-11, 0x1.a019f20dc5ebbp-11,
                        -0x1.6c16c16b0a5a1p-9, 0x1.555555555554bp-4};
constexpr double B[] = {-0x1.589018ff36761p+10, -0x1.2f234355bb93ep+15, -0x1.43d73f89089e5p+18,
                        -0x1.1bc82f994db51p+20, -0x1.a45890219f20bp+20, -0x1.a0c675418055ep+19};
constexpr double C[] = {-0x1.5fd0d1cf312b2p+8, -0x1.0aa0d7b89d757p+14, -0x1.aeb84b9744c9bp+17,
                        -0x1.1628671950043p+20, -0x1.35255892ff34cp+21, -0x1.ece4b6a11e14ap+20};
constexpr double LS2PI = 0x1.d67f1c864beb5p-1;
constexpr double LOGPI = 0x1.250d048e7a1bdp+0;
constexpr double MAXLGM = 2.556348e305;
constexpr double PI = 3.14159265358979323846;

inline double polevl(double x, const double* p, int n) {
    double y = *p++;
    do {
        y = y * x + *p++;
    } while (--n);
    return y;
}

inline double p1evl(double x, const double* p, int n) {
    n -= 1;
    double y = x + *p++;
    do {
        y = y * x + *p++;
    } while (--n);
    return y;
}

inline double lgamma_r(double x, int* sgngam) {
    double p, q, u, w, z;
    *sgngam = 1;
    if (std::isnan(x)) return x;
    if (!std::isfinite(x)) return HUGE_VAL;
    if (x < -34.0) {
        q = -x;
        w = lgamma_r(q, sgngam);
        p = std::floor(q);
        if (p == q) return HUGE_VAL;
        const int i = static_cast<int>(p);
        *sgngam = ((i & 1) == 0) ? -1 : 1;
        z = q - p;
        if (z > 0.5) {
            p += 1.0;
            z = p - q;
        }
        z = q * lsm::r_sin(PI * z);
        if (z == 0.0) return HUGE_VAL;
        return LOGPI - lsm::r_log(z) - w;
    }
    if (x < 13.0) {
        z = 1.0;
        p = 0.0;
        u = x;
        while (u >= 3.0) {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while (u < 2.0) {
            if (u == 0.0) return HUGE_VAL;
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if (z < 0.0) {
            *sgngam = -1;
            z = -z;
        } else {
            *sgngam = 1;
        }
        if (u == 2.0) return lsm::r_log(z);
        p -= 2.0;
        x = x + p;
        p = x * polevl(x, B, 5) / p1evl(x, C, 6);
        return lsm::r_log(z) + p;
    }
    if (x > MAXLGM) return *sgngam * HUGE_VAL;
    q = (x - 0.5) * lsm::r_log(x) - x + LS2PI;
    if (x > 1.0e8) return q;
    p = 1.0 / (x * x);
    if (x >= 1000.0)
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p + 0.0833333333333333333333) / x;
    else
        q += polevl(p, A, 4) / x;
    return q;
}
}  // namespace cephes

inline double lgamma(double x) {
    int sign;
    return cephes::lgamma_r(x, &sign);
}

// bessel_i_ex() calls it for nu < 0 only, which r_bessel_i_scaled() refuses
double bessel_k_ex(double, double, double, double*) { return std::numeric_limits<double>::quiet_NaN(); }

// the generator rgamma() draws from (r_rgamma sets it)
nlm::RRng* current_rng = nullptr;
double norm_rand(void) { return current_rng->norm_rand(); }
double unif_rand(void) { return current_rng->unif_rand(); }
double exp_rand(void) { return current_rng->exp_rand(); }

#include "vendor/rmath-4.4.2/nmath.h"
#include "vendor/rmath-4.4.2/chebyshev.c"
#include "vendor/rmath-4.4.2/lgammacor.c"
#include "vendor/rmath-4.4.2/cospi.c"
#include "vendor/rmath-4.4.2/gamma.c"
#include "vendor/rmath-4.4.2/lgamma.c"
#include "vendor/rmath-4.4.2/pgamma_part.c"
#include "vendor/rmath-4.4.2/stirlerr.c"
#include "vendor/rmath-4.4.2/bd0.c"
#include "vendor/rmath-4.4.2/dpois.c"
#include "vendor/rmath-4.4.2/dgamma.c"
#include "vendor/rmath-4.4.2/rgamma.c"
// lgammacor.c's and gamma.c's macros (in R each file is compiled on its own)
#undef xbig
#undef xmax
#undef xmin
#include "vendor/rmath-4.4.2/gamma_cody.c"
#include "vendor/rmath-4.4.2/bessel_i.c"

}  // namespace rnmath

double r_dgamma(double x, double shape, double scale, bool give_log) {
    return rnmath::dgamma(x, shape, scale, give_log ? 1 : 0);
}

double r_lgammafn(double x) { return rnmath::lgammafn(x); }

double r_gammafn(double x) { return rnmath::gammafn(x); }

double r_bessel_i_scaled(double x, double nu) {
    // besselI(x, nu, expon.scaled = TRUE): bessel_i(x, nu, 2), its work array here
    if (!(nu >= 0) || nu > 100) return std::numeric_limits<double>::quiet_NaN();
    double bi[102];
    return rnmath::bessel_i_ex(x, nu, 2.0, bi);
}

double r_rgamma(nlm::RRng& rng, double shape, double scale) {
    // rgamma() keeps its constants for the last shape in statics, as in R
    rnmath::current_rng = &rng;
    const double v = rnmath::rgamma(shape, scale);
    rnmath::current_rng = nullptr;
    return v;
}

}  // namespace move

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
