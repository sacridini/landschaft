/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Stands in for R's src/nmath/nmath.h (with Rmath.h and R_ext/Arith.h) for
 * the R 4.4.2 nmath files of this folder, which move/rnmath.cpp #includes
 * inside namespace move::rnmath. It holds what those files use: the IEEE
 * build's constants and tests, the warnings (dropped: R only reports them),
 * and the declarations of the functions they call one another by. The
 * elementary functions (log, exp, sqrt, log1p, lgamma...) are those of
 * move/rnmath.cpp's namespace, R's mingw-w64 ones.
 */
#ifndef MOVE_RNMATH_SHIM_H
#define MOVE_RNMATH_SHIM_H

#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>

#define IEEE_754 1
#define attribute_hidden
#define LDOUBLE long double

typedef int Rboolean;
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ISNAN(x) (std::isnan(x))
#define R_FINITE(x) (std::isfinite(x))
#define ML_POSINF (std::numeric_limits<double>::infinity())
#define ML_NEGINF (-std::numeric_limits<double>::infinity())
#define ML_NAN (std::numeric_limits<double>::quiet_NaN())

#define ME_NONE 0
#define ME_DOMAIN 1
#define ME_RANGE 2
#define ME_NOCONV 4
#define ME_PRECISION 8
#define ME_UNDERFLOW 16
#define ML_WARNING(x, s) do {} while (0)
#define ML_WARN_return_NAN { return ML_NAN; }
#define MATHLIB_WARNING(fmt, x) do {} while (0)
#define MATHLIB_WARNING2(fmt, x, x2) do {} while (0)
#define MATHLIB_WARNING3(fmt, x, x2, x3) do {} while (0)
#define MATHLIB_WARNING4(fmt, x, x2, x3, x4) do {} while (0)
#define MATHLIB_WARNING5(fmt, x, x2, x3, x4, x5) do {} while (0)

#define R_forceint(x) nearbyint(x)
#define R_nonint(x) (fabs((x) - R_forceint(x)) > 1e-9 * fmax2(1., fabs(x)))

#ifndef M_PI
#define M_PI 3.141592653589793238462643383280
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458
#endif
#define M_2PI 6.283185307179586476925286766559
#define M_LN_SQRT_2PI 0.918938533204672741780329736406
#define M_LN_SQRT_PI 0.572364942924700087071713675677
#define M_LN_2PI 1.837877066409345483560659472811
#define M_LN_SQRT_PId2 0.225791352644727432363097614947
#define M_1_SQRT_2PI 0.398942280401432677939946059934
#define M_SQRT_32 5.656854249492380195206754896838

#define NA_INTEGER INT_MIN
#define NA_REAL ML_NAN
#define R_POW(x, y) pow(x, y)

inline double fmax2(double x, double y) {
    if (ISNAN(x) || ISNAN(y)) return x + y;
    return (x < y) ? y : x;
}
inline double fmin2(double x, double y) {
    if (ISNAN(x) || ISNAN(y)) return x + y;
    return (x < y) ? x : y;
}

// R's main/arithmetic.c R_pow_di()
inline double R_pow_di(double x, int n) {
    double xn = 1.0;
    if (ISNAN(x)) return x;
    if (n == NA_INTEGER) return NA_REAL;
    if (n != 0) {
        if (!R_FINITE(x)) return R_POW(x, (double)n);
        Rboolean is_neg = (n < 0);
        if (is_neg) n = -n;
        for (;;) {
            if (n & 01) xn *= x;
            if (n >>= 1) x *= x;
            else break;
        }
        if (is_neg) xn = 1. / xn;
    }
    return xn;
}

double sinpi(double);
double Rf_gamma_cody(double);
double bessel_i_ex(double, double, double, double*);
double bessel_k_ex(double, double, double, double*);
double chebyshev_eval(double, const double*, const int);
int chebyshev_init(double*, int, double);
double lgammacor(double);
double stirlerr(double);
double bd0(double, double);
void ebd0(double, double, double*, double*);
double gammafn(double);
double lgammafn(double);
double lgammafn_sign(double, int*);
double lgamma1p(double);
double log1pmx(double);
double dpois_raw(double, double, int);
double cospi(double);
double sinpi(double);
double tanpi(double);
double norm_rand(void);
double unif_rand(void);
double exp_rand(void);

#endif
