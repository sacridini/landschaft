// SPDX-License-Identifier: GPL-3.0-or-later
//
// cov.c (R 4.4) for cor(x, y): the MEAN_, cov_na_2 and COV_SDEV steps in R's
// LDOUBLE (lsm::Extended), the standard deviations rounded to double from
// sqrtl() of the 80-bit variance (lsm::sqrt_ext), then ANS /= (sx * sy) in
// double and clamped to [-1, 1].

#include "sdm/rstats.hpp"

#include <cmath>
#include <limits>

#include "lsm/extended.hpp"

namespace sdm {

namespace {

using lsm::Extended;

Extended r_mean_ext(const double* x, i64 n) {
    Extended sum = 0.0;
    for (i64 k = 0; k < n; ++k) sum += x[k];
    Extended tmp = sum / Extended(n);
    if (std::isfinite(static_cast<double>(tmp))) {
        sum = 0.0;
        for (i64 k = 0; k < n; ++k) sum += Extended(x[k]) - tmp;
        tmp = tmp + sum / Extended(n);
    }
    return tmp;
}

inline double sign(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }

}  // namespace

double r_cor2(const double* x, const double* y, i64 n, bool kendall) {
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    if (n < 2) return NaN;
    double ans, sx, sy;
    if (!kendall) {
        // the means are rounded to double (xm[i] = (double)tmp), then used as LDOUBLE (xxm)
        const double xm = static_cast<double>(r_mean_ext(x, n));
        const double ym = static_cast<double>(r_mean_ext(y, n));
        const Extended xxm = xm, yym = ym, n1 = Extended(n - 1);
        Extended sum = 0.0;
        for (i64 k = 0; k < n; ++k) sum += (Extended(x[k]) - xxm) * (Extended(y[k]) - yym);
        ans = static_cast<double>(sum / n1);
        Extended vx = 0.0, vy = 0.0;
        for (i64 k = 0; k < n; ++k) {
            const Extended dx = Extended(x[k]) - xxm, dy = Extended(y[k]) - yym;
            vx += dx * dx;
            vy += dy * dy;
        }
        vx /= n1;
        vy /= n1;
        sx = static_cast<double>(lsm::sqrt_ext(vx));
        sy = static_cast<double>(lsm::sqrt_ext(vy));
    } else {
        Extended sum = 0.0, cx = 0.0, cy = 0.0;
        for (i64 k = 0; k < n; ++k)
            for (i64 l = 0; l < n; ++l) {
                sum += sign(x[k] - x[l]) * sign(y[k] - y[l]);
                if (x[k] != x[l]) cx += 1.0;
                if (y[k] != y[l]) cy += 1.0;
            }
        ans = static_cast<double>(sum);
        sx = static_cast<double>(lsm::sqrt_ext(cx));
        sy = static_cast<double>(lsm::sqrt_ext(cy));
    }
    if (sx == 0.0 || sy == 0.0) return NaN;
    ans /= (sx * sy);
    return ans >= 1.0 ? 1.0 : (ans <= -1.0 ? -1.0 : ans);
}

}  // namespace sdm
