// SPDX-License-Identifier: GPL-3.0-or-later
//
// geodiv's amplitude parameters (basics.R): sa, sq, ssk, sku, sv, sph, smean,
// with R's arithmetic: mean() and sd() and sum() in 80 bits, x^3 and x^4 by
// mingw-w64's powl, the scalar sqrt of the byte-code interpreter.

#include <cmath>
#include <limits>
#include <vector>

#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// The values that are not NA (R's na.rm = TRUE), in order.
std::vector<double> present(const double* z, i64 n) {
    std::vector<double> v;
    v.reserve(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i)
        if (!std::isnan(z[i])) v.push_back(z[i]);
    return v;
}

// sum((z - zbar)^p) / N over the values, as R: the powers rounded to double,
// then an 80-bit sum.
double central_moment(const std::vector<double>& v, double zbar, double p) {
    const i64 n = static_cast<i64>(v.size());
    std::vector<double> t(v.size());
#pragma omp parallel for schedule(static) if (n > 65536)
    for (i64 i = 0; i < n; ++i) t[static_cast<std::size_t>(i)] = lsm::r_pow(v[static_cast<std::size_t>(i)] - zbar, p);
    return lsm::r_sum(t.data(), n) / static_cast<double>(n);
}

}  // namespace

Amplitude amplitude(const double* z, i64 n, unsigned parts) {
    Amplitude a{};
    if (parts & (kSa | kSq | kSsk | kSku)) {
        const std::vector<double> v = present(z, n);
        const i64 N = static_cast<i64>(v.size());
        const double dn = static_cast<double>(N);
        const double zbar = lsm::r_mean(v.data(), N);

        if (parts & kSa) {  // sum(abs(z - zbar), na.rm = TRUE) / N
            std::vector<double> t(v.size());
#pragma omp parallel for schedule(static) if (N > 65536)
            for (i64 i = 0; i < N; ++i)
                t[static_cast<std::size_t>(i)] = std::fabs(v[static_cast<std::size_t>(i)] - zbar);
            a.sa = lsm::r_sum(t.data(), N) / dn;
        }
        if (parts & (kSq | kSsk | kSku)) {
            // sq: sd(z, na.rm = TRUE)
            const double s = lsm::r_sd(v.data(), N, zbar);
            a.sq = s;
            if (parts & kSsk) {  // (sum((z - zbar)^3) / N) / s^3, times sqrt(N (N - 1)) / (N - 2)
                a.ssk_unadj = N ? central_moment(v, zbar, 3.0) / lsm::r_pow(s, 3.0) : NaN;
                a.ssk = (std::sqrt(dn * (dn - 1.0)) / (dn - 2.0)) * a.ssk_unadj;
            }
            if (parts & kSku) {  // (sum((z - zbar)^4) / N) / s^4, minus 3 for the excess
                a.sku_raw = N ? central_moment(v, zbar, 4.0) / lsm::r_pow(s, 4.0) : NaN;
                a.sku = a.sku_raw - 3.0;
            }
        }
    }
    if (parts & kExtremes) {  // sv, sph: abs(min(z)), abs(max(z)) (Inf / -Inf without values, as R)
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (i64 i = 0; i < n; ++i) {
            const double x = z[i];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        a.sv = std::fabs(lo);
        a.sph = std::fabs(hi);
    }
    if (parts & kSmean) {  // smean: mean(z[z > 0], na.rm = TRUE)
        std::vector<double> pos;
        for (i64 i = 0; i < n; ++i)
            if (z[i] > 0) pos.push_back(z[i]);
        a.smean = lsm::r_mean(pos.data(), static_cast<i64>(pos.size()));
    }
    return a;
}

}  // namespace surf
