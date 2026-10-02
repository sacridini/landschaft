// SPDX-License-Identifier: GPL-3.0-or-later
//
// usdm's .vif(): for column i, 1 / (1 - summary(lm(x[, i] ~ ., x[-i]))$r.squared):
// lm.fit (LINPACK dqrdc2 / dqrsl over R's reference BLAS: lsm::r_lm_residuals),
// fitted = y - residuals, mss = sum((f - mean(f))^2), rss = sum(r^2) (R's
// LDOUBLE mean and sums), r.squared = mss / (mss + rss), or 0 for a rank-1 fit.

#include "sdm/collinear.hpp"

#include <cstddef>

#include "lsm/landscape.hpp"
#include "lsm/rmath.hpp"

namespace sdm {

std::vector<double> vif(const double* data, i64 n, i64 m, const std::vector<i64>& which) {
    const i64 nw = static_cast<i64>(which.size());
    std::vector<double> out(static_cast<std::size_t>(nw));
    const int ni = static_cast<int>(n), p = static_cast<int>(m);  // intercept + the m - 1 others
#pragma omp parallel
    {
        std::vector<double> x(static_cast<std::size_t>(n * m)), rsd(static_cast<std::size_t>(n)),
            f(static_cast<std::size_t>(n));
#pragma omp for schedule(dynamic, 1)
        for (i64 w = 0; w < nw; ++w) {
            const i64 i = which[static_cast<std::size_t>(w)];
            const double* y = data + i * n;
            for (i64 r = 0; r < n; ++r) x[static_cast<std::size_t>(r)] = 1.0;
            i64 c = 1;
            for (i64 j = 0; j < m; ++j) {
                if (j == i) continue;
                std::copy(data + j * n, data + (j + 1) * n, x.begin() + c * n);
                ++c;
            }
            const int rank = lsm::r_lm_residuals(x.data(), y, ni, p, rsd.data());
            for (i64 r = 0; r < n; ++r) f[static_cast<std::size_t>(r)] = y[r] - rsd[static_cast<std::size_t>(r)];
            double r2 = 0.0;
            if (rank != 1) {
                const double mean = lsm::r_mean(f.data(), n);
                const double mss = lsm::sum_of(0, n, [&](i64 r) {
                    const double d = f[static_cast<std::size_t>(r)] - mean;
                    return d * d;
                });
                const double rss = lsm::sum_of(0, n, [&](i64 r) {
                    const double e = rsd[static_cast<std::size_t>(r)];
                    return e * e;
                });
                r2 = mss / (mss + rss);
            }
            out[static_cast<std::size_t>(w)] = 1.0 / (1.0 - r2);
        }
    }
    return out;
}

}  // namespace sdm
