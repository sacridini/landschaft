// SPDX-License-Identifier: GPL-3.0-or-later
//
// Carbon emissions and removals of land change (see emissions.hpp).

#include "serv/emissions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace serv {

namespace {

constexpr i64 kBlock = 1 << 16;

double regrown(int form, double mx, double rate, double shape, double age) {
    if (form == kLinear) return std::min(rate * age, mx);
    return mx * std::pow(1.0 - std::exp(-rate * age), shape);
}

}  // namespace

void emission_stocks(const std::int32_t* cls, int t, i64 n, const double* times, const double* stock,
                     const int* form, const double* rmax, const double* rate, const double* shape, double* out,
                     int threads) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        bool born = false;  // the current class appeared in the cell at a date of the series
        double since = 0.0;
        for (int d = 0; d < t; ++d) {
            const std::int32_t c = cls[static_cast<std::size_t>(d) * n + i];
            double& o = out[static_cast<std::size_t>(d) * n + i];
            if (c < 0) {
                born = false;
                o = nan;
                continue;
            }
            if (d > 0 && cls[static_cast<std::size_t>(d - 1) * n + i] != c) {
                born = true;
                since = times[d];
            }
            if (form[c] != kNoRegrowth) {
                o = born ? regrown(form[c], rmax[c], rate[c], shape[c], times[d] - since)
                         : (std::isnan(stock[c]) ? rmax[c] : stock[c]);
            } else {
                o = stock[c];
            }
        }
    }
}

void emission_flows(const double* stocks, const std::int32_t* cls, int t, i64 n, int k, double factor, double ha,
                    double* net, double* periods, double* trans, int threads) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill(net, net + n, nan);
    const i64 blocks = (n + kBlock - 1) / kBlock;
    const std::size_t kk = static_cast<std::size_t>(k) * k;
    for (int d = 1; d < t; ++d) {
        const double* a = stocks + static_cast<std::size_t>(d - 1) * n;
        const double* b = stocks + static_cast<std::size_t>(d) * n;
        const std::int32_t* ca = cls + static_cast<std::size_t>(d - 1) * n;
        const std::int32_t* cb = cls + static_cast<std::size_t>(d) * n;
        // per block: emissions, removals, then k x k x 3 by pair
        std::vector<double> part(static_cast<std::size_t>(blocks) * (2 + 3 * kk), 0.0);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 bl = 0; bl < blocks; ++bl) {
            double* p = part.data() + static_cast<std::size_t>(bl) * (2 + 3 * kk);
            const i64 end = std::min(n, (bl + 1) * kBlock);
            for (i64 i = bl * kBlock; i < end; ++i) {
                if (std::isnan(a[i]) || std::isnan(b[i])) continue;
                const double diff = (a[i] - b[i]) * factor;
                double de = 0.0, dr = 0.0;
                if (diff > 0.0)
                    de = diff * ha;
                else
                    dr = -diff * ha;
                p[0] += de;
                p[1] += dr;
                net[i] = std::isnan(net[i]) ? diff : net[i] + diff;
                double* q = p + 2 + 3 * (static_cast<std::size_t>(ca[i]) * k + cb[i]);
                q[0] += 1.0;
                q[1] += de;
                q[2] += dr;
            }
        }
        double* per = periods + static_cast<std::size_t>(d - 1) * 2;
        double* tr = trans + static_cast<std::size_t>(d - 1) * 3 * kk;
        per[0] = per[1] = 0.0;
        std::fill(tr, tr + 3 * kk, 0.0);
        for (i64 bl = 0; bl < blocks; ++bl) {
            const double* p = part.data() + static_cast<std::size_t>(bl) * (2 + 3 * kk);
            per[0] += p[0];
            per[1] += p[1];
            for (std::size_t j = 0; j < 3 * kk; ++j) tr[j] += p[2 + j];
        }
    }
}

}  // namespace serv
