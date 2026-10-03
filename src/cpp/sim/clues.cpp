// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLUE-S allocation (lulcc 1.0.4's src/allocateclues.c, step for step, with
// R's generator) and the focal mean of the neighbourhood maps (see
// clues.hpp). lulcc adds the class differences with their signs for the mean
// criterion (they sum to zero, so it never acts); here it is the mean of
// their absolute values, as CLUE-S defines it.

#include "sim/clues.hpp"

#include <cmath>
#include <stdexcept>

#include "lsm/parallel.hpp"

namespace sim {

CluesResult clues_allocate(const double* tprob, const double* landuse, std::int64_t ncell, const double* demand,
                           const double* codes, int ncode, double jitter, double scale, double max_iter,
                           double max_diff, double mean_diff, nlm::RRng& rng) {
    CluesResult out;
    out.allocation.assign(static_cast<std::size_t>(ncell), 0.0);
    const std::size_t total_cells = static_cast<std::size_t>(ncell) * static_cast<std::size_t>(ncode);
    std::vector<double> pprob(total_cells), iter(ncode, 0.0), total(ncode), diff(ncode);
    std::vector<int> lu_index(static_cast<std::size_t>(ncell));
    for (std::int64_t i = 0; i < ncell; ++i) {
        int ix = -1;
        for (int j = 0; j < ncode; ++j)
            if (codes[j] == landuse[i]) ix = j;
        if (ix == -1) throw std::invalid_argument("land use map contains invalid codes");
        lu_index[static_cast<std::size_t>(i)] = ix;
    }
    double count = 0;
    int passes = 0;
    double maxd = 0, aved = 0;
    do {
        ++passes;
        for (std::size_t k = 0; k < total_cells; ++k) {
            pprob[k] = std::isnan(tprob[k]) ? tprob[k] : tprob[k] + (-jitter + (jitter - (-jitter)) * rng.unif_rand());
        }
        for (int j = 0; j < ncode; ++j) total[j] = 0;
        for (std::int64_t i = 0; i < ncell; ++i) {
            int lu1 = lu_index[static_cast<std::size_t>(i)];
            double maxp = -999;
            for (int j = 0; j < ncode; ++j) {
                const double p = pprob[static_cast<std::size_t>(j) * static_cast<std::size_t>(ncell) + i];
                if (!std::isnan(p)) {
                    const double t = p + iter[j];
                    if (maxp < t) {
                        maxp = t;
                        lu1 = j;
                    }
                }
            }
            out.allocation[static_cast<std::size_t>(i)] = codes[lu1];
            total[lu1] += 1;
        }
        for (int j = 0; j < ncode; ++j) diff[j] = demand[j] - total[j];
        maxd = 0;
        aved = 0;
        for (int j = 0; j < ncode; ++j) {
            const double a = diff[j] < 0 ? diff[j] * -1 : diff[j];
            if (maxd < a) maxd = a;
            aved = aved + a / ncode;
        }
        if (count >= max_iter) break;
        for (int j = 0; j < ncode; ++j) iter[j] = (iter[j] + diff[j] * scale);
        count += 1;
    } while (maxd > max_diff || aved > mean_diff);
    out.iterations = passes;
    out.converged = !(maxd > max_diff || aved > mean_diff);
    out.max_diff = maxd;
    out.mean_diff = aved;
    return out;
}

std::vector<double> focal_mean(const double* x, std::int64_t ny, std::int64_t nx, const double* w, int wy, int wx) {
    std::vector<double> out(static_cast<std::size_t>(ny * nx), std::nan(""));
    const int ry = wy / 2, rx = wx / 2;
    lsm::parallel_for(ny * nx >= (1 << 14), ny, [&](std::int64_t r) {
        for (std::int64_t c = 0; c < nx; ++c) {
            double sum = 0;
            int n = 0;
            for (int a = 0; a < wy; ++a) {
                const std::int64_t rr = r + a - ry;
                if (rr < 0 || rr >= ny) continue;
                for (int b = 0; b < wx; ++b) {
                    const std::int64_t cc = c + b - rx;
                    if (cc < 0 || cc >= nx) continue;
                    const double v = x[rr * nx + cc];
                    const double wt = w[a * wx + b];
                    if (std::isnan(v) || std::isnan(wt)) continue;
                    sum += v * wt;
                    ++n;
                }
            }
            if (n > 0) out[static_cast<std::size_t>(r * nx + c)] = sum / n;
        }
    });
    return out;
}

}  // namespace sim
