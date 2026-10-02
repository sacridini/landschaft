// SPDX-License-Identifier: GPL-3.0-or-later
//
// mop's distance step (mop_distance.R, eucdist_mm.cpp) without the ng x nm
// matrix: each row of g scans the reference rows, keeps its k smallest
// distances (partial selection, then sorted as sort() leaves them) and takes
// R's mean of them.

#include "sdm/extrapolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "lsm/rmath.hpp"

namespace sdm {

std::vector<double> knn_mean_distance(const double* m, i64 nm, const double* g, i64 ng, i64 p, i64 k, bool squared) {
    std::vector<double> out(static_cast<std::size_t>(ng));
    k = std::max<i64>(1, std::min(k, nm));
#pragma omp parallel
    {
        std::vector<double> d(static_cast<std::size_t>(nm));
#pragma omp for schedule(dynamic, 64)
        for (i64 i = 0; i < ng; ++i) {
            const double* gi = g + i * p;
            for (i64 j = 0; j < nm; ++j) {
                const double* mj = m + j * p;
                double s = 0.0;
                for (i64 v = 0; v < p; ++v) {
                    const double diff = gi[v] - mj[v];
                    s += diff * diff;
                }
                d[static_cast<std::size_t>(j)] = squared ? s : std::sqrt(s);
            }
            std::nth_element(d.begin(), d.begin() + (k - 1), d.end());
            std::sort(d.begin(), d.begin() + k);
            out[static_cast<std::size_t>(i)] = lsm::r_mean(d.data(), k);
        }
    }
    return out;
}

}  // namespace sdm
