// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fuzzy similarity (Dinamica EGO's CalcReciprocalSimilarityMap): see similarity.hpp.

#include "sim/similarity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/parallel.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;

// membership of class c at (r, col) in map m: the largest weight of a cell of class c in the window
inline float membership(const int* m, i64 ny, i64 nx, i64 r, i64 col, int c, int half, const std::vector<float>& w,
                        int side) {
    float best = 0.0f;
    const i64 r0 = std::max<i64>(0, r - half), r1 = std::min(ny - 1, r + half);
    const i64 c0 = std::max<i64>(0, col - half), c1 = std::min(nx - 1, col + half);
    for (i64 y = r0; y <= r1; ++y)
        for (i64 x = c0; x <= c1; ++x)
            if (m[y * nx + x] == c) {
                const float v = w[static_cast<std::size_t>((y - r + half) * side + (x - col + half))];
                if (v > best) best = v;
            }
    return best;
}

}  // namespace

SimilarityResult reciprocal_similarity(i64 ny, i64 nx, const int* a, const int* b, int window, bool exponential) {
    if (window < 1 || window % 2 == 0) throw std::invalid_argument("the window must be odd");
    const int half = window / 2;
    std::vector<float> w(static_cast<std::size_t>(window) * window);
    for (int dy = -half; dy <= half; ++dy)
        for (int dx = -half; dx <= half; ++dx) {
            const double d = std::sqrt(static_cast<double>(dy * dy + dx * dx));
            w[static_cast<std::size_t>((dy + half) * window + (dx + half))] =
                exponential ? static_cast<float>(std::pow(2.0, -d / 2)) : 1.0f;
        }
    const i64 n = ny * nx;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    SimilarityResult res;
    res.first.assign(static_cast<std::size_t>(n), nan);
    res.second.assign(static_cast<std::size_t>(n), nan);
    lsm::parallel_for(n * window >= 65536, ny, [&](i64 r) {
        for (i64 col = 0; col < nx; ++col) {
            const i64 i = r * nx + col;
            const int ca = a[i], cb = b[i];
            const float in_b = ca >= 0 ? membership(b, ny, nx, r, col, ca, half, w, window) : 0.0f;
            const float in_a = cb >= 0 ? membership(a, ny, nx, r, col, cb, half, w, window) : 0.0f;
            if (ca >= 0) res.first[i] = cb >= 0 ? std::min(in_b, in_a) : in_b;
            if (cb >= 0) res.second[i] = ca >= 0 ? std::min(in_b, in_a) : in_a;
        }
    });
    double s1 = 0, s2 = 0;
    for (i64 i = 0; i < n; ++i) {
        if (a[i] >= 0) {
            s1 += res.first[i];
            ++res.first_cells;
        }
        if (b[i] >= 0) {
            s2 += res.second[i];
            ++res.second_cells;
        }
    }
    res.first_mean = res.first_cells ? s1 / static_cast<double>(res.first_cells) : std::nan("");
    res.second_mean = res.second_cells ? s2 / static_cast<double>(res.second_cells) : std::nan("");
    return res;
}

}  // namespace sim
