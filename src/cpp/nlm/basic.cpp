// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR's random, percolation and gradient models, util_rescale() and the
// classification helpers of landscape_classify().

#include <algorithm>
#include <cmath>
#include <limits>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"

namespace nlm {

namespace {

constexpr i64 kParallelCells = 1 << 16;

// all.equal(target, current) of two numbers (tolerance 1.5e-8, relative
// when |target| exceeds it).
bool all_equal(double target, double current) {
    if (target == current) return true;
    const double tolerance = 1.5e-8;
    double xy = std::fabs(target - current);
    const double xn = std::fabs(target);
    if (std::isfinite(xn) && xn > tolerance) xy /= xn;
    return !(std::isnan(xy) || xy > tolerance);
}

}  // namespace

void rescale(double* x, i64 n) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    bool any = false;
    for (i64 i = 0; i < n; ++i) {
        const double v = x[i];
        if (std::isnan(v)) continue;
        any = true;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    if (!any) return;
    if (all_equal(lo, hi)) {
        for (i64 i = 0; i < n; ++i)
            if (!std::isnan(x[i])) x[i] = 0.0;
        return;
    }
    const double range = hi - lo;
    lsm::parallel_for(n >= kParallelCells, n, [&](i64 i) { x[i] = (x[i] - lo) / range; });
}

std::vector<double> random_field(RRng& rng, i64 nrow, i64 ncol) {
    std::vector<double> byc(static_cast<std::size_t>(nrow * ncol));
    rng.runif(byc.data(), nrow * ncol, 0.0, 1.0);
    std::vector<double> out(byc.size());
    lsm::parallel_for(nrow * ncol >= kParallelCells, ncol, [&](i64 c) {
        for (i64 r = 0; r < nrow; ++r) out[static_cast<std::size_t>(r * ncol + c)] = byc[static_cast<std::size_t>(c * nrow + r)];
    });
    return out;
}

std::vector<double> percolation(RRng& rng, i64 nrow, i64 ncol, double prob) {
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol));
    for (i64 c = 0; c < ncol; ++c)
        for (i64 r = 0; r < nrow; ++r)
            out[static_cast<std::size_t>(r * ncol + c)] = rng.runif(0.0, 1.0) < prob ? 1.0 : 0.0;
    return out;
}

std::vector<double> planar_gradient(i64 nrow, i64 ncol, double direction) {
    // R: sin((pi / 180) * direction), cos(...) * -1, with mingw's x87 sin/cos.
    const double rad = (3.141592653589793238462643383280 / 180) * direction;
    const double eastness = lsm::r_sin(rad);
    const double southness = lsm::r_cos(rad) * -1;
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol));
    lsm::parallel_for(nrow * ncol >= kParallelCells, nrow, [&](i64 r) {
        const double s = southness * static_cast<double>(r);
        for (i64 c = 0; c < ncol; ++c) out[static_cast<std::size_t>(r * ncol + c)] = s + eastness * static_cast<double>(c);
    });
    return out;
}

void edge_from_planar(double* g, i64 n) {
    lsm::parallel_for(n >= kParallelCells, n, [&](i64 i) { g[i] = (std::fabs(0.5 - g[i]) * -2) + 1; });
}

std::vector<double> jenks_breaks(const std::vector<double>& d, int k) {
    // rcpp_get_jenksbreaks(): the O(k n^2) dynamic programme of seqInt's
    // getJenksBreaks. Row l depends on the rows above it only through column
    // j - 1, so the columns run one after the other and the rows of a column
    // in parallel; each (l, j) sees the same sums in the same order as the
    // original loop.
    const i64 n = static_cast<i64>(d.size());
    if (n == 0) return {};
    const double max_double = std::numeric_limits<double>::max();
    std::vector<double> mat1(static_cast<std::size_t>(n * k), 1.0);
    std::vector<double> mat2(static_cast<std::size_t>(n * k), 0.0);
    for (i64 i = 1; i < n; ++i)
        for (int j = 0; j < k; ++j) mat2[static_cast<std::size_t>(i * k + j)] = max_double;
    auto at = [k](i64 row, int col) { return static_cast<std::size_t>(row * k + col); };

    // First column: the within-class sum of squares of d[0..l-1], summed
    // from the end as the original's inner loop does.
    lsm::parallel_for_dynamic(n > 2000, n - 1, [&](i64 idx) {
        const i64 l = idx + 2;
        double s1 = 0.0, s2 = 0.0, w = 0.0, v = 0.0;
        for (i64 m = 1; m <= l; ++m) {
            const double val = d[static_cast<std::size_t>(l - m)];
            s2 += val * val;
            s1 += val;
            w += 1.0;
            v = s2 - (s1 * s1) / w;
        }
        mat1[at(l - 1, 0)] = 1.0;
        mat2[at(l - 1, 0)] = v;
    });
    for (int j = 2; j <= k; ++j) {
        lsm::parallel_for_dynamic(n > 2000, n - 1, [&](i64 idx) {
            const i64 l = idx + 2;
            double s1 = 0.0, s2 = 0.0, w = 0.0;
            double best1 = mat1[at(l - 1, j - 1)];
            double best2 = mat2[at(l - 1, j - 1)];
            for (i64 m = 1; m <= l; ++m) {
                const i64 i3 = l - m + 1;
                const double val = d[static_cast<std::size_t>(i3 - 1)];
                s2 += val * val;
                s1 += val;
                w += 1.0;
                const double v = s2 - (s1 * s1) / w;
                const i64 i4 = i3 - 1;
                if (i4 != 0) {
                    const double cand = v + mat2[at(i4 - 1, j - 2)];
                    if (best2 >= cand) {
                        best1 = static_cast<double>(i3);
                        best2 = cand;
                    }
                }
            }
            mat1[at(l - 1, j - 1)] = best1;
            mat2[at(l - 1, j - 1)] = best2;
        });
    }
    std::vector<i64> kclass(static_cast<std::size_t>(k));
    for (int i = 1; i <= k; ++i) kclass[static_cast<std::size_t>(i - 1)] = i;
    kclass[static_cast<std::size_t>(k - 1)] = n;
    i64 k_temp = n;
    for (int j = k; j > 1; --j) {
        const i64 id = static_cast<i64>(mat1[at(k_temp - 1, j - 1)]) - 1;
        kclass[static_cast<std::size_t>(j - 2)] = id;
        k_temp = id;
    }
    std::vector<double> brks(static_cast<std::size_t>(k + 1));
    brks[0] = d[0];
    for (int i = 1; i < k + 1; ++i) brks[static_cast<std::size_t>(i)] = d[static_cast<std::size_t>(kclass[static_cast<std::size_t>(i - 1)] - 1)];
    return brks;
}

void cut_include_lowest(const double* x, i64 n, const std::vector<double>& breaks, double* out) {
    // .bincode(x, breaks, right = TRUE, include.lowest = TRUE)
    const i64 nb1 = static_cast<i64>(breaks.size()) - 1;
    lsm::parallel_for(n >= kParallelCells, n, [&](i64 i) {
        const double v = x[i];
        out[i] = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(v) || nb1 < 1) return;
        i64 lo = 0, hi = nb1;
        if (v < breaks[static_cast<std::size_t>(lo)] || breaks[static_cast<std::size_t>(hi)] < v) return;
        while (hi - lo >= 2) {
            const i64 mid = (hi + lo) / 2;
            if (v > breaks[static_cast<std::size_t>(mid)])
                lo = mid;
            else
                hi = mid;
        }
        out[i] = static_cast<double>(lo + 1);
    });
}

}  // namespace nlm
