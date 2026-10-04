// SPDX-License-Identifier: GPL-3.0-or-later
//
// Three-map comparison (lulcc's ThreeMapComparison): see validation.hpp.
//
// The blocks go through in chunks: the per-block quantities of a chunk are
// computed in parallel (doubles, lulcc's order of operations), then each
// group of table cells adds the chunk's terms to its 80-bit sums in block
// order, so the sums are R's whatever the number of threads.

#include "sim/validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;
constexpr i64 kChunk = 1 << 15;

inline double fmin2(double a, double b) {  // pmin(a, b, na.rm = TRUE)
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return a < b ? a : b;
}

}  // namespace

std::vector<double> tmc_table(int n, i64 blocks, const double* x, const double* x1, const double* y1,
                              const double* w) {
    const int N = n + 1;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> tab(static_cast<std::size_t>(N) * N * N, nan);
    lsm::Extended sw = 0.0;
    for (i64 b = 0; b < blocks; ++b)
        if (!std::isnan(w[b])) sw += w[b];
    const double swt = static_cast<double>(sw);
    // per-block quantities of a chunk
    const int stride = n * n + 5 * n + 2;  // eq1, eq2, eq3 (j, i), Qq, Rr, Ss, 1 - a, bsum7
    std::vector<double> buf(static_cast<std::size_t>(kChunk) * stride);
    // accumulators: per (j, k): k == j holds Rule 1, Rule 2 and Rule 3 (i); k != j Rule 4 (i)
    const int groups = n * n;
    std::vector<lsm::Extended> acc(static_cast<std::size_t>(groups) * (n + 2), lsm::Extended(0.0));
    for (i64 start = 0; start < blocks; start += kChunk) {
        const i64 len = std::min(kChunk, blocks - start);
        lsm::parallel_for(len * n * n >= 4096, len, [&](i64 q) {
            const i64 b = start + q;
            double* v = &buf[static_cast<std::size_t>(q) * stride];
            double* eq1 = v;
            double* eq2 = v + n;
            double* eq3 = v + 2 * n;  // [j * n + i]
            double* qq = v + 2 * n + n * n;
            double* rr = qq + n;
            double* ss = rr + n;
            double Q[64], R[64], S[64];
            for (int c = 0; c < n; ++c) {
                const std::size_t at = static_cast<std::size_t>(c) * blocks + b;
                Q[c] = x[at] / w[b];
                R[c] = x1[at] / w[b];
                S[c] = y1[at] / w[b];
            }
            for (int j = 0; j < n; ++j) {
                eq1[j] = fmin2(R[j], S[j]);                       // Equation 1
                eq2[j] = fmin2(fmin2(Q[j], R[j]), S[j]);          // Equation 2
            }
            for (int j = 0; j < n; ++j) {                         // Equation 3
                double bsum = 0;
                bool first = true;
                for (int i = 0; i < n; ++i) {
                    if (i == j) continue;
                    const double t = Q[i] - eq2[i];
                    bsum = first ? t : bsum + t;
                    first = false;
                }
                for (int i = 0; i < n; ++i)
                    eq3[j * n + i] = i == j ? nan : ((eq1[j] - eq2[j]) * (Q[i] - eq2[i])) / bsum;
            }
            for (int i = 0; i < n; ++i) {                         // Equation 4 (rowSums, na.rm)
                lsm::Extended s = 0.0;
                for (int j = 0; j < n; ++j) {
                    const double t = j == i ? eq2[i] : eq3[j * n + i];
                    if (!std::isnan(t)) s += t;
                }
                qq[i] = Q[i] - static_cast<double>(s);
            }
            for (int j = 0; j < n; ++j) {
                rr[j] = R[j] - eq1[j];                            // Equation 5
                ss[j] = S[j] - eq1[j];                            // Equation 6
            }
            double a = eq1[0];                                    // Equation 7
            for (int j = 1; j < n; ++j) a = a + eq1[j];
            double bsum = 0;
            bool first = true;
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int i = 0; i < n; ++i) {
                        if (j == k) continue;
                        const double t = qq[i] * rr[j] * ss[k];
                        bsum = first ? t : bsum + t;
                        first = false;
                    }
            v[stride - 2] = 1 - a;
            v[stride - 1] = bsum;
        });
        lsm::parallel_for(len * n >= 4096, groups, [&](i64 g) {
            const int j = static_cast<int>(g / n), k = static_cast<int>(g % n);
            lsm::Extended* A = &acc[static_cast<std::size_t>(g) * (n + 2)];
            for (i64 q = 0; q < len; ++q) {
                const double* v = &buf[static_cast<std::size_t>(q) * stride];
                const double wb = w[start + q];
                auto add = [&](lsm::Extended& s, double t) {
                    const double p = t * wb;
                    if (!std::isnan(p)) s += p;
                };
                if (k == j) {
                    add(A[n], v[j]);                             // Rule 1
                    add(A[n + 1], v[n + j]);                     // Rule 2
                    for (int i = 0; i < n; ++i)
                        if (i != j) add(A[i], v[2 * n + j * n + i]);  // Rule 3
                } else {
                    const double* qq = v + 2 * n + n * n;
                    const double rr = qq[n + j], ss = qq[2 * n + k];
                    const double oma = v[stride - 2], bsum = v[stride - 1];
                    for (int i = 0; i < n; ++i) add(A[i], (oma * (qq[i] * rr * ss)) / bsum);  // Rule 4
                }
            }
        });
    }
    auto at = [&](int row, int col) -> double& { return tab[static_cast<std::size_t>(row) * N + col]; };
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            const lsm::Extended* A = &acc[static_cast<std::size_t>(j * n + k) * (n + 2)];
            if (k == j) {
                at((j + 1) * N - 1, j) = static_cast<double>(A[n]) / swt;
                at(j + j * N, j) = static_cast<double>(A[n + 1]) / swt;
                for (int i = 0; i < n; ++i)
                    if (i != j) at(i + j * N, j) = static_cast<double>(A[i]) / swt;
            } else {
                for (int i = 0; i < n; ++i) at(i + j * N, k) = static_cast<double>(A[i]) / swt;
            }
        }
    return tab;
}

}  // namespace sim
