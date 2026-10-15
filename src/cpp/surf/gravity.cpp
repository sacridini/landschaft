// SPDX-License-Identifier: GPL-3.0-or-later
//
// Gravitational interaction potential (Dinamica EGO's CalcInteractionPotentialMap,
// clean-room from its documentation and black-box runs): every cell gets
// sum_k m_k / d_k^beta over the source cells k, d_k the distance between the cell
// centres, a source's own cell at `self_distance`.
//
// The weight depends only on the offset (|di|, |dj|), so one table of ny x nx
// weights serves every pair; each output row adds the sources in row order, a
// source at a time along the row (the same sum, in the same order, whatever the
// thread count). Dinamica sums float32 terms in float32; this sums in double.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include "surf/surface.hpp"

namespace surf {

namespace {

// 1 / d^beta from d^2 (exact squared offsets in cells: portable sqrt and division
// for the usual exponents).
double weight(double d2, double beta) {
    if (beta == 0.0) return 1.0;
    if (beta == 1.0) return 1.0 / std::sqrt(d2);
    if (beta == 2.0) return 1.0 / d2;
    if (beta == 3.0) return 1.0 / (d2 * std::sqrt(d2));
    if (beta == 4.0) return 1.0 / (d2 * d2);
    return std::pow(d2, -0.5 * beta);
}

}  // namespace

std::vector<double> gravity_weights(int ny, int nx, double beta, double dx, double dy, double self_distance) {
    if (ny <= 0 || nx <= 0) return {};
    std::vector<double> t(static_cast<std::size_t>(ny) * nx);
    for (int a = 0; a < ny; ++a) {
        const double ya = a * dy;
        for (int b = 0; b < nx; ++b) {
            const double xb = b * dx;
            const double d2 = (a == 0 && b == 0) ? self_distance * self_distance : ya * ya + xb * xb;
            t[static_cast<std::size_t>(a) * nx + b] = weight(d2, beta);
        }
    }
    return t;
}

std::vector<double> gravity(const double* mass, const double* mask, int ny, int nx, double beta, double dx,
                            double dy, double self_distance) {
    if (!(self_distance > 0.0) || !std::isfinite(self_distance))
        throw std::invalid_argument("the distance of a source to its own cell must be positive");
    if (!std::isfinite(beta)) throw std::invalid_argument("the exponent must be finite");
    const std::size_t n = static_cast<std::size_t>(ny) * nx;
    std::vector<int> si, sj;
    std::vector<double> sm;
    for (std::size_t c = 0; c < n; ++c) {
        const double m = mass[c];
        if (std::isnan(m) || m == 0.0) continue;
        si.push_back(static_cast<int>(c / nx));
        sj.push_back(static_cast<int>(c % nx));
        sm.push_back(m);
    }
    std::vector<double> out(n, 0.0);
    const std::vector<double> t = gravity_weights(ny, nx, beta, dx, dy, self_distance);
    const auto ns = static_cast<std::int64_t>(sm.size());
#pragma omp parallel for schedule(dynamic, 1) if (ns * static_cast<std::int64_t>(n) > 65536)
    for (int i = 0; i < ny; ++i) {
        double* row = out.data() + static_cast<std::size_t>(i) * nx;
        for (std::int64_t k = 0; k < ns; ++k) {
            const double* w = t.data() + static_cast<std::size_t>(std::abs(i - si[k])) * nx;
            const double m = sm[k];
            const int jk = sj[k];
            for (int j = 0; j < jk; ++j) row[j] += m * w[jk - j];
            for (int j = jk; j < nx; ++j) row[j] += m * w[j - jk];
        }
        if (mask != nullptr) {
            const double* mk = mask + static_cast<std::size_t>(i) * nx;
            for (int j = 0; j < nx; ++j)
                if (std::isnan(mk[j])) row[j] = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return out;
}

}  // namespace surf
