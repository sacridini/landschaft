// SPDX-License-Identifier: GPL-3.0-or-later
//
// The decision rules of CLUE-S as lulcc applies them (see rules.hpp), the
// same operations as landschaft's numpy code that came before, cell by cell.

#include "sim/rules.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/parallel.hpp"
#include "sim/clues.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;
constexpr i64 kParallelCells = 1 << 15;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int class_index(double v, const double* classes, int ncode) {
    for (int j = 0; j < ncode; ++j)
        if (classes[j] == v) return j;
    return -1;
}

}  // namespace

std::vector<double> allow_matrix(const double* x, const std::int64_t* hist, std::int64_t ncell, const double* classes,
                                 int ncode, const double* cd, const std::int64_t* rules) {
    std::vector<int> lu(static_cast<std::size_t>(ncell));
    for (i64 i = 0; i < ncell; ++i) {
        lu[static_cast<std::size_t>(i)] = class_index(x[i], classes, ncode);
        if (lu[static_cast<std::size_t>(i)] < 0) throw std::invalid_argument("the landscape has classes the model does not know");
    }
    std::vector<double> out(static_cast<std::size_t>(ncell) * static_cast<std::size_t>(ncode));
    lsm::parallel_for(ncell >= kParallelCells, ncell, [&](i64 i) {
        const int from = lu[static_cast<std::size_t>(i)];
        const i64 h = hist[i];
        const bool decreasing = cd[from] < 0;
        for (int k = 0; k < ncode; ++k) {
            const i64 r = rules[static_cast<std::size_t>(from) * static_cast<std::size_t>(ncode) + k];
            bool ok = true;
            if (r == 0) ok = false;
            else if (r == -1) ok = decreasing;
            else if (r == -2) ok = cd[k] > 0;
            else if (r >= 100 && r < 1000) ok = h < r - 100;
            else if (r >= 1000) ok = !(h < r - 1000);
            out[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncode) + k] = ok ? 1.0 : kNaN;
        }
    });
    return out;
}

std::vector<double> auto_convert(const double* x, const double* prob, std::int64_t ncell, const double* classes,
                                 int ncode, const std::int64_t* mask) {
    std::vector<int> lu(static_cast<std::size_t>(ncell));
    for (i64 i = 0; i < ncell; ++i) {
        lu[static_cast<std::size_t>(i)] = class_index(x[i], classes, ncode);
        if (lu[static_cast<std::size_t>(i)] < 0) throw std::invalid_argument("the landscape has classes the model does not know");
    }
    std::vector<double> out(static_cast<std::size_t>(ncell), kNaN);
    lsm::parallel_for(ncell >= kParallelCells, ncell, [&](i64 i) {
        const double* p = prob + static_cast<std::size_t>(i) * static_cast<std::size_t>(ncode);
        if (std::isnan(p[lu[static_cast<std::size_t>(i)]])) {
            // the first largest suitability; lulcc starts the search at -1
            int arg = 0;
            double best = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < ncode; ++k) {
                const double v = std::isnan(p[k]) ? -std::numeric_limits<double>::infinity() : p[k];
                if (v > best) {
                    best = v;
                    arg = k;
                }
            }
            out[static_cast<std::size_t>(i)] = best > -1 ? classes[arg] : x[i];
        } else if (mask[i] == 0) {
            out[static_cast<std::size_t>(i)] = x[i];
        }
    });
    return out;
}

std::vector<double> neighbourhood_allow(const double* landscape, std::int64_t ny, std::int64_t nx,
                                        const double* classes, int ncode,
                                        const std::vector<NeighbourhoodRule>& rules) {
    const i64 ncell = ny * nx;
    std::vector<double> out(static_cast<std::size_t>(ncell) * static_cast<std::size_t>(ncode), 1.0);
    std::vector<double> presence(static_cast<std::size_t>(ncell));
    for (const auto& rule : rules) {
        const int j = class_index(rule.cls, classes, ncode);
        if (j < 0) continue;
        if (rule.wy % 2 == 0 || rule.wx % 2 == 0 || static_cast<i64>(rule.weights.size()) != i64{rule.wy} * rule.wx)
            throw std::invalid_argument("a neighbourhood window needs odd sides and a weight per cell");
        for (i64 i = 0; i < ncell; ++i)
            presence[static_cast<std::size_t>(i)] = std::isnan(landscape[i]) ? kNaN : (landscape[i] == rule.cls ? 1.0 : 0.0);
        const auto frac = focal_mean(presence.data(), ny, nx, rule.weights.data(), rule.wy, rule.wx);
        for (i64 i = 0; i < ncell; ++i)  // NaN (no neighbours) fails the threshold
            out[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncode) + j] =
                frac[static_cast<std::size_t>(i)] >= rule.threshold ? 1.0 : kNaN;
    }
    return out;
}

}  // namespace sim
