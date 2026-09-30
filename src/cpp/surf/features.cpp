// SPDX-License-Identifier: GPL-3.0-or-later
//
// geodiv's local features (localsurface.R): findpeaks, findvalleys, sds, ssc,
// s10z. A peak is an interior cell above zero and above its 8 neighbours (all
// with a value); a valley one below zero and below them. geodiv's findvalleys
// fills its matrix by columns (findpeaks by rows), so it compares each cell
// with the 8 neighbours of its place in the matrix read by columns:
// `transposed_valleys` reproduces that.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// The interior cells (row order) whose value v satisfies cmp(v, neighbour)
// for its 8 neighbours n(r, c) and cmp(v, 0).
template <class Neighbour, class Cmp>
std::vector<i64> local_extremes(const Grid& g, Neighbour n, Cmp cmp) {
    std::vector<std::vector<i64>> rows(static_cast<std::size_t>(g.ny > 2 ? g.ny - 2 : 0));
#pragma omp parallel for schedule(static) if (g.size() > 65536)
    for (int r = 1; r < g.ny - 1; ++r) {
        auto& out = rows[static_cast<std::size_t>(r - 1)];
        for (int c = 1; c < g.nx - 1; ++c) {
            const double v = g.z[static_cast<i64>(r) * g.nx + c];
            if (!cmp(v, 0.0)) continue;
            bool all = true;
            for (int dr = -1; dr <= 1 && all; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (!cmp(v, n(r + dr, c + dc))) {
                        all = false;
                        break;
                    }
                }
            if (all) out.push_back(static_cast<i64>(r) * g.nx + c);
        }
    }
    std::vector<i64> cells;
    for (auto& r : rows) cells.insert(cells.end(), r.begin(), r.end());
    return cells;
}

}  // namespace

std::vector<i64> find_peaks(const Grid& g) {
    auto n = [&](int r, int c) { return g.z[static_cast<i64>(r) * g.nx + c]; };
    return local_extremes(g, n, [](double a, double b) { return a > b; });
}

std::vector<i64> find_valleys(const Grid& g, bool transposed) {
    auto below = [](double a, double b) { return a < b; };
    if (!transposed) {
        auto n = [&](int r, int c) { return g.z[static_cast<i64>(r) * g.nx + c]; };
        return local_extremes(g, n, below);
    }
    // matrix(z, nrow = ny, ncol = nx)[r, c]: element c * ny + r of the values
    auto n = [&](int r, int c) { return g.z[static_cast<i64>(c) * g.ny + r]; };
    return local_extremes(g, n, below);
}

double sds(const Grid& g) {
    const double n = static_cast<double>(find_peaks(g).size());
    return n / ((static_cast<double>(g.nx) - 1) * (static_cast<double>(g.ny) - 1));
}

double ssc(const Grid& g) {
    const std::vector<i64> peaks = find_peaks(g);
    if (peaks.empty()) return NaN;
    // zshift(scale = TRUE) and deltax / deltay as geodiv (see hybrid.cpp)
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (i64 i = 0; i < g.size(); ++i) {
        const double v = g.z[i];
        if (std::isnan(v)) continue;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const double range = hi - lo;
    auto zn = [&](i64 i) { return (g.z[i] - lo) / range; };
    const double rx = (g.xmax - g.xmin) / g.nx, ry = (g.ymax - g.ymin) / g.ny;
    const double dx = rx / rx, dy = ry / rx;
    const i64 m = static_cast<i64>(peaks.size());
    const double sx = lsm::sum_of(0, m, [&](i64 k) {
        const i64 i = peaks[static_cast<std::size_t>(k)];
        const double v = zn(i);
        return (((v - zn(i + 1)) + (v - zn(i - 1))) / 2) / dx;
    });
    const double sy = lsm::sum_of(0, m, [&](i64 k) {
        const i64 i = peaks[static_cast<std::size_t>(k)];
        const double v = zn(i);
        return (((v - zn(i + g.nx)) + (v - zn(i - g.nx))) / 2) / dy;
    });
    return -(1 / (2 * static_cast<double>(m))) * (sx + sy);
}

namespace {

// The first 5 values of the cells in a stable sort by `before` (R's order()):
// one pass, a later cell passing only cells it strictly precedes.
template <class Before>
std::vector<double> first5(const Grid& g, const std::vector<i64>& cells, Before before) {
    std::vector<double> top;
    for (i64 i : cells) {
        const double v = g.z[i];
        if (top.size() == 5 && !before(v, top.back())) continue;
        auto at = std::upper_bound(top.begin(), top.end(), v, [&](double a, double b) { return before(a, b); });
        top.insert(at, v);
        if (top.size() > 5) top.pop_back();
    }
    return top;
}

}  // namespace

double s10z(const Grid& g, bool transposed_valleys) {
    // the 5 highest peaks and the 5 deepest valleys, in R's stable order
    const std::vector<double> sp = first5(g, find_peaks(g), [](double a, double b) { return a > b; });
    const std::vector<double> sv = first5(g, find_valleys(g, transposed_valleys),
                                          [](double a, double b) { return a < b; });
    const double hi = lsm::sum_of(0, static_cast<i64>(sp.size()), [&](i64 k) { return sp[static_cast<std::size_t>(k)]; });
    const double lo = lsm::sum_of(0, static_cast<i64>(sv.size()),
                                  [&](i64 k) { return std::fabs(sv[static_cast<std::size_t>(k)]); });
    return (hi + lo) / 5;
}

}  // namespace surf
