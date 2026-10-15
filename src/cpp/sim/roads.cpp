// SPDX-License-Identifier: GPL-3.0-or-later
//
// Endogenous road building: see roads.hpp.

#include "sim/roads.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "sim/rng.hpp"
#include "sim/weighted.hpp"

namespace sim {

namespace {

// the 3 x 3 window in row order and Dinamica's direction codes
constexpr int kDr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int kDc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr std::uint8_t kCode[8] = {1, 2, 3, 8, 4, 7, 6, 5};

int window_index(std::uint8_t code) {
    for (int k = 0; k < 8; ++k)
        if (kCode[k] == code) return k;
    return -1;
}

}  // namespace

FrictionCost friction_cost(std::int64_t ny, std::int64_t nx, const std::uint8_t* source, const double* friction,
                           double diagonal, bool directions) {
    const std::int64_t n = ny * nx;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> cost(n, inf);
    using Item = std::pair<double, std::int64_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> heap;
    for (std::int64_t c = 0; c < n; ++c) {
        if (source[c] == 0) continue;
        cost[c] = 0;
        heap.emplace(0.0, c);
    }
    while (!heap.empty()) {
        const auto [d, c] = heap.top();
        heap.pop();
        if (d > cost[c]) continue;
        const std::int64_t r = c / nx, k = c % nx;
        for (int w = 0; w < 8; ++w) {
            const std::int64_t rr = r + kDr[w], kk = k + kDc[w];
            if (rr < 0 || rr >= ny || kk < 0 || kk >= nx) continue;
            const std::int64_t nb = rr * nx + kk;
            const double f = friction[nb];
            if (!(f == f) || source[nb] != 0) continue;  // missing friction: impassable
            const double step = (kDr[w] != 0 && kDc[w] != 0) ? f * diagonal : f;
            const double nd = d + step;
            if (nd < cost[nb]) {
                cost[nb] = nd;
                heap.emplace(nd, nb);
            }
        }
    }
    FrictionCost out;
    out.dir.assign(directions ? n : 0, 0);
    for (std::int64_t c = 0; c < n; ++c) {
        if (cost[c] == inf || (source[c] == 0 && !(friction[c] == friction[c]))) {
            cost[c] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        if (!directions || source[c] != 0) continue;
        const std::int64_t r = c / nx, k = c % nx;
        for (int w = 0; w < 8; ++w) {  // the first neighbour of row order the cheapest path comes through
            const std::int64_t rr = r + kDr[w], kk = k + kDc[w];
            if (rr < 0 || rr >= ny || kk < 0 || kk >= nx) continue;
            const double cn = cost[rr * nx + kk];
            if (cn == inf || !(cn == cn)) continue;
            const double step = (kDr[w] != 0 && kDc[w] != 0) ? friction[c] * diagonal : friction[c];
            if (cn + step == cost[c]) {
                out.dir[c] = kCode[w];
                break;
            }
        }
    }
    out.cost = std::move(cost);
    return out;
}

std::vector<std::int64_t> destinations(std::int64_t ny, std::int64_t nx, const double* attraction, int horizontal,
                                       int vertical, int per_quadrant, double min_distance, int max_attempts,
                                       double dx, double dy, std::uint64_t seed) {
    if (horizontal < 1 || vertical < 1) throw std::invalid_argument("the quadrants must be at least 1 x 1");
    if (per_quadrant < 0 || max_attempts < 1) throw std::invalid_argument("invalid destinations or attempts");
    Xoshiro rng(seed);
    std::vector<std::int64_t> chosen;
    const double min2 = min_distance * min_distance;
    auto far_enough = [&](std::int64_t c) {
        if (!(min_distance > 0)) return true;
        const double y = static_cast<double>(c / nx) * dy, x = static_cast<double>(c % nx) * dx;
        for (const auto o : chosen) {
            const double oy = static_cast<double>(o / nx) * dy - y, ox = static_cast<double>(o % nx) * dx - x;
            if (ox * ox + oy * oy < min2) return false;
        }
        return true;
    };
    for (int i = 0; i < vertical; ++i) {
        const std::int64_t r0 = i * ny / vertical, r1 = (i + 1) * ny / vertical;
        for (int j = 0; j < horizontal; ++j) {
            const std::int64_t c0 = j * nx / horizontal, c1 = (j + 1) * nx / horizontal;
            std::vector<std::int64_t> cells;
            std::vector<double> w;
            for (std::int64_t r = r0; r < r1; ++r)
                for (std::int64_t k = c0; k < c1; ++k) {
                    const double a = attraction[r * nx + k];
                    if (a > 0 && std::isfinite(a)) {
                        cells.push_back(r * nx + k);
                        w.push_back(a);
                    }
                }
            if (cells.empty()) continue;
            Weighted draw(w);
            for (int d = 0; d < per_quadrant; ++d) {
                bool placed = false;
                for (int t = 0; t < max_attempts && draw.left() > 0; ++t) {
                    const double total = draw.total();
                    if (!(total > 0)) break;
                    const std::size_t pick = draw.draw(rng.uniform() * total);
                    draw.zero(pick);
                    if (far_enough(cells[pick])) {
                        chosen.push_back(cells[pick]);
                        placed = true;
                        break;
                    }
                }
                if (!placed && draw.left() == 0) break;
            }
        }
    }
    return chosen;
}

std::vector<std::uint8_t> pathways(std::int64_t ny, std::int64_t nx, const std::uint8_t* network,
                                   const std::uint8_t* sources, const double* cost, const std::uint8_t* dir,
                                   bool lottery, std::uint64_t seed) {
    const std::int64_t n = ny * nx;
    std::vector<std::uint8_t> out(n, 0);
    Xoshiro rng(seed);
    auto where = [&](std::int64_t c) {
        return "(row " + std::to_string(c / nx) + ", column " + std::to_string(c % nx) + ")";
    };
    std::int64_t tie[8];
    for (std::int64_t s = 0; s < n; ++s) {
        if (sources[s] == 0) continue;
        std::int64_t c = s;
        if (cost != nullptr) {
            if (!(cost[c] == cost[c])) continue;  // a source of missing cost is skipped
            std::int64_t steps = 0;
            while (cost[c] > 0) {
                out[c] = 1;
                const std::int64_t r = c / nx, k = c % nx;
                double best = std::numeric_limits<double>::infinity();
                int nt = 0;
                for (int w = 0; w < 8; ++w) {
                    const std::int64_t rr = r + kDr[w], kk = k + kDc[w];
                    if (rr < 0 || rr >= ny || kk < 0 || kk >= nx) continue;
                    const std::int64_t nb = rr * nx + kk;
                    const double v = cost[nb];
                    if (!(v == v)) continue;
                    if (v < best) {
                        best = v;
                        nt = 0;
                        tie[nt++] = nb;
                    } else if (v == best && lottery) {
                        tie[nt++] = nb;
                    }
                }
                if (nt == 0 || !(best < cost[c]))
                    throw std::runtime_error("the costs have a local minimum at " + where(c) +
                                             ": no neighbour is cheaper on the way to the network");
                c = nt == 1 ? tie[0] : tie[static_cast<int>(rng.uniform() * nt)];
                if (++steps > n) throw std::runtime_error("a pathway does not reach the network");
            }
        } else {
            if (dir[c] == 255) continue;
            std::int64_t steps = 0;
            while (network[c] == 0 && dir[c] != 0) {
                out[c] = 1;
                const int w = window_index(dir[c]);
                if (w < 0) throw std::runtime_error("missing or invalid direction at " + where(c));
                const std::int64_t rr = c / nx + kDr[w], kk = c % nx + kDc[w];
                if (rr < 0 || rr >= ny || kk < 0 || kk >= nx)
                    throw std::runtime_error("a direction leaves the map at " + where(c));
                c = rr * nx + kk;
                if (dir[c] == 255 && network[c] == 0) throw std::runtime_error("missing direction at " + where(c));
                if (++steps > n) throw std::runtime_error("the directions loop at " + where(c));
            }
        }
    }
    return out;
}

}  // namespace sim
