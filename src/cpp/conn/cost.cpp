// SPDX-License-Identifier: GPL-3.0-or-later
#include "conn/cost.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "lsm/rmath.hpp"

namespace conn {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Distances between neighbouring cell centres as raster::pointDistance gives
// them: sqrt((x1 - x2)^2 + (y1 - y2)^2) with the centres of xyFromCell and R's
// vector sqrt. Horizontal steps depend on the column only, vertical ones on
// the row; diagonal ones on both, from a table over the distinct values.
struct StepLengths {
    std::vector<double> dx, dy;        // x(c) - x(c + 1), y(r) - y(r + 1)
    std::vector<double> horizontal, vertical;
    std::vector<std::int32_t> kx, ky;  // index of dx[c], dy[r] among the distinct values
    std::vector<double> diagonal;      // [kx * n_dy + ky]
    std::size_t n_dy = 0;
    bool table = false;

    StepLengths(const CostGrid& g) {
        const double xres = (g.xmax - g.xmin) / static_cast<double>(g.nx);
        const double yres = (g.ymax - g.ymin) / static_cast<double>(g.ny);
        auto x = [&](i64 c) { return (static_cast<double>(c) + 0.5) * xres + g.xmin; };
        auto y = [&](i64 r) { return g.ymax - (static_cast<double>(r) + 0.5) * yres; };
        for (i64 c = 0; c + 1 < g.nx; ++c) {
            dx.push_back(x(c) - x(c + 1));
            horizontal.push_back(lsm::r_sqrt(dx.back() * dx.back() + 0.0 * 0.0));
        }
        for (i64 r = 0; r + 1 < g.ny; ++r) {
            dy.push_back(y(r) - y(r + 1));
            vertical.push_back(lsm::r_sqrt(0.0 * 0.0 + dy.back() * dy.back()));
        }
        if (g.directions != 8) return;
        auto index = [](const std::vector<double>& v, std::vector<double>& distinct, std::vector<std::int32_t>& k) {
            distinct = v;
            for (double& d : distinct) d = std::fabs(d);
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            k.resize(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) {
                k[i] = static_cast<std::int32_t>(std::lower_bound(distinct.begin(), distinct.end(), std::fabs(v[i])) -
                                                 distinct.begin());
            }
        };
        std::vector<double> ux, uy;
        index(dx, ux, kx);
        index(dy, uy, ky);
        if (ux.size() * uy.size() > (1u << 20)) return;
        table = true;
        n_dy = uy.size();
        diagonal.resize(ux.size() * uy.size());
        for (std::size_t a = 0; a < ux.size(); ++a) {
            for (std::size_t b = 0; b < uy.size(); ++b) diagonal[a * n_dy + b] = lsm::r_sqrt(ux[a] * ux[a] + uy[b] * uy[b]);
        }
    }

    // The step between (r, c) and (r2, c2), neighbours.
    double length(i64 r, i64 c, i64 r2, i64 c2) const {
        if (r == r2) return horizontal[std::min(c, c2)];
        if (c == c2) return vertical[std::min(r, r2)];
        const i64 cl = std::min(c, c2), rt = std::min(r, r2);
        if (table) return diagonal[static_cast<std::size_t>(kx[cl]) * n_dy + ky[rt]];
        return lsm::r_sqrt(dx[cl] * dx[cl] + dy[rt] * dy[rt]);
    }
};

struct Neighbour {
    int dr, dc;
};
constexpr Neighbour kRook[4] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
constexpr Neighbour kQueen[8] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

}  // namespace

ConductanceEdges conductance_edges(const double* resistance, const CostGrid& grid) {
    if (grid.directions != 4 && grid.directions != 8) throw std::invalid_argument("directions must be 4 or 8");
    const i64 nx = grid.nx, ny = grid.ny;
    const StepLengths steps(grid);
    // The neighbours after a cell in scan order.
    constexpr Neighbour kForward8[4] = {{0, 1}, {1, -1}, {1, 0}, {1, 1}};
    constexpr Neighbour kForward4[2] = {{0, 1}, {1, 0}};
    const Neighbour* nb = grid.directions == 8 ? kForward8 : kForward4;
    const int nnb = grid.directions / 2;
    ConductanceEdges out;
    for (i64 r = 0; r < ny; ++r) {
        for (i64 c = 0; c < nx; ++c) {
            const i64 a = r * nx + c;
            if (!std::isfinite(resistance[a])) continue;
            for (int k = 0; k < nnb; ++k) {
                const i64 r2 = r + nb[k].dr, c2 = c + nb[k].dc;
                if (r2 >= ny || c2 < 0 || c2 >= nx) continue;
                const i64 b = r2 * nx + c2;
                if (!std::isfinite(resistance[b])) continue;
                double f;
                if (grid.function == CostFunction::Destination) {
                    f = 1.0 / resistance[b];
                } else {
                    const double pair[2] = {resistance[a], resistance[b]};
                    f = 1.0 / lsm::r_mean(pair, 2);
                }
                out.a.push_back(a);
                out.b.push_back(b);
                out.conductance.push_back((1.0 / steps.length(r, c, r2, c2)) * f);
            }
        }
    }
    return out;
}

PairList least_cost_distances(const double* resistance, const CostGrid& grid, const i64* node_cell,
                              std::int32_t n_nodes, double max_cost, int max_threads) {
    if (grid.directions != 4 && grid.directions != 8) throw std::invalid_argument("directions must be 4 or 8");
    const i64 nx = grid.nx, ny = grid.ny, ncell = nx * ny;
    const StepLengths steps(grid);
    const Neighbour* nb = grid.directions == 8 ? kQueen : kRook;
    const int nnb = grid.directions;

    // The cost of the step between two cells (both with a resistance).
    auto step = [&](i64 a, i64 b) {
        const i64 lo = std::min(a, b), hi = std::max(a, b);
        double conductance;
        if (grid.function == CostFunction::Destination) {
            conductance = 1.0 / resistance[hi];
        } else {
            const double pair[2] = {resistance[lo], resistance[hi]};
            conductance = 1.0 / lsm::r_mean(pair, 2);
        }
        const double d = steps.length(a / nx, a % nx, b / nx, b % nx);
        return 1.0 / ((1.0 / d) * conductance);
    };

    // Target cells: a bitmap, and the nodes of each cell (sorted by cell).
    std::vector<std::uint64_t> is_target(static_cast<std::size_t>((ncell + 63) / 64), 0);
    std::vector<std::pair<i64, std::int32_t>> by_cell;
    for (std::int32_t v = 0; v < n_nodes; ++v) {
        if (node_cell[v] < 0 || node_cell[v] >= ncell) continue;
        by_cell.emplace_back(node_cell[v], v);
        is_target[static_cast<std::size_t>(node_cell[v] >> 6)] |= std::uint64_t(1) << (node_cell[v] & 63);
    }
    std::sort(by_cell.begin(), by_cell.end());

    std::vector<PairList> parts(n_nodes);
    int threads = 1;
#if defined(_OPENMP)
    threads = omp_get_max_threads();
    if (max_threads > 0) threads = std::min(threads, max_threads);
#endif
    (void)threads;
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> cost(static_cast<std::size_t>(ncell), kInf);
        std::vector<char> done(static_cast<std::size_t>(ncell), 0);
        std::vector<i64> touched;
        using Item = std::pair<double, i64>;
        std::priority_queue<Item, std::vector<Item>, std::greater<>> heap;
        std::vector<double> found(static_cast<std::size_t>(n_nodes), kInf);
#pragma omp for schedule(dynamic)
        for (std::int32_t i = 0; i < n_nodes; ++i) {
            const i64 start = node_cell[i];
            if (start < 0 || start >= ncell) continue;
            // Targets: the nodes before i. gdistance's costDistance keeps the
            // lower triangle of its matrix (as.dist), so the distance of a pair
            // is summed from its later node.
            std::int32_t remaining = 0;
            for (const auto& [cell, v] : by_cell) {
                (void)cell;
                if (v < i) ++remaining;
            }
            if (remaining == 0) continue;
            std::vector<std::int32_t> reached;
            cost[start] = 0;
            touched.push_back(start);
            heap.emplace(0.0, start);
            while (!heap.empty() && remaining > 0) {
                const auto [d, a] = heap.top();
                heap.pop();
                if (done[a] || d != cost[a]) continue;
                if (d > max_cost) break;
                done[a] = 1;
                if (is_target[static_cast<std::size_t>(a >> 6)] >> (a & 63) & 1) {
                    auto it = std::lower_bound(by_cell.begin(), by_cell.end(), std::make_pair(a, std::int32_t(-1)));
                    for (; it != by_cell.end() && it->first == a; ++it) {
                        if (it->second < i) {
                            found[it->second] = d;
                            reached.push_back(it->second);
                            --remaining;
                        }
                    }
                }
                if (!std::isfinite(resistance[a])) continue;  // no edges out of an NA cell
                const i64 r = a / nx, c = a % nx;
                for (int k = 0; k < nnb; ++k) {
                    const i64 r2 = r + nb[k].dr, c2 = c + nb[k].dc;
                    if (r2 < 0 || r2 >= ny || c2 < 0 || c2 >= nx) continue;
                    const i64 b = r2 * nx + c2;
                    if (done[b] || !std::isfinite(resistance[b])) continue;
                    const double nd = d + step(a, b);
                    if (nd < cost[b]) {
                        if (cost[b] == kInf) touched.push_back(b);
                        cost[b] = nd;
                        heap.emplace(nd, b);
                    }
                }
            }
            for (const i64 t : touched) {
                cost[t] = kInf;
                done[t] = 0;
            }
            touched.clear();
            while (!heap.empty()) heap.pop();
            std::sort(reached.begin(), reached.end());
            PairList& p = parts[i];
            for (const std::int32_t j : reached) {
                p.i.push_back(j);
                p.j.push_back(i);
                p.distance.push_back(found[j]);
                found[j] = kInf;
            }
        }
    }
    // Sorted by (i, j): part j holds the pairs (i, j) for i < j.
    std::vector<std::size_t> order;
    std::vector<std::int32_t> pi, pj;
    std::vector<double> pd;
    for (PairList& p : parts) {
        pi.insert(pi.end(), p.i.begin(), p.i.end());
        pj.insert(pj.end(), p.j.begin(), p.j.end());
        pd.insert(pd.end(), p.distance.begin(), p.distance.end());
        p = PairList();
    }
    order.resize(pi.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return pi[a] != pi[b] ? pi[a] < pi[b] : pj[a] < pj[b];
    });
    PairList out;
    out.i.reserve(order.size());
    out.j.reserve(order.size());
    out.distance.reserve(order.size());
    for (const std::size_t k : order) {
        out.i.push_back(pi[k]);
        out.j.push_back(pj[k]);
        out.distance.push_back(pd[k]);
    }
    return out;
}

}  // namespace conn
