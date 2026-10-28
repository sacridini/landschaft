// SPDX-License-Identifier: GPL-3.0-or-later
//
// Distances along a river network (see hydro.hpp): between points, down from where
// their paths meet, and the stretches up and down the river within a distance of each.

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hydro/hydro.hpp"

namespace hydro {

namespace {

// The path from a cell down to the edge of the grid: each cell and its distance from the
// first (a circle stopped after every cell).
std::vector<std::pair<i64, double>> path_down(i64 k, const i64* down, const double* step, i64 n) {
    std::vector<std::pair<i64, double>> out{{k, 0.0}};
    i64 at = k;
    while (down[at] >= 0 && static_cast<i64>(out.size()) <= n) {
        const double d = out.back().second + step[at];
        at = down[at];
        out.emplace_back(at, d);
    }
    return out;
}

}  // namespace

RiverDistances river_distances(const i64* down, const double* step, i64 n, const i64* cells, i64 p) {
    std::vector<std::vector<std::pair<i64, double>>> paths(static_cast<std::size_t>(p));
    for (i64 i = 0; i < p; ++i)
        if (cells[i] >= 0) paths[i] = path_down(cells[i], down, step, n);
    RiverDistances out;
    const auto pairs = static_cast<std::size_t>(p * (p - 1) / 2);
    out.network.assign(pairs, std::numeric_limits<double>::quiet_NaN());
    out.connected.assign(pairs, 0);
    out.flow.assign(pairs, 0);
    std::size_t row = 0;
    for (i64 i = 0; i < p; ++i) {
        std::unordered_map<i64, double> index;
        if (cells[i] >= 0) index.insert(paths[i].begin(), paths[i].end());
        for (i64 j = i + 1; j < p; ++j, ++row) {
            if (cells[i] < 0 || cells[j] < 0) continue;
            for (const auto& [m, db] : paths[j]) {
                const auto it = index.find(m);
                if (it == index.end()) continue;
                out.network[row] = it->second + db;
                out.connected[row] = 1;
                out.flow[row] = (m == cells[i] || m == cells[j]) ? 1 : 0;
                break;
            }
        }
    }
    return out;
}

void along_network(const i64* down, const double* step, const std::uint8_t* on, i64 n, const i64* cells, i64 p,
                   double upstream, double downstream, std::int32_t* label, double* dist) {
    // the stream cells draining into each, in a row (compressed)
    std::vector<i64> first(static_cast<std::size_t>(n + 1), 0);
    for (i64 k = 0; k < n; ++k)
        if (on[k] && down[k] >= 0) ++first[down[k] + 1];
    for (i64 k = 0; k < n; ++k) first[k + 1] += first[k];
    std::vector<i64> ups(static_cast<std::size_t>(first[n]));
    std::vector<i64> fill(first.begin(), first.end());
    for (i64 k = 0; k < n; ++k)
        if (on[k] && down[k] >= 0) ups[fill[down[k]]++] = k;
    for (i64 k = 0; k < n; ++k) {
        label[k] = 0;
        dist[k] = std::numeric_limits<double>::infinity();
    }
    auto reach = [&](i64 k, double d, std::int32_t id) {
        if (d < dist[k]) {
            dist[k] = d;
            label[k] = id;
        }
    };
    std::vector<std::pair<i64, double>> stack;
    for (i64 i = 0; i < p; ++i) {
        const i64 k = cells[i];
        if (k < 0) continue;
        const auto id = static_cast<std::int32_t>(i + 1);
        for (const auto& [c, d] : path_down(k, down, step, n)) {
            if (d > downstream * (1.0 + 1e-12)) break;
            reach(c, d, id);
        }
        stack.assign(1, {k, 0.0});
        while (!stack.empty()) {
            const auto [c, d] = stack.back();
            stack.pop_back();
            for (i64 q = first[c]; q < first[c + 1]; ++q) {
                const i64 u = ups[q];
                const double du = d + step[u];
                if (du <= upstream * (1.0 + 1e-12)) {
                    reach(u, du, id);
                    stack.emplace_back(u, du);
                }
            }
        }
    }
}

}  // namespace hydro
