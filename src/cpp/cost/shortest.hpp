// SPDX-License-Identifier: GPL-3.0-or-later
//
// Least-cost distances and paths over a transition (gdistance's accCost,
// costDistance and shortestPath): the cost of a step from cell a to its
// neighbour b is 1 / T[a, b]. Dijkstra's algorithm as igraph runs it for
// gdistance: igraph's two-way indexed heap (a new or improved element rises
// above equal ones), neighbours in increasing cell order, a parent replaced
// only by a strictly shorter path. The distances are sums along the cheapest
// path, so any correct Dijkstra gives their bits; the order matters for
// which of several equally cheap paths shortestPath returns.
#pragma once

#include <cstdint>
#include <vector>

#include "cost/transition.hpp"

namespace cost {

struct Layout {
    const Grid* grid;
    int directions;
    bool symmetric;
    const double* values;
};

// Accumulated cost from the nearest of `sources` to every cell (inf where
// unreachable); gdistance's accCost.
std::vector<double> accumulated_cost(const Layout& t, const std::vector<i64>& sources);

// Costs from every `from` cell to every `to` cell (row-major n_from x n_to),
// sources in parallel (max_threads: 0 = all).
std::vector<double> cost_matrix(const Layout& t, const std::vector<i64>& from, const std::vector<i64>& to);

// The cells of the least-cost path from `origin` to each goal (empty when
// unreachable), with igraph's choice among equally cheap paths.
std::vector<std::vector<i64>> shortest_paths(const Layout& t, i64 origin, const std::vector<i64>& goals);

}  // namespace cost
