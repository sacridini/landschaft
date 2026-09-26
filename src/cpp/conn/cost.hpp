// SPDX-License-Identifier: GPL-3.0-or-later
//
// Least-cost distances between habitat nodes over a resistance raster, as
// gdistance computes them for Makurhini (transition(), geoCorrection(),
// costDistance()): between two neighbouring cells (4 or 8 directions) the
// conductance is f(neighbour, cell) with the cell of larger index last
// (Makurhini's default f = 1 / x[2], or 1 / mean(x)), divided by the
// distance between the cell centres (raster::pointDistance, R's vector sqrt);
// the cost of the step is its reciprocal, and a node's distance to another is
// the cheapest sum of steps between their cells (Dijkstra). With the same
// step costs every correct Dijkstra gives the same sums, so the distances are
// gdistance's to the last bit.
#pragma once

#include <cstdint>
#include <vector>

#include "conn/nodes.hpp"

namespace conn {

enum class CostFunction { Destination, Mean };

struct CostGrid {
    i64 nx = 0, ny = 0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    int directions = 8;
    CostFunction function = CostFunction::Destination;
};

// `resistance`: row-major, NaN where there is no cell. `node_cell`: the cell
// of every node (-1 outside the raster). Pairs i < j with a finite cost up to
// max_cost (unreachable pairs are left out). `max_threads` caps the threads
// (each keeps a cost per cell); 0 = all.
PairList least_cost_distances(const double* resistance, const CostGrid& grid, const i64* node_cell,
                              std::int32_t n_nodes, double max_cost, int max_threads);

// The conductance graph itself (gdistance's transition() and geoCorrection()
// on a projected raster): every pair of neighbouring cells a < b with a
// resistance, and f(a, b) / (the distance between their centres).
struct ConductanceEdges {
    std::vector<i64> a, b;
    std::vector<double> conductance;
};
ConductanceEdges conductance_edges(const double* resistance, const CostGrid& grid);

// The transient part Q of a spatial absorbing Markov chain as the samc package
// builds it (model RW, no fidelity): the states are the cells with a
// resistance (row-major); from cell i, a step goes to neighbour j (directions 4
// or 8, in samc's order) with probability (1 - absorption_i) w_ij / sum_k w_ik,
// w_ij = f(r_i, r_j) / (1 or sqrt(2) for diagonal steps, in cells), f = 1 / the
// mean of both (samc's and Makurhini's default) or 1 / r_j.
struct SamcTransitions {
    std::vector<i64> cell;               // raster cell of every state
    std::vector<std::int32_t> from, to;  // states
    std::vector<double> q;
};
SamcTransitions samc_transitions(const double* resistance, const double* absorption, i64 nx, i64 ny, int directions,
                                 CostFunction function);

}  // namespace conn
