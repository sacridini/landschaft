// SPDX-License-Identifier: GPL-3.0-or-later
//
// Random walks on a symmetric transition (gdistance's commuteDistance and
// passage without theta): the graph of the cells with a connection
// (transitionSolidify), its Laplacian L = diag(colSums(T)) - T, grounded and
// factorised per connected component with the circuit module's multifrontal
// Cholesky. gdistance solves with CHOLMOD, so results agree to about 1e-12.
#pragma once

#include <cstdint>
#include <vector>

#include "cost/shortest.hpp"

namespace cost {

struct WalkGraph {
    std::vector<i64> cell;                // node -> cell, ascending
    std::vector<std::int32_t> node_of;    // cell -> node (-1: not on the graph)
    std::vector<std::int64_t> ptr;        // CSR over nodes, neighbours ascending
    std::vector<std::int32_t> adj;
    std::vector<double> w;                // conductances
    std::vector<double> degree;           // colSums (Matrix: double, ascending neighbours)
    std::vector<std::int32_t> component;  // node -> component
    std::int32_t n_components = 0;
    double volume = 0;                    // sum of the whole matrix
    std::vector<double> component_volume;
};

// The graph of a symmetric transition.
WalkGraph walk_graph(const Layout& t);

// Effective resistances between the given nodes (-1: not on the graph):
// row-major k x k, inf between components, NaN for nodes off the graph.
// `geometric`: 4 or 8 directions without wrapping, so the grid's nested
// dissection orders the unknowns (else the graph's own); nx: the grid width.
std::vector<double> effective_resistance(const WalkGraph& g, const std::vector<std::int32_t>& nodes, bool geometric,
                                         i64 nx);

// Net passages of a random walk from the origins to the goals (gdistance's
// passage without theta): half the absolute current through every node when a
// unit current enters at the origins (1/n_origins each) and leaves at the
// goals, 1 at the origins and goals, 0 off the graph. The circuit is grounded
// at the last node of the component; with `quirk` the ground's potential is
// taken as its predecessor's, as gdistance's .currentR does.
std::vector<double> random_walk_passages(const WalkGraph& g, const std::vector<std::int32_t>& origins,
                                         const std::vector<std::int32_t>& goals, bool quirk, bool geometric, i64 nx,
                                         i64 ncell);

// gdistance's .currentM for pathInc: for each location, the current on every
// connection (CSR order of the graph, both directions) when a unit current
// enters at `origin` and leaves at the location: |V_k - V_l| g_kl. Rows of
// locations in another part of the graph (or off it) are NaN.
std::vector<double> walk_flows(const WalkGraph& g, std::int32_t origin, const std::vector<std::int32_t>& locs,
                               bool quirk, bool geometric, i64 nx);

}  // namespace cost
