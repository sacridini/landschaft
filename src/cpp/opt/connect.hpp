// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cuts of the connected structures of the MILP prioritization, separated inside HiGHS's branch-and-bound
// (the user's separator of highs_patch.cmake): at the root and at every node, told the LP's solution.
//
// * A patch (contiguity, feature contiguity: prioritize._milp_spatial._add_connected): the selected nodes
//   form one arborescence from one root. Directed cuts r(W) + y(in(W)) >= x_k for every set W of nodes
//   holding node k: a patch reaches k from its root, inside W or through an arc into W. Separated by a
//   maximum flow from a super root (capacity r_v to every node v, y_a on the arcs) to k.
// * A corridor network (prioritize._milp_spatial._apply_corridors): node cuts x(S) >= 1 for every set S of
//   units that separates a terminal from the source (a selected path crosses it), by a maximum flow through
//   the units (capacity x_v, the terminals unbounded).
//
// The cuts are valid for every feasible selection: they only tighten the relaxation.
#pragma once

#include <cstdint>
#include <vector>

#include "opt/milp.hpp"

namespace opt {

// One set of node columns that must form one patch: the nodes' columns, the first column of their roots
// (the root of nodes[v] is root + v) and of the arcs (arc + a), and every arc's tail and head (positions in
// `nodes`). The arcs come in pairs, the second half the reverse of the first (as _add_connected makes them).
struct Connected {
    std::vector<i64> nodes;
    i64 root = 0;
    i64 arc = 0;
    std::vector<int> tail, head;
};

// A corridor network on the first `n` columns (the units): its source, its other terminals, and its edges
// (each once, both directions usable).
struct Corridor {
    int n = 0;
    int source = 0;
    std::vector<int> sinks;
    std::vector<int> tail, head;
};

// Cuts a x <= upper in compressed rows.
struct Cuts {
    std::vector<int> start{0};
    std::vector<int> index;
    std::vector<double> value;
    std::vector<double> upper;

    std::size_t size() const { return upper.size(); }
};

struct SeparateOptions {
    int max_candidates = 200;   // nodes tried per patch and call (the most selected first)
    double min_value = 0.05;    // nodes selected less than this are not tried
    double min_violation = 1e-4;
};

// The violated directed cuts of a solution x (the model's columns), as x_k - r(W) - y(in(W)) <= 0.
void separate_connected(const Connected& c, const std::vector<double>& x, const SeparateOptions& o, Cuts& out);

// The violated node cuts of a solution x, as -x(S) <= -1.
void separate_corridor(const Corridor& c, const std::vector<double>& x, const SeparateOptions& o, Cuts& out);

// Both, as a separator for SolveOptions::separator (it keeps its own copies; thread safe).
Separator connect_separator(std::vector<Connected> connected, std::vector<Corridor> corridors,
                            SeparateOptions options = {});

}  // namespace opt
