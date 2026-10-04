// SPDX-License-Identifier: GPL-3.0-or-later
//
// Minimum s-t cuts by the Boykov-Kolmogorov max-flow algorithm (Y. Boykov and
// V. Kolmogorov, "An experimental comparison of min-cut/max-flow algorithms
// for energy minimization in vision", IEEE PAMI 26(9), 2004), written here from
// the paper: two search trees grown from the terminals, augmentation along the
// paths where they meet and adoption of the orphans, with the paper's
// distance and timestamp heuristics. Capacities are doubles.
//
// Built for energies of binary labels, sum_i u_i x_i + sum_ij w_ij [x_i != x_j]
// with w_ij >= 0: the graph (the pairs) is fixed and the terminal capacities
// (the unit terms) are set before every solve.
#pragma once

#include <cstdint>
#include <vector>

namespace opt {

class MaxFlow {
public:
    // n nodes; edges (i[k], j[k]) of capacity cap_ij[k] from i to j and cap_ji[k] back
    MaxFlow(int n, const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& cap_ij,
            const std::vector<double>& cap_ji);

    // every node's capacities from the source and to the sink (>= 0), then the max flow (the min cut's
    // capacity); the edges' capacities are restored first
    double solve(const std::vector<double>& source, const std::vector<double>& sink);

    // after solve(): whether a node is on the source's side of the minimum cut (the nodes that the source
    // reaches in the residual graph)
    bool source_side(int v) const { return parent_[v] != kNone && tree_[v] == kSourceTree; }

    int nodes() const { return n_; }

private:
    static constexpr int kTerminal = -1, kOrphan = -2, kNone = -3;
    static constexpr std::uint8_t kSourceTree = 1, kSinkTree = 2;

    int tail(int a) const { return head_[sister_[a]]; }
    void activate(int v);
    int next_active();
    double augment(int a);  // the flow pushed
    void set_orphan(int v);
    void adopt(int v);
    // the distance of v's root from the terminal, through valid parents (-1: none)
    int origin_distance(int v);

    int n_;
    std::vector<std::int64_t> first_;  // n + 1: the arcs out of every node
    std::vector<int> head_, sister_;
    std::vector<double> cap_, rcap_;  // every arc's capacity and residual capacity
    std::vector<double> tr_;          // > 0: residual from the source; < 0: to the sink
    std::vector<int> parent_;         // the arc to the parent in the node's tree (or a mark above)
    std::vector<std::uint8_t> tree_;
    std::vector<int> dist_;
    std::vector<std::int64_t> ts_;
    std::int64_t time_ = 0;
    std::vector<int> queue_;  // the active nodes (FIFO)
    std::size_t queue_head_ = 0;
    std::vector<std::uint8_t> active_;
    std::vector<int> orphans_;
};

}  // namespace opt
