// SPDX-License-Identifier: GPL-3.0-or-later
//
// Habitat connectivity graphs: the indices of Conefor Sensinode (NC, NL, H,
// CCP, LCP, IIC, F, AWF, PC) and the node importances (dI and its intra,
// flux and connector fractions) as Makurhini's MK_dPCIIC and Conefor compute
// them.
//
// Every shortest path is found once per source (breadth-first for links,
// Dijkstra over -log p for probabilities), in parallel over the sources.
// Removing a node k only changes the paths that ran through it: in the
// shortest-path tree of a source, those are the subtree of k, whose new
// lengths come from a Dijkstra restricted to that subtree and seeded from its
// untouched neighbours. The connector fraction of every node is summed from
// these repairs instead of recomputing the whole graph once per node.
#pragma once

#include <cstdint>
#include <vector>

namespace conn {

using i64 = std::int64_t;

// Symmetric adjacency (compressed rows). `cost` is -log p on probabilistic
// graphs and empty on graphs of links.
struct Csr {
    std::int32_t n = 0;
    std::vector<i64> start;
    std::vector<std::int32_t> to;
    std::vector<double> cost;
    i64 n_edges() const { return static_cast<i64>(to.size()) / 2; }
};

// Node pairs (i < j) with a value: a distance, a probability or a link (0/1).
struct Pairs {
    std::vector<std::int32_t> i, j;
    std::vector<double> value;
};

enum class Kind { Distance, Probability, Link };

// How a threshold turns pairs into edges.
struct Scenario {
    Kind kind = Kind::Distance;
    double threshold = 0;    // link distance (or minimum probability); distance of `probability`
    double probability = 0.5;
    double pij_min = 0;      // probabilistic edges below it are dropped
    bool conefor = false;    // Conefor's rules instead of Makurhini's (see make_link_graph/make_prob_graph)
};

// Binary graph: Makurhini links pairs closer than the threshold (d < t),
// Conefor up to it (d <= t); probabilities link when p >= t, links when != 0.
Csr make_link_graph(std::int32_t n, const Pairs& pairs, const Scenario& s);

// Probabilistic graph: p = exp(d log(P) / t) (Makurhini) or exp((log(P) / t) d)
// (Conefor), kept when p >= pij_min (Makurhini) or p > pij_min (Conefor), with
// cost -log p. `direct` receives the probability of every pair (uncut, for F
// and AWF) as sums per node: sum_j p_kj and sum_j a_j p_kj.
Csr make_prob_graph(std::int32_t n, const Pairs& pairs, const Scenario& s, const double* area,
                    std::vector<double>* f_row, std::vector<double>* awf_row);

struct LinkResult {
    i64 nl = 0;
    std::int32_t nc = 0;
    double h = 0;           // sum over pairs i < j of 1 / nl_ij (connected pairs)
    double iic_reach = 0;   // sum over ordered pairs (i = j included) of a_i a_j / (1 + nl_ij)
    double iic_direct = 0;  // ordered pairs one link apart: a_i a_j / 2
    double unreach_aa = 0;  // ordered pairs without a path: a_i a_j
    double comp_sq = 0;     // sum over components of their attribute squared
    std::vector<std::int32_t> component;
    // Per node (importance only).
    std::vector<double> iic_row;    // sum_j a_j / (1 + nl_kj), j = k included
    std::vector<double> unreach_a;  // sum of a_j over the nodes k cannot reach
    std::vector<double> h_row;      // sum over j != k of 1 / nl_kj
    std::vector<double> iic_conn;   // ordered pairs i, j != k: a_i a_j (1/(1+nl) - 1/(1+nl')) once k is removed
    std::vector<double> h_conn;     // pairs i < j, both != k: 1/nl - 1/nl'
    std::vector<i64> degree;
    std::vector<std::int32_t> pieces;  // components the component of k splits into without k
    std::vector<double> pieces_sq;     // sum of their attributes squared
};

LinkResult analyse_links(const Csr& g, const double* area, bool importance);

struct ProbResult {
    double pc_num = 0;     // sum over ordered pairs (i = j included) of a_i a_j p*_ij
    double pc_direct = 0;  // ordered pairs i != j joined by an edge: a_i a_j p_ij
    std::vector<double> pc_row;   // sum_j a_j p*_kj, j = k included (importance only)
    std::vector<double> pc_conn;  // ordered pairs i, j != k: a_i a_j (p*_ij - p*'_ij) once k is removed
};

// `expand_limit`: Conefor's minProb heuristic, nodes reached with a cost above
// it are not expanded (infinity to expand every node).
ProbResult analyse_probabilities(const Csr& g, const double* area, double expand_limit, bool importance);

}  // namespace conn
