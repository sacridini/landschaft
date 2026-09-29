// SPDX-License-Identifier: GPL-3.0-or-later
//
// Randomised shortest paths (Saerens et al. 2009) on a transition, as
// gdistance's passage(theta = ...) and rSPDistance compute them: the
// reference random walk P = T / rowSums(T), W = exp(-theta C) * P (method 1,
// C = 1 / T the step costs) or W = P^theta (method 2), and the fundamental
// matrix of I - W with the goals made absorbing. gdistance solves with sparse
// LU; for symmetric transitions I - W scaled row by row is symmetric positive
// definite, so it is factorised here with the circuit module's multifrontal
// Cholesky (sparse LU for asymmetric ones). The results agree with
// gdistance to about 1e-12.
#pragma once

#include <cstdint>
#include <vector>

#include "cost/shortest.hpp"

namespace cost {

struct RspOptions {
    double theta = 0;
    int method = 1;      // 1: exp(-theta C) P; 2: P^theta
    bool net = true;     // net (true) or total passages
    bool solidify = true;  // passage: the cells with outgoing connections only (transitionSolidify)
};

// gdistance's passage(x, origin, goal, theta): per cell, the expected number
// of visits (total) or net passages of the walk from the origins (1/n each)
// to the goals (their rows scaled by 1 - 1/n_goals). NaN off the graph.
// With `edges`, the passages of every connection instead (values layout of
// an asymmetric transition, directions x cells).
std::vector<double> rsp_passages(const Layout& t, const std::vector<i64>& origins, const std::vector<i64>& goals,
                                 const RspOptions& o, bool edges);

// gdistance's rSPDistance: the expected cost of the walk from each origin to
// each goal (row-major n_from x n_to).
std::vector<double> rsp_distances(const Layout& t, const std::vector<i64>& from, const std::vector<i64>& to,
                                  const RspOptions& o);

// The connections of the graph of cells with outgoing connections
// (transitionSolidify), in the order of the flows below: from, to, value.
struct GraphEdges {
    std::vector<i64> from, to;
    std::vector<double> t;
};
GraphEdges rsp_edges(const Layout& t);

// gdistance's .randomSP for pathInc: for each location, the net passages of
// the randomised shortest paths from `origin` to it on every connection
// (skewpart(N) * 2, negatives set to 0); rows of zeros when the location
// cannot be reached, NaN off the graph.
std::vector<double> rsp_flows(const Layout& t, i64 origin, const std::vector<i64>& locs, double theta);

// pathInc's reduction: out[(f, r, p, q)] = sum_e fn_f(fa[p, e], fb[q, e]) r[r, e]
// for fn 0 = overlap (pmin), 1 = nonoverlap; `upper`: only q >= p.
void path_overlap(const double* fa, std::int64_t na, const double* fb, std::int64_t nb, std::int64_t ne,
                  const double* r, std::int64_t nr, const std::vector<int>& functions, bool upper, double* out);

}  // namespace cost
