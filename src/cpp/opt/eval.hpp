// SPDX-License-Identifier: GPL-3.0-or-later
//
// prioritizr 8.0.6's evaluation of solutions (eval_*_summary, eval_ferrier_importance), the same
// arithmetic in the same order as its C++ (Armadillo) and R (Matrix) code, so that the results are
// equal bit for bit. Sparse matrices come in compressed columns (Matrix's dgCMatrix, Armadillo's
// sp_mat layout).
#pragma once

#include <cstdint>
#include <vector>

namespace opt::eval {

using i64 = std::int64_t;

struct Csc {
    i64 nrow = 0, ncol = 0;
    std::vector<i64> start;  // ncol + 1
    std::vector<i64> index;  // row of each non-zero
    std::vector<double> value;
};

// R's pnorm(x) (nmath's pnorm5 with mean 0, sd 1, the lower tail).
double pnorm(double x);

// Armadillo's arrayops::accumulate: two interleaved partial sums, then their sum.
double accumulate(const double* x, i64 n);

// rcpp_boundary for one zone: lower holds the shared lengths below the diagonal (diagonal removed),
// exposed and total every unit's exposed and total boundary.
double boundary(double edge_factor, const Csc& lower, const std::vector<double>& exposed,
                const std::vector<double>& total, const std::vector<double>& solution);

// rcpp_connectivity for one zone (the matrix scaled by the solution by rows and columns, then summed).
double connectivity(const Csc& data, const std::vector<double>& solution);

// rcpp_asym_connectivity for one zone.
double asym_connectivity(const Csc& data, const std::vector<double>& solution);

// rcpp_ferrier_score: rij features x units; selected: the units of the solution (portfolio); the
// scores of the (feature, selected unit) pairs with an amount, as features x units in compressed columns.
Csc ferrier_score(const Csc& rij, const std::vector<double>& targets, const std::vector<std::uint8_t>& selected);

}  // namespace opt::eval
