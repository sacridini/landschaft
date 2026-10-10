// SPDX-License-Identifier: GPL-3.0-or-later
//
// How much land each class needs at each step: lulcc's linear demand
// (Hmisc::approxExtrap and roundSum) and Markov chains of land change
// (transition probabilities from cell counts, the principal root of a
// period's matrix for one time step, as Dinamica EGO's
// DetermineTransitionMatrix, and the projection of areas).
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

// Hmisc::approxExtrap(x, y, xout): R's approx (linear) between the points
// (duplicated x after the first dropped, sorted by a stable sort), the lines
// of the end segments outside them. Needs two distinct x at least.
std::vector<double> approx_extrap(const std::vector<double>& x, const std::vector<double>& y,
                                  const std::vector<double>& xout);

// lulcc's roundSum on each row of x (nrow x ncol, row-major): the row scaled
// to add up to ncell (R's 80-bit sum), floored, and the cells still missing
// given to the largest fractions (a stable sort, as R's sort with
// index.return = TRUE).
std::vector<double> round_sum(const std::vector<double>& x, std::int64_t nrow, std::int64_t ncol, std::int64_t ncell);

// The transition probabilities of a k x k matrix of cell counts (rows the
// classes at the start): the classes with cells at the start, and those only
// reached from them, are kept (`kept` gets their indices, increasing); a
// kept class without cells at the start stays as it is.
std::vector<double> markov_from_counts(const std::vector<double>& counts, int k, std::vector<int>& kept);

// The principal n-th root of a k x k transition matrix: expm(logm(P) / n);
// for a singular matrix (no logarithm) V diag(w^(1/n)) V^-1 from its
// eigendecomposition. Throws when P has a negative real eigenvalue (no real
// root) or is singular and not diagonalisable.
std::vector<double> markov_root(const std::vector<double>& p, int k, int n);

// The areas of each class after each of `steps` steps of a k x k matrix:
// (steps + 1) x k, row 0 the given areas, row s = row s-1 x P.
std::vector<double> markov_project(const std::vector<double>& areas, const std::vector<double>& p, int k, int steps);

}  // namespace sim
