// SPDX-License-Identifier: GPL-3.0-or-later
//
// Distances of new conditions to a reference set (the R package mop).
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

// For each row of g (ng x p, row-major), the mean (R's 80-bit mean) of its k
// smallest Euclidean distances to the rows of m (nm x p): mop's
// mean(sort(eucdist(g_i, m))[1:k]). Distances summed over the variables in
// order, squared by multiplication, square root correctly rounded (mop's
// eucdist_mm). `squared` takes the squared distances (for Mahalanobis on
// whitened data). Rows of g in parallel.
std::vector<double> knn_mean_distance(const double* m, i64 nm, const double* g, i64 ng, i64 p, i64 k, bool squared);

}  // namespace sdm
