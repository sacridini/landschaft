// SPDX-License-Identifier: GPL-3.0-or-later
//
// Variance inflation factors as the R package usdm computes them:
// 1 / (1 - R^2) of each variable regressed on the others, R^2 from
// summary(lm()) (R's lm.fit QR, R's 80-bit means and sums).
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

// VIF of the columns `which` of the n x m table `data` (column-major), each
// regressed on all the other m - 1 columns (with an intercept). Columns in
// parallel; each value is R's to the last bit.
std::vector<double> vif(const double* data, i64 n, i64 m, const std::vector<i64>& which);

}  // namespace sdm
