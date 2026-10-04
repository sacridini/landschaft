// SPDX-License-Identifier: GPL-3.0-or-later
//
// The three-map comparison of Pontius, Peethambaram & Castella (2011) as
// lulcc's ThreeMapComparison computes it, for one resolution: for every
// block b, the shares Q (reference t0), R (reference t1) and S (simulated
// t1) of each class, Equations 1-7 in lulcc's operation order (doubles),
// and the weighted shares of the table (Rules 1-4) summed over the blocks
// in R's 80-bit LDOUBLE, NaN skipped (sum(v * w, na.rm = TRUE)), in block
// order. Equation 4's row sums are R's rowSums(na.rm = TRUE).
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

// x, x1, y1: (n, blocks) cells of each class per block; w: blocks (cells
// with a value per block). Returns lulcc's table ((n + 1)^2 x (n + 1),
// row-major) with Rules 1-4 filled and NaN elsewhere (the totals are left
// to the caller).
std::vector<double> tmc_table(int n, std::int64_t blocks, const double* x, const double* x1, const double* y1,
                              const double* w);

}  // namespace sim
