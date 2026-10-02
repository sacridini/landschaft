// SPDX-License-Identifier: GPL-3.0-or-later
//
// R statistics the SDM evaluation tools call, to the last bit.
#pragma once

#include <cstdint>

namespace sdm {

using i64 = std::int64_t;

// R's cor(x, y) of two vectors without missing values (cov.c's cov_na_2):
// two-pass 80-bit means, 80-bit cross products, standard deviations as the
// 80-bit sqrtl of the 80-bit variance; NaN when a standard deviation is 0.
// `kendall`: Kendall's tau (exact sign sums over sqrtl of the tie-free pair
// counts). Spearman is Pearson on average ranks.
double r_cor2(const double* x, const double* y, i64 n, bool kendall);

}  // namespace sdm
