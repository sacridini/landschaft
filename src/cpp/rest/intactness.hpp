// SPDX-License-Identifier: GPL-3.0-or-later
//
// Ecological intactness (Beyer et al. 2020, Conservation Letters 13:e12692), as plangea's
// multi_calc_it computes it (R's arithmetic kept), with the window clipped at the raster's edges:
// habitat quality q = exp(-gamma HFI) from the human footprint, and the intactness of a cell
// Q_i = sum_j (q_i q_j)^z w_ij / sum_j w_ij over the cells j within `radius` cells (w = exp(beta d),
// d in km). Restoration lowers the footprint: HFI - reduction x, at least 0.
#pragma once

#include <cstdint>
#include <vector>

namespace rest {

using i64 = std::int64_t;

struct IntactnessParams {
    double gamma, beta, z;
    int radius;    // cells
    double dx, dy;  // a cell's width and height, km
};

// The footprint after restoring the shares `restored` of the cells (null: none) with `reduction`
// (the footprint a cell loses per unit restored; null: none): ny x nx, NaN where hfi is.
std::vector<double> restored_footprint(const double* hfi, const double* reduction, const double* restored, i64 ny,
                                       i64 nx, int threads);

// Q of every cell (NaN where the footprint is NaN).
std::vector<double> intactness(const double* hfi, i64 ny, i64 nx, const IntactnessParams& p, int threads);

// d(sum of Q over the cells) / d(the share of cell k restored), where restoring lowers the cell's
// footprint by reduction_k per unit (0 where the footprint is already 0 and would go below).
std::vector<double> intactness_gradient(const double* hfi, const double* reduction, i64 ny, i64 nx,
                                        const IntactnessParams& p, int threads);

}  // namespace rest
