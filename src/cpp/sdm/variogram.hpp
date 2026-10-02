// SPDX-License-Identifier: GPL-3.0-or-later
//
// gstat's sample variogram with given distance classes (variogram(...,
// boundaries =)): for each class, the number of pairs, their mean distance
// and the semivariance (half the mean squared difference).
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

struct SampleVariogram {
    std::vector<double> np, dist, gamma;  // one entry per class (np = 0: empty)
};

// Pairs i > j with distance <= bounds[k] go to the first such class k (pairs
// beyond the last bound are left out). Distances: Euclidean, or with `lonlat`
// gstat's great-circle distance in km (Andoyer-Lambert on WGS 84). Rows of
// pairs in parallel, their sums added in row order: the same result with any
// number of threads.
SampleVariogram sample_variogram(const double* x, const double* y, const double* z, i64 n,
                                 const std::vector<double>& bounds, bool lonlat);

// gstat's fit.variogram (fit.method 7) of a nugget plus one model ("Sph",
// "Exp", "Gau", "Ste" with smoothness kappa): Gauss-Newton on the weighted
// residuals (weights np / dist^2), steps capped when the ratio of the squared
// norms of step and parameters passes 0.2, stopping at a relative change of
// the weighted squared error below 1e-5 (or a singular system, or 200
// iterations); a negative nugget or partial sill is fixed at zero and the fit
// repeated from the initial range. `ok` false when no fit was possible.
struct VariogramFit {
    double nugget, psill, range, sserr;
    bool ok;
};

VariogramFit fit_variogram(const char* model, double kappa, const double* dist, const double* gamma,
                           const double* np, i64 n, double nugget, double psill, double range);

}  // namespace sdm
