// SPDX-License-Identifier: GPL-3.0-or-later
//
// Interpolation of points as R's gstat: the sample variogram (Matheron's
// estimator in classes of distance), fit.variogram's weighted least squares
// (N/h^2 weights, Gauss-Newton steps bounded to 20%), inverse distance
// weighting and ordinary kriging with every point or the nearest. The same
// operations in the same order as Terraflux's stats.variogram / idw / kriging.
#pragma once

#include <cstdint>
#include <vector>

namespace stats {

// gstat's Sph, Exp (practical range 3 times the range) and Gau (sqrt(3) times).
enum class Shape : int { kSpherical = 0, kExponential = 1, kGaussian = 2 };

struct VarioModel {
    Shape shape = Shape::kSpherical;
    double nugget = 0.0;
    double psill = 0.0;
    double range = 0.0;

    // gstat's get_semivariance: the sills beyond a spherical model's range
    double gamma(double h) const;
    double covariance(double h) const { return nugget + psill - gamma(h); }
};

struct SampleVariogram {
    std::vector<double> np, dist, gamma;
};

// A third of the diagonal of the points' bounding box (gstat's default cutoff).
double default_cutoff(const double* x, const double* y, std::int64_t n);

// The pairs at a distance (lower, upper] of a class of `width` up to `cutoff`:
// their number, mean distance and half their mean squared difference.
SampleVariogram sample_variogram(const double* x, const double* y, const double* z, std::int64_t n, double cutoff,
                                 double width);

struct VarioFit {
    VarioModel model;
    double sserr = 0.0;
    bool singular = false;
    bool converged = true;
};

// gstat's fit.variogram (fit.method 7): a negative sill fixed at 0 and fitted
// again from the initial range.
VarioFit fit_variogram(const SampleVariogram& s, VarioModel init, bool fit_nugget, bool fit_psill, bool fit_range);

// Inverse distance weighting at `m` places, with every point (nmax <= 0) or
// the nmax nearest; a point's value at its own place.
void idw(const double* x, const double* y, const double* z, std::int64_t n, const double* px, const double* py,
         std::int64_t m, double power, std::int64_t nmax, double* out, int threads);

// Ordinary kriging at `m` places: the prediction and its variance. False when
// the covariances of the points (or of a place's nearest) are singular.
bool krige(const double* x, const double* y, const double* z, std::int64_t n, const VarioModel& model,
           const double* px, const double* py, std::int64_t m, std::int64_t nmax, double* pred, double* var,
           int threads);

}  // namespace stats
