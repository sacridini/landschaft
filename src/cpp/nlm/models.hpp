// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR's neutral landscape models. Rasters are row-major (nrow x ncol)
// arrays of doubles, NaN for missing values; R fills matrices by column,
// which the models reproduce when they draw random numbers.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlm/rng.hpp"

namespace nlm {

// -- utilities -------------------------------------------------------------------

// util_rescale(): (x - min) / (max - min) over the non-NaN cells; 0 when
// min and max are all.equal() (NaN stays NaN).
void rescale(double* x, i64 n);

// -- simple models ----------------------------------------------------------------

// nlm_random(): runif() by column.
std::vector<double> random_field(RRng& rng, i64 nrow, i64 ncol);
// nlm_percolation(): runif(1) < prob by column (1 or 0).
std::vector<double> percolation(RRng& rng, i64 nrow, i64 ncol, double prob);
// nlm_planargradient() before rescaling: southness * row + eastness * col.
std::vector<double> planar_gradient(i64 nrow, i64 ncol, double direction);
// The edge gradient of a rescaled planar gradient: (|0.5 - g| * -2) + 1.
void edge_from_planar(double* g, i64 n);

// -- distances ---------------------------------------------------------------------

// Distance of every cell to the nearest cell with origin[i] != 0.
//  - "gdal": terra::distance() on a planar raster (GDALComputeProximity,
//    cell units times `res`, float32 values);
//  - "haversine": terra::distance() on a lon/lat raster (terra's fast
//    haversine path on a 6378137 m sphere; cell centres from the extent
//    xmin, ymax and the cell sizes);
//  - "exact": exact Euclidean distance between cell centres times `res`.
std::vector<double> distance_gdal(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res);
std::vector<double> distance_haversine(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double xmin,
                                       double ymax, double xres, double yres);
std::vector<double> distance_exact(const std::vector<std::uint8_t>& origin, i64 nrow, i64 ncol, double res);

// -- classification -----------------------------------------------------------------

// NLMR's rcpp_get_jenksbreaks(sorted d, k): k + 1 break values.
std::vector<double> jenks_breaks(const std::vector<double>& d, int k);
// base::cut(x, breaks, include.lowest = TRUE, right = TRUE, labels = FALSE)
// (breaks sorted and unique); NaN outside the breaks or for NaN.
void cut_include_lowest(const double* x, i64 n, const std::vector<double>& breaks, double* out);

}  // namespace nlm
