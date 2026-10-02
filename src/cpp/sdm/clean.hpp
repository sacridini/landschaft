// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geographic outliers of occurrence records (CoordinateCleaner::cc_outl): the
// distance of each record to the others of its species, as R computes it.
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

// geosphere::distHaversine (R = 6378137 m) in km, as R evaluates it on
// Windows (x87 sin / cos / atan2 / sqrt of vectors).
double haversine_km(double lon1, double lat1, double lon2, double lat2);

// cc_outl's statistic of each record: over the other records at a different
// location (distance != 0), the mean distance (R's mean(), 80-bit) or, with
// `use_min`, the smallest one. distm(): the distance between records i and j
// is haversine(point max(i, j), point min(i, j)). NaN when there is none.
std::vector<double> outlier_statistic(const double* lon, const double* lat, i64 n, bool use_min);

// The raster variant for large species (cc_outl's ras_dist with weights):
// cells with `count` records; a cell's distance to itself is `within`; the
// pairwise distances times count_i * count_j rounded half to even; the
// statistic is the row sum of those over the row sum of the weights, or
// with `use_min` the smallest of those rounded weighted distances.
std::vector<double> outlier_statistic_weighted(const double* lon, const double* lat, const double* count, i64 n,
                                               double within, bool use_min);

}  // namespace sdm
