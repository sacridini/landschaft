// SPDX-License-Identifier: GPL-3.0-or-later
//
// Weights of evidence on rasters: the bucket of each cell (an increment of a
// continuous variable or a category), the cells of each bucket by final
// class, and the probability map of a transition.
#pragma once

#include <cstdint>
#include <vector>

#include "sim/change.hpp"

namespace sim {

// Bucket of each cell: for a continuous variable the increment k with
// start + k * inc - 1e-14 <= v (the largest such k, Dinamica EGO's
// rule), for a categorical one the index of its class in `classes`;
// -1 where the variable is missing (or not a class).
std::vector<std::int32_t> woe_buckets(const MapView& var, bool categorical, const std::vector<double>& classes,
                                      double start, double inc, std::int64_t n_buckets);

// Cells with initial class `from` (and a value in the three maps), by bucket
// (rows) and final class (columns, the index in `final_classes`):
// n_buckets x final_classes.size(), row-major.
std::vector<std::int64_t> woe_counts(const MapView& t0, const MapView& t1, const std::vector<std::int32_t>& buckets,
                                     std::int64_t n_buckets, double from, const std::vector<double>& final_classes);

// The sum over the variables of the weight of each cell's bucket (the
// log-odds of a transition from class `from`; a variable missing at the cell
// adds nothing, as in Dinamica EGO); NaN for cells of other classes or
// missing in the landscape.
std::vector<double> woe_probability(const MapView& landscape, double from,
                                    const std::vector<const std::int32_t*>& buckets,
                                    const std::vector<std::vector<double>>& weights, std::int64_t n);

}  // namespace sim
