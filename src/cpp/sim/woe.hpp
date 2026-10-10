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

// The weights of a range with nb transitions among its b possible cells, of n
// transitions among a possible cells: W+ = ln(P(B|D) / P(B|~D)), the
// contrast W+ - W- and whether it exceeds 1.96 standard deviations. Ranges
// with no transition or only transitions (and the like) are abnormal: W+ is
// 0, or with `fix` -ln(b n + 1) and ln(b (a - n) + 1); their contrast is 0.
struct WoeWeights {
    double plus = 0.0;
    double contrast = 0.0;
    bool significant = false;
};
WoeWeights woe_weights(double nb, double b, double n, double a, bool fix);

// The ranges of a continuous variable: the bucket indices (1..n) that end
// each range, from the line generalisation of the curve (A, A exp(W+)) of
// the cumulative buckets (Dinamica EGO's): a range ends where the direction
// from its start deviates from that of the previous range by more than
// `tolerance` degrees, or where it would exceed `max_delta` cells, never
// below `min_delta` cells; the last bucket always ends one.
std::vector<int> woe_generalise(const std::vector<std::int64_t>& possible, const std::vector<std::int64_t>& executed,
                                double min_delta, double max_delta, double tolerance, bool fix);

// From the sums of weights S of k transitions from the same class
// (k x n, row-major, NaN where they do not apply) to their probabilities
// exp(S_j) / (1 + sum_i exp(S_i)), computed as exp(S_j - t) / (exp(-t) +
// sum_i exp(S_i - t)), t = max(0, max_i S_i), in place.
void woe_normalise(double* sums, int k, std::int64_t n);

}  // namespace sim
