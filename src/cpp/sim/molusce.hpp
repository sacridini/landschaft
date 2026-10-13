// SPDX-License-Identifier: GPL-3.0-or-later
//
// MOLUSCE's simulator (the QGIS plugin, GPL >= 2; ls.simulate's MolusceCA):
// the transition each cell is predicted to take and its confidence, from
// MOLUSCE's weights of evidence or from a potential per class, and one
// iteration of the cellular automaton, which makes the observed transitions
// where the confidence is highest, with numpy's legacy generator as MOLUSCE
// draws after numpy.random.seed(seed).
#pragma once

#include <cstdint>
#include <vector>

#include "nlm/rng.hpp"

namespace sim {

// numpy's legacy RandomState seeded with an integer (mt19937_seed is
// std::mt19937's seeding) and its random_sample (genrand_res53: 27 + 26 bits).
class NumpyRandomState {
public:
    explicit NumpyRandomState(std::uint32_t seed = 0) : mt_(seed) {}
    double random_sample() {
        const std::uint32_t a = mt_() >> 5, b = mt_() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) / 9007199254740992.0;
    }

private:
    nlm::StdMt19937 mt_;
};

// A factor's categories as MOLUSCE cuts it: with sorted bounds b0 < b1 < ...,
// (-inf, b0) -> 1, [b0, b1) -> 2 ... [bn, inf) -> n + 2 (numpy's searchsorted
// on the right, plus 1); without bounds the values themselves. NaN stays NaN.
std::vector<double> molusce_categories(const double* values, std::int64_t n, const std::vector<double>& bounds);

// The weights of one transition code for one factor: the weight of each
// category (cells of other categories get 0).
struct MolusceFactorWeights {
    std::vector<double> categories, weights;
};

// The total weight of each transition code: the sum over the factors (in
// their order) of the weight of each cell's category; NaN where a factor has
// none. factors: nf x n categories (molusce_categories); weights[code][f].
// Returns ncode x n.
std::vector<double> molusce_total(const double* factors, int nf, std::int64_t n,
                                  const std::vector<std::vector<MolusceFactorWeights>>& weights);

struct MolusceForecast {
    std::vector<double> prediction;  // the class predicted (NaN: none)
    std::vector<double> confidence;  // 0-100, truncated (NaN outside the cells of the model's classes)
};

// MolusceWoE.prediction: for each cell of class i (classes[i]), the transition
// i -> j of largest total weight among the codes i * m + j given (codes, with
// totals: ncode x n, NaN skipped) and its confidence 100 x (sigmoid(largest) -
// sigmoid(second largest)), truncated; both weights start at -1000.
MolusceForecast molusce_prediction_woe(const double* state, std::int64_t n, const std::vector<double>& classes,
                                       const std::vector<std::int64_t>& codes, const double* totals);

// MolusceCA with a potential per class (layers: m x n, NaN as -inf, a missing
// class as a layer of NaN; infinities as numpy.nan_to_num makes them): the class of largest potential (the first in
// case of ties) and 100 x the gap to the second largest, truncated (the gap to
// 0 when the second is not finite); no prediction and confidence 0 where the
// largest is not finite, NaN where the state has none.
MolusceForecast molusce_prediction_classes(const double* state, std::int64_t n, int m, const double* layers,
                                           const std::vector<double>& classes);

struct MolusceTransition {
    double from = 0.0, to = 0.0;
    std::int64_t cells = 0;
};

struct MolusceStep {
    std::vector<double> state;
    // per transition of `counts` that applies (both classes in the map, a
    // count > 0): the cells changed
    std::vector<MolusceTransition> changed;
};

// One iteration of MOLUSCE's Simulator: for each pair of distinct classes of
// the map, in order, the min(count, cells of `from` predicted to take `to`)
// cells of highest confidence + U(0, 1) / 1000 change (draws over the whole
// map for each such pair, as numpy's random_sample(size=shape)); among equal
// values the later cells first (numpy's stable ordering).
MolusceStep molusce_ca_step(const double* state, std::int64_t n, const double* prediction, const double* confidence,
                            const std::vector<MolusceTransition>& counts, NumpyRandomState& rng);

}  // namespace sim
