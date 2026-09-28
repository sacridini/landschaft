// SPDX-License-Identifier: GPL-3.0-or-later
//
// Randomisations of a basedata (Biodiverse::Randomise), drawing from
// Biodiverse's Mersenne Twister in the same order. Groups and labels are
// numbered in name order, which is the order Biodiverse sorts them in.

#pragma once

#include <cstdint>
#include <vector>

#include "bio/conditions.hpp"
#include "bio/phylocom.hpp"

namespace bio {

// groups x labels sample counts (CSR, labels ascending); explicit zeros are
// labels present with no samples
struct CountsCsr {
    i64 n_groups = 0, n_labels = 0;
    std::vector<i64> indptr;
    std::vector<i32> indices;
    std::vector<double> counts;
};

// rand_structured: labels, in random order, go to random groups until each
// group reaches its target richness (floor (richness * multiplier + addition)),
// then swaps reach the targets (swap_to_reach_richness_targets).
CountsCsr rand_structured(const CountsCsr& bd, MT64& rng, double multiplier, double addition);

}  // namespace bio
