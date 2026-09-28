// SPDX-License-Identifier: GPL-3.0-or-later
//
// Randomisations of a basedata (Biodiverse::Randomise), drawing from
// Biodiverse's Mersenne Twister in the same order. Groups and labels are
// numbered in name order, which is the order Biodiverse sorts them in.

#pragma once

#include <cstdint>
#include <string>
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

// Spatial allocation of rand_structured (rand_spatially_structured,
// rand_diffusion, rand_random_walk): label occurrences go to the neighbours
// of a seed group. Its caches live across iterations, as Biodiverse's do.
class SpatialAlloc {
public:
    // order: "", "diffusion", "random_walk", "random" or "proximity";
    // backtracking: "from_end", "from_start", "random" or ""; nbrs[group][set]
    // are Biodiverse's stored neighbour lists (sorted); `same` when every
    // condition gives every group the same list; coords are the group names
    // as numbers (proximity); seeds the groups passing the seed condition.
    SpatialAlloc(std::string order, std::string backtracking, double reseed_prob,
                 std::vector<std::vector<std::vector<i32>>> nbrs, bool same, std::vector<std::vector<double>> coords,
                 bool have_seeds, std::vector<i32> seeds);
    const std::vector<std::vector<i32>>& lists(i32 group, MT64& rng);

    std::string order, backtracking;
    double reseed_prob;
    bool have_seeds;
    std::vector<i32> seeds;               // sorted
    std::vector<std::uint8_t> seeded;     // labels whose seed groups were looked up (the cache)

private:
    std::vector<std::vector<std::vector<i32>>> nbrs_, cache_;
    std::vector<std::uint8_t> cached_;
    bool same_;
    std::vector<std::vector<double>> coords_;
};

// rand_structured: labels, in random order, go to random groups until each
// group reaches its target richness (floor (richness * multiplier + addition)),
// then swaps reach the targets (swap_to_reach_richness_targets).
CountsCsr rand_structured(const CountsCsr& bd, MT64& rng, double multiplier, double addition,
                          SpatialAlloc* spatial = nullptr);

// Swap options (0: Biodiverse's defaults, twice the non-zero cells and 100
// times as many attempts).
struct SwapOptions {
    std::int64_t swap_count = 0;
    std::int64_t max_swap_attempts = 0;
    bool stop_on_all_swapped = false;
    bool use_max_swap = false;  // curveball
};

// rand_independent_swaps and rand_independent_swaps_modified (Gotelli 2000;
// Miklos & Podani 2004); label_range is Biodiverse's get_range.
CountsCsr rand_independent_swaps(const CountsCsr& bd, MT64& rng, const std::vector<double>& label_range,
                                 const SwapOptions& opt, bool modified);

// rand_curveball (Strona et al. 2014), without the hypergeometric draws.
CountsCsr rand_curveball(const CountsCsr& bd, MT64& rng, const SwapOptions& opt);

}  // namespace bio
