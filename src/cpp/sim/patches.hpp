// SPDX-License-Identifier: GPL-3.0-or-later
//
// Expander and Patcher (Soares-Filho, Cerqueira & Pennachin 2002), written
// from their publications (Dinamica EGO's code is closed): the allocation
// of the cells each transition must change as expansions of existing patches
// of the target class and as new patches.
//
//  - Expander: the probability of a cell of the origin class becomes
//    P' = P where more than 3 cells of the 3 x 3 window around it hold the
//    target class, else P n_j / 4 (Soares-Filho et al. 2002, eq. of the
//    expander), so expansion happens only next to existing patches.
//  - Patcher: new patches seeded in cells with no target-class neighbour.
//  - Seeds: the cells of positive probability ranked from the highest, the
//    first prune_factor x changes of them kept, drawn in proportion to their
//    probability.
//  - Patch size: lognormal with the given mean and variance (cells).
//  - Patch growth: each step adds a cell of the origin class next to the
//    patch, drawn in proportion to P x isometry^k, k the patch cells in its
//    window: isometry 0 makes single cells, 1 follows the probabilities,
//    2 makes compact patches (the descriptions of Dinamica's isometry).
//  - A changed cell has its probabilities corroded (0).
//
// Random numbers: xoshiro256** seeded through SplitMix64.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct PatchTransition {
    int from = 0, to = 0;
    std::vector<double> probability;  // per cell
    std::int64_t expand = 0, patch = 0;  // cells to change by each function
    double e_mean = 1, e_var = 0, e_iso = 1;
    double p_mean = 1, p_var = 0, p_iso = 1;
};

struct PatchResult {
    std::vector<int> landscape;
    std::vector<std::int64_t> expanded, patched;  // per transition: cells changed by each function
    std::vector<std::int64_t> patches;            // per transition: patches formed (both)
};

// landscape: class of each cell (INT32_MIN: no data); transitions in order;
// window: the neighbourhood (odd sides) for n_j and growth.
PatchResult allocate_patches(std::int64_t ny, std::int64_t nx, const int* landscape,
                             std::vector<PatchTransition>& transitions, double prune_factor, int win_rows,
                             int win_cols, std::uint64_t seed);

}  // namespace sim
