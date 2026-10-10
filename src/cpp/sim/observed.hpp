// SPDX-License-Identifier: GPL-3.0-or-later
//
// The patches of an observed transition (Dinamica EGO's Expander and Patcher
// fitted on two maps): the cells that went from one class to another grouped
// in patches of 8 neighbours, a patch touching (8 neighbours) a cell of the
// target class in the first map being an expansion, the others new patches.
#pragma once

#include <cstdint>

namespace sim {

struct ObservedPatches {
    std::int64_t cells = 0;              // cells that made the transition
    double expansion = 0.0;              // share of them in expansions (0 without any)
    std::int64_t expansion_patches = 0;  // patches of each kind, their mean and variance (cells)
    double expansion_mean = 1.0, expansion_variance = 0.0;
    std::int64_t new_patches = 0;
    double new_mean = 1.0, new_variance = 0.0;
};

// a, b: the two maps (ny x nx, NaN missing). Patches are numbered in the
// order of their first cell, row by row (scipy.ndimage.label), and their mean
// and (population) variance are numpy's (pairwise sums in that order); a kind
// without patches has mean 1 and variance 0.
ObservedPatches observed_patches(const double* a, const double* b, std::int64_t ny, std::int64_t nx, double from,
                                 double to);

}  // namespace sim
