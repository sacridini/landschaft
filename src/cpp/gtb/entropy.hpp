// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

struct SpatialEntropyResultCpp {
    double entropy_index = 0.0;             // Normalized global entropy [0, 100]%
    std::vector<std::uint64_t> histogram;  // Counts of transitions (size 256)
};

// Computes whole-image spatial joint entropy index matching GuidosToolbox entropy_mspainp.
SpatialEntropyResultCpp compute_spatial_entropy(const ImageU8& im);

// Computes moving-window spatial entropy map matching GuidosToolbox batch_ent.
// Returns a float image of local normalized entropy [0, 100].
Image<float> compute_spatial_entropy_map(
    const ImageU8& im,
    int window_size = 50
);

}  // namespace gtb
