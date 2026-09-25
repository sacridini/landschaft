// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

struct DistanceStatsCpp {
    double mean_fg_distance = 0.0;
    double max_fg_distance = 0.0;
    double mean_bg_distance = 0.0;
    double max_bg_distance = 0.0;
    std::int64_t fg_pixel_count = 0;
    std::int64_t bg_pixel_count = 0;
};

struct EuclideanDistanceResultCpp {
    std::vector<float> signed_distance; // float distance (positive in FG, negative in BG)
    ImageU8 byte_map;                   // byte map (10..129 for BG, 131..250 for FG, 130 boundary)
    DistanceStatsCpp stats;
};

// Computes Euclidean Distance Transform matching GuidosToolbox batch_eucldist.
EuclideanDistanceResultCpp compute_euclidean_distance(const ImageU8& im);

}  // namespace gtb
