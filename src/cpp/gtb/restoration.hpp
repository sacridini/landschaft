// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

struct RestorationStatusResult {
    i64 data_area = 0;       // Valid pixels (total - missing 0b)
    i64 nr_comp = 0;         // Number of 8-connected foreground components
    i64 eca_max = 0;         // Total foreground area (sum of component areas)
    i64 larg_obj = 0;        // Area of largest component
    double aps = 0.0;        // Average patch size (eca_max / nr_comp)
    double cnoa = 0.0;       // Critical New Object Area
    double eca = 0.0;        // Equivalent Connected Area sqrt(sum(A_i^2))
    double doc = 0.0;        // Degree of Coherence (eca / eca_max * 100.0)
    double rest_pot = 0.0;   // Restoration Potential (100.0 - doc)
    double rac = 0.0;        // Relative Area of Connected Habitat (eca_max / data_area * 100.0)
};

// Compute Restoration Status (RSS) on single-band image:
// 0: missing / nodata
// 1: background (or resistance <= 100)
// 2: foreground habitat
RestorationStatusResult compute_restoration_status(const ImageU8& im);

struct CostDistanceResult {
    Image<float> distance;  // Accumulated cost distance from sources
};

// Compute 8-neighbor anisotropic cost distance from source_mask across resistance surface.
// Cells with resistance == 0 or missing are impassable barriers.
Image<float> compute_cost_distance(
    const ImageU8& resistance,
    const ImageU8& source_mask
);

struct LeastCostPathResult {
    Image<float> combined_cost;  // cost_a + cost_b
    ImageU8 lcp_mask;            // 1 on least-cost path, 0 elsewhere
    i64 path_length = 0;         // Total pixel length of path
    i64 rest_pixels = 0;         // Number of restore pixels (in background)
    double expense = 0.0;        // Accumulated resistance along restore pixels
    double avg_distance = 0.0;   // Average distance of restore pixels (if dist_map provided)
};

// Compute two-ended least cost path between marker_a and marker_b.
LeastCostPathResult compute_least_cost_path(
    const ImageU8& resistance,
    const ImageU8& marker_a,
    const ImageU8& marker_b,
    const Image<float>* dist_map = nullptr
);

}  // namespace gtb
