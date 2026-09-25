// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

struct ClassParcellationStats {
    i64 class_id = 0;       // Raster class value
    i64 count = 0;          // Number of parcels (patches)
    i64 area = 0;           // Total class area in pixels
    double aps = 0.0;       // Average Patch Size (area / count)
    double awaps_rel = 0.0; // Area-Weighted APS relative to class (effective mesh size)
    double awaps_data = 0.0;// Area-Weighted APS relative to total valid data area
    double division = 0.0;  // Division index (1.0 - sum((a_i / area)^2))
    double parc = 0.0;      // Logarithmic Parcellation Index [%]
};

struct GlobalParcellationStats {
    i64 total_parcels = 0;  // Total parcels across all valid classes
    i64 data_area = 0;      // Total valid pixels (excluding nodata 0)
    double aps = 0.0;       // Global Average Patch Size (data_area / total_parcels)
    double awaps = 0.0;     // Global Area-Weighted APS (sum(a_k^2) / data_area)
    double division = 0.0;  // Global Division index
    double parc = 0.0;      // Global Parcellation Index [%]
};

struct ParcellationResult {
    std::vector<ClassParcellationStats> class_stats;
    GlobalParcellationStats global_stats;
    Image<std::uint32_t> parcel_labels; // Optional labeled parcels raster (empty if not requested)
};

// Compute Parcellation metrics across all classes in raster (nodata = 0).
// connectivity: 4 or 8.
ParcellationResult compute_parcellation(
    const ImageU8& im,
    int connectivity = 8,
    bool return_labels = false
);

// Compute Contortion metric for foreground habitat components:
// - Labels 8-connected components
// - Discards components without a core (eroded by 3x3) or area < min_area
// - Traces boundary and counts directional transitions
// Output raster:
// 0: nodata (from input == 0)
// 1: background (from input == 1)
// 2: neglected foreground (< min_area or no core)
// >= 3: contortion value (contortion + 1, so contortion=2 becomes 3)
Image<std::uint32_t> compute_contortion(
    const ImageU8& im,
    i64 min_area = 1
);

}  // namespace gtb
