// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "image.hpp"

namespace gtb {

// Statistics row for one scale or the multiscale average
struct FADStatsRow {
    int scale = 0;              // window size (e.g. 7, 13, 27, 81, 243, or 0 for multiscale)
    double intact_pct = 0.0;     // FAD == 100%
    double interior_pct = 0.0;   // 90% <= FAD < 100%
    double dominant_pct = 0.0;   // 60% <= FAD < 90%
    double transitional_pct = 0.0;// 40% <= FAD < 60%
    double patchy_pct = 0.0;     // 10% <= FAD < 40%
    double rare_pct = 0.0;       // FAD < 10%
    double average_fad = 0.0;    // Mean FAD on foreground
};

struct FADMultiscaleResultCpp {
    std::vector<int> window_sizes;
    std::vector<ImageU8> single_scale_maps; // single scale FAD maps with GTB masking (101 bg, 102 missing)
    std::vector<ImageU8> raw_densities;     // raw density values [0, 100]
    ImageU8 multiscale_map;                 // average over scales with GTB masking
    ImageU8 multiscale_classes;             // classes of the average (see FADSingleResultCpp::classes)
    std::vector<FADStatsRow> stats;
};

// Single-scale FAD in one pass over the map.
struct FADSingleResultCpp {
    ImageU8 density;               // [0, 100], GTB's codes (101, 102, 105, 106) off the foreground
    ImageU8 raw;                   // [0, 100] everywhere
    ImageU8 classes;               // 1 (rare) .. 6 (intact) on the foreground (2), GTB's codes elsewhere
    i64 foreground = 0;            // pixels coded 2
    i64 class_counts[6] = {};      // foreground pixels of classes 1 .. 6
    i64 density_sum = 0;           // raw density summed over the foreground
};

// Computes local Forest Area Density (FAD) for a single window size.
// Returns density in [0, 100] for each pixel.
// If apply_gtb_mask is true, background is set to 101 and missing to 102.
ImageU8 compute_fad_density(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask = true
);

// The density with and without GTB's codes, the classes and the counts of the
// statistics for one window size, the integral images built once.
FADSingleResultCpp compute_fad(const ImageU8& im, int window_size);

// Computes multiscale FAD across multiple window sizes (default: 7, 13, 27, 81, 243).
FADMultiscaleResultCpp compute_fad_multiscale(
    const ImageU8& im,
    const std::vector<int>& window_sizes = {7, 13, 27, 81, 243}
);

}  // namespace gtb
