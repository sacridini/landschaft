// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

// Lookup table mapping 103-class LM codes to 19-class LM codes (recodelm103.sav).
extern const u8 kRecode103To19[256];

struct LandscapeMosaicStats {
    std::vector<int> class_codes;
    std::vector<std::uint64_t> pixel_counts;
    std::vector<double> percentages;
};

// Computes Landscape Mosaic tri-polar metric (GuidosToolbox / Spatcon rule 7).
//
// Input raster contains:
//   - 0: Missing
//   - 1: Agriculture (or Class 1)
//   - 2: Forest / Natural (or Class 2)
//   - 3: Developed / Antropic (or Class 3)
//
// Parameters:
//   - window_size: odd integer >= 3 (default 27 in GTB)
//   - output_classes: 19 (default) or 103
//
// Returns ImageU8 with either 19-class or 103-class byte values.
// Original missing pixels (im == 0) remain 0 in the output.
ImageU8 compute_landscape_mosaic(
    const ImageU8& im,
    int window_size,
    int output_classes = 19
);

// Computes 103-class frequency distribution and percentages on non-missing pixels.
LandscapeMosaicStats compute_landscape_mosaic_stats(
    const ImageU8& im103
);

}  // namespace gtb
