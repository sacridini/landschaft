// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "image.hpp"

namespace gtb {

// Spatial convolution across continuous / grayscale maps (0..100, missing=255).
// Supports metrics:
// - "mean": mean pixel value
// - "median": median pixel value
// - "majority": modal pixel value
// - "min": minimum pixel value
// - "max": maximum pixel value
// - "range": max - min
// - "stddev": standard deviation
// - "shannon_diversity": Shannon entropy of pixel values
// - "shannon_evenness": Shannon evenness
// - "diagonal_contagion": sum of diagonal co-occurrences
Image<float> compute_gray_convolution(
    const ImageU8& im,
    const std::string& metric,
    int window_size = 7,
    bool mask_missing = true,
    int n_threads = 0
);

}  // namespace gtb
