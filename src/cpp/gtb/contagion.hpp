// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

// Computes Contagion (Pff) moving-window metric (GuidosToolbox / Spatcon rule 77).
// Pff is the proportion of foreground edges that are self-joins (foreground-foreground).
//
// If as_byte is true, returns uint8 with GTB scaling:
//   - [0..100]: foreground contagion percentage
//   - 101: background
//   - 102: missing
// If as_byte is false, returns raw float [0.0, 1.0] where background and missing are NaN.
ImageU8 compute_contagion_pff(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask = true
);

Image<float> compute_contagion_pff_float(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask = true
);

// Computes Foreground Area Clustering (FAC) moving-window metric (GuidosToolbox / Spatcon rule 76).
// FAC is the proportion of non-missing edges in the window that are foreground self-joins.
//
// If as_byte is true, returns uint8 with GTB scaling:
//   - [0..100]: foreground clustering percentage
//   - 101: background
//   - 102: missing
// If as_byte is false, returns raw float [0.0, 1.0] where background and missing are NaN.
ImageU8 compute_clustering_fac(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask = true
);

Image<float> compute_clustering_fac_float(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask = true
);

}  // namespace gtb
