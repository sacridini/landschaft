// SPDX-License-Identifier: GPL-3.0-or-later
//
// GuidosToolbox "Pattern -> Morphological -> SPA2/3/5/6": simplified pattern
// analysis of the 8-connected foreground.
#pragma once

#include "image.hpp"

namespace gtb {

struct SPAResult {
    // GTB codes: 17 core (SPA2: contiguous), 1 margin (SPA2: small linear
    // features), 3 edge, 5 perforation, 9 islet, 100 core-opening, 0 other
    // background, 129 missing.
    ImageU8 map;
    i64 nr_fg = 0;           // 8-connected foreground objects
    i64 nr_cores = 0;
    i64 nr_margin = 0;
    i64 nr_holes = -1;       // SPA3/5/6 only (4-connected core-openings)
    i64 nr_edge = -1;        // SPA5/6
    i64 nr_perf = -1;        // SPA5/6
    i64 nr_islet = -1;       // SPA6
    i64 nr_contiguous = -1;  // SPA5/6: core + edge + perforation
};

// SPA2 (variant 2), SPA3, SPA5 or SPA6 of a map coded 0 missing,
// 1 background, 2 foreground, as GTB's 'spa' event computes it.
SPAResult compute_spa(const ImageU8& im, int variant);

}  // namespace gtb
