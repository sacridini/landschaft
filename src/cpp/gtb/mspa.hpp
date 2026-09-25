// SPDX-License-Identifier: GPL-3.0-or-later
//
// Morphological Spatial Pattern Analysis (MSPA), Soille & Vogt (2009),
// doi:10.1016/j.patrec.2008.10.015 - port of the GTB/GWB `mspa` v2.3 core.
#pragma once

#include "image.hpp"

namespace gtb {

struct MspaParams {
    int connectivity = 8;  // foreground connectivity (4 or 8), GTB "FGconn"
    int edge_width = 1;    // effective edge width in pixels, GTB "EdgeWidth"
    bool internal = true;  // add +100 to internal features, GTB "Intext"
    int threads = 0;       // 1 = sequential; otherwise independent stages run concurrently
};

// MSPA classes (external values; internal pixels get +100 when internal=true).
namespace mspa_class {
constexpr u8 kBackground = 0;
constexpr u8 kBranch = 1;
constexpr u8 kEdge = 3;
constexpr u8 kPerforation = 5;
constexpr u8 kIslet = 9;
constexpr u8 kCore = 17;
constexpr u8 kBridge = 33;
constexpr u8 kLoop = 65;
constexpr u8 kCoreOpening = 100;
constexpr u8 kMissing = 129;
constexpr u8 kBorderOpening = 220;
}  // namespace mspa_class

// `input` uses the GTB encoding: 0 = missing, 1 = background, 2 = foreground.
// Returns the MSPA class image with the same shape as `input`.
ImageU8 mspa(const ImageU8& input, const MspaParams& params);

}  // namespace gtb
