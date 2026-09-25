// SPDX-License-Identifier: GPL-3.0-or-later
//
// GuidosToolbox Restoration Planner: isochrone maps (geodesic time from
// markers across a resistance map, computed by GTB's `ggeo` program).
#pragma once

#include <cstdint>

#include "image.hpp"

namespace gtb {

// uc_sqtgsym (miallib ggeo.c) with graph 8: symmetric geodesic time from the
// pixels with ref == 1 across the grey-level mask `m`; a step between
// neighbours p, q costs m[p] + m[q]. Pixels with ref == 3 and the 1-pixel
// image frame are not crossed. Unreached pixels keep UINT32_MAX, the frame 0.
Image<std::uint32_t> geodesic_time(const ImageU8& m, const ImageU8& ref);

}  // namespace gtb
