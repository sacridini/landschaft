// SPDX-License-Identifier: GPL-3.0-or-later
//
// GraySpatCon metrics used by GuidosToolbox's FOS (Fixed Observation Scale)
// fragmentation analysis: metric 1 (mean, FAD on grayscale maps) and metric 52
// (clustering, FED), as GTB runs them (P 0, G 0, F 1, A 1; B 6 for metric 1,
// B 1 for metric 52).
#pragma once

#include "image.hpp"

namespace gtb {

// Byte output of grayspatcon for a map of values 0..100 with 255 = missing
// (other values are rejected): moving window of size `window` (odd, >= 3),
// missing input pixels -> 255. Reproduces GraySpatCon.c 1.3.x bit for bit.
ImageU8 grayspatcon_fos(const ImageU8& im, int metric, int window);

}  // namespace gtb
