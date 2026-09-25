// SPDX-License-Identifier: GPL-3.0-or-later
//
// GuidosToolbox "Accounting": foreground objects grouped into size classes.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

// Viewport colour codes of GTB (mspacolorston colour table).
constexpr std::array<u8, 6> kAccountingClassCodes = {103, 33, 65, 1, 9, 17};  // size classes 1..6
constexpr u8 kAccountingLargest = 80;   // the three largest objects (pink)
constexpr u8 kAccountingMissing = 129;  // input 0
constexpr u8 kAccountingSpecial1 = 105; // input 3
constexpr u8 kAccountingSpecial2 = 176; // input 4

struct AccountingResult {
    std::vector<i64> thresholds;    // size class thresholds used
    ImageU8 classes;                // GTB colour codes; 0 = background (input 1)
    Image<std::int32_t> ids;        // object id 1..n; 0 background; -1/-3/-4 for input 0/3/4
    Image<std::int32_t> area;       // object area [pixels] on its pixels; codes as `ids`
    std::vector<i64> object_area;   // object_area[k] = area of object k (index 0 unused)
    // Ids of the largest, second and third largest objects (ties share a
    // rank; a rank is empty when fewer objects exist).
    std::array<std::vector<std::int32_t>, 3> largest;
};

// Size thresholds GTB's interactive Accounting proposes: the 0.1, 1, 10, 50
// and 90% points of the [min, max] range of the 8-connected object areas
// `areas8` (index 0 unused), in IDL single-precision arithmetic, with
// repeated values dropped.
std::vector<i64> accounting_auto_thresholds(const std::vector<i64>& areas8);

// Accounting of the foreground (value 2) objects of a map coded 0 missing,
// 1 background, 2 foreground, 3/4 special background. Objects are
// `graph`-connected (4 or 8) and numbered as GTB does (raster order from the
// bottom row). `thresholds` are the strictly increasing upper bounds of the
// size classes (1 to 5 values >= 1): class k holds objects with
// thresholds[k-1] < area <= thresholds[k]; the last class the larger ones.
// An empty `thresholds` selects the defaults of GTB's interactive Accounting
// (accounting_auto_thresholds). With `highlight_largest` and at least 3
// objects, the three largest are painted kAccountingLargest.
AccountingResult compute_accounting(const ImageU8& im, int graph, const std::vector<i64>& thresholds,
                                    bool highlight_largest);

}  // namespace gtb
