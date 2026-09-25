// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

struct SimpleChangeResult {
    ImageU8 change_map;         // 11: Stable BG, 12: Gain, 21: Loss, 22: Stable FG, 255: Missing
    i64 stable_bg = 0;          // 11
    i64 gain = 0;               // 12
    i64 loss = 0;               // 21
    i64 stable_fg = 0;          // 22
    i64 missing = 0;            // 255
    i64 total_valid = 0;
};

// Compute Simple Change between T1 and T2 habitat rasters (0=missing, 1=bg, 2=fg).
SimpleChangeResult compute_simple_change(
    const ImageU8& im1,
    const ImageU8& im2
);

struct MorphologicalChangeResult {
    ImageU8 change_map;         // 11: Stable BG, 12: MCD Gain, 21: MCD Loss, 22: Stable FG, 176: Spurious, 255: Missing
    i64 fg_area_t1 = 0;
    i64 fg_area_t2 = 0;
    i64 net_change = 0;
    i64 stable_bg = 0;          // 11
    i64 stable_fg = 0;          // 22
    i64 mcd_gain = 0;           // 12
    i64 mcd_loss = 0;           // 21
    i64 spurious = 0;           // 176
    i64 missing = 0;            // 255
    double elasticity = 0.0;    // Rate of core change / Rate of total FG change
};

// Compute Morphological Change Detection (MCD) between T1 and T2 habitat rasters.
// Uses geodesic reconstruction from core transitions to filter out boundary edge chatter.
MorphologicalChangeResult compute_morphological_change(
    const ImageU8& im1,
    const ImageU8& im2
);

struct FOSChangeResult {
    ImageU8 change_map;         // Delta FOS shifted by 100, plus 250: Gain, 251: Loss, 252: Stable BG, 254: Missing
    i64 fg_gain = 0;            // 250
    i64 fg_loss = 0;            // 251
    i64 stable_bg = 0;          // 252
    i64 forcom = 0;             // Common foreground cover at both times
    i64 diag_same_class = 0;    // Common FG in the same fragmentation class
    i64 diff_class = 0;         // Common FG in different fragmentation classes
    double frag_decrease_pct = 0.0; // Above matrix diagonal (%)
    double frag_increase_pct = 0.0; // Below matrix diagonal (%)
    std::vector<std::vector<i64>> matrix_6x6; // 6x6 transition matrix [Rare, Patchy, Transitional, Dominant, Interior, Intact]
};

// Compute Fragmentation (FOS) Change between T1 and T2 FOS maps (0..100: FG FOS, 101: BG, 102/255: Missing).
FOSChangeResult compute_fos_change(
    const ImageU8& fos1,
    const ImageU8& fos2
);

// GTB "Change -> FAD Multiscale" pixel counts for one observation scale.
// Inputs are GTB FAD maps: 0..100 foreground FAD [%], 101 background,
// 102 missing, 105/106 special background. Classes: 0 background (101..149
// except 102), 1 rare (<10), 2 patchy (10-39), 3 transitional (40-59),
// 4 dominant (60-89), 5 interior (90-99), 6 intact (100). Pixels missing at
// either time, or with values >= 150, are not counted in the matrix.
struct FADChangeCounts {
    std::array<i64, 49> matrix{};  // matrix[7 * class_A + class_B]
    i64 fg_area_a = 0;             // pixels <= 100 in A
    i64 fg_area_b = 0;             // pixels <= 100 in B
    i64 forcom = 0;                // pixels <= 100 in both
    i64 fg_sum_a = 0;              // sum of the values <= 100 in A (for the mean FAD)
    i64 fg_sum_b = 0;
    i64 ru_area_a = 0;             // non-missing pixels (!= 102) in A (GTB reporting unit)
    i64 ru_area_b = 0;
};

FADChangeCounts compute_fad_change_counts(const ImageU8& a, const ImageU8& b);

}  // namespace gtb
