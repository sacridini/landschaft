// SPDX-License-Identifier: GPL-3.0-or-later
//
// Omniscape 0.7's moving-window circuits (run_omniscape): a target every
// block_size cells (where the block's sources add up to more than 0) gets a
// window of radius + buffer; the block's strength is injected into the
// sources within `radius` (outside the target block, in proportion to their
// strength), the target cell is a direct ground, and the node currents of
// that circuit are added to the cumulative map. Omniscape's arithmetic and
// its quirks are kept: windows and circles clipped at the map edges, the
// target block zeroed at radius + buffer from the window's corner (not at the
// target when the window is clipped on the left or top), the block-artifact
// correction and the flow potential on a window of conductance 1.
#pragma once

#include <cstdint>
#include <vector>

#include "circuit/grid.hpp"

namespace circuit {

struct OmniscapeParams {
    int radius = 0;
    int buffer = 0;
    int block_size = 1;  // odd
    bool four_neighbors = false;
    bool resistance_is_conductance = false;
    bool correct_artifacts = true;
    bool flow_potential = false;
};

struct OmniscapeResult {
    std::vector<double> current;         // cumulative current
    std::vector<double> flow_potential;  // empty unless asked for
    i64 n_targets = 0;
};

// `resistance`: row-major ny × nx, NaN = NoData (values must be > 0);
// `source`: source strengths, NaN = 0. Targets are solved in parallel and
// summed in a fixed order: the result does not depend on the threads.
OmniscapeResult omniscape(const double* resistance, const double* source, i64 nx, i64 ny,
                          const OmniscapeParams& params, int max_threads);

// Omniscape's calc_correction: the (2 (radius + buffer) + 1)² factors that
// undo the artifacts of solving blocks of targets together.
std::vector<double> omniscape_correction(const OmniscapeParams& params, i64 nx, i64 ny);

}  // namespace circuit
