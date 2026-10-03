// SPDX-License-Identifier: GPL-3.0-or-later
//
// The CLUE-S allocation (Verburg et al. 2002) as lulcc's allocateclues runs
// it, and the focal mean of lulcc's neighbourhood maps (raster::focal).
#pragma once

#include <cstdint>
#include <vector>

#include "nlm/rng.hpp"

namespace sim {

struct CluesResult {
    std::vector<double> allocation;  // the class of each cell
    int iterations = 0;
    bool converged = false;
    double max_diff = 0.0;
    double mean_diff = 0.0;
};

// tprob: ncode x ncell, class-major (class j, cell i at j * ncell + i), NaN =
// not allowed; landuse: the class of each cell (a value of codes). Each pass
// jitters every allowed suitability by U(-jitter, jitter) (R's generator,
// class-major order), gives each cell the class of highest suitability plus
// its iteration variable (the first in case of ties; a cell with none keeps
// its class), and moves the iteration variables by scale * (demand -
// allocated) until the largest absolute difference is at most max_diff and
// their mean at most mean_diff, or after max_iter passes.
CluesResult clues_allocate(const double* tprob, const double* landuse, std::int64_t ncell, const double* demand,
                           const double* codes, int ncode, double jitter, double scale, double max_iter,
                           double max_diff, double mean_diff, nlm::RRng& rng);

// raster::focal(x, w, fun = mean, pad = TRUE, na.rm = TRUE): the mean of
// value * weight over the window cells with a value (NaN outside the map and
// for missing cells); NaN where none.
std::vector<double> focal_mean(const double* x, std::int64_t ny, std::int64_t nx, const double* w, int wy, int wx);

}  // namespace sim
