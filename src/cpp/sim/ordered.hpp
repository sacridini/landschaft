// SPDX-License-Identifier: GPL-3.0-or-later
//
// The ordered allocation of Fuchs et al. (2013) with stochastic choice, as
// lulcc 1.0.4's OrderedModel runs it, with R's generator.
#pragma once

#include <cstdint>
#include <vector>

#include "nlm/rng.hpp"

namespace sim {

struct OrderedResult {
    std::vector<double> allocation;  // the class of each cell
    int iterations = 0;              // 0 when the map already meets the demand
    bool converged = true;
    std::int64_t short_cells = 0;    // changes the demand asked for that no cell could make
};

// tprob: ncell x ncode (row-major, NaN = not allowed); x the class of each
// cell; demand the cells of each class; order the classes in the order they
// are served (each of `classes` once). A class whose demand grows takes the
// cells of other classes in decreasing suitability, one whose demand shrinks
// releases its cells of lowest suitability. With `stochastic`, the cells are
// drawn by comparing their suitability with runif() over the ranked
// candidates (up to 1001 draws for enough of them, as lulcc); else the
// ranked cells are taken in order.
OrderedResult ordered_allocate(const double* tprob, const double* x, std::int64_t ncell, const double* demand,
                               const double* classes, int ncode, const double* order, bool stochastic,
                               nlm::RRng& rng);

}  // namespace sim
