// SPDX-License-Identifier: GPL-3.0-or-later
//
// The cut engine: minimum-set problems with nonnegative pair terms (prioritizr's
// boundary penalties, symmetric connectivity rewards) in Marxan's form,
//
//     min  sum_i c_i x_i + sum_{i<j} w_ij [x_i != x_j]   (w_ij >= 0)
//     s.t. sum_i a_ki x_i >= t_k  for every feature k,  x binary, locked units fixed,
//
// by Lagrangian relaxation of the targets: for multipliers y >= 0,
//
//     L(y) = y't + min_x  sum_i (c_i - (A'y)_i) x_i + sum_{i<j} w_ij [x_i != x_j],
//
// whose inner problem is a minimum s-t cut (exact: the pair terms are
// submodular), so max L is the linear relaxation's bound or better. The
// multipliers by a box-step bundle method (as opt/wide.cpp), one maximum flow
// per evaluation; every cut's selection is repaired to meet the targets and
// trimmed (marxan::Repair), the best one kept. The repairs run on other threads
// while the bundle goes on; its decisions use the repairs of the evaluations up
// to a fixed number back, so the result is the same on any number of threads.
#pragma once

#include <string>
#include <vector>

#include "opt/marxan.hpp"

namespace opt::cut {

struct Options {
    double gap = 0.1;          // stop when (best - bound) / best is below it
    double time_limit = -1.0;  // seconds (< 0: none)
    int max_evaluations = 1000;
    bool verbose = false;
    int threads = 1;           // repairs at the same time
};

struct Result {
    std::string status;      // "optimal" (gap reached), "time_limit", "stalled", "infeasible"
    std::vector<int> x;      // the best selection (0/1)
    double objective = 0.0;  // its value (Marxan's score at BLM 1)
    double bound = 0.0;      // the best Lagrangian bound
    double gap = 0.0;
    int evaluations = 0;
    double seconds = 0.0;
};

// data: Marxan's problem with BLM 1 (cost, status 0/2/3, features by unit, targets; fixed boundaries and
// symmetric neighbour lists with boundaries >= 0); initial: a selection to start from (empty: none)
Result solve(const marxan::Data& data, const Options& options, const std::vector<int>& initial = {});

}  // namespace opt::cut
