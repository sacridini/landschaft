// SPDX-License-Identifier: GPL-3.0-or-later
//
// Minimum sets with management zones and pair terms (boundary penalties, connectivity rewards with
// matrices of zones >= 0) in Marxan's form over the unit-zone columns j = z * n + i:
//
//     min  sum_j c_j x_j + sum_{(a,b)} w_ab [x_a != x_b]   (w_ab >= 0, pairs within or across zones)
//     s.t. A x >= t (the targets),  sum_z x_zi <= 1 (= 1 with mandatory allocation),  x binary, locks.
//
// The Lagrangian of the targets with the GUB rows dropped (a relaxation) is a binary energy with
// submodular pairs: one minimum s-t cut over all the columns (opt/maxflow.hpp) gives L(y), a valid bound;
// a box-step bundle method (opt/bundle.hpp) maximises it. Every few cuts the selection becomes a solution:
// a unit chosen in several zones keeps the one of least reduced cost, the targets left unmet are repaired
// greedily (the move of a unit into a zone with the most amount per cost, boundary changes included) and a
// local search moves units (out, into another zone) while that lowers the objective and keeps every target
// met. The result is the best solution and the best bound; it does not depend on the threads.
#pragma once

#include <string>
#include <vector>

#include "opt/milp.hpp"

namespace opt::zcut {

struct Input {
    i64 n = 0, nz = 0;                      // units and zones (columns: nz * n)
    std::vector<double> cost, lb, ub;       // per column (ub = 0: closed; lb = 1: locked in)
    bool mandatory = false;
    std::vector<i64> t_start, t_index;      // the targets by column (CSC over the target rows)
    std::vector<double> t_value, target;    // amounts (>= 0) and right-hand sides
    std::vector<int> pi, pj;                // pairs of columns
    std::vector<double> pw;                 // their weights (>= 0)
};

struct Options {
    double gap = 0.1;
    double time_limit = -1.0;  // seconds (< 0: none)
    int max_evaluations = 1000;
    bool verbose = false;
};

struct Result {
    std::string status;      // "optimal" (gap reached), "time_limit", "stalled", "infeasible"
    std::vector<double> x;   // the best solution (columns), empty if none
    double objective = 0.0;
    double bound = 0.0;
    double gap = 0.0;
    int evaluations = 0;
    double seconds = 0.0;
};

Result solve(const Input& in, const Options& options, const std::vector<double>& start = {});

}  // namespace opt::zcut
