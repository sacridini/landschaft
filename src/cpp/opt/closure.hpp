// SPDX-License-Identifier: GPL-3.0-or-later
//
// The closure engine: binary problems on nodes with requirements (z_v <= z_w: an action needs its
// planning unit monitored, a downstream action its upstream one), nonnegative pair penalties (the
// boundary of the selected units, of the actions against the same threat) and a few side rows with
// nonnegative coefficients (targets, ">=", or budgets, "<="):
//
//     min  sum_v c_v z_v + sum_(i,j) w_ij z_i (1 - z_j)
//     s.t. sum_v a_kv z_v >= t_k  (or <= t_k),  z_v <= z_w for the requirements,  some z fixed.
//
// Relaxing the side rows with multipliers y >= 0 leaves a closure problem with pair terms, solved
// exactly: per group in closed form when the requirements are stars (an action needs only its unit)
// and there are no pair terms, by a minimum s-t cut otherwise (the requirements as infinite arcs).
// Its value is a valid lower bound, as strong as the linear relaxation's (the inner problem is
// integral); the multipliers by the bundle method (opt/bundle.hpp). Every inner solution is repaired
// into a feasible one (budgets met by dropping the groups that cost least per amount freed, targets by
// adding the groups that cover most per cost, then the groups that improve the objective added and
// those that do not pay for themselves dropped), the best kept.
//
// Without pair terms and with star requirements, the engine also says which nodes no better solution
// can flip (the bound with the node forced to its other value is above the best solution): a model
// left to a MILP solver can fix them; and how much flipping each node costs the bound, from which a
// caller picks a core of the most promising nodes for a MILP solver. With integral costs and weights
// the bound is rounded up.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "opt/milp.hpp"

namespace opt::closure {

struct Problem {
    int n = 0;                          // nodes
    std::vector<double> cost;           // n
    std::vector<std::int8_t> fixed;     // n: -1 free, 0 or 1
    std::vector<int> req_from, req_to;  // z[from] <= z[to]
    std::vector<int> pen_i, pen_j;      // w z_i (1 - z_j)
    std::vector<double> pen_w;          // >= 0
    int m = 0;                          // side rows
    std::vector<double> rhs;            // m
    std::vector<std::uint8_t> geq;      // m: 1 ">=", 0 "<="
    std::vector<i64> col_start;         // n + 1: every node's row entries
    std::vector<int> col_row;
    std::vector<double> col_val;        // >= 0

    void check() const;  // throws std::invalid_argument
};

struct Options {
    double gap = 0.1;          // stop when (best - bound) / |best| is below it
    double time_limit = -1.0;  // seconds (< 0: none)
    int max_evaluations = 500;
    int threads = 1;
    bool verbose = false;
};

struct Result {
    std::string status;  // "optimal" (gap reached), "time_limit", "stalled", "infeasible", "no_solution"
    std::vector<std::uint8_t> z;
    double objective = kInf;
    double bound = -kInf;
    double gap = kInf;
    int evaluations = 0;
    double seconds = 0.0;
    std::vector<std::int8_t> fixable;  // n, when computed: -1, or the value no better solution changes
    // n, when computed (star requirements, no pair terms): every node's value in the inner minimum at
    // the best multipliers and what forcing it the other way adds to the bound (inf: fixed)
    std::vector<std::uint8_t> centre_z;
    std::vector<double> flip;
};

// initial: a selection to start from (empty: none)
Result solve(const Problem& p, const Options& o, const std::vector<std::uint8_t>& initial = {});

}  // namespace opt::closure
