// SPDX-License-Identifier: GPL-3.0-or-later
//
// Budgeted objectives with management zones (prioritizr's min shortfall, min largest shortfall, max
// utility, max features, max cover, phylogenetic objectives, min penalties with zones): few rows (targets,
// amounts, one budget or one per zone), up to millions of unit columns z * n + i (zone after zone) with one
// GUB row per unit (sum_z x_zi <= 1, = 1 with mandatory allocation) and a few other columns after them
// (shortfalls, amounts, targets met, branches). As opt/wide.cpp for one zone, with the GUB rows kept out of
// the pricing:
//
// 1. the LP by sifting over units: HiGHS solves it on a working set of units (their columns and GUB rows)
//    and every other column, the other units at their best shares for the duals (a continuous knapsack over
//    the unit's zones, as opt/zones.cpp); units whose best shares the duals change join, until none does:
//    the LP is then solved and its objective is the bound;
// 2. binary units: the fractional units leave (with mandatory allocation: their largest share), the other
//    columns are re-optimised for those units (a MIP on them alone), then units join one by one, the most
//    attractive reduced cost first, while every row allows it, and the better of the two is kept;
// 3. if the gap is still wider than asked for, HiGHS's MIP goes on from the solution after Lagrangian
//    fixing (a unit-zone column whose forcing raises the bound beyond the incumbent keeps its value), within
//    the time left and the size HiGHS keeps time on.
#pragma once

#include "opt/milp.hpp"

namespace opt {

// Whether the model has the form above (n_units units with one GUB row each; few other columns after
// the unit columns; finite bounds on the unit columns).
bool is_zone_wide(const Model& model, i64 n_units);

SolveResult solve_zone_wide(const Model& model, i64 n_units, const SolveOptions& options);

}  // namespace opt
