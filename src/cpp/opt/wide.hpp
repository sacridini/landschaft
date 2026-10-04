// SPDX-License-Identifier: GPL-3.0-or-later
//
// Wide models with a budget (prioritizr's min shortfall, min largest
// shortfall, max utility, max features, max cover, ...): a few rows (one or
// two per target and the budget), up to millions of "unit" columns (the
// planning units, the first n_units columns, with finite bounds) and a few
// other columns (shortfalls, amounts, targets met: always kept). HiGHS's MIP
// stalls on such models for minutes without looking at the time (IDEAS.md);
// here:
//
// 1. the LP relaxation by sifting. Wide models first get near-optimal duals:
//    the LP on an evenly spaced 2% sample of the units (its row bounds scaled
//    by the share of each row's amount it holds), then a proximal bundle
//    method on the Lagrangian (as cover.cpp; the signs of the duals follow the
//    rows' senses, and the other columns with an infinite bound are linear
//    constraints of its master), with at most 40% of the time left. Then
//    HiGHS solves the LP on the other columns and a working set of units,
//    kept in one instance from round to round (columns added and deleted,
//    row bounds changed: it goes on from its basis), every other unit at the
//    bound its reduced cost points to; all units are priced with the duals
//    (OpenMP, one column each: the same bits on any number of threads) and
//    those whose bound breaks optimality join the working set, until none
//    does: the LP is then solved and its objective is a valid bound;
// 2. integer units: the LP solution rounded down, and rounded up then
//    trimmed (the units the rows do not need dropped, those weighing most on
//    the budget first), the other columns re-optimised for those units (a MIP
//    on the few other columns). When that leaves a wider gap than asked:
//    the same from the cheapest LP solution (the other columns at their LP
//    values, the least spent: the LP's optimum may spend the whole budget);
//    with binary other columns (targets met, features present), a dive: they
//    are ranked by LP value, the first k fixed at 1 and the rest at 0, the
//    cheapest LP for that rounded up and trimmed, the largest k that fits
//    found by binary search; then units join one by one, the most attractive
//    reduced cost first, while the rows allow it and the objective improves;
// 3. if the gap is still wider than asked, the columns whose reduced cost
//    exceeds it keep their LP bound (exact) and HiGHS's MIP goes on from the
//    solution on the rest (with a time limit, only when at most 1e5 columns
//    are left, as in cover.hpp).
//
// Out of time before the LP is solved: the restricted LP's solution (or the
// units at their lower bounds) rounded the same way, with the best
// Lagrangian value as the bound.
#pragma once

#include "opt/milp.hpp"

namespace opt {

// Units [0, n_units) with finite bounds, other columns at most kMaxOther.
bool is_wide(const Model& model, i64 n_units);

// Solves a wide model (see above); options as solve_highs (threads: the
// pricing's threads; HiGHS runs on one).
SolveResult solve_wide(const Model& model, i64 n_units, const SolveOptions& options);

}  // namespace opt
